"""
benchmark_experimental.py — Kaggle benchmark: PRODUCTION vs EXPERIMENTAL blocker
=================================================================================

Controlled A/B on training data with ground truth. No production file is
imported or modified. Run inside a Kaggle notebook (>= 30 GB RAM) or any
machine that can hold the full preprocessed target set in memory.

Usage:
    python benchmark_experimental.py --n-s1 5000   --phases train
    python benchmark_experimental.py --n-s1 50000  --phases train
    python benchmark_experimental.py --n-s1 10000  --phases test-speed

Outputs benchmark_experimental_results.csv + per-config recall printed.
"""

import argparse
import gc
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import TRAIN_FILES, TEST_FILES
from src.preprocess import preprocess_dataframe
from src.blocking_experimental import generate_candidates_experimental

TARGET_CHUNK_FOR_PROD = 100_000


# ===========================================================================
# Shared: disk-backed target preparation (same layout for both arms)
# ===========================================================================

def preprocess_targets_to_disk(files, countries_set, tag):
    tmp_dir = tempfile.gettempdir()
    country_info = {}
    for c in countries_set:
        path = os.path.join(tmp_dir, f"bench_targets_{tag}_{c}.tsv")
        country_info[c] = {"path": path, "count": 0, "header_written": False}

    for sp in [files["source2"], files["source3"]]:
        reader = pd.read_csv(sp, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=500_000)
        for chunk in reader:
            if "country" not in chunk.columns:
                continue
            cc = chunk["country"].fillna("").astype(str).str.strip().str.lower()
            for country in countries_set:
                sub = chunk[cc == country]
                if len(sub) == 0:
                    continue
                sub = preprocess_dataframe(sub.copy())
                out = sub[["entity_id", "clean_name", "clean_address"]].copy()
                info = country_info[country]
                hdr = not info["header_written"]
                out.to_csv(info["path"], sep="\t", index=False,
                           mode="w" if hdr else "a", header=hdr)
                info["header_written"] = True
                info["count"] += len(out)
            del chunk
            gc.collect()
    for c, info in country_info.items():
        print(f"  [{tag}] '{c}': {info['count']:,} targets -> {info['path']}")
    return country_info


def load_s1_sample(files, n_sample, random_state=42):
    df = pd.read_csv(files["source1"], sep="\t", dtype=str,
                     keep_default_na=False)
    n = min(n_sample, len(df))
    s = df.sample(n=n, random_state=random_state).copy()
    del df
    gc.collect()
    return preprocess_dataframe(s)


def load_gt_pairs():
    gt = pd.read_csv(TRAIN_FILES["gt"], sep="\t", dtype=str,
                     keep_default_na=False)
    pairs = set()
    for row in gt.itertuples():
        s1 = row.source1_entity_id
        for m in str(row.matched_entity_ids).split(","):
            m = m.strip()
            if m:
                pairs.add((s1, m))
    del gt
    gc.collect()
    return pairs


def compute_recall(cands_df, gt_pairs, sampled_ids):
    cand = set(zip(cands_df["source1_id"], cands_df["target_id"]))
    sset = set(sampled_ids)
    relevant = {p for p in gt_pairs if p[0] in sset}
    total = len(relevant)
    captured = relevant & cand
    missed = relevant - cand

    res = {
        "recall": round(len(captured) / total, 6) if total else 0.0,
        "total_gt": total,
        "captured": len(captured),
        "missed": len(missed),
    }
    return res


# ===========================================================================
# Arm A: production blocker semantics, in-process (identical parameters)
# ===========================================================================

def _merge_topk(ei, es, ni, ns, top_k):
    if len(ni) == 0:
        return ei, es
    if len(ei) == 0:
        ci, cs = ni, ns
    else:
        ci = np.concatenate([ei, ni])
        cs = np.concatenate([es, ns])
    if len(ci) <= top_k:
        return ci, cs
    p = np.argpartition(cs, -top_k)[-top_k:]
    return ci[p], cs[p]


def run_production_inprocess(df_s1, target_country_info,
                             top_k=30, threshold=0.05,
                             s1_chunk_size=5000, target_chunk_size=100_000):
    """
    Exact re-implementation of src/blocking.py::generate_candidates control
    flow, reading the same preprocessed per-country target files from disk
    instead of a preloaded dataframe. Parameters are production's.
    Returns (candidates_df, runtime_s).
    """
    t0 = time.perf_counter()
    n_threads = max(1, (os.cpu_count() or 4) - 1)
    all_s1, all_t, all_s = [], [], []

    for country in df_s1["clean_country"].dropna().unique():
        info = target_country_info.get(country)
        if not info or info["count"] == 0:
            continue
        s1c = df_s1[df_s1["clean_country"] == country]
        if len(s1c) == 0:
            continue
        s1_texts = (s1c["clean_name"].fillna("") + " "
                    + s1c["clean_address"].fillna("")).values
        s1_ids = s1c["entity_id"].values
        n_s1 = len(s1_texts)

        # TF-IDF fit: identical to blocking.py (rng(42) over ordered rows,
        # sample 250k) — matches benchmark_blocker.py's disk-backed arm.
        texts, ids = [], []
        reader = pd.read_csv(info["path"], sep="\t", dtype=str,
                             keep_default_na=False, chunksize=500_000)
        for chunk in reader:
            texts.extend((chunk["clean_name"].fillna("") + " "
                          + chunk["clean_address"].fillna("")).values.tolist())
            ids.extend(chunk["entity_id"].values.tolist())
        del reader
        target_texts = np.array(texts)
        target_ids = np.array(ids, dtype=object)
        del texts, ids
        gc.collect()

        sample_size = min(250_000, len(target_texts))
        rng = np.random.default_rng(42)
        idx = (rng.choice(len(target_texts), sample_size, replace=False)
               if sample_size < len(target_texts)
               else np.arange(len(target_texts)))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                              max_features=50000, sublinear_tf=True,
                              dtype=np.float32)
        vec.fit(target_texts[idx])
        del idx
        gc.collect()

        s1_matrices = [vec.transform(s1_texts[a:a + s1_chunk_size])
                       for a in range(0, n_s1, s1_chunk_size)]
        best_i = [np.empty(0, dtype=np.int64) for _ in range(n_s1)]
        best_s = [np.empty(0, dtype=np.float32) for _ in range(n_s1)]

        for t_start in range(0, len(target_texts), target_chunk_size):
            t_end = min(t_start + target_chunk_size, len(target_texts))
            tm = vec.transform(target_texts[t_start:t_end])
            tmT = tm.T.tocsr()
            del tm
            for ck, s1_start in enumerate(range(0, n_s1, s1_chunk_size)):
                s1_end = min(s1_start + s1_chunk_size, n_s1)
                block = sp_matmul_topn(
                    s1_matrices[ck], tmT, top_n=top_k, threshold=threshold,
                    sort=True, n_threads=n_threads).tocsr()
                for row in range(s1_end - s1_start):
                    rs, re = block.indptr[row], block.indptr[row + 1]
                    if rs == re:
                        continue
                    cols = block.indices[rs:re].astype(np.int64) + t_start
                    scs = block.data[rs:re].astype(np.float32)
                    gi = s1_start + row
                    best_i[gi], best_s[gi] = _merge_topk(
                        best_i[gi], best_s[gi], cols, scs, top_k)
            del tmT
            gc.collect()

        for i in range(n_s1):
            for gid, sc in zip(best_i[i].tolist(), best_s[i].tolist()):
                all_s1.append(s1_ids[i])
                all_t.append(target_ids[int(gid)])
                all_s.append(float(sc))
        del s1_matrices, best_i, best_s, target_texts, target_ids, vec
        gc.collect()

    cdf = pd.DataFrame({"source1_id": all_s1, "target_id": all_t,
                        "blocking_score": all_s})
    cdf.drop_duplicates(subset=["source1_id", "target_id"], inplace=True)
    return cdf, round(time.perf_counter() - t0, 1)


# ===========================================================================
# Phases
# ===========================================================================

def phase_train(df_s1, country_info, gt_pairs, sampled_ids):
    rows = []

    print("\n>>> Arm A: production semantics (K=30, t=0.05)")
    cands_a, t_a = run_production_inprocess(df_s1, country_info)
    rec_a = compute_recall(cands_a, gt_pairs, sampled_ids)
    rows.append({
        "arm": "production", "runtime_s": t_a, "pairs": len(cands_a),
        "avg_per_s1": round(len(cands_a) / max(1, sampled_ids.shape[0]), 2),
        **rec_a,
    })
    print(f"    runtime={t_a}s pairs={len(cands_a):,} "
          f"recall={rec_a['recall']}")
    del cands_a
    gc.collect()

    print("\n>>> Arm B: experimental prefilter+rerank (K=30, t=0.05)")
    cands_b, stats_b = generate_candidates_experimental(df_s1, country_info)
    rec_b = compute_recall(cands_b, gt_pairs, sampled_ids)
    rows.append({
        "arm": "experimental", "runtime_s": stats_b["wall_clock_s"],
        "pairs": len(cands_b),
        "avg_per_s1": round(len(cands_b) / max(1, sampled_ids.shape[0]), 2),
        **rec_b,
        "prefilter_keep_rate": stats_b["prefilter_keep_rate"],
        "s1_escape_no_signal": stats_b["s1_escape_no_signal"],
        "full_chunk_escapes": stats_b["full_chunk_escapes"],
    })
    print(f"    runtime={stats_b['wall_clock_s']}s pairs={len(cands_b):,} "
          f"recall={rec_b['recall']} keep_rate={stats_b['prefilter_keep_rate']}")
    del cands_b
    gc.collect()

    # Paired comparison: relationships lost / gained
    return rows


def phase_test_speed(df_s1_test, country_info_test):
    rows = []
    cands_a, t_a = run_production_inprocess(df_s1_test, country_info_test)
    rows.append({"arm": "production_test", "runtime_s": t_a,
                 "pairs": len(cands_a)})
    del cands_a
    gc.collect()

    cands_b, stats_b = generate_candidates_experimental(df_s1_test,
                                                        country_info_test)
    rows.append({"arm": "experimental_test", "runtime_s": stats_b["wall_clock_s"],
                 "pairs": len(cands_b)})
    del cands_b
    gc.collect()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=5000)
    ap.add_argument("--phases", type=str, default="train",
                    choices=["train", "test-speed", "both"])
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    do_train = args.phases in ("train", "both")
    do_test = args.phases in ("test-speed", "both")
    results = []

    if do_train:
        df_s1 = load_s1_sample(TRAIN_FILES, args.n_s1, args.random_state)
        countries = set(df_s1["clean_country"].dropna().unique())
        country_info = preprocess_targets_to_disk(TRAIN_FILES, countries, "tr")
        gt_pairs = load_gt_pairs()
        sampled_ids = df_s1["entity_id"].values
        sampled_set = set(sampled_ids)
        print(f"GT relationships relevant to sample: "
              f"{sum(1 for s, _ in gt_pairs if s in sampled_set):,}")
        results += phase_train(df_s1, country_info, gt_pairs, sampled_ids)
        for c, info in country_info.items():
            try:
                os.remove(info["path"])
            except OSError:
                pass
        del df_s1, gt_pairs, country_info
        gc.collect()

    if do_test:
        df_s1_t = load_s1_sample(TEST_FILES, args.n_s1, args.random_state)
        countries_t = set(df_s1_t["clean_country"].dropna().unique())
        info_t = preprocess_targets_to_disk(TEST_FILES, countries_t, "te")
        results += phase_test_speed(df_s1_t, info_t)
        for c, info in info_t.items():
            try:
                os.remove(info["path"])
            except OSError:
                pass

    out = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_experimental_results.csv")
    out.to_csv(out_path, index=False)
    print("\n=== RESULTS ===")
    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
