"""
EXPERIMENTAL BLOCKER — NOT PRODUCTION. DO NOT WIRE INTO main.py.
================================================================

Status
------
Isolated experiment for benchmarking only. Production remains src/blocking.py
(country partition -> streaming TF-IDF -> sp_matmul_topn top-K, K=30,
threshold=0.05). No production file imports this module.

Motivation
----------
Production multiplies every S1 chunk against EVERY target chunk of a country,
so its sparse-matmul work scales O(n_s1 x n_targets) even though, at cosine
threshold 0.05, only a tiny fraction of pairs can ever qualify. This module
tests whether a cheap prefilter can shrink the set of target rows that get
transformed and multiplied, WITHOUT losing true pairs.

Design (V1)
-----------
Per country, two passes over the preprocessed target file:

Pass 1 (index build, streaming):
  - collect target texts + entity ids
  - word-token inverted index: token -> ascending row ids (array('q'));
    tokens with document frequency > WORD_BUCKET_MAX are NOT indexed
    (IDF signal too weak, mirrors blocking_old.py intent)
  - char 3-gram document frequency over the first CHAR3_SAMPLE_ROWS rows;
    grams with df <= CHAR3_DF_MAX form "rare_gram_vocab"
  - fit a TfidfVectorizer with IDENTICAL hyperparameters to production
    (char_wb, (3,4), max_features=50000, sublinear_tf, float32) on the same
    rng(42)-selected min(250k, n) sample of the same ordered rows, so the
    fitted vocabulary matches production's disk-backed benchmark harness.

Pass 2 (prefilter -> rerank, chunked):
  - per target chunk, build a gram->positions index restricted to grams that
    appear in the S1 side (s1_gram_union) and in rare_gram_vocab
  - per S1 entity, keep-set = target positions that share
      (a) an indexed word token, or
      (b) a rare char 3-gram
  - safety valves:
      * S1 with an empty text is inactive (production gives it no candidates
        anyway: zero vector, everything below threshold)
      * S1 with NO indexed token and NO rare gram escapes -> full-chunk rerank
        for that S1 (production-parity path, counted in stats)
      * keep-set larger than PREFILTER_UNION_CAP escapes likewise
  - targets to transform = union of keep-sets (or full chunk if any escape);
    if that union is larger than KEEP_TRANSFORM_RATIO of the chunk, the full
    chunk is transformed instead (masking still applies)
  - sp_matmul_topn(s1_matrix, kept_T, top_n=top_k, threshold=tfidf_threshold)
    — identical maths to production — then per-S1 masking drops pairs outside
    that S1's keep-set, and the same _merge_topk accumulation as production
    builds the global top-K.

Because the prefilter can only REMOVE target rows before an identical
scoring step, experimental pairs are a subset of production pairs apart from
top-K tie-flipping at chunk boundaries. Whether the removed rows contained
true matches (>= 0.05 cosine but no shared indexed token / rare 3-gram) is an
EMPIRICAL question — the whole point of benchmark_experimental.py. Nothing in
this file has been measured on the real dataset yet.

Known V1 limitation (next iteration if benchmarks justify it)
-------------------------------------------------------------
All surviving targets of a chunk are multiplied against all S1 rows and
masked afterwards, so the matmul itself is not yet shrunk — only the
transform step and the multiplied column count are. A V2 would group S1 rows
by overlapping keep-sets and multiply group-by-group. Measure first.
"""

import gc
import os
import time
from array import array

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tunables (experiment-local; production values in src/blocking.py untouched)
# ---------------------------------------------------------------------------

WORD_BUCKET_MAX = 2000        # token df over the country; above -> not indexed
CHAR3_SAMPLE_ROWS = 50_000    # rows used to estimate char-3-gram document freq
CHAR3_DF_MAX = 500            # gram df (within sample) above -> not a signal
PREFILTER_UNION_CAP = 1_000_000   # per-S1 keep-set size above -> escape
TFIDF_FIT_SAMPLE = 250_000    # same as production's fit sample size
KEEP_TRANSFORM_RATIO = 0.6    # transform full chunk if union exceeds this share
S1_CHUNK_SIZE = 5000          # same as production

_STOP_WORDS = {
    "the", "and", "of", "in", "for", "a", "on", "at",
    "inc", "ltd", "llc", "co", "corp", "corporation",
    "limited", "private", "pvt", "company", "group",
    "st", "street", "rd", "road", "ave", "avenue", "blvd",
    "floor", "room", "suite", "building", "no", "number",
}

_INACTIVE = object()   # S1 with empty text: production also yields nothing
_ESCAPE = None         # rerank the full chunk for this S1


def _merge_topk(existing_indices, existing_scores, new_indices, new_scores, top_k):
    """Byte-for-byte semantics of src/blocking.py::_merge_topk."""
    if len(new_indices) == 0:
        return existing_indices, existing_scores
    if len(existing_indices) == 0:
        combined_idx = new_indices
        combined_sc = new_scores
    else:
        combined_idx = np.concatenate([existing_indices, new_indices])
        combined_sc = np.concatenate([existing_scores, new_scores])
    if len(combined_idx) <= top_k:
        return combined_idx, combined_sc
    partition = np.argpartition(combined_sc, -top_k)[-top_k:]
    return combined_idx[partition], combined_sc[partition]


def _word_tokens(text):
    if not text:
        return ()
    return tuple(t for t in text.split() if len(t) > 2 and t not in _STOP_WORDS)


def _iter_char3(text):
    """char_wb-style padded 3-grams of one text (space-joined words)."""
    s = " " + text.strip() + " "
    n = len(s)
    if n < 3:
        return
    for i in range(n - 2):
        yield s[i:i + 3]


def generate_candidates_experimental(
    df_source1,
    target_country_files,      # {country: {"path": tsv, "count": int}}
    top_k=30,
    tfidf_threshold=0.05,
    target_chunk_size=100_000,
):
    """
    Two-stage experimental blocker: inverted-token/char-3-gram prefilter ->
    production-identical TF-IDF rerank.

    Returns (candidates_df[source1_id, target_id, blocking_score], stats dict).
    """
    t_start = time.perf_counter()

    stats = {
        "targets_seen": 0,
        "s1_input": 0,
        "s1_active": 0,
        "s1_escape_no_signal": 0,
        "full_chunk_escapes": 0,
        "pair_window_total": 0,        # active S1 x targets (upper bound)
        "pair_kept_prefilter": 0,      # pairs surviving the prefilter
        "rerank_pairs_evaluated": 0,   # pairs >= tfidf_threshold pre-masking
        "pairs_emitted": 0,            # pairs merged into top-K buffers
        "avg_transform_ratio": None,
        "country_seconds": {},
    }

    all_s1_ids, all_target_ids, all_scores = [], [], []
    n_threads = max(1, (os.cpu_count() or 4) - 1)

    countries = [
        c for c in df_source1["clean_country"].dropna().unique()
        if c in target_country_files and target_country_files[c]["count"] > 0
    ]

    for country in countries:
        t_country = time.perf_counter()
        path = target_country_files[country]["path"]

        s1_country = df_source1[df_source1["clean_country"] == country]
        if len(s1_country) == 0:
            continue

        s1_texts = (
            s1_country["clean_name"].fillna("")
            + " "
            + s1_country["clean_address"].fillna("")
        ).values
        s1_ids = s1_country["entity_id"].values
        n_s1 = len(s1_texts)
        stats["s1_input"] += n_s1

        print(f"\n[{country}] S1={n_s1:,}  targets_on_disk="
              f"{target_country_files[country]['count']:,}")

        # ---------------------------------------------------------------
        # Pass 1: stream file once — texts, entity ids, indexes
        # ---------------------------------------------------------------
        texts = []
        entity_ids = []
        word_index = {}
        gram_df = {}
        n_targets = 0

        reader = pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False, chunksize=500_000
        )
        for chunk in reader:
            t_col = (
                chunk["clean_name"].fillna("")
                + " "
                + chunk["clean_address"].fillna("")
            ).values
            for t, e in zip(t_col, chunk["entity_id"].values):
                texts.append(t)
                entity_ids.append(e)
                for tok in _word_tokens(t):
                    arr = word_index.get(tok)
                    if arr is None:
                        word_index[tok] = array("q", [n_targets])
                    else:
                        arr.append(n_targets)
                if n_targets < CHAR3_SAMPLE_ROWS:
                    for g in _iter_char3(t):
                        gram_df[g] = gram_df.get(g, 0) + 1
                n_targets += 1
        del reader

        stats["targets_seen"] += n_targets

        rare_vocab = {g for g, c in gram_df.items() if c <= CHAR3_DF_MAX}
        del gram_df

        frozen_index = {
            tok: np.frombuffer(arr, dtype=np.int64)
            for tok, arr in word_index.items()
            if len(arr) <= WORD_BUCKET_MAX
        }
        del word_index
        gc.collect()

        # ---------------------------------------------------------------
        # Vectorizer: identical hyperparameters + identical fit sample
        # (rng(42).choice over the same ordered rows) as production's
        # disk-backed harness -> same vocabulary -> comparable scores.
        # ---------------------------------------------------------------
        rng = np.random.default_rng(42)
        sample_size = min(TFIDF_FIT_SAMPLE, n_targets)
        if sample_size < n_targets:
            sample_idx = rng.choice(n_targets, sample_size, replace=False)
        else:
            sample_idx = np.arange(n_targets)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            max_features=50000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        vectorizer.fit([texts[i] for i in sample_idx])
        del sample_idx
        gc.collect()

        # ---------------------------------------------------------------
        # S1-side signal prep
        # ---------------------------------------------------------------
        s1_token_lists = [_word_tokens(t) for t in s1_texts]
        s1_gram_sets = []
        for t in s1_texts:
            gs = set()
            for g in _iter_char3(t):
                if g in rare_vocab:
                    gs.add(g)
            s1_gram_sets.append(gs)

        s1_gram_union = set()
        for gs in s1_gram_sets:
            s1_gram_union |= gs

        s1_active = np.array([len(t.strip()) > 0 for t in s1_texts])
        stats["s1_active"] += int(s1_active.sum())

        s1_escape = np.empty(n_s1, dtype=bool)
        for i in range(n_s1):
            has_tok = any(tok in frozen_index for tok in s1_token_lists[i])
            s1_escape[i] = (not has_tok) and (len(s1_gram_sets[i]) == 0)
        stats["s1_escape_no_signal"] += int(s1_escape.sum())

        s1_matrices = [
            vectorizer.transform(s1_texts[a:a + S1_CHUNK_SIZE])
            for a in range(0, n_s1, S1_CHUNK_SIZE)
        ]

        entity_arr = np.array(entity_ids, dtype=object)
        del entity_ids
        gc.collect()

        # ---------------------------------------------------------------
        # Pass 2: prefilter + rerank, chunk by chunk
        # ---------------------------------------------------------------
        best_indices = [np.empty(0, dtype=np.int64) for _ in range(n_s1)]
        best_scores = [np.empty(0, dtype=np.float32) for _ in range(n_s1)]

        n_chunks = (n_targets + target_chunk_size - 1) // target_chunk_size
        chunk_offset = 0
        ratio_sum, ratio_n = 0.0, 0

        for _ in tqdm(range(n_chunks), desc=f"prefilter+rerank [{country}]"):
            start = chunk_offset
            end = min(start + target_chunk_size, n_targets)
            clen = end - start
            chunk_texts = texts[start:end]

            # gram -> positions, restricted to grams the S1 side can query
            cgi = {}
            for pos in range(clen):
                t = chunk_texts[pos]
                s = " " + t.strip() + " "
                for k in range(len(s) - 2):
                    g = s[k:k + 3]
                    if g in s1_gram_union:
                        lst = cgi.get(g)
                        if lst is None:
                            cgi[g] = [pos]
                        else:
                            lst.append(pos)

            # per-S1 keep-sets for this chunk
            keeps = [_INACTIVE] * n_s1
            escape_in_chunk = bool(s1_escape.any())
            for i in range(n_s1):
                if not s1_active[i]:
                    continue
                if s1_escape[i]:
                    keeps[i] = _ESCAPE
                    continue
                keep = set()
                for tok in s1_token_lists[i]:
                    arr = frozen_index.get(tok)
                    if arr is not None:
                        lo = np.searchsorted(arr, start, side="left")
                        hi = np.searchsorted(arr, end, side="left")
                        if hi > lo:
                            keep.update((arr[lo:hi] - start).tolist())
                for g in s1_gram_sets[i]:
                    lst = cgi.get(g)
                    if lst is not None:
                        keep.update(lst)
                if len(keep) > PREFILTER_UNION_CAP:
                    keeps[i] = _ESCAPE
                    escape_in_chunk = True
                    stats["full_chunk_escapes"] += 1
                else:
                    keeps[i] = keep

            if escape_in_chunk:
                kept_pos = np.arange(clen, dtype=np.int64)
            else:
                union_all = set()
                for k in keeps:
                    if k is not _INACTIVE and k is not _ESCAPE:
                        union_all |= k
                kept_pos = np.fromiter(
                    union_all, dtype=np.int64, count=len(union_all)
                )
                kept_pos.sort()

            ratio_sum += len(kept_pos) / clen if clen else 1.0
            ratio_n += 1

            active_rows = sum(
                1 for k in keeps if k is not _INACTIVE
            )
            kept_pairs = sum(
                clen if k is _ESCAPE else len(k)
                for k in keeps if k is not _INACTIVE
            )
            stats["pair_window_total"] += active_rows * clen
            stats["pair_kept_prefilter"] += kept_pairs

            if len(kept_pos) == 0:
                chunk_offset = end
                del cgi, keeps, chunk_texts
                continue

            kept_texts = [chunk_texts[p] for p in kept_pos.tolist()]
            kept_matrix = vectorizer.transform(kept_texts)
            kept_T = kept_matrix.T.tocsr()
            del kept_matrix, kept_texts

            for ck in range(len(s1_matrices)):
                a = ck * S1_CHUNK_SIZE
                b = min(a + S1_CHUNK_SIZE, n_s1)
                sim = sp_matmul_topn(
                    s1_matrices[ck],
                    kept_T,
                    top_n=top_k,
                    threshold=tfidf_threshold,
                    sort=True,
                    n_threads=n_threads,
                ).tocsr()
                stats["rerank_pairs_evaluated"] += int(sim.nnz)

                for row in range(b - a):
                    gi = a + row
                    keep = keeps[gi]
                    if keep is _INACTIVE:
                        continue
                    rs, re = sim.indptr[row], sim.indptr[row + 1]
                    if rs == re:
                        continue
                    cols = sim.indices[rs:re]
                    scores = sim.data[rs:re]
                    pos = kept_pos[cols]

                    if keep is _ESCAPE:
                        sel = slice(None)
                    elif len(keep) == 0:
                        continue
                    else:
                        pos_list = pos.tolist()
                        sel = [p in keep for p in pos_list]
                        if not any(sel):
                            continue
                        sel = np.array(sel, dtype=bool)

                    gcols = pos[sel] + start
                    gsc = scores[sel].astype(np.float32)
                    stats["pairs_emitted"] += len(gcols)

                    best_indices[gi], best_scores[gi] = _merge_topk(
                        best_indices[gi],
                        best_scores[gi],
                        gcols.astype(np.int64),
                        gsc,
                        top_k,
                    )

                del sim
            del kept_T, cgi, keeps, chunk_texts
            gc.collect()
            chunk_offset = end

        # ---------------------------------------------------------------
        # Emit country candidates
        # ---------------------------------------------------------------
        for i in range(n_s1):
            if len(best_indices[i]) == 0:
                continue
            sid = s1_ids[i]
            for gid, sc in zip(
                best_indices[i].tolist(), best_scores[i].tolist()
            ):
                eid = entity_arr[int(gid)]
                all_s1_ids.append(sid)
                all_target_ids.append(eid)
                all_scores.append(sc)

        del texts, entity_arr, s1_matrices, best_indices, best_scores
        del vectorizer, s1_token_lists, s1_gram_sets
        gc.collect()

        stats["country_seconds"][country] = round(
            time.perf_counter() - t_country, 1
        )
        print(f"[{country}] done in {stats['country_seconds'][country]}s")

    stats["avg_transform_ratio"] = (
        round(ratio_sum / ratio_n, 4) if ratio_n else None
    )
    stats["prefilter_keep_rate"] = (
        round(stats["pair_kept_prefilter"] / stats["pair_window_total"], 6)
        if stats["pair_window_total"] else None
    )
    stats["wall_clock_s"] = round(time.perf_counter() - t_start, 2)

    candidates_df = pd.DataFrame(
        {
            "source1_id": all_s1_ids,
            "target_id": all_target_ids,
            "blocking_score": all_scores,
        }
    )
    candidates_df.drop_duplicates(
        subset=["source1_id", "target_id"], inplace=True
    )
    return candidates_df, stats
