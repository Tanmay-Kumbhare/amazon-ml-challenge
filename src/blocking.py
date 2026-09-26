import pandas as pd
import numpy as np
import os
import tempfile
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn


# -----------------------------------------------------------------
# Streaming helper: merge two top-K sets per S1 entity
# -----------------------------------------------------------------

def _merge_topk(
    existing_indices,
    existing_scores,
    new_indices,
    new_scores,
    top_k,
):
    """
    For one S1 entity, merge its current best candidates with
    newly discovered candidates and retain only the top-K by score.

    Parameters
    ----------
    existing_indices : np.ndarray   – current best target indices
    existing_scores  : np.ndarray   – corresponding scores
    new_indices      : np.ndarray   – new target indices from this chunk
    new_scores       : np.ndarray   – corresponding scores
    top_k            : int

    Returns
    -------
    (top_indices, top_scores) – both np.ndarray, length <= top_k
    """
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

    # Partial argsort: keep top_k largest scores
    partition = np.argpartition(combined_sc, -top_k)[-top_k:]
    return combined_idx[partition], combined_sc[partition]


# -----------------------------------------------------------------
# Main candidate generation
# -----------------------------------------------------------------

def generate_candidates(
    df_source1,
    df_targets,
    top_k=30,
    s1_chunk_size=5000,
    target_chunk_size=100000,
):
    """
    Memory-safe STREAMING TF-IDF candidate generation.

    Pipeline per country:

        fit TF-IDF on target sample
            ↓
        for each S1 chunk:
            transform S1 chunk  (small matrix, kept in memory)
            initialise empty top-K accumulators
                ↓
            for each target chunk:
                transform target chunk  (moderate matrix)
                sp_matmul_topn → per-S1 top-K from this target chunk
                merge into running global top-K for each S1
                discard target chunk matrix
                ↓
            emit global top-K for this S1 chunk
            discard S1 chunk matrix

    Memory ceiling at any instant:
        one S1 chunk matrix   (s1_chunk_size × vocab)
      + one target chunk matrix (target_chunk_size × vocab)
      + small top-K arrays    (s1_chunk_size × top_k)
    """

    print(
        f"Generating top-{top_k} candidates "
        "using Streaming Sparse TF-IDF..."
    )

    countries = df_source1["clean_country"].dropna().unique()

    chunk_files = []
    total_pairs = 0

    n_threads = max(1, (os.cpu_count() or 4) - 1)

    for country in countries:

        print("\n=======================================================")
        print(f"Processing country: '{country}'")
        print("=======================================================")

        s1_country = df_source1[
            df_source1["clean_country"] == country
        ].copy()

        target_country = df_targets[
            df_targets["clean_country"] == country
        ].copy()

        if len(s1_country) == 0 or len(target_country) == 0:
            print("  No S1/target rows for this country. Skipping.")
            continue

        s1_texts = (
            s1_country["clean_name"].fillna("")
            + " "
            + s1_country["clean_address"].fillna("")
        ).values

        target_texts = (
            target_country["clean_name"].fillna("")
            + " "
            + target_country["clean_address"].fillna("")
        ).values

        target_ids = target_country["entity_id"].values
        s1_ids = s1_country["entity_id"].values

        print(f"  S1 entities:     {len(s1_ids):,}")
        print(f"  Target entities: {len(target_ids):,}")

        del s1_country, target_country
        gc.collect()

        # -------------------------------------------------
        # TF-IDF fitting  (on a sample, same as before)
        # -------------------------------------------------

        print("  Fitting TF-IDF on representative sample...")

        sample_size = min(250_000, len(target_texts))

        rng = np.random.default_rng(42)

        sample_idx = rng.choice(
            len(target_texts),
            sample_size,
            replace=False,
        )

        target_sample = target_texts[sample_idx]

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            max_features=50000,
            sublinear_tf=True,
            dtype=np.float32,
        )

        vectorizer.fit(target_sample)

        print(
            f"  Vocabulary size: "
            f"{len(vectorizer.vocabulary_):,}"
        )

        del target_sample, sample_idx
        gc.collect()

        # -------------------------------------------------
        # Streaming single-chunked retrieval
        # -------------------------------------------------

        n_s1 = len(s1_texts)
        n_targets = len(target_texts)

        country_s1_ids = []
        country_target_ids = []
        country_scores = []

        n_s1_chunks = (n_s1 + s1_chunk_size - 1) // s1_chunk_size
        n_target_chunks = (
            (n_targets + target_chunk_size - 1) // target_chunk_size
        )

        print(
            f"  Streaming: {n_s1_chunks} S1 chunk(s) "
            f"× {n_target_chunks} target chunk(s)"
        )
        print("  Pre-transforming S1 entities...")

        s1_matrices = []
        for s1_start in range(0, n_s1, s1_chunk_size):
            s1_end = min(s1_start + s1_chunk_size, n_s1)
            s1_matrices.append(vectorizer.transform(s1_texts[s1_start:s1_end]))

        # Initialize global top-K accumulators for ALL S1 entities
        best_indices = [np.empty(0, dtype=np.int64) for _ in range(n_s1)]
        best_scores = [np.empty(0, dtype=np.float32) for _ in range(n_s1)]

        # Iterate over every target chunk ONCE
        for t_start in tqdm(range(0, n_targets, target_chunk_size), desc="Target Chunks"):
            t_end = min(t_start + target_chunk_size, n_targets)
            target_chunk_texts = target_texts[t_start:t_end]

            # Transform this target chunk and transpose
            target_matrix = vectorizer.transform(target_chunk_texts)
            target_matrix_T = target_matrix.T.tocsr()

            for chunk_idx, s1_start in enumerate(range(0, n_s1, s1_chunk_size)):
                s1_end = min(s1_start + s1_chunk_size, n_s1)
                chunk_len = s1_end - s1_start
                s1_matrix = s1_matrices[chunk_idx]

                # Sparse top-K: s1_matrix @ target_matrix.T
                sim_block = sp_matmul_topn(
                    s1_matrix,
                    target_matrix_T,
                    top_n=top_k,
                    threshold=0.05,
                    sort=True,
                    n_threads=n_threads,
                )

                sim_block = sim_block.tocsr()

                # Merge per-S1 results into global running top-K
                for row in range(chunk_len):
                    row_start = sim_block.indptr[row]
                    row_end = sim_block.indptr[row + 1]

                    if row_start == row_end:
                        continue

                    # Column indices are local to this target chunk
                    local_cols = sim_block.indices[
                        row_start:row_end
                    ].astype(np.int64)
                    global_cols = local_cols + t_start
                    scores = sim_block.data[
                        row_start:row_end
                    ].astype(np.float32)

                    global_row = s1_start + row
                    best_indices[global_row], best_scores[global_row] = _merge_topk(
                        best_indices[global_row],
                        best_scores[global_row],
                        global_cols,
                        scores,
                        top_k,
                    )

            del target_matrix, target_matrix_T, target_chunk_texts
            gc.collect()

        # -------------------------------------------------
        # Emit global top-K for all S1 entities
        # -------------------------------------------------

        for row in range(n_s1):
            if len(best_indices[row]) == 0:
                continue

            s1_id = s1_ids[row]

            for g_idx, sc in zip(
                best_indices[row],
                best_scores[row],
            ):
                country_s1_ids.append(s1_id)
                country_target_ids.append(
                    target_ids[g_idx]
                )
                country_scores.append(float(sc))

        del s1_matrices, best_indices, best_scores
        del s1_texts, target_texts, s1_ids, target_ids
        del vectorizer
        gc.collect()

        # -------------------------------------------------
        # Flush country results to temp file
        # -------------------------------------------------

        if country_s1_ids:

            chunk_df = pd.DataFrame(
                {
                    "source1_id": country_s1_ids,
                    "target_id": country_target_ids,
                    "blocking_score": country_scores,
                }
            )

            chunk_df.drop_duplicates(
                subset=["source1_id", "target_id"],
                inplace=True,
            )

            chunk_path = os.path.join(
                tempfile.gettempdir(),
                f"blocking_topn_{country}.csv",
            )

            chunk_df.to_csv(
                chunk_path,
                index=False,
            )

            chunk_files.append(chunk_path)

            total_pairs += len(chunk_df)

            print(
                f"  Flushed "
                f"{len(chunk_df):,} candidate pairs "
                f"for '{country}'."
            )

            del chunk_df
            del country_s1_ids
            del country_target_ids
            del country_scores

            gc.collect()

        else:
            print(
                f"  WARNING: No candidates generated "
                f"for country '{country}'."
            )

    # ---------------------------------------------------------
    # Combine country chunks
    # ---------------------------------------------------------

    print(
        f"\nReading "
        f"{len(chunk_files)} country chunks back..."
    )

    if chunk_files:

        chunks = []

        for f in chunk_files:

            chunks.append(
                pd.read_csv(
                    f,
                    dtype={
                        "source1_id": str,
                        "target_id": str,
                        "blocking_score": np.float32,
                    },
                )
            )

        candidates_df = pd.concat(
            chunks,
            ignore_index=True,
        )

        for f in chunk_files:
            try:
                os.remove(f)
            except OSError:
                pass

    else:

        candidates_df = pd.DataFrame(
            columns=[
                "source1_id",
                "target_id",
                "blocking_score",
            ]
        )

    candidates_df.drop_duplicates(
        subset=["source1_id", "target_id"],
        inplace=True,
    )

    print(
        f"Total candidate pairs generated: "
        f"{len(candidates_df):,}"
    )

    return candidates_df