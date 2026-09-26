import pandas as pd
import numpy as np
import jellyfish
from tqdm import tqdm
import gc


def extract_features(df_pairs, df_s1, df_targets, batch_size=100000):
    """
    Extract string-similarity features for candidate pairs.

    Memory strategy:
    - NEVER build maps for all target entities.
    - First identify target IDs actually present in candidate pairs.
    - Subset df_targets to only those IDs.
    - Build lookup dictionaries only for that subset.
    - Process candidate pairs in batches.
    """

    n_samples = len(df_pairs)
    print(f"Extracting features for {n_samples:,} pairs...")

    if n_samples == 0:
        return pd.DataFrame(columns=[
            'source1_id',
            'target_id',
            'blocking_score',
            'name_jaro_winkler',
            'name_levenshtein',
            'name_len_diff',
            'name_token_jaccard',
            'name_exact_match',
            'address_jaro_winkler',
            'address_levenshtein',
            'address_len_diff',
            'address_token_jaccard',
            'address_exact_match'
        ])

    # ------------------------------------------------------------------
    # 1. Source 1 lookup
    # ------------------------------------------------------------------
    # Source 1 is small compared with the 10.3M target table, so
    # dictionary lookup here is acceptable.
    s1_name_map = dict(
        zip(
            df_s1['entity_id'],
            df_s1['clean_name'].fillna("")
        )
    )

    s1_addr_map = dict(
        zip(
            df_s1['entity_id'],
            df_s1['clean_address'].fillna("")
        )
    )

    # ------------------------------------------------------------------
    # 2. Candidate IDs
    # ------------------------------------------------------------------
    s1_ids = df_pairs['source1_id'].to_numpy()
    target_ids = df_pairs['target_id'].to_numpy()

    if 'blocking_score' in df_pairs.columns:
        blocking_scores = df_pairs['blocking_score'].to_numpy(
            dtype=np.float32,
            copy=False
        )
    else:
        blocking_scores = np.zeros(
            n_samples,
            dtype=np.float32
        )

    # Only IDs that actually occur in candidate pairs.
    unique_target_ids = pd.unique(target_ids)

    print(
        f"Candidate pairs reference "
        f"{len(unique_target_ids):,} unique target entities."
    )

    # ------------------------------------------------------------------
    # 3. CRITICAL MEMORY FIX
    # ------------------------------------------------------------------
    # DO NOT create:
    #
    # target_name_map = dict(zip(df_targets['entity_id'], ...))
    #
    # because that materializes all 10.3M targets as Python objects.
    #
    # Instead, filter the existing dataframe down to the candidate IDs.
    # This keeps memory proportional to the candidates rather than the
    # complete target universe.
    # ------------------------------------------------------------------
    print("Subsetting target dataframe to candidate target IDs...")

    target_subset = df_targets.loc[
        df_targets['entity_id'].isin(unique_target_ids),
        ['entity_id', 'clean_name', 'clean_address']
    ].copy()

    print(
        f"Target subset contains "
        f"{len(target_subset):,} rows."
    )

    # ------------------------------------------------------------------
    # 4. Build lookup dictionaries ONLY for candidate targets
    # ------------------------------------------------------------------
    target_name_map = dict(
        zip(
            target_subset['entity_id'],
            target_subset['clean_name'].fillna("")
        )
    )

    target_addr_map = dict(
        zip(
            target_subset['entity_id'],
            target_subset['clean_address'].fillna("")
        )
    )

    # We no longer need the pandas target subset after constructing
    # the limited lookup dictionaries.
    del target_subset
    gc.collect()

    # ------------------------------------------------------------------
    # 5. Allocate output arrays
    # ------------------------------------------------------------------
    name_jw = np.zeros(n_samples, dtype=np.float32)
    name_lev = np.zeros(n_samples, dtype=np.float32)
    name_len_diff = np.zeros(n_samples, dtype=np.int16)
    name_jaccard = np.zeros(n_samples, dtype=np.float32)
    name_exact = np.zeros(n_samples, dtype=np.int8)

    addr_jw = np.zeros(n_samples, dtype=np.float32)
    addr_lev = np.zeros(n_samples, dtype=np.float32)
    addr_len_diff = np.zeros(n_samples, dtype=np.int16)
    addr_jaccard = np.zeros(n_samples, dtype=np.float32)
    addr_exact = np.zeros(n_samples, dtype=np.int8)

    # ------------------------------------------------------------------
    # 6. Process candidate pairs in batches
    # ------------------------------------------------------------------
    print(
        f"Computing pairwise string metrics "
        f"in batches of {batch_size:,}..."
    )

    for start in tqdm(
        range(0, n_samples, batch_size),
        desc="Feature extraction"
    ):
        end = min(start + batch_size, n_samples)

        # Only construct token sets for entities used by THIS batch.
        batch_s1_ids = s1_ids[start:end]
        batch_target_ids = target_ids[start:end]

        unique_batch_s1 = pd.unique(batch_s1_ids)
        unique_batch_target = pd.unique(batch_target_ids)

        # --------------------------------------------------------------
        # Token sets for current batch only
        # --------------------------------------------------------------
        s1_name_tokens = {
            eid: set(s1_name_map[eid].split())
            for eid in unique_batch_s1
            if s1_name_map.get(eid)
        }

        s1_addr_tokens = {
            eid: set(s1_addr_map[eid].split())
            for eid in unique_batch_s1
            if s1_addr_map.get(eid)
        }

        target_name_tokens = {
            eid: set(target_name_map[eid].split())
            for eid in unique_batch_target
            if target_name_map.get(eid)
        }

        target_addr_tokens = {
            eid: set(target_addr_map[eid].split())
            for eid in unique_batch_target
            if target_addr_map.get(eid)
        }

        # --------------------------------------------------------------
        # Pairwise calculations
        # --------------------------------------------------------------
        for local_i, (s1_id, t_id) in enumerate(
            zip(batch_s1_ids, batch_target_ids)
        ):
            i = start + local_i

            s1_n = s1_name_map.get(s1_id, "")
            t_n = target_name_map.get(t_id, "")

            s1_a = s1_addr_map.get(s1_id, "")
            t_a = target_addr_map.get(t_id, "")

            # ----------------------------------------------------------
            # Name metrics
            # ----------------------------------------------------------
            if s1_n and t_n:

                name_jw[i] = jellyfish.jaro_winkler_similarity(
                    s1_n,
                    t_n
                )

                max_l = max(len(s1_n), len(t_n))

                name_lev[i] = (
                    1.0
                    - (
                        jellyfish.levenshtein_distance(
                            s1_n,
                            t_n
                        ) / max_l
                    )
                    if max_l > 0
                    else 1.0
                )

                name_len_diff[i] = abs(
                    len(s1_n) - len(t_n)
                )

                set1 = s1_name_tokens.get(s1_id, set())
                set2 = target_name_tokens.get(t_id, set())

                union = set1 | set2

                name_jaccard[i] = (
                    len(set1 & set2) / len(union)
                    if union
                    else 0.0
                )

                name_exact[i] = int(s1_n == t_n)

            else:
                name_len_diff[i] = abs(
                    len(s1_n) - len(t_n)
                )

            # ----------------------------------------------------------
            # Address metrics
            # ----------------------------------------------------------
            if s1_a and t_a:

                addr_jw[i] = jellyfish.jaro_winkler_similarity(
                    s1_a,
                    t_a
                )

                max_l = max(len(s1_a), len(t_a))

                addr_lev[i] = (
                    1.0
                    - (
                        jellyfish.levenshtein_distance(
                            s1_a,
                            t_a
                        ) / max_l
                    )
                    if max_l > 0
                    else 1.0
                )

                addr_len_diff[i] = abs(
                    len(s1_a) - len(t_a)
                )

                set1 = s1_addr_tokens.get(s1_id, set())
                set2 = target_addr_tokens.get(t_id, set())

                union = set1 | set2

                addr_jaccard[i] = (
                    len(set1 & set2) / len(union)
                    if union
                    else 0.0
                )

                addr_exact[i] = int(s1_a == t_a)

            else:
                addr_len_diff[i] = abs(
                    len(s1_a) - len(t_a)
                )

        # --------------------------------------------------------------
        # Free batch token sets immediately
        # --------------------------------------------------------------
        del (
            s1_name_tokens,
            s1_addr_tokens,
            target_name_tokens,
            target_addr_tokens
        )

        gc.collect()

    # ------------------------------------------------------------------
    # 7. Construct final dataframe
    # ------------------------------------------------------------------
    print("Constructing features dataframe...")

    features_df = pd.DataFrame({
        'source1_id': s1_ids,
        'target_id': target_ids,
        'blocking_score': blocking_scores,

        'name_jaro_winkler': name_jw,
        'name_levenshtein': name_lev,
        'name_len_diff': name_len_diff,
        'name_token_jaccard': name_jaccard,
        'name_exact_match': name_exact,

        'address_jaro_winkler': addr_jw,
        'address_levenshtein': addr_lev,
        'address_len_diff': addr_len_diff,
        'address_token_jaccard': addr_jaccard,
        'address_exact_match': addr_exact
    })

    return features_df
