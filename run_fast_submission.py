"""
High-Precision Fast Rule-Based Entity Matcher for Amazon ML Challenge.
Memory-Safe Version: Uses numpy arrays instead of Python dicts.
Computes tokens on-the-fly only for candidates, not all targets.
"""

import os
import gc
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import jellyfish
from src import config

STOP_WORDS = set([
    'the', 'and', 'of', 'in', 'for', 'a', 'on', 'at', 'to',
    'inc', 'ltd', 'llc', 'co', 'corp', 'corporation',
    'limited', 'private', 'pvt', 'company', 'group', 'enterprises',
    'st', 'street', 'rd', 'road', 'ave', 'avenue', 'blvd',
    'floor', 'room', 'suite', 'building', 'no', 'number'
])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def get_tokens(text):
    if not text:
        return set()
    return {w for w in text.split() if len(w) > 2 and w not in STOP_WORDS}

def token_jaccard(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0

def main():
    print("=" * 65)
    print("  AMAZON ML CHALLENGE: HIGH-PRECISION FAST MATCHER (v2)")
    print("=" * 65)

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    test_files = config.TEST_FILES
    print("\n1. Loading Test Datasets...")
    df_s1 = pd.read_csv(test_files['source1'], sep='\t', dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(test_files['source2'], sep='\t', dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(test_files['source3'], sep='\t', dtype=str, keep_default_na=False)

    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    print(f"  Test Source 1 entities: {len(df_s1):,}")
    print(f"  Total Target entities:  {len(df_targets):,}")

    # Preprocessing
    print("\n2. Normalizing text...")
    df_s1['clean_name'] = df_s1['business_name'].apply(clean_text)
    df_s1['clean_address'] = df_s1['business_address'].apply(clean_text)
    df_s1['clean_country'] = df_s1['country'].apply(clean_text)

    df_targets['clean_name'] = df_targets['business_name'].apply(clean_text)
    df_targets['clean_address'] = df_targets['business_address'].apply(clean_text)
    df_targets['clean_country'] = df_targets['country'].apply(clean_text)

    countries = df_s1['clean_country'].unique()
    all_s1_ids = []
    all_matched_targets = []

    cand_pairs_path = os.path.join(config.OUTPUT_DIR, 'candidate_pairs.tsv')
    matches_path = os.path.join(config.OUTPUT_DIR, 'matching_results.tsv')

    with open(cand_pairs_path, 'w', encoding='utf-8') as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    for country in countries:
        print(f"\n-------------------------------------------------------")
        print(f"Processing Country: '{country}'")
        print(f"-------------------------------------------------------")

        s1_c = df_s1[df_s1['clean_country'] == country]
        t_c = df_targets[df_targets['clean_country'] == country]

        n_s1 = len(s1_c)
        n_t = len(t_c)
        print(f"  S1 entities: {n_s1:,} | Target entities: {n_t:,}")

        if n_s1 == 0:
            continue

        if n_t == 0:
            for eid in s1_c['entity_id'].values:
                all_s1_ids.append(eid)
                all_matched_targets.append("")
            with open(cand_pairs_path, 'a', encoding='utf-8') as f_cand:
                for eid in s1_c['entity_id'].values:
                    f_cand.write(f"{eid}\t\n")
            continue

        # Extract numpy arrays (much more memory-efficient than Python dicts)
        target_eids = t_c['entity_id'].values
        target_names = t_c['clean_name'].values
        target_addrs = t_c['clean_address'].values

        # Build inverted indexes using INTEGER ROW INDICES (not string IDs)
        # This saves ~50% memory vs storing string entity_ids in every list
        print("  Building inverted lookup indexes...")
        exact_name_idx = defaultdict(list)
        name_token_idx = defaultdict(list)

        for i in range(n_t):
            name = target_names[i]
            if name:
                exact_name_idx[name].append(i)
                for tok in get_tokens(name):
                    name_token_idx[tok].append(i)

        # Free the country DataFrame subset
        del t_c
        gc.collect()

        print(f"  Matching {n_s1:,} S1 entities...")
        cand_lines = []

        for row in tqdm(s1_c.itertuples(), total=n_s1, desc=f"Matching {country}"):
            s1_id = row.entity_id
            name = row.clean_name
            addr = row.clean_address
            n_tokens = get_tokens(name)

            # Build candidate set (integer indices into target arrays)
            candidates = set()

            # 1. Exact Name match
            if name and name in exact_name_idx:
                for idx in exact_name_idx[name]:
                    candidates.add(idx)

            # 2. Token overlap (skip hyper-frequent tokens)
            if n_tokens:
                for tok in n_tokens:
                    if tok in name_token_idx:
                        bucket = name_token_idx[tok]
                        if len(bucket) <= 1500:
                            for idx in bucket:
                                candidates.add(idx)
                                if len(candidates) >= 200:
                                    break
                    if len(candidates) >= 200:
                        break

            # Write candidate pairs (convert indices back to entity_ids)
            cand_eids = [target_eids[idx] for idx in list(candidates)[:30]]
            cand_lines.append(f"{s1_id}\t{';'.join(cand_eids)}\n")

            if not candidates:
                all_s1_ids.append(s1_id)
                all_matched_targets.append("")
                continue

            # Score candidates ON-THE-FLY (no pre-computed dicts needed)
            a_tokens = get_tokens(addr)
            matched = []

            for idx in candidates:
                t_name = target_names[idx]
                t_addr = target_addrs[idx]
                tid = target_eids[idx]

                # Compute target tokens on-the-fly (only ~200 candidates, not 4.7M!)
                t_ntoks = get_tokens(t_name)
                t_atoks = get_tokens(t_addr)

                # Rule 1: Exact Name + non-contradictory address
                if name and t_name and name == t_name:
                    addr_jac = token_jaccard(a_tokens, t_atoks)
                    if not a_tokens or not t_atoks or addr_jac >= 0.15 or addr == t_addr:
                        matched.append(tid)
                        continue

                # Rule 2: High Jaro-Winkler + Address verification
                if name and t_name:
                    jw = jellyfish.jaro_winkler_similarity(name, t_name)
                    if jw >= 0.90:
                        addr_jac = token_jaccard(a_tokens, t_atoks)
                        if addr_jac >= 0.30 or (addr and addr == t_addr) or (jw >= 0.97 and (not a_tokens or not t_atoks)):
                            matched.append(tid)
                            continue

                # Rule 3: Exact Address + decent Name similarity
                if addr and t_addr and len(addr) > 8 and addr == t_addr:
                    if name and t_name:
                        jw = jellyfish.jaro_winkler_similarity(name, t_name)
                        if jw >= 0.75:
                            matched.append(tid)
                            continue

            # Deduplicate: keep at most 1 per source (S2, S3)
            if len(matched) > 2:
                s2 = [m for m in matched if m.startswith('S2')]
                s3 = [m for m in matched if m.startswith('S3')]
                final = []
                if s2: final.append(s2[0])
                if s3: final.append(s3[0])
                matched_str = ";".join(final) if final else ";".join(matched[:2])
            else:
                matched_str = ";".join(matched)

            all_s1_ids.append(s1_id)
            all_matched_targets.append(matched_str)

        # Flush candidate lines to disk
        with open(cand_pairs_path, 'a', encoding='utf-8') as f_cand:
            f_cand.writelines(cand_lines)

        # Free ALL country-level data
        del exact_name_idx, name_token_idx
        del target_eids, target_names, target_addrs
        del s1_c, cand_lines
        gc.collect()

    print("\n3. Generating Final Submission File...")
    sub_df = pd.DataFrame({
        'source1_entity_id': all_s1_ids,
        'matched_entity_ids': all_matched_targets
    })

    # Ensure ALL S1 entities are present (left join preserves order)
    final_sub = df_s1[['entity_id']].merge(
        sub_df, left_on='entity_id', right_on='source1_entity_id', how='left'
    )
    final_sub['matched_entity_ids'] = final_sub['matched_entity_ids'].fillna("")
    final_sub = final_sub[['source1_entity_id', 'matched_entity_ids']]

    # Fill any NaN source1_ids from the merge
    mask = final_sub['source1_entity_id'].isna()
    if mask.any():
        final_sub.loc[mask, 'source1_entity_id'] = df_s1.loc[mask, 'entity_id'].values
        final_sub.loc[mask, 'matched_entity_ids'] = ""

    final_sub.to_csv(matches_path, sep='\t', index=False)
    print(f"  Saved: {matches_path}")
    print(f"  Total records:           {len(final_sub):,}")
    print(f"  Entities with matches:   {sum(final_sub['matched_entity_ids'] != ''):,}")
    print(f"  Singletons (no match):   {sum(final_sub['matched_entity_ids'] == ''):,}")

    print("\n4. Verifying Submission Integrity...")
    assert len(final_sub) == len(df_s1), f"Row count mismatch! {len(final_sub)} vs {len(df_s1)}"
    print(f"  Row count: {len(final_sub):,} == {len(df_s1):,} [PASSED]")
    print(f"  Candidate pairs: {cand_pairs_path} [SAVED]")
    print("\n  SUCCESS! Submission is ready to upload!")

if __name__ == "__main__":
    main()
