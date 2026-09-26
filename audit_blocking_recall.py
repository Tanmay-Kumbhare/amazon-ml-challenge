"""
Phase 2: Blocking Recall Audit
Uses the exact same 5,000-S1 training sample (random_state=42) and the
current blocking implementation to measure how many GT relationships
are captured vs missed.

RUN THIS IN VS CODE — NOT ANTIGRAVITY
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.preprocess import preprocess_dataframe
from src.blocking import generate_candidates

def p(msg):
    print(msg, flush=True)

def main():
    PILOT_N = 5000
    
    p("=" * 60)
    p("BLOCKING RECALL AUDIT")
    p(f"Sample: {PILOT_N} S1 entities (random_state=42)")
    p("=" * 60)
    
    # 1. Load data exactly as main.py does
    p("\n1. Loading data...")
    df_s1 = pd.read_csv(config.TRAIN_FILES['source1'], sep='\t', dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(config.TRAIN_FILES['source2'], sep='\t', dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(config.TRAIN_FILES['source3'], sep='\t', dtype=str, keep_default_na=False)
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    
    # 2. Sample exactly as main.py --pilot 5000 does
    df_s1 = df_s1.sample(n=min(PILOT_N, len(df_s1)), random_state=42).copy()
    
    # 3. Preprocess
    p("2. Preprocessing...")
    df_s1 = preprocess_dataframe(df_s1)
    df_targets = preprocess_dataframe(df_targets)
    
    # 4. Generate candidates using current blocking
    p("3. Generating candidates...")
    candidates = generate_candidates(df_s1, df_targets, top_k=config.BLOCKING_TOP_K)
    
    # 5. Load GT (corrected)
    p("\n4. Loading ground truth...")
    df_gt = pd.read_csv(config.TRAIN_FILES['gt'], sep='\t', dtype=str, keep_default_na=False)
    
    true_dict = {}
    for row in df_gt.itertuples():
        raw = str(row.matched_entity_ids).strip()
        matches = set(m.strip() for m in raw.split(',') if m.strip()) if raw else set()
        true_dict[row.source1_entity_id] = matches
    
    # 6. Build candidate lookup
    cand_dict = candidates.groupby('source1_id')['target_id'].apply(set).to_dict()
    
    # Build sets of S2 and S3 entity IDs for source-specific recall
    s2_ids = set(df_s2['entity_id'])
    s3_ids = set(df_s3['entity_id'])
    target_id_set = set(df_targets['entity_id'])
    
    # 7. Measure recall
    p("\n5. Computing blocking recall...")
    
    total_gt_rels = 0
    total_captured = 0
    total_missed = 0
    s1_with_missed = 0
    
    s2_gt = 0; s2_captured = 0
    s3_gt = 0; s3_captured = 0
    
    sampled_s1_ids = list(df_s1['entity_id'])
    
    for s1_id in sampled_s1_ids:
        true_matches = true_dict.get(s1_id, set())
        # Only count GT matches that exist in the target pool
        true_in_pool = true_matches & target_id_set
        pred_cands = cand_dict.get(s1_id, set())
        
        captured = true_in_pool & pred_cands
        missed = true_in_pool - pred_cands
        
        total_gt_rels += len(true_in_pool)
        total_captured += len(captured)
        total_missed += len(missed)
        
        if len(missed) > 0:
            s1_with_missed += 1
        
        # Source-specific
        for m in true_in_pool:
            if m in s2_ids:
                s2_gt += 1
                if m in pred_cands:
                    s2_captured += 1
            elif m in s3_ids:
                s3_gt += 1
                if m in pred_cands:
                    s3_captured += 1
    
    # 8. Candidate statistics
    total_cands = len(candidates)
    s1_grouped = candidates.groupby('source1_id')['target_id'].count()
    s1_with_cands = len(s1_grouped)
    s1_zero_cands = len(df_s1) - s1_with_cands
    
    # 9. Report
    p("\n" + "=" * 60)
    p("BLOCKING RECALL AUDIT RESULTS")
    p("=" * 60)
    
    p(f"\n--- Ground Truth Coverage ---")
    p(f"  S1 entities sampled:              {len(sampled_s1_ids):,}")
    p(f"  Total GT relationships (in pool): {total_gt_rels:,}")
    p(f"  GT relationships captured:        {total_captured:,}")
    p(f"  GT relationships missed:          {total_missed:,}")
    if total_gt_rels > 0:
        p(f"  BLOCKING RECALL:                  {total_captured/total_gt_rels*100:.2f}%")
    else:
        p(f"  BLOCKING RECALL:                  N/A (no GT rels)")
    p(f"  S1 entities with missed matches:  {s1_with_missed:,}")
    
    p(f"\n--- Source-Specific Recall ---")
    if s2_gt > 0:
        p(f"  Source 2: {s2_captured}/{s2_gt} captured ({s2_captured/s2_gt*100:.2f}%)")
    else:
        p(f"  Source 2: 0 GT relationships in sample")
    if s3_gt > 0:
        p(f"  Source 3: {s3_captured}/{s3_gt} captured ({s3_captured/s3_gt*100:.2f}%)")
    else:
        p(f"  Source 3: 0 GT relationships in sample")
    
    p(f"\n--- Candidate Statistics ---")
    p(f"  Total candidate pairs:            {total_cands:,}")
    p(f"  S1 with candidates:               {s1_with_cands:,}")
    p(f"  S1 with zero candidates:          {s1_zero_cands:,}")
    p(f"  Avg candidates / S1:              {s1_grouped.mean():.1f}")
    p(f"  Median candidates / S1:           {s1_grouped.median():.1f}")
    p(f"  Max candidates / S1:              {s1_grouped.max()}")
    
    p("\nDone.")

if __name__ == "__main__":
    main()
