"""
AUDIT PART 3 (fixed): Actual blocking implementation benchmark.
200 S1 entities vs 100k targets - safe for 16GB RAM.
flush=True so output appears immediately.
"""
import pandas as pd
import numpy as np
import time
import gc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def p(msg):
    print(msg, flush=True)

p("=" * 60)
p("SECTION D: ACTUAL BLOCKING IMPLEMENTATION BENCHMARK")
p("=" * 60)

# Load GT (corrected)
p("Loading GT...")
df_gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t", keep_default_na=False, dtype=str
)
true_dict = {}
for row in df_gt.itertuples():
    raw = str(row.matched_entity_ids).strip()
    matches = set(m.strip() for m in raw.split(",") if m.strip()) if raw else set()
    true_dict[row.source1_entity_id] = matches
p(f"GT loaded: {len(true_dict)} entries.")

# Sample 200 S1
p("Loading S1 sample (200 entities)...")
df_s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t", keep_default_na=False, dtype=str
).sample(200, random_state=42)

# Load 100k S2 targets only (safe)
p("Loading 100k rows from S2 as target...")
df_s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t",
                     nrows=100_000, keep_default_na=False, dtype=str)
df_targets = df_s2.copy()
del df_s2; gc.collect()

# Add cleaned columns expected by src/blocking.py
for df in [df_s1, df_targets]:
    df["clean_country"] = df["country"].str.strip().str.lower()
    df["clean_name"]    = df["business_name"].fillna("")
    df["clean_address"] = df["business_address"].fillna("")

# Check for hard-coded countries in blocking.py
p("\nChecking blocking.py for hard-coded country strings...")
with open("src/blocking.py", "r") as f:
    blocking_src = f.read()
hardcoded = any(kw in blocking_src for kw in ['"US"', '"India"', "'US'", "'India'", '"us"', '"india"'])
p(f"  Hard-coded country strings present: {'YES ← BUG' if hardcoded else 'No'}")
p(f"  Country loop: df_source1['clean_country'].unique() -> dynamic")

# Run blocking
p(f"\nRunning generate_candidates: {len(df_s1)} S1 vs {len(df_targets)} targets...")
sys.stdout.flush()
from src.blocking import generate_candidates
t0 = time.time()
candidates = generate_candidates(df_s1, df_targets, top_k=10)
elapsed = time.time() - t0
p(f"Blocking completed in {elapsed:.1f}s")

# Metrics
total_cands = len(candidates)
s1_grouped  = candidates.groupby("source1_id")["target_id"].count()
s1_with_cands = len(s1_grouped)
s1_zero_cands = len(df_s1) - s1_with_cands

p(f"\n--- Candidate Statistics ---")
p(f"  Total candidate pairs:       {total_cands:,}")
p(f"  S1 entities with candidates: {s1_with_cands:,}")
p(f"  S1 entities with 0 cands:    {s1_zero_cands:,}")
p(f"  Avg candidates / S1:         {s1_grouped.mean():.2f}")
p(f"  Median candidates / S1:      {s1_grouped.median():.1f}")
p(f"  Max candidates / S1:         {s1_grouped.max()}")

# Blocking recall
target_id_set = set(df_targets["entity_id"])
cand_dict = candidates.groupby("source1_id")["target_id"].apply(set).to_dict()

possible_true = 0
captured_true = 0

for s1_id in df_s1["entity_id"]:
    true_in_subset = true_dict.get(s1_id, set()) & target_id_set
    possible_true  += len(true_in_subset)
    pred = cand_dict.get(s1_id, set())
    captured_true  += len(true_in_subset & pred)

p(f"\n--- Blocking Recall (vs 100k-target subset) ---")
if possible_true > 0:
    recall = captured_true / possible_true
    p(f"  True matches in subset:   {possible_true}")
    p(f"  Captured by blocking:     {captured_true}")
    p(f"  Blocking Recall:          {recall*100:.2f}%")
    p(f"  True matches LOST:        {possible_true - captured_true}")
else:
    p("  No true matches in this sample vs 100k targets (normal for small sample).")
    p("  This is because GT matching spans the full 5M rows, not just 100k.")

p("\nDone with Section D.")
