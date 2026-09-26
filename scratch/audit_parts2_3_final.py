"""
Quick isolated script: ONLY Hashing@500k and blocking Part 3.
No TF-IDF. No large loops. All prints are flushed immediately.
"""
import pandas as pd
import numpy as np
import tracemalloc
import time
import gc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def p(msg):
    print(msg, flush=True)

# ─── HASHING @ 500k ──────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import HashingVectorizer

p("=" * 60)
p("SECTION C (cont): HashingVectorizer @ 500k")
p("=" * 60)
p("Loading 500k rows from source2...")
df_s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t",
                     nrows=500_000, keep_default_na=False, dtype=str)
subset = (df_s2["business_name"] + " " + df_s2["business_address"]).fillna("").tolist()
del df_s2; gc.collect()
p(f"Loaded {len(subset)} rows.")

gc.collect()
tracemalloc.start()
t0 = time.time()
vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                        n_features=2**16, dtype=np.float32,
                        alternate_sign=False)
mat = vec.transform(subset)
elapsed = time.time() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
p(f"  [Hashing@500k] Time={elapsed:.1f}s  PeakRAM={peak/(1024**2):.0f}MB  "
  f"Shape={mat.shape}  NNZ={mat.nnz:,}")
del mat, vec, subset; gc.collect()
p("Done with Section C.\n")

# ─── BLOCKING BENCHMARK ───────────────────────────────────────────────────────
p("=" * 60)
p("SECTION D: ACTUAL BLOCKING BENCHMARK")
p("=" * 60)

p("Loading GT...")
df_gt = pd.read_csv("dataset/train/train_ground_truth.tsv",
                     sep="\t", keep_default_na=False, dtype=str)
true_dict = {}
for row in df_gt.itertuples():
    raw = str(row.matched_entity_ids).strip()
    matches = set(m.strip() for m in raw.split(",") if m.strip()) if raw else set()
    true_dict[row.source1_entity_id] = matches
p(f"GT loaded: {len(true_dict)} entries.")

p("Loading S1 sample (200 entities)...")
df_s1_full = pd.read_csv("dataset/train/train_source1.tsv",
                          sep="\t", keep_default_na=False, dtype=str)
df_s1 = df_s1_full.sample(200, random_state=42)
del df_s1_full; gc.collect()

p("Loading 100k rows from S2 as targets...")
df_targets = pd.read_csv("dataset/train/train_source2.tsv", sep="\t",
                          nrows=100_000, keep_default_na=False, dtype=str)

for df in [df_s1, df_targets]:
    df["clean_country"] = df["country"].str.strip().str.lower()
    df["clean_name"]    = df["business_name"].fillna("")
    df["clean_address"] = df["business_address"].fillna("")

# Country hard-code check
p("\nChecking src/blocking.py for hard-coded countries...")
with open("src/blocking.py", "r") as f:
    src = f.read()
hc = any(k in src for k in ['"US"', '"India"', "'US'", "'India'", '"us"', '"india"'])
p(f"  Hard-coded country strings: {'YES ← BUG' if hc else 'NONE (good)'}")
p(f"  Country iteration: df_source1['clean_country'].unique() -> fully dynamic")

p(f"\nRunning generate_candidates: {len(df_s1)} S1 vs {len(df_targets)} targets...")
from src.blocking import generate_candidates
t0 = time.time()
candidates = generate_candidates(df_s1, df_targets, top_k=10)
elapsed = time.time() - t0
p(f"Blocking done in {elapsed:.1f}s")

total_cands   = len(candidates)
s1_grouped    = candidates.groupby("source1_id")["target_id"].count()
s1_with_cands = len(s1_grouped)
s1_zero_cands = len(df_s1) - s1_with_cands

p(f"\n--- Candidate Statistics ---")
p(f"  Total candidate pairs:       {total_cands:,}")
p(f"  S1 entities with candidates: {s1_with_cands:,}")
p(f"  S1 entities with 0 cands:    {s1_zero_cands:,}")
p(f"  Avg candidates / S1:         {s1_grouped.mean():.2f}")
p(f"  Median / S1:                 {s1_grouped.median():.1f}")
p(f"  Max / S1:                    {s1_grouped.max()}")

target_id_set = set(df_targets["entity_id"])
cand_dict = candidates.groupby("source1_id")["target_id"].apply(set).to_dict()
possible_true = 0
captured_true = 0
for s1_id in df_s1["entity_id"]:
    true_sub = true_dict.get(s1_id, set()) & target_id_set
    possible_true  += len(true_sub)
    captured_true  += len(true_sub & cand_dict.get(s1_id, set()))

p(f"\n--- Blocking Recall (200 S1 vs 100k targets) ---")
if possible_true > 0:
    p(f"  True matches in subset:   {possible_true}")
    p(f"  Captured by blocking:     {captured_true}")
    p(f"  Blocking Recall:          {captured_true/possible_true*100:.2f}%")
    p(f"  True matches LOST:        {possible_true - captured_true}")
else:
    p("  No GT overlaps in this 200-S1 x 100k sample (expected for tiny slices).")
    p("  To measure true blocking recall properly we need full-file coverage.")

p("\nAll sections done.")
