import pandas as pd
import numpy as np
import time
import tracemalloc
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def p(msg):
    print(msg, flush=True)

def run_benchmark():
    p("=" * 60)
    p("ACTUAL BLOCKING IMPLEMENTATION BENCHMARK")
    p("=" * 60)

    try:
        from src.blocking import generate_candidates
    except ImportError as e:
        p(f"Error importing blocker: {e}")
        return

    p("1. Loading Ground Truth...")
    try:
        df_gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", keep_default_na=False, dtype=str)
        true_dict = {}
        for row in df_gt.itertuples():
            raw = str(row.matched_entity_ids).strip()
            matches = set(m.strip() for m in raw.split(",") if m.strip()) if raw else set()
            true_dict[row.source1_entity_id] = matches
        del df_gt
        gc.collect()
    except Exception as e:
        p(f"Failed loading GT: {e}")
        return

    p("2. Loading S1 Sample (500 entities)...")
    try:
        df_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", keep_default_na=False, dtype=str).sample(500, random_state=42)
    except Exception as e:
        p(f"Failed loading S1: {e}")
        return

    p("3. Loading S2 and S3 full targets...")
    tracemalloc.start()
    try:
        df_s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", keep_default_na=False, dtype=str)
        df_s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", keep_default_na=False, dtype=str)
        df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
        del df_s2, df_s3
        gc.collect()
    except MemoryError:
        p("EXACT REASON: MemoryError during data loading/concat.")
        p("Loading 10.3 million S2+S3 target rows exceeded available RAM before blocking even started.")
        tracemalloc.stop()
        return

    p("4. Formatting columns...")
    for df in [df_s1, df_targets]:
        df["clean_country"] = df["country"].str.strip().str.lower()
        df["clean_name"]    = df["business_name"].fillna("")
        df["clean_address"] = df["business_address"].fillna("")

    p(f"5. Running src.blocking.generate_candidates on {len(df_s1)} S1 vs {len(df_targets)} targets...")
    t0 = time.time()
    try:
        candidates = generate_candidates(df_s1, df_targets, top_k=10)
        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        p(f"\n--- BENCHMARK RESULTS ---")
        p(f"Runtime: {elapsed:.2f} seconds")
        p(f"Peak RAM (tracemalloc delta during blocking): {peak/(1024**2):.0f} MB")
        
        total_cands = len(candidates)
        s1_grouped = candidates.groupby("source1_id")["target_id"].count() if total_cands > 0 else pd.Series(dtype=int)
        s1_zero_cands = len(df_s1) - len(s1_grouped)
        
        p(f"Candidate pairs total: {total_cands}")
        if total_cands > 0:
            p(f"Average candidates per S1: {s1_grouped.mean():.2f}")
            p(f"Median candidates per S1: {s1_grouped.median():.2f}")
            p(f"Maximum candidates per S1: {s1_grouped.max()}")
        else:
            p("Average candidates per S1: 0")
            p("Median candidates per S1: 0")
            p("Maximum candidates per S1: 0")
            
        p(f"Number of S1 entities with zero candidates: {s1_zero_cands}")
        
        target_id_set = set(df_targets["entity_id"])
        cand_dict = candidates.groupby("source1_id")["target_id"].apply(set).to_dict() if total_cands > 0 else {}
        possible_true = 0
        captured_true = 0
        
        for s1_id in df_s1["entity_id"]:
            true_sub = true_dict.get(s1_id, set()) & target_id_set
            possible_true += len(true_sub)
            captured_true += len(true_sub & cand_dict.get(s1_id, set()))
            
        p(f"True matches captured: {captured_true}")
        p(f"True matches missed: {possible_true - captured_true}")
        if possible_true > 0:
            p(f"Blocking recall: {captured_true/possible_true*100:.2f}%")
        else:
            p("Blocking recall: N/A (no true matches exist for this sample in the target set)")

    except MemoryError:
        tracemalloc.stop()
        p("\n--- BENCHMARK FAILED ---")
        p("EXACT REASON: MemoryError")
        p("The actual blocking implementation crashed because TfidfVectorizer attempted to fit millions of target rows in memory (or matrix multiplication failed), exceeding available 16GB RAM.")
    except Exception as e:
        tracemalloc.stop()
        p("\n--- BENCHMARK FAILED ---")
        p(f"EXACT REASON: Exception -> {e}")

if __name__ == "__main__":
    run_benchmark()
