import pandas as pd
import numpy as np
import time
import tracemalloc
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import gc

# 1. Macro F0.5 per S1
def f05_macro_per_s1(y_true_dict, y_pred_dict):
    f05_scores = []
    for s1_id in y_true_dict:
        true_targets = y_true_dict[s1_id]
        pred_targets = y_pred_dict.get(s1_id, set())
        
        if len(true_targets) == 0 and len(pred_targets) == 0:
            f05_scores.append(1.0)
            continue
        if len(true_targets) == 0 and len(pred_targets) > 0:
            f05_scores.append(0.0)
            continue
        if len(true_targets) > 0 and len(pred_targets) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_targets.intersection(pred_targets))
        fp = len(pred_targets - true_targets)
        fn = len(true_targets - pred_targets)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision == 0 and recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
            f05_scores.append(f05)
            
    return np.mean(f05_scores) if f05_scores else 0.0

def run_f05_unit_tests():
    print("--- B. MACRO F0.5 UNIT TESTS ---")
    tests = [
        ("A", set(), set(), 1.0),
        ("B", set(), {'S2-1'}, 0.0),
        ("C", {'S2-1'}, set(), 0.0),
        ("D", {'S2-1'}, {'S2-1'}, 1.0),
        ("E", {'S2-1', 'S3-1'}, {'S2-1', 'S3-1'}, 1.0),
        ("F", {'S2-1', 'S3-1'}, {'S2-1', 'S3-2'}, 0.5)
    ]
    for name, t_true, t_pred, expected in tests:
        res = f05_macro_per_s1({'mock_s1': t_true}, {'mock_s1': t_pred})
        status = "PASS" if abs(res - expected) < 1e-5 else f"FAIL (Got {res})"
        print(f"Test {name}: true={t_true} pred={t_pred} -> {res:.4f} (Expected {expected}) [{status}]")

def run_audit():
    print("--- A. CORRECTED GROUND TRUTH STATISTICS ---")
    df_gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', keep_default_na=False, dtype=str)
    
    zero_match_s1 = 0
    nonzero_match_s1 = 0
    total_relationships = 0
    
    true_dict_audit = {}
    
    for row in df_gt.itertuples():
        # Correct parsing of empty strings
        raw_val = str(row.matched_entity_ids).strip()
        if raw_val == "":
            matches = set()
        else:
            matches = set([m.strip() for m in raw_val.split(',') if m.strip()])
            
        true_dict_audit[row.source1_entity_id] = matches
        
        if len(matches) == 0:
            zero_match_s1 += 1
        else:
            nonzero_match_s1 += 1
            total_relationships += len(matches)
            
    print(f"Total S1 entities: {len(df_gt)}")
    print(f"Zero-match S1 entities (Singletons): {zero_match_s1}")
    print(f"Nonzero-match S1 entities: {nonzero_match_s1}")
    print(f"Total GT relationships: {total_relationships}")
    
    run_f05_unit_tests()
    
    print("\n--- C. TF-IDF VS HASHINGVECTORIZER BENCHMARK ---")
    # Load 1M targets for benchmarking
    df_s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=1000000, keep_default_na=False, dtype=str)
    target_text = df_s2['business_name'] + " " + df_s2['business_address']
    
    for N in [100000, 500000, 1000000]:
        print(f"\nTarget size: {N}")
        subset = target_text.head(N)
        
        # TF-IDF
        gc.collect()
        tracemalloc.start()
        t0 = time.time()
        vec_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=20000, dtype=np.float32)
        mat_tfidf = vec_tfidf.fit_transform(subset)
        t_tfidf = time.time() - t0
        mem_tfidf = tracemalloc.get_traced_memory()[1] / (1024**2)
        tracemalloc.stop()
        
        print(f"  [TF-IDF] Time: {t_tfidf:.2f}s | Peak RAM: {mem_tfidf:.2f} MB")
        print(f"           Shape: {mat_tfidf.shape} | NNZ: {mat_tfidf.nnz}")
        
        # Hashing
        gc.collect()
        tracemalloc.start()
        t0 = time.time()
        vec_hash = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 4), n_features=2**16, dtype=np.float32)
        mat_hash = vec_hash.fit_transform(subset)
        t_hash = time.time() - t0
        mem_hash = tracemalloc.get_traced_memory()[1] / (1024**2)
        tracemalloc.stop()
        
        print(f"  [Hashing] Time: {t_hash:.2f}s | Peak RAM: {mem_hash:.2f} MB")
        print(f"            Shape: {mat_hash.shape} | NNZ: {mat_hash.nnz}")
        
        del mat_tfidf, mat_hash, vec_tfidf, vec_hash
        gc.collect()
        
    print("\n--- D. ACTUAL BLOCKING IMPLEMENTATION BENCHMARK ---")
    # We will import the actual blocking logic from src.blocking
    from src.blocking import generate_candidates
    
    # Load 1000 S1 entities
    df_s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', keep_default_na=False, dtype=str).sample(1000, random_state=42)
    # Give them clean_country columns as expected by the script
    df_s1['clean_country'] = df_s1['country'].str.strip().str.lower()
    df_s2['clean_country'] = df_s2['country'].str.strip().str.lower()
    
    # Add clean_name / clean_address as expected by blocking script
    df_s1['clean_name'] = df_s1['business_name']
    df_s1['clean_address'] = df_s1['business_address']
    df_s2['clean_name'] = df_s2['business_name']
    df_s2['clean_address'] = df_s2['business_address']
    
    # We use df_s2 (1,000,000 rows) as our target subset
    print(f"Running src.blocking.generate_candidates on {len(df_s1)} S1 vs {len(df_s2)} targets...")
    t0 = time.time()
    candidates = generate_candidates(df_s1, df_s2, top_k=10)
    t_block = time.time() - t0
    
    print(f"Blocking completed in {t_block:.2f}s")
    total_cands = len(candidates)
    s1_grouped = candidates.groupby('source1_id').size()
    
    print(f"Total candidate pairs: {total_cands}")
    print(f"Avg candidates/S1: {total_cands/len(df_s1):.2f}")
    print(f"Max candidates/S1: {s1_grouped.max() if not s1_grouped.empty else 0}")
    print(f"S1 entities with 0 candidates: {len(df_s1) - len(s1_grouped)}")
    
    # Check recall against true_dict_audit
    s1_ids = df_s1['entity_id'].unique()
    target_ids = set(df_s2['entity_id'].unique())
    
    total_possible_true = 0
    captured_true = 0
    
    # Convert candidates to a fast lookup dict
    cand_dict = candidates.groupby('source1_id')['target_id'].apply(set).to_dict()
    
    for s1_id in s1_ids:
        # Only count true matches that ACTUALLY exist in our 1M target subset
        true_matches_in_subset = true_dict_audit[s1_id].intersection(target_ids)
        total_possible_true += len(true_matches_in_subset)
        
        pred_cands = cand_dict.get(s1_id, set())
        captured_true += len(true_matches_in_subset.intersection(pred_cands))
        
    if total_possible_true > 0:
        print(f"Blocking Recall (vs 1M targets): {captured_true} / {total_possible_true} ({(captured_true/total_possible_true)*100:.2f}%)")
        print(f"True matches lost by blocking: {total_possible_true - captured_true}")
    else:
        print("No true matches found in this random S1 sample vs 1M target subset.")

if __name__ == "__main__":
    run_audit()
