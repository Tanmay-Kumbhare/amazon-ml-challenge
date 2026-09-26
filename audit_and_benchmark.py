import pandas as pd
import numpy as np
import time
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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
            
    return np.mean(f05_scores)

def run_audit():
    print("--- STARTING AUDIT ---")
    start_time = time.time()
    
    # 1. COUNTRY CHECK
    print("\n1. Loading Country Data for Audit...")
    df_s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', usecols=['entity_id', 'country'])
    df_s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', usecols=['entity_id', 'country'])
    df_s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', usecols=['entity_id', 'country'])
    df_gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
    
    df_s1['country'] = df_s1['country'].fillna('').str.lower().str.strip()
    df_s2['country'] = df_s2['country'].fillna('').str.lower().str.strip()
    df_s3['country'] = df_s3['country'].fillna('').str.lower().str.strip()
    
    country_map = {}
    country_map.update(dict(zip(df_s1['entity_id'], df_s1['country'])))
    country_map.update(dict(zip(df_s2['entity_id'], df_s2['country'])))
    country_map.update(dict(zip(df_s3['entity_id'], df_s3['country'])))
    
    print("Evaluating Country Matches in Ground Truth...")
    same_country = 0
    diff_country = 0
    missing_country = 0
    total_pairs = 0
    
    for row in df_gt.itertuples():
        s1 = row.source1_entity_id
        matches = str(row.matched_entity_ids).split(',')
        s1_c = country_map.get(s1, '')
        for m in matches:
            m = m.strip()
            if not m: continue
            total_pairs += 1
            t_c = country_map.get(m, '')
            if not s1_c or not t_c:
                missing_country += 1
            elif s1_c == t_c:
                same_country += 1
            else:
                diff_country += 1
                
    print(f"Total GT Pairs: {total_pairs}")
    print(f"Same Country: {same_country} ({(same_country/total_pairs)*100:.2f}%)")
    print(f"Diff Country: {diff_country} ({(diff_country/total_pairs)*100:.2f}%)")
    print(f"Missing Country in either: {missing_country} ({(missing_country/total_pairs)*100:.2f}%)")
    
    # Free RAM
    del df_s1, df_s2, df_s3, country_map
    gc.collect()
    
    print("\n2. Running Small Blocking Benchmark...")
    # Load data again but small sample for text blocking
    df_s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
    s1_sample = df_s1.sample(1000, random_state=42)
    s1_ids = set(s1_sample['entity_id'])
    
    gt_sample = df_gt[df_gt['source1_entity_id'].isin(s1_ids)]
    true_dict = {}
    total_true_matches_in_sample = 0
    for row in gt_sample.itertuples():
        matches = set([m.strip() for m in str(row.matched_entity_ids).split(',') if m.strip()])
        true_dict[row.source1_entity_id] = matches
        total_true_matches_in_sample += len(matches)
        
    print(f"Sample contains {total_true_matches_in_sample} true relationships.")
    
    df_s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
    df_s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    
    # We only block within US to save benchmark time
    s1_us = s1_sample[s1_sample['country'].str.lower() == 'us'].copy()
    target_us = df_targets[df_targets['country'].str.lower() == 'us'].copy()
    
    s1_text = (s1_us['business_name'].fillna("") + " " + s1_us['business_address'].fillna(""))
    target_text = (target_us['business_name'].fillna("") + " " + target_us['business_address'].fillna(""))
    
    print(f"TF-IDF Vectorization for {len(s1_us)} S1 queries against {len(target_us)} targets...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=20000, dtype=np.float32)
    target_tfidf = vectorizer.fit_transform(target_text)
    s1_tfidf = vectorizer.transform(s1_text)
    
    print("Running CPU NearestNeighbors...")
    nn = NearestNeighbors(n_neighbors=10, metric='cosine', n_jobs=-1)
    nn.fit(target_tfidf)
    dist, ind = nn.kneighbors(s1_tfidf)
        
    s1_us_ids = s1_us['entity_id'].values
    target_us_ids = target_us['entity_id'].values
    
    pred_dict = {}
    candidate_count = 0
    for i, s1_id in enumerate(s1_us_ids):
        preds = set()
        for j in range(10):
            if (1.0 - dist[i, j]) > 0.3:
                preds.add(target_us_ids[ind[i, j]])
                candidate_count += 1
        pred_dict[s1_id] = preds
        
    print(f"\n3. AUDIT METRICS RESULTS")
    print(f"Total Candidate Pairs (Top-10, >0.3 sim): {candidate_count}")
    print(f"Candidates per S1 (Average): {candidate_count / len(s1_us):.2f}")
    
    captured_true = 0
    possible_true_us = 0
    for s1_id in s1_us_ids:
        true_t = true_dict.get(s1_id, set())
        pred_t = pred_dict.get(s1_id, set())
        captured_true += len(true_t.intersection(pred_t))
        possible_true_us += len(true_t)
        
    recall = captured_true / possible_true_us if possible_true_us > 0 else 0
    print(f"Blocking Recall (True Matches Kept): {captured_true} / {possible_true_us} ({recall*100:.2f}%)")
    
    # Fill in empty sets for singletons so the metric works correctly
    for s1_id in true_dict:
        if s1_id not in pred_dict:
            pred_dict[s1_id] = set()
            
    macro_f05 = f05_macro_per_s1(true_dict, pred_dict)
    print(f"Macro F0.5 per S1 (Dummy Threshold 0.3): {macro_f05:.4f}")
    
    print(f"\nAudit completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_audit()
