import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from src import config

def f05_macro_per_s1(y_true_dict, y_pred_dict):
    """
    Computes the challenge-specific Macro F0.5 per S1 entity.
    """
    f05_scores = []
    
    for s1_id in y_true_dict:
        true_targets = y_true_dict[s1_id]
        pred_targets = y_pred_dict.get(s1_id, set())
        
        # True negatives (Singleton correctly predicted as empty)
        if len(true_targets) == 0 and len(pred_targets) == 0:
            f05_scores.append(1.0)
            continue
            
        # False positives on singleton (Singleton predicted to have matches)
        if len(true_targets) == 0 and len(pred_targets) > 0:
            f05_scores.append(0.0)
            continue
            
        # False negatives (Has matches, predicted none)
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

def tune_threshold(val_df, val_probs, true_dict):
    """
    Finds the best threshold for F0.5 score by grouping predictions by S1.
    """
    best_thresh = 0.5
    best_f05 = 0
    
    val_df = val_df.copy()
    val_df['prob'] = val_probs
    
    # Check thresholds from 0.1 to 0.9. Allow 0.99 for strictness (empty predictions)
    for thresh in np.arange(0.1, 0.95, 0.05):
        # Filter to matches passing threshold
        matches = val_df[val_df['prob'] > thresh]
        
        # Group by S1 to get sets of predicted targets
        pred_dict = matches.groupby('source1_id')['target_id'].apply(set).to_dict()
        
        # Calculate macro f05 over validation S1s
        f05 = f05_macro_per_s1(true_dict, pred_dict)
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"Optimal Threshold: {best_thresh:.2f}, Validation Macro F0.5: {best_f05:.4f}")
    return best_thresh

def train_model(df_features, df_gt, params, all_s1_entities=None):
    """
    Trains a LightGBM model using leakage-free group splits.
    df_features must have source1_id, target_id, and feature columns.
    df_gt must be the raw ground truth DataFrame (loaded with keep_default_na=False).
    all_s1_entities is an optional array/Series of all S1 entity IDs sampled for training,
    ensuring zero-candidate S1 entities are represented in validation split and threshold tuning.
    """
    print("Preparing training data...")
    
    # Safely construct the true dictionaries for ALL S1 entities in GT
    all_true_dict = {}
    gt_pair_set = set()  # Memory-safe: set of tuples instead of exploded DataFrame
    
    for row in df_gt.itertuples():
        s1 = row.source1_entity_id
        raw_val = str(row.matched_entity_ids).strip()
        
        if raw_val == "":
            matches = set()
        else:
            matches = set([m.strip() for m in raw_val.split(',') if m.strip()])
            
        all_true_dict[s1] = matches
        
        for m in matches:
            gt_pair_set.add((s1, m))
    
    # Memory-safe label assignment: vectorized set lookup instead of pandas merge
    print("Assigning labels via set lookup (memory-safe)...")
    s1_col = df_features['source1_id'].values
    t_col = df_features['target_id'].values
    is_match = np.array(
        [1 if (str(s1_col[i]), str(t_col[i])) in gt_pair_set else 0 for i in range(len(s1_col))],
        dtype=np.int8
    )
    df_features = df_features.copy()
    df_features['is_match'] = is_match
    
    del gt_pair_set  # Free ~500MB at full scale
    
    pos_count = int(is_match.sum())
    neg_count = len(is_match) - pos_count
    print(f"Positive pairs: {pos_count:,} | Negative pairs: {neg_count:,}")
    
    # Features to use: exact 11 columns in deterministic order
    feature_cols = [
        'name_jaro_winkler',
        'name_levenshtein',
        'name_len_diff',
        'name_token_jaccard',
        'name_exact_match',
        'address_jaro_winkler',
        'address_levenshtein',
        'address_len_diff',
        'address_token_jaccard',
        'address_exact_match',
        'blocking_score'
    ]

    # Split by source1_id to prevent data leakage.
    # Grouped split using all sampled S1 entities so zero-candidate entities are represented.
    if all_s1_entities is not None:
        unique_s1 = list(pd.unique(all_s1_entities))
    else:
        unique_s1 = list(df_features['source1_id'].unique())
    train_s1, val_s1 = train_test_split(unique_s1, test_size=0.2, random_state=42)
    
    # Prepare validation true dict for ALL validation S1 entities (including zero-candidate S1s)
    val_true_dict = {s1: all_true_dict.get(s1, set()) for s1 in val_s1}
    
    train_s1_set = set(train_s1)
    val_s1_set = set(val_s1)
    
    train_mask = df_features['source1_id'].isin(train_s1_set)
    val_mask = df_features['source1_id'].isin(val_s1_set)
    
    X_train = df_features[train_mask][feature_cols]
    y_train = df_features[train_mask]['is_match']
    
    X_val = df_features[val_mask][feature_cols]
    y_val = df_features[val_mask]['is_match']
    
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
    
    # We train using standard binary logloss, which handles class imbalance gracefully 
    # if scale_pos_weight is passed in params, without disrupting F0.5 calibration later.
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=True)
    
    print("Training LightGBM model...")
    # Ensure a metric is present for early stopping
    if 'metric' not in params or params['metric'] == 'custom':
        params['metric'] = 'binary_logloss'
        
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    print("Tuning threshold on validation set using Macro F0.5...")
    val_df_subset = df_features[val_mask][['source1_id', 'target_id']].copy()
    val_probs = model.predict(X_val)
    
    best_thresh = tune_threshold(val_df_subset, val_probs, val_true_dict)
    
    # Save model and threshold to disk for instant inference reuse
    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)
    model.save_model(os.path.join(config.MODELS_DIR, 'lgbm_model.txt'))
    with open(os.path.join(config.MODELS_DIR, 'best_threshold.txt'), 'w') as f:
        f.write(str(best_thresh))
    print(f"Model saved to {os.path.join(config.MODELS_DIR, 'lgbm_model.txt')}")
    
    return model, best_thresh
