import pandas as pd
import numpy as np

def predict_test(model, df_features, best_thresh):
    """
    Predicts matches on the test set features using the trained model and best threshold.
    """
    print("Running inference on test pairs...")
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
    X_test = df_features[feature_cols]
    
    probs = model.predict(X_test)
    preds = (probs > best_thresh).astype(int)
    
    df_results = df_features[['source1_id', 'target_id']].copy()
    df_results['probability'] = probs
    df_results['is_match'] = preds
    
    return df_results

def generate_submission(df_results, df_s1_test, output_path):
    """
    Generates the final submission file:
    source1_entity_id, matched_entity_ids (comma separated)
    Ensures EXACTLY ONE row per Source 1 entity (including singletons as empty strings).
    """
    print(f"Formatting submission file to {output_path}...")
    
    # Filter only positive matches
    matches = df_results[df_results['is_match'] == 1].copy()
    
    # Group by source1_id and aggregate targets
    submission = matches.groupby('source1_id')['target_id'].apply(lambda x: ','.join(x)).reset_index()
    submission.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Ensure ALL source1 IDs are present exactly once, even singletons
    all_s1_df = pd.DataFrame({'source1_entity_id': df_s1_test['entity_id']})
    final_submission = all_s1_df.merge(submission, on='source1_entity_id', how='left')
    final_submission['matched_entity_ids'] = final_submission['matched_entity_ids'].fillna('')
    
    final_submission.to_csv(output_path, index=False, sep='\t')
    print("Submission saved successfully.")
    
def save_candidate_pairs(df_candidates, df_s1_test, output_path):
    """
    Saves the generated candidate pairs as expected by the official validator.
    source1_entity_id, candidate_entity_ids (comma separated)
    Ensures EXACTLY ONE row per Source 1 entity (including singletons as empty strings).
    """
    print(f"Formatting candidate pairs file to {output_path}...")
    
    # Group by source1_id and aggregate targets
    cands_grouped = df_candidates.groupby('source1_id')['target_id'].apply(lambda x: ','.join(x)).reset_index()
    cands_grouped.columns = ['source1_entity_id', 'candidate_entity_ids']
    
    # Ensure ALL source1 IDs are present exactly once, even singletons
    all_s1_df = pd.DataFrame({'source1_entity_id': df_s1_test['entity_id']})
    final_cands = all_s1_df.merge(cands_grouped, on='source1_entity_id', how='left')
    final_cands['candidate_entity_ids'] = final_cands['candidate_entity_ids'].fillna('')
    
    final_cands.to_csv(output_path, index=False, sep='\t')
    print("Candidate pairs saved successfully.")
