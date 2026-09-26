import os
import argparse
import pandas as pd
from src import config
from src.preprocess import preprocess_dataframe
from src.blocking import generate_candidates
from src.features import extract_features
from src.train import train_model
from src.inference import predict_test, generate_submission, save_candidate_pairs
import warnings

warnings.filterwarnings('ignore')

def load_data(is_train=True):
    files = config.TRAIN_FILES if is_train else config.TEST_FILES
    print(f"Loading {'train' if is_train else 'test'} data...")
    # use keep_default_na=False to avoid "nan" string issues on empty fields
    df_s1 = pd.read_csv(files['source1'], sep='\t', dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(files['source2'], sep='\t', dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(files['source3'], sep='\t', dtype=str, keep_default_na=False)
    
    df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
    return df_s1, df_targets

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge Pipeline")
    parser.add_argument('--pilot', type=int, default=0, help="Run a quick pilot test with N Source 1 entities for both train and test")
    parser.add_argument('--train-sample', type=int, default=0, help="Train on N sampled S1 entities, but evaluate 100% of the TEST set")
    parser.add_argument('--skip-train', action='store_true', help="Skip training and load existing model from models/lgbm_model.txt")
    args = parser.parse_args()
    
    print("=== Amazon ML Challenge Pipeline ===")
    
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    model_path = os.path.join(config.MODELS_DIR, 'lgbm_model.txt')
    thresh_path = os.path.join(config.MODELS_DIR, 'best_threshold.txt')
    
    if args.skip_train:
        # ---- Load existing model & threshold from disk ----
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"--skip-train was specified but the model file does not exist: {model_path}\n"
                f"Run training first (without --skip-train) to produce the model."
            )
        print(f"Skipping training. Loading saved model from {model_path}...")
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
        if os.path.exists(thresh_path):
            with open(thresh_path, 'r') as f:
                best_thresh = float(f.read().strip())
        else:
            best_thresh = 0.60
            print(f"  Threshold file not found; using default {best_thresh}")
        print(f"Loaded threshold: {best_thresh}")
    else:
        # ---- Full training path ----
        # Load & preprocess training data
        df_s1_train, df_targets_train = load_data(is_train=True)

        train_n = args.train_sample if args.train_sample > 0 else args.pilot
        if train_n > 0:
            print(f"TRAIN SAMPLING: Sampling {train_n:,} entities from Train Source 1.")
            df_s1_train = df_s1_train.sample(n=min(train_n, len(df_s1_train)), random_state=42).copy()

        df_s1_train = preprocess_dataframe(df_s1_train)
        df_targets_train = preprocess_dataframe(df_targets_train)

        # Candidate Generation (Blocking)
        train_candidates = generate_candidates(df_s1_train, df_targets_train, top_k=config.BLOCKING_TOP_K)
        if train_candidates is None or len(train_candidates) == 0:
            raise RuntimeError(
                "Candidate generation produced zero train candidates. "
                "Blocking failed to retrieve any pairs between df_s1_train and df_targets_train."
            )

        # Feature Extraction
        train_features = extract_features(train_candidates, df_s1_train, df_targets_train)

        # Free memory
        del df_targets_train, train_candidates

        # Model Training
        df_gt = pd.read_csv(config.TRAIN_FILES['gt'], sep='\t', dtype=str, keep_default_na=False)
        all_train_s1 = df_s1_train['entity_id'].values
        model, best_thresh = train_model(train_features, df_gt, config.LGBM_PARAMS, all_s1_entities=all_train_s1)

        del train_features, df_s1_train, df_gt
    
    # 5. Load & Preprocess Test
    df_s1_test, df_targets_test = load_data(is_train=False)
    
    if args.pilot > 0:
        print(f"PILOT MODE: Sampling {args.pilot:,} entities from Test Source 1.")
        df_s1_test = df_s1_test.sample(n=min(args.pilot, len(df_s1_test)), random_state=42).copy()
    else:
        print(f"FULL TEST MODE: Processing all {len(df_s1_test):,} entities from Test Source 1.")
        
    df_s1_test = preprocess_dataframe(df_s1_test)
    df_targets_test = preprocess_dataframe(df_targets_test)
    
    # 6. Candidate Generation Test
    test_candidates = generate_candidates(df_s1_test, df_targets_test, top_k=config.BLOCKING_TOP_K)
    if test_candidates is None or len(test_candidates) == 0:
        raise RuntimeError(
            "Candidate generation produced zero test candidates. "
            "Blocking failed to retrieve any pairs between df_s1_test and df_targets_test."
        )
    
    # The signature takes df_s1_test to ensure ALL test singletons are saved as empty lists
    save_candidate_pairs(test_candidates, df_s1_test, os.path.join(config.OUTPUT_DIR, 'candidate_pairs.tsv'))
    
    # 7. Feature Extraction Test
    test_features = extract_features(test_candidates, df_s1_test, df_targets_test)
    
    # 8. Inference
    test_results = predict_test(model, test_features, best_thresh)
    
    # 9. Output Generation
    generate_submission(test_results, df_s1_test, os.path.join(config.OUTPUT_DIR, 'matching_results.tsv'))
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
