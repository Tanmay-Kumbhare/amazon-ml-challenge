import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File paths
TRAIN_FILES = {
    'source1': os.path.join(TRAIN_DIR, 'train_source1.tsv'),
    'source2': os.path.join(TRAIN_DIR, 'train_source2.tsv'),
    'source3': os.path.join(TRAIN_DIR, 'train_source3.tsv'),
    'gt': os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')
}

TEST_FILES = {
    'source1': os.path.join(TEST_DIR, 'test_source1.tsv'),
    'source2': os.path.join(TEST_DIR, 'test_source2.tsv'),
    'source3': os.path.join(TEST_DIR, 'test_source3.tsv')
}

# Hyperparameters
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2
BLOCKING_TOP_K = 30  # Number of candidates to retrieve per entity during blocking
TFIDF_MAX_FEATURES = 100000

# Feature engineering
USE_NAME_FEATURES = True
USE_ADDRESS_FEATURES = True

# LightGBM Params
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',  # Early stopping metric; challenge F0.5 is evaluated post-training
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}
