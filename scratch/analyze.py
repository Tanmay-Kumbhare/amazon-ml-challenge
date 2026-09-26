import pandas as pd
import json
import os
import random
from collections import defaultdict

# Setting up paths
DATA_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

files_to_analyze = {
    'train_source1': os.path.join(TRAIN_DIR, 'train_source1.tsv'),
    'train_source2': os.path.join(TRAIN_DIR, 'train_source2.tsv'),
    'train_source3': os.path.join(TRAIN_DIR, 'train_source3.tsv'),
    'train_gt': os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'),
    'test_source1': os.path.join(TEST_DIR, 'test_source1.tsv'),
    'test_source2': os.path.join(TEST_DIR, 'test_source2.tsv'),
    'test_source3': os.path.join(TEST_DIR, 'test_source3.tsv'),
}

report = {}

def get_file_stats(filepath):
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    total_rows = 0
    missing = {}
    uniques = {}
    columns = []
    dtypes = {}
    samples = []
    
    chunksize = 100000
    first_chunk = True
    
    for chunk in pd.read_csv(filepath, sep='\t', chunksize=chunksize, dtype=str, keep_default_na=False):
        if first_chunk:
            columns = list(chunk.columns)
            for c in columns:
                missing[c] = 0
                uniques[c] = set()
                dtypes[c] = 'string'
            samples = chunk.head(5).to_dict('records')
            first_chunk = False
            
        total_rows += len(chunk)
        for c in columns:
            empty_mask = chunk[c].isin(['', 'NaN', 'null', 'NA', 'None'])
            missing[c] += int(empty_mask.sum())
            uniques[c].update(chunk[c].dropna().unique())
            
    stats = {
        'file_size_mb': round(size_mb, 2),
        'total_rows': total_rows,
        'columns': columns,
        'dtypes': dtypes,
        'samples': samples,
        'missing': missing,
        'unique_counts': {c: len(uniques[c]) for c in columns}
    }
    return stats, uniques

print("Starting file analysis...")
file_uniques = {}
for name, filepath in files_to_analyze.items():
    print(f"Analyzing {name}...")
    stats, uniques = get_file_stats(filepath)
    report[name] = stats
    file_uniques[name] = uniques

print("Starting field-level analysis...")
field_analysis = {}

def analyze_field(uniques_set):
    sample_set = random.sample(list(uniques_set), min(10000, len(uniques_set)))
    lengths = [len(str(x)) for x in sample_set if str(x) != '']
    return {
        'avg_length': sum(lengths)/len(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'examples': sample_set[:10]
    }

for name in ['train_source1', 'train_source2', 'train_source3', 'test_source1', 'test_source2', 'test_source3']:
    field_analysis[name] = {}
    for c in ['business_name', 'business_address', 'country']:
        if c in file_uniques[name]:
            field_analysis[name][c] = analyze_field(file_uniques[name][c])

print("Starting Ground Truth analysis...")
gt_df = pd.read_csv(files_to_analyze['train_gt'], sep='\t', dtype=str)
total_s1_entities = len(gt_df)
total_relationships = 0
zero_matches = 0
one_match = 0
multiple_matches = 0
max_matches = 0
s2_rels = 0
s3_rels = 0
both_rels = 0

for _, row in gt_df.iterrows():
    matches = str(row['matched_entity_ids']).split(',')
    matches = [m for m in matches if m.strip()]
    if not matches:
        zero_matches += 1
        continue
    
    match_count = len(matches)
    total_relationships += match_count
    
    if match_count == 1:
        one_match += 1
    else:
        multiple_matches += 1
        
    if match_count > max_matches:
        max_matches = match_count
        
    has_s2 = any(m.startswith('S2-') for m in matches)
    has_s3 = any(m.startswith('S3-') for m in matches)
    
    if has_s2: s2_rels += 1
    if has_s3: s3_rels += 1
    if has_s2 and has_s3: both_rels += 1

gt_stats = {
    'total_s1_entities_in_gt_file': total_s1_entities,
    'total_relationships': total_relationships,
    'zero_matches': zero_matches,
    'one_match': one_match,
    'multiple_matches': multiple_matches,
    'max_matches_for_s1': max_matches,
    's2_rels': s2_rels,
    's3_rels': s3_rels,
    'both_rels': both_rels
}

print("Saving report...")
with open('scratch/report.json', 'w') as f:
    json.dump({
        'file_stats': report,
        'field_analysis': field_analysis,
        'gt_stats': gt_stats
    }, f, indent=2)
print("Done.")
