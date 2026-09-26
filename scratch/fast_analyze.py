import csv
import json
import os
import random

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
file_uniques = {}

def fast_file_stats(filepath):
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    total_rows = 0
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        try:
            columns = next(reader)
        except StopIteration:
            return {}, {}
            
        missing = {c: 0 for c in columns}
        uniques = {c: set() for c in columns}
        samples = []
        
        for row in reader:
            if not row: continue
            total_rows += 1
            if total_rows <= 5:
                # pad row to column length if necessary
                padded_row = row + [''] * (len(columns) - len(row))
                samples.append({columns[i]: padded_row[i] for i in range(len(columns))})
            
            for i, val in enumerate(row):
                if i >= len(columns): break
                c = columns[i]
                if val in ('', 'NaN', 'null', 'NA', 'None'):
                    missing[c] += 1
                if val:
                    uniques[c].add(val)
                    
            # if row has missing columns at the end, count them as missing
            if len(row) < len(columns):
                for i in range(len(row), len(columns)):
                    missing[columns[i]] += 1
                    
    stats = {
        'file_size_mb': round(size_mb, 2),
        'total_rows': total_rows,
        'columns': columns,
        'dtypes': {c: 'string' for c in columns},
        'samples': samples,
        'missing': missing,
        'unique_counts': {c: len(uniques[c]) for c in columns}
    }
    return stats, uniques

print("Starting fast analysis...")
for name, filepath in files_to_analyze.items():
    print(f"Analyzing {name}...")
    stats, uniques = fast_file_stats(filepath)
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
total_s1_entities = report['train_gt']['total_rows']
total_relationships = 0
zero_matches = 0
one_match = 0
multiple_matches = 0
max_matches = 0
s2_rels = 0
s3_rels = 0
both_rels = 0

with open(files_to_analyze['train_gt'], 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader) # skip header
    for row in reader:
        if not row: continue
        if len(row) < 2:
            matches_str = ''
        else:
            matches_str = row[1]
            
        matches = [m.strip() for m in matches_str.split(',') if m.strip()]
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
# We must convert sets to list to save as json if they exist, but we didn't store sets in json.
# Only thing is we need to write to JSON
with open('scratch/report.json', 'w') as f:
    json.dump({
        'file_stats': report,
        'field_analysis': field_analysis,
        'gt_stats': gt_stats
    }, f, indent=2)
print("Done.")
