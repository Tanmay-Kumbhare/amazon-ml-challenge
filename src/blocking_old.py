import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import gc
import os
import tempfile

# Stop words to prevent extremely dense buckets in the inverted index
STOP_WORDS = set([
    'the', 'and', 'of', 'in', 'for', 'a', 'on', 'at', 
    'inc', 'ltd', 'llc', 'co', 'corp', 'corporation', 
    'limited', 'private', 'pvt', 'company', 'group',
    'st', 'street', 'rd', 'road', 'ave', 'avenue', 'blvd', 
    'floor', 'room', 'suite', 'building', 'no', 'number'
])

def get_tokens(text):
    if not isinstance(text, str) or not text:
        return set()
    # Basic tokenization: split by whitespace
    tokens = set(text.lower().split())
    # Keep tokens > 2 chars that aren't stop words
    return {t for t in tokens if len(t) > 2 and t not in STOP_WORDS}

def generate_candidates(df_source1, df_targets, top_k=None):
    """
    Generate candidate pairs using a Memory-Safe Multi-Pass Inverted Index.
    Writes candidate chunks per country to temp CSV files, then reads back.
    Pass 1: Exact Name Match
    Pass 2: Name Token Overlap
    Pass 3: Exact Address Match
    """
    print("Generating candidates using Memory-Safe Multi-Pass Blocking...")
    
    countries = df_source1['clean_country'].unique()
    chunk_files = []
    total_pairs = 0
    
    for country in countries:
        print(f"\nProcessing country: '{country}'")
        s1_country = df_source1[df_source1['clean_country'] == country]
        target_country = df_targets[df_targets['clean_country'] == country]
        
        if len(s1_country) == 0 or len(target_country) == 0:
            continue
            
        print("  Building inverted indexes for targets...")
        exact_name_idx = defaultdict(list)
        name_token_idx = defaultdict(list)
        exact_addr_idx = defaultdict(list)
        
        # Populate indexes
        for row in target_country.itertuples():
            t_id = row.entity_id
            c_name = getattr(row, 'clean_name', '').strip()
            c_addr = getattr(row, 'clean_address', '').strip()
            
            if c_name:
                exact_name_idx[c_name].append(t_id)
                for token in get_tokens(c_name):
                    name_token_idx[token].append(t_id)
            
            if c_addr:
                exact_addr_idx[c_addr].append(t_id)
        
        # Collect pairs for this country in a local list, then flush to disk
        country_s1_ids = []
        country_target_ids = []
        
        print(f"  Searching for {len(s1_country)} S1 entities...")
        for row in tqdm(s1_country.itertuples(), total=len(s1_country)):
            s1_id = row.entity_id
            c_name = getattr(row, 'clean_name', '').strip()
            c_addr = getattr(row, 'clean_address', '').strip()
            
            candidates = set()
            
            # Pass 1: Exact Name Match
            if c_name and c_name in exact_name_idx:
                candidates.update(exact_name_idx[c_name])
            
            # Pass 2: Name Token Overlap
            if c_name:
                for token in get_tokens(c_name):
                    if token in name_token_idx:
                        bucket = name_token_idx[token]
                        # IDF cutoff equivalent: drop massively common words that slip through stop-words
                        if len(bucket) < 2000:  
                            candidates.update(bucket)
                            
            # Pass 3: Exact Address Match
            if c_addr and len(c_addr) > 10 and c_addr in exact_addr_idx:
                bucket = exact_addr_idx[c_addr]
                if len(bucket) < 1000:
                    candidates.update(bucket)
            
            # Cap candidates to avoid runaway pairwise explosion later
            cand_list = list(candidates)
            if len(cand_list) > 300:
                cand_list = cand_list[:300]
                
            for t_id in cand_list:
                country_s1_ids.append(s1_id)
                country_target_ids.append(t_id)
                
        # Free inverted indexes for this country partition
        del exact_name_idx, name_token_idx, exact_addr_idx
        gc.collect()
        
        # Flush this country's pairs to a temp CSV file
        if country_s1_ids:
            chunk_df = pd.DataFrame({
                'source1_id': country_s1_ids,
                'target_id': country_target_ids
            })
            chunk_df.drop_duplicates(inplace=True)
            
            chunk_path = os.path.join(tempfile.gettempdir(), f'blocking_chunk_{country}.csv')
            chunk_df.to_csv(chunk_path, index=False)
            chunk_files.append(chunk_path)
            total_pairs += len(chunk_df)
            print(f"  Flushed {len(chunk_df):,} deduplicated pairs for '{country}' to disk.")
            
            del chunk_df, country_s1_ids, country_target_ids
            gc.collect()
        else:
            del country_s1_ids, country_target_ids

    # Read all chunks back and concatenate
    print(f"\nReading {len(chunk_files)} country chunks back...")
    if chunk_files:
        chunks = [pd.read_csv(f, dtype=str) for f in chunk_files]
        candidates_df = pd.concat(chunks, ignore_index=True)
        
        # Clean up temp files
        for f in chunk_files:
            try:
                os.remove(f)
            except OSError:
                pass
    else:
        candidates_df = pd.DataFrame(columns=['source1_id', 'target_id'])
    
    # Add dummy blocking score for feature compatibility
    candidates_df['blocking_score'] = np.ones(len(candidates_df), dtype=np.float32)
    
    # Final global dedup (cross-country shouldn't happen, but safety net)
    candidates_df.drop_duplicates(subset=['source1_id', 'target_id'], inplace=True)
    print(f"Total candidate pairs generated: {len(candidates_df):,}")
    return candidates_df
