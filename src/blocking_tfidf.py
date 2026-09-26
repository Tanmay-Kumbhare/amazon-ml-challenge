import pandas as pd
import numpy as np
import os
import tempfile
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def generate_candidates(df_source1, df_targets, top_k=30):
    """
    Generate candidate pairs using Ranked Sparse Char-Ngram TF-IDF.
    - Fits TF-IDF per country.
    - Computes sparse cosine similarity in small S1 chunks (RAM safe).
    - Retrieves strictly Top-K ranked candidates per S1 entity.
    """
    print(f"Generating top-{top_k} candidates using Sparse Char-Ngram TF-IDF...")
    
    countries = df_source1['clean_country'].unique()
    chunk_files = []
    total_pairs = 0
    
    for country in countries:
        print(f"\nProcessing country: '{country}'")
        s1_country = df_source1[df_source1['clean_country'] == country].copy()
        target_country = df_targets[df_targets['clean_country'] == country].copy()
        
        if len(s1_country) == 0 or len(target_country) == 0:
            continue
            
        # Combine name and address for richer char n-grams
        s1_texts = (s1_country['clean_name'].fillna('') + ' ' + s1_country['clean_address'].fillna('')).values
        target_texts = (target_country['clean_name'].fillna('') + ' ' + target_country['clean_address'].fillna('')).values
        
        target_ids = target_country['entity_id'].values
        s1_ids = s1_country['entity_id'].values
        
        del s1_country, target_country
        gc.collect()
        
        print("  Fitting TF-IDF on a sample to prevent memory explosion...")
        # Fit on a random subset to prevent the char n-gram vocabulary dictionary from exploding RAM
        sample_size = min(250000, len(target_texts))
        np.random.seed(42)
        sample_idx = np.random.choice(len(target_texts), sample_size, replace=False)
        target_sample = target_texts[sample_idx]
        
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=100000, dtype=np.float32)
        vectorizer.fit(target_sample)
        
        del target_sample
        gc.collect()
        
        print("  Transforming targets in chunks...")
        import scipy.sparse as sp
        
        def transform_in_chunks(texts, vec, chunk_size=100000):
            chunks = []
            for i in range(0, len(texts), chunk_size):
                chunks.append(vec.transform(texts[i:i+chunk_size]))
            return sp.vstack(chunks)
            
        target_matrix = transform_in_chunks(target_texts, vectorizer)
        s1_matrix = transform_in_chunks(s1_texts, vectorizer)
        
        del s1_texts, target_texts
        gc.collect()
        
        # Process S1 in chunks. We use chunk_size=50 because:
        # Sparse (6M, F) @ Dense (F, 50) -> Dense (6M, 50) = 1.2 GB RAM (very safe).
        # This completely avoids scipy's sparse-sparse matrix alignment overhead that caused the MemoryError.
        chunk_size = 50
        n_s1 = s1_matrix.shape[0]
        
        country_s1_ids = []
        country_target_ids = []
        country_scores = []
        
        print(f"  Computing ranked similarities for {n_s1} S1 entities...")
        for start_idx in tqdm(range(0, n_s1, chunk_size), desc=f"Scoring {country}"):
            end_idx = min(start_idx + chunk_size, n_s1)
            s1_chunk = s1_matrix[start_idx:end_idx].toarray().T # Shape: (F, chunk_size)
            
            # target_matrix is CSR. csr_matrix.dot(dense_matrix) is highly optimized in C++
            # Result is dense (n_targets, chunk_size)
            sim_dense = target_matrix.dot(s1_chunk).T # Shape: (chunk_size, n_targets)
            
            for i in range(end_idx - start_idx):
                global_s1_idx = start_idx + i
                s1_id = s1_ids[global_s1_idx]
                row_scores = sim_dense[i]
                
                # We only want scores > 0, up to top_k
                # A quick optimization: if max score is 0, skip
                if row_scores.max() == 0:
                    continue
                    
                if len(row_scores) > top_k:
                    top_k_idx = np.argpartition(row_scores, -top_k)[-top_k:]
                    # Filter out zero scores that might have been included if less than top_k matches existed
                    valid = row_scores[top_k_idx] > 0
                    best_indices = top_k_idx[valid]
                    best_scores = row_scores[top_k_idx][valid]
                else:
                    valid = row_scores > 0
                    best_indices = np.arange(len(row_scores))[valid]
                    best_scores = row_scores[valid]
                    
                for t_idx, score in zip(best_indices, best_scores):
                    country_s1_ids.append(s1_id)
                    country_target_ids.append(target_ids[t_idx])
                    country_scores.append(score)
            
            del sim_dense, s1_chunk
                    
        # Free matrices before writing chunk
        del target_matrix, s1_matrix, vectorizer
        gc.collect()
        
        if country_s1_ids:
            chunk_df = pd.DataFrame({
                'source1_id': country_s1_ids,
                'target_id': country_target_ids,
                'blocking_score': country_scores
            })
            
            # Safety deduplication (should naturally be unique per S1)
            chunk_df.drop_duplicates(subset=['source1_id', 'target_id'], inplace=True)
            
            chunk_path = os.path.join(tempfile.gettempdir(), f'blocking_chunk_tfidf_{country}.csv')
            chunk_df.to_csv(chunk_path, index=False)
            chunk_files.append(chunk_path)
            total_pairs += len(chunk_df)
            print(f"  Flushed {len(chunk_df):,} candidate pairs for '{country}' to disk.")
            
            del chunk_df, country_s1_ids, country_target_ids, country_scores
            gc.collect()

    print(f"\nReading {len(chunk_files)} country chunks back...")
    if chunk_files:
        chunks = [pd.read_csv(f, dtype={'source1_id': str, 'target_id': str, 'blocking_score': np.float32}) for f in chunk_files]
        candidates_df = pd.concat(chunks, ignore_index=True)
        
        for f in chunk_files:
            try:
                os.remove(f)
            except OSError:
                pass
    else:
        candidates_df = pd.DataFrame(columns=['source1_id', 'target_id', 'blocking_score'])
    
    print(f"Total candidate pairs generated: {len(candidates_df):,}")
    return candidates_df
