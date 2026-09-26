import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import gc
import torch

def gpu_topk_cosine_similarity(query_tfidf, target_tfidf, top_k=10, batch_size=2000, device='cuda'):
    """
    Computes top-k cosine similarity using GPU PyTorch tensor operations.
    Fast, memory-efficient, and utilizes RTX 3050 Tensor Cores.
    """
    n_queries = query_tfidf.shape[0]
    n_targets = target_tfidf.shape[0]
    
    actual_k = min(top_k, n_targets)
    
    top_indices = np.zeros((n_queries, actual_k), dtype=np.int32)
    top_scores = np.zeros((n_queries, actual_k), dtype=np.float32)
    
    # Target TF-IDF conversion
    # Target matrix transposed: (V, N_target)
    target_csc = target_tfidf.tocsc()
    
    # Process target matrix in GPU column chunks if target is huge
    target_chunk_size = 100000
    
    for i in tqdm(range(0, n_queries, batch_size), desc="GPU Search Batches"):
        q_batch = query_tfidf[i:i+batch_size].toarray()
        q_tensor = torch.from_numpy(q_batch).float().to(device)
        
        # We will collect topk scores across target chunks if needed
        all_top_scores = []
        all_top_indices = []
        
        for t_start in range(0, n_targets, target_chunk_size):
            t_end = min(t_start + target_chunk_size, n_targets)
            t_sub = target_csc[t_start:t_end].toarray().T # (V, N_sub)
            t_tensor = torch.from_numpy(t_sub).float().to(device)
            
            # Matrix multiplication on CUDA: (batch_size, N_sub)
            sim_matrix = torch.mm(q_tensor, t_tensor)
            
            # Top-k for this chunk
            sub_k = min(actual_k, t_end - t_start)
            scores, indices = torch.topk(sim_matrix, k=sub_k, dim=1)
            
            # Adjust indices relative to global target index
            indices = indices + t_start
            
            all_top_scores.append(scores)
            all_top_indices.append(indices)
            
            del t_tensor, sim_matrix
            torch.cuda.empty_cache()
            
        # Merge results across target chunks
        all_scores_cat = torch.cat(all_top_scores, dim=1)
        all_indices_cat = torch.cat(all_top_indices, dim=1)
        
        final_scores, final_sub_indices = torch.topk(all_scores_cat, k=actual_k, dim=1)
        final_indices = torch.gather(all_indices_cat, 1, final_sub_indices)
        
        top_scores[i:i+batch_size] = final_scores.cpu().numpy()
        top_indices[i:i+batch_size] = final_indices.cpu().numpy()
        
        del q_tensor, all_scores_cat, all_indices_cat, final_scores, final_sub_indices
        torch.cuda.empty_cache()
        
    return top_indices, top_scores

def generate_candidates_gpu(df_source1, df_targets, top_k=10, output_parquet_path=None):
    """
    Generate candidate pairs by querying df_source1 against df_targets.
    Uses GPU PyTorch CUDA for cosine search and streams directly to disk via Parquet.
    """
    print("Generating candidates using GPU Acceleration...")
    
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU'})")
    
    countries = df_source1['clean_country'].unique()
    
    writer = None
    total_pairs = 0
    
    for country in countries:
        print(f"\nProcessing country: '{country}'")
        s1_country = df_source1[df_source1['clean_country'] == country]
        target_country = df_targets[df_targets['clean_country'] == country]
        
        if len(s1_country) == 0 or len(target_country) == 0:
            continue
            
        s1_text = (s1_country['clean_name'] + " " + s1_country['clean_address']).fillna("")
        target_text = (target_country['clean_name'] + " " + target_country['clean_address']).fillna("")
        
        print("Fitting TF-IDF Vectorizer...")
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=30000, dtype=np.float32)
        target_tfidf = vectorizer.fit_transform(target_text)
        s1_tfidf = vectorizer.transform(s1_text)
        
        actual_k = min(top_k, len(target_country))
        
        print(f"Running GPU top-{actual_k} search across {len(s1_country)} queries and {len(target_country)} targets...")
        
        if use_cuda:
            top_indices, top_scores = gpu_topk_cosine_similarity(s1_tfidf, target_tfidf, top_k=actual_k, batch_size=2000, device=device)
        else:
            # Fallback to sklearn NearestNeighbors if CPU
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=actual_k, metric='cosine', n_jobs=-1)
            nn.fit(target_tfidf)
            dist, top_indices = nn.kneighbors(s1_tfidf)
            top_scores = 1.0 - dist
            
        s1_ids = s1_country['entity_id'].values
        target_ids = target_country['entity_id'].values
        
        # Flatten to dataframe in chunks to prevent RAM overflow
        print("Formatting and streaming candidate pairs...")
        chunk_s1 = []
        chunk_target = []
        chunk_score = []
        
        for i in range(len(s1_ids)):
            curr_s1 = s1_ids[i]
            for j in range(actual_k):
                t_idx = top_indices[i, j]
                score = top_scores[i, j]
                chunk_s1.append(curr_s1)
                chunk_target.append(target_ids[t_idx])
                chunk_score.append(score)
                
        df_chunk = pd.DataFrame({
            'source1_id': chunk_s1,
            'target_id': chunk_target,
            'blocking_score': np.array(chunk_score, dtype=np.float32)
        })
        
        total_pairs += len(df_chunk)
        print(f"Generated {len(df_chunk)} pairs for country '{country}'. Total candidate pairs: {total_pairs}")
        
        del target_tfidf, s1_tfidf, s1_text, target_text, top_indices, top_scores, chunk_s1, chunk_target, chunk_score
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
            
    return df_chunk
