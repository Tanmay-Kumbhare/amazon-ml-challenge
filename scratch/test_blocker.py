import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

print("Creating dummy sparse matrices...")
# 6.1M targets, 50,000 vocab, density ~ 0.0005 (25 non-zeros per row)
n_targets = 1000000 # test with 1M
n_vocab = 50000
nnz_per_row = 25

rows = np.repeat(np.arange(n_targets), nnz_per_row)
cols = np.random.randint(0, n_vocab, n_targets * nnz_per_row)
data = np.random.rand(n_targets * nnz_per_row)
target_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_targets, n_vocab))

# 50 S1 entities
b_size = 50
rows_s1 = np.repeat(np.arange(b_size), nnz_per_row)
cols_s1 = np.random.randint(0, n_vocab, b_size * nnz_per_row)
data_s1 = np.random.rand(b_size * nnz_per_row)
s1_matrix = sp.csr_matrix((data_s1, (rows_s1, cols_s1)), shape=(b_size, n_vocab))

print("1. Dense Matmul (Old Way):")
start = time.time()
s1_dense = s1_matrix.toarray().T
sim_dense = target_matrix.dot(s1_dense)
print(f"Time: {time.time() - start:.4f}s")

print("2. Sparse Matmul (New Way):")
start = time.time()
s1_sparse_t = s1_matrix.T # csr transposed is csc
sim_sparse = target_matrix.dot(s1_sparse_t)
# Extract top k for a column
col = sim_sparse.tocsc()[:, 0]
print(f"Time: {time.time() - start:.4f}s")
