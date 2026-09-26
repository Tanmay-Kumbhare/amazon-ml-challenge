"""
AUDIT PART 2 (fixed): TF-IDF vs HashingVectorizer memory/speed benchmark.
Uses ONLY 100k and 500k rows. Skips 1M TF-IDF to avoid OOM.
flush=True so output appears immediately in log.
"""
import pandas as pd
import numpy as np
import tracemalloc
import time
import gc
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

def p(msg):
    print(msg, flush=True)

p("=" * 60)
p("SECTION C: TF-IDF vs HashingVectorizer BENCHMARK")
p("=" * 60)
p("Loading source2 rows for benchmark...")
sys.stdout.flush()

df_s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t", nrows=500_000,
    keep_default_na=False, dtype=str
)
target_text = (df_s2["business_name"] + " " + df_s2["business_address"]).fillna("").tolist()
del df_s2; gc.collect()
p(f"Loaded {len(target_text)} rows.")

for N in [100_000, 500_000]:
    subset = target_text[:N]
    p(f"\n--- Target N={N:,} ---")
    sys.stdout.flush()

    # TF-IDF (only at N<=500k to avoid OOM)
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                              max_features=20_000, dtype=np.float32)
        mat = vec.fit_transform(subset)
        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        p(f"  [TF-IDF]  Time={elapsed:.1f}s  PeakRAM={peak/(1024**2):.0f}MB  "
          f"Shape={mat.shape}  NNZ={mat.nnz:,}")
        del mat, vec
    except MemoryError:
        tracemalloc.stop()
        p(f"  [TF-IDF]  *** MemoryError at N={N:,} ***")
    gc.collect()
    sys.stdout.flush()

    # Hashing
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                            n_features=2**16, dtype=np.float32,
                            alternate_sign=False)
    mat = vec.transform(subset)
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    p(f"  [Hashing] Time={elapsed:.1f}s  PeakRAM={peak/(1024**2):.0f}MB  "
      f"Shape={mat.shape}  NNZ={mat.nnz:,}")
    del mat, vec
    gc.collect()
    sys.stdout.flush()

p("\nDone with Section C.")
