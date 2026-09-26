# Amazon ML Challenge 2026 – Complete Session Notes
> Last updated: 2026-09-26 15:59 IST  
> Read this file at the START of any new conversation to pick up exactly where we left off.

---

## 1. Competition Overview

| Item | Value |
|------|-------|
| Competition | Amazon ML Challenge 2026 (Unstop platform) |
| Deadline | **27 September 2026 at 11:59 PM IST** (~32 hours from now) |
| Metric | **Macro-averaged F0.5** (precision weighted 2× over recall) |
| Daily submission limit | **5 per day** |
| Submissions used today | **2** (1 failed on header, 1 scored **0.384**) |
| Remaining today | **3** |

### Key Metric Rules
- F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
- **Singletons** (S1 entities with no real match): predicting **empty** = score of **1.0** for that entity
- False merges are penalized **2× more** than missed matches
- ~36% of test entities ARE singletons (626,849 of 1,732,544)

### Data Scale
| File | Rows |
|------|------|
| `dataset/train/train_source1.tsv` | 2,206,821 |
| `dataset/train/train_source2.tsv` | ~4M |
| `dataset/train/train_source3.tsv` | ~6M |
| `dataset/test/test_source1.tsv` | **1,732,544** (every row MUST appear in submission) |
| `dataset/test/test_source2.tsv + test_source3.tsv` combined | **9,969,589** targets |
| Countries in test | `us`, `france`, `india` (france is UNSEEN in training) |

---

## 2. Hardware & Environment

| Item | Detail |
|------|--------|
| Machine | Lenovo LOQ |
| CPU | Intel i5-13450HX |
| GPU | NVIDIA RTX 3050 6GB VRAM |
| RAM | **16 GB** ← main bottleneck |
| Python env | `D:\AmazonML\venv` (activate with `D:\AmazonML\venv\Scripts\Activate.ps1`) |
| Project root | `C:\Users\Tanmay\Downloads\6ab10eb3b23ba_student_resource\student_resource` |

### Why GPU is NOT Used
The pipeline is 99% variable-length string manipulation and hash-map lookups.
GPUs need dense matrix math. Moving 10M text records to 6GB VRAM is a bigger
bottleneck than just using the CPU. CPU is the right choice here.

---

## 3. Project Structure

```
student_resource/
├── dataset/
│   ├── train/   (train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv)
│   └── test/    (test_source1.tsv, test_source2.tsv, test_source3.tsv)
├── models/
│   ├── lgbm_model.txt          ← TRAINED LightGBM model (DO NOT DELETE)
│   └── best_threshold.txt      ← optimal threshold (value: 0.60)
├── output/
│   ├── matching_results.tsv    ← current best submission (rule-based, score 0.384)
│   └── candidate_pairs.tsv     ← current candidate pairs (rule-based)
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── blocking.py             ← MODIFIED: switched to sparse-sparse matmul
│   ├── features.py
│   ├── train.py                ← MODIFIED: saves model to models/lgbm_model.txt
│   └── inference.py
├── main.py                     ← MODIFIED: added --train-sample, --skip-train
├── run_fast_submission.py      ← the rule-based heuristic script (already ran successfully)
├── create_zip.py               ← bundles final submission zip
├── utils/
│   └── validate_submission.py  ← official validator (use before every upload)
├── Documentation_template.md
└── SESSION_NOTES.md            ← THIS FILE
```

---

## 4. What We Built & Why

### Phase 1 – LightGBM ML Pipeline (`main.py`)
**Goal:** Train a high-precision ML model using TF-IDF blocking + string similarity features + LightGBM classifier.

**Architecture:**
1. **Blocking** (`src/blocking.py`): TF-IDF with `char_wb` (3,4) n-grams. For each S1 entity, finds top-30 most similar targets using cosine similarity.
2. **Feature Extraction** (`src/features.py`): For each candidate pair, computes:
   - `name_jaro_winkler`, `name_levenshtein`, `name_token_jaccard`, `name_exact_match`, `name_len_diff`
   - `address_jaro_winkler`, `address_levenshtein`, `address_token_jaccard`, `address_exact_match`, `address_len_diff`
   - `blocking_score` (TF-IDF cosine similarity from step 1)
3. **LightGBM** (`src/train.py`): Binary classifier. Trained on 50k S1 entities (1.5M pairs).

**Training Result:**
- Trained on: 50,000 S1 entities (sampled from 2.2M)
- Validation F0.5: **0.8638**
- Optimal threshold: **0.60**
- Model saved: `models/lgbm_model.txt`

**Why we couldn't run full test inference with this:**
- Full TF-IDF blocking on 1.7M test entities against 9.9M targets requires
  dense matrix multiplication → **96 hours** on this hardware → not feasible.

### Phase 2 – Fast Heuristic Baseline (`run_fast_submission.py`)
**Goal:** Get a valid baseline submission on the leaderboard quickly.

**Method:**
- Inverted index on exact business name tokens + 3-character shingles.
- JaroWinkler similarity (threshold > 0.90) + address token jaccard check.
- Caps candidates at **30 per entity** (important for ranking criterion).
- Processes each country separately (US, France, India) to stay within RAM.

**Results:**
- Runtime: ~61 minutes total (US: 31min, France: 9min, India: 20min)
- Output: 1,732,544 rows, 626,849 singletons, 1,105,695 matched
- Submission score: **0.384** (low because strict thresholds missed many true positives)

### Phase 3 – ML Inference (CURRENT GOAL – NOT YET COMPLETE)
**Goal:** Use the saved LightGBM model on the full test set to jump score.

**The blocker we hit:**
The `main.py --skip-train` command was supposed to:
1. Load saved `lgbm_model.txt` ✅ (this part works)
2. Skip training ✅ (fixed in code)
3. Load test data and preprocess ← **CRASHING HERE**

**Root Cause of Current Error:**
`src/preprocess.py` uses `df['business_name'].apply(clean_text)` which calls
`pandas.apply()` on 10M rows. This materialises a Python-object array of 10M
strings → **OOM crash** with:
```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 78.7 MiB
for an array with shape (10320219,) and dtype uint64
```
The fix: replace `.apply()` with vectorized `pandas str` operations which
work column-by-column and don't materialise intermediate Python objects.

---

## 5. The Critical Bug to Fix in New Session

### File: `src/preprocess.py`
**Current broken code (lines 59-64):**
```python
def preprocess_dataframe(df):
    print("Preprocessing dataframe...")
    if 'business_name' in df.columns:
        df['clean_name'] = df['business_name'].apply(clean_text).apply(normalize_legal_suffixes)
    if 'business_address' in df.columns:
        df['clean_address'] = df['business_address'].apply(clean_text)
    if 'country' in df.columns:
        df['clean_country'] = df['country'].apply(lambda x: str(x).strip().lower() if pd.notna(x) else "")
    return df
```

**Why it crashes:** `.apply()` on 10M rows triggers pyarrow/pandas internals
that try to allocate a 10M-element int64 array → OOM on 16GB RAM.

**The Fix:** Replace with vectorized `str` operations:
```python
def preprocess_dataframe(df):
    print("Preprocessing dataframe...")
    if 'business_name' in df.columns:
        s = df['business_name'].fillna('').astype(str).str.lower()
        s = s.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
        s = s.str.replace(r'\.(com|org|net|co|us|in)\b', '', regex=True)
        s = s.str.replace(r'[,.\-_()"\':|]', ' ', regex=True)
        s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
        # legal suffix normalization
        replacements = {
            r'\bpvt\b': 'private', r'\bltd\b': 'limited',
            r'\binc\b': 'incorporated', r'\bcorp\b': 'corporation',
            r'\bllc\b': 'llc', r'\bllp\b': 'llp',
            r'\bsarl\b': 'sarl', r'\bsas\b': 'sas'
        }
        for pat, rep in replacements.items():
            s = s.str.replace(pat, rep, regex=True)
        df['clean_name'] = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    if 'business_address' in df.columns:
        s = df['business_address'].fillna('').astype(str).str.lower()
        s = s.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
        s = s.str.replace(r'\.(com|org|net|co|us|in)\b', '', regex=True)
        s = s.str.replace(r'[,.\-_()"\':|]', ' ', regex=True)
        df['clean_address'] = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    if 'country' in df.columns:
        df['clean_country'] = df['country'].fillna('').astype(str).str.strip().str.lower()
    return df
```
**Why this works:** pandas `str` operations are C-level vectorized. They
process data in fixed-width internal arrays without creating Python object
arrays per element. No OOM.

---

## 6. Also Fixed Already (Don't Touch These)

### `main.py` – `--skip-train` now correctly skips training data loading
The `if not args.skip_train:` block was added so that when `--skip-train` is
passed, the script **never calls `load_data(is_train=True)`** and therefore
never tries to preprocess the 10M training targets.

### `src/blocking.py` – Switched to Sparse-Sparse Matmul
- **Old:** `target_matrix.dot(s1_dense)` → dense column extraction → OOM for chunk_size > 50
- **New:** `target_matrix.dot(s1_sparse_t).tocsc()` → pure sparse, reads CSC column indices directly
- chunk_size increased from 50 → 10,000 (safe because sparse uses far less RAM)

### `run_fast_submission.py` – Header and delimiter fixed
- `source1_id` → `source1_entity_id`  
- Semicolon separators `;` → comma separators `,`

---

## 7. Exact Commands to Run in New Session

**Step 1: Fix preprocess.py (apply the vectorized version above)**

**Step 2: Verify the fix works on a tiny pilot first:**
```powershell
python main.py --skip-train --pilot 1000
```
This should finish in < 5 minutes. If it prints "Pipeline completed successfully!" with no error → proceed.

**Step 3: Run full test inference:**
```powershell
python main.py --skip-train
```
Expected runtime: ~60–90 minutes (most of it is the sparse TF-IDF blocker).

**Step 4: Validate:**
```powershell
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
Must print: `PASS — no blocking issues found. Safe to submit.`

**Step 5: Submit `output/matching_results.tsv`** to the Unstop portal.

---

## 8. Submission Format Rules (Exact)

### `output/matching_results.tsv`
```
source1_entity_id[TAB]matched_entity_ids
S1-714132312[TAB]S2-786029403,S3-123456
S1-106407869[TAB]
```
- Column 1 header: `source1_entity_id` (NOT `source1_id`)
- Column 2 header: `matched_entity_ids`
- Separator: TAB between columns, COMMA between IDs
- Every S1 entity must appear (empty second column = singleton)
- No S1 IDs in the matched list (only S2- and S3-)

### `output/candidate_pairs.tsv`
```
source1_entity_id[TAB]candidate_entity_ids
S1-714132312[TAB]S2-786029403,S3-123456,S2-999
S1-106407869[TAB]
```
- Same rules, different column name: `candidate_entity_ids`
- Every matched ID in matching_results must appear in candidates

### Final Submission ZIP Structure
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/             (all .py source files)
│       ├── main.py
│       ├── run_fast_submission.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```
Run `python create_zip.py` to generate this automatically.

---

## 9. Score Improvement Plan (Priority Order)

| Priority | Action | Expected Score Gain | Time to Run |
|----------|--------|-------------------|-------------|
| 🔴 **NOW** | Fix preprocess.py, run `main.py --skip-train` | **0.384 → ~0.75+** | ~90 min |
| 🟡 Next | If score < 0.75, train with more data: `main.py --train-sample 200000` | +0.05–0.10 | ~3 hours |
| 🟡 Next | Tune LightGBM threshold (try 0.50, 0.55, 0.65) | +0.02–0.05 | ~30 min |
| 🟢 Optional | Add more features (token overlap ratio, prefix match, numeric token match) | +0.02–0.05 | ~1 hour |
| 🟢 Optional | Fill in `Documentation_template.md` for final zip | Required for final ranking | ~30 min |

---

## 10. Things That Were Tried and Failed (Do NOT Repeat)

| Attempt | What Happened | Lesson |
|---------|--------------|--------|
| `python main.py --train-sample 50000` | OOM on chunk_size=200 dense matmul | Fixed: sparse-sparse matmul |
| `python run_fast_submission.py` v1 | MemoryError: 4.7M target dict exceeded RAM | Fixed: compute tokens on-the-fly |
| First submission (header `source1_id`) | **Rejected by portal** | Fixed: must be `source1_entity_id` |
| Semicolon separators in IDs | Would fail validation | Fixed: must use commas |
| `python main.py --skip-train` without preprocess fix | OOM in `preprocess_dataframe` on 10M targets | **Current bug to fix** |

---

## 11. LightGBM Model Details

- **Saved at:** `models/lgbm_model.txt`
- **Threshold:** `models/best_threshold.txt` → `0.60`
- **Validation F0.5:** 0.8638
- **Features used (11 total):**
  - `blocking_score`, `name_jaro_winkler`, `name_levenshtein`, `name_len_diff`
  - `name_token_jaccard`, `name_exact_match`, `address_jaro_winkler`
  - `address_levenshtein`, `address_len_diff`, `address_token_jaccard`, `address_exact_match`
- **Trained on:** 50,000 S1 entities → ~1.5M candidate pairs
- **LGBM Params:** `objective=binary`, `metric=binary_logloss`, `num_leaves=31`, `lr=0.1`, `feature_fraction=0.8`

---

## 12. Key Files Modified in This Session

| File | Change Made |
|------|-------------|
| `src/blocking.py` | Switched from dense to sparse-sparse matmul; chunk_size 50→10000 |
| `src/train.py` | Added model.save_model() and best_threshold.txt persistence |
| `main.py` | Added `--train-sample`, `--skip-train` args; fixed conditional training block |
| `run_fast_submission.py` | Full heuristic matcher; fixed header and delimiter |
| `create_zip.py` | New file: builds final submission zip |
| `src/preprocess.py` | **NEEDS FIX** – replace .apply() with vectorized str operations |

---

## 14. Complete File Inventory

### ✅ CORE / PRODUCTION FILES (Do NOT delete or modify unless intentional)

| File | Purpose |
|------|---------|
| `main.py` | Master pipeline entry point. Run with `--skip-train` or `--train-sample N` |
| `run_fast_submission.py` | Rule-based heuristic matcher (already ran, scored 0.384) |
| `create_zip.py` | Builds the final `baseline_submission.zip` for portal submission |
| `src/config.py` | All paths, LGBM params, constants. `BLOCKING_TOP_K=30` |
| `src/preprocess.py` | **⚠️ NEEDS FIX** – text cleaning. Replace `.apply()` with vectorized `str` ops (see Section 5) |
| `src/blocking.py` | **Active blocker.** Sparse TF-IDF char_wb (3,4). chunk_size=10000 |
| `src/features.py` | Extracts 11 string similarity features per candidate pair |
| `src/train.py` | Trains LightGBM, saves model + threshold to `models/` |
| `src/inference.py` | Loads model, predicts, generates final TSV outputs |
| `src/__init__.py` | Package init |
| `utils/validate_submission.py` | Official validator. Always run before uploading |
| `models/lgbm_model.txt` | **THE TRAINED MODEL** – F0.5=0.8638 on validation |
| `models/best_threshold.txt` | `0.60` – classification threshold for LightGBM |
| `output/matching_results.tsv` | Current best submission file (rule-based, 0.384). WILL be overwritten by ML run |
| `output/candidate_pairs.tsv` | Current candidate pairs (rule-based). WILL be overwritten by ML run |
| `requirements.txt` | `pandas, numpy, jellyfish, lightgbm, scikit-learn, tqdm, scipy` |
| `README.md` | Brief project README |
| `Documentation_template.md` | Fill this in for the final zip submission |
| `SESSION_NOTES.md` | THIS FILE – read at start of every new conversation |
| `AMAZON_ML_COMPLETE_SETUP.md` | Earlier high-level setup doc (superseded by SESSION_NOTES.md) |

---

### 🗄️ ARCHIVED / OLD BLOCKING VARIANTS (in `src/` – keep but do not use)

| File | Why It Exists | Status |
|------|--------------|--------|
| `src/blocking_old.py` | Original dense matmul blocker – crashes on 16GB RAM | **Archived – do not use** |
| `src/blocking_fast.py` | Intermediate attempt – inverted index approach | **Archived – do not use** |
| `src/blocking_gpu.py` | GPU-based attempt – abandoned (GPU not useful for text hashing) | **Archived – do not use** |
| `src/blocking_tfidf.py` | Earlier TF-IDF variant before sparse-sparse fix | **Archived – do not use** |

> `src/blocking.py` is the **only active blocker**. All others are dead code.

---

### 🧪 SCRATCH / TEST SCRIPTS (in `scratch/` and root – safe to ignore)

| File | What it was used for |
|------|----------------------|
| `scratch/analyze.py` | Early data exploration |
| `scratch/fast_analyze.py` | Quick statistics on dataset sizes |
| `scratch/audit_actual_blocking.py` | Verified that the TF-IDF blocker was actually finding real matches |
| `scratch/audit_part1_gt_f05.py` | Computed ground-truth F0.5 ceiling on training data |
| `scratch/audit_part2_vectorizer.py` | Tested TF-IDF vocabulary size vs recall tradeoffs |
| `scratch/audit_part3_blocking.py` | Tested blocking recall on a 5k sample → result: **94.01% recall** |
| `scratch/audit_parts2_3_final.py` | Combined audit script |
| `scratch/test_blocker.py` | Speed test: dense matmul vs sparse-sparse (written in this session) |
| `scratch/test_lgbm_stopping.py` | Tested LightGBM early stopping parameters |
| `scratch/test_split.py` | Verified train/val split logic |
| `scratch/report.json` | Output from one of the audit scripts |
| `audit_and_benchmark.py` | Root-level comprehensive audit/benchmark script |
| `audit_blocking_recall.py` | Root-level blocking recall measurement script |
| `audit_fast_blocking.py` | Root-level fast-blocking test |
| `audit_v2.py` | Root-level v2 audit |
| `progress.txt` | Text notes from earlier in the session |

> All scratch/audit files are **read-only reference** – no need to run any of them.

---

### 📤 OUTPUT FILES

| File | Description |
|------|-------------|
| `output/matching_results.tsv` | **Upload this to portal.** 1,732,544 rows. Header: `source1_entity_id\tmatched_entity_ids`. Score: 0.384 (rule-based). Will be replaced after ML run. |
| `output/candidate_pairs.tsv` | Required in final zip. Header: `source1_entity_id\tcandidate_entity_ids`. 1,732,544 rows. |
| `baseline_submission.zip` | Zip bundle (may be outdated – regenerate with `python create_zip.py` after ML run) |


---

## 15. The Single Most Important Thing

> **The `models/lgbm_model.txt` file is our competitive advantage.**
> It validates at **F0.5 = 0.8638** on held-out data.
> All we need to do is run it on the test data without running out of RAM.
>
> **3-step plan:**
> 1. Fix `src/preprocess.py` (Section 5 has the exact replacement code)
> 2. Run `python main.py --skip-train`
> 3. Upload `output/matching_results.tsv` to the portal
>
> That's it. Everything else is already done.
