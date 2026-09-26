# DATASET RECONNAISSANCE REPORT
## Amazon ML Challenge: Business Entity Resolution

This report summarizes an extensive, read-only inspection of the dataset provided for the Business Entity Resolution challenge. No data was modified during this analysis.

---

### 1. DATASET OVERVIEW

The dataset consists of 4 training files and 3 test files in TSV format. All columns are inferred as `string`.

| File | Size (MB) | Rows | Columns |
|------|-----------|------|---------|
| `train_source1.tsv` | 200.34 | 2,206,821 | `entity_id`, `business_name`, `business_address`, `country` |
| `train_source2.tsv` | 466.63 | 5,034,616 | `entity_id`, `business_name`, `business_address`, `country` |
| `train_source3.tsv` | 480.37 | 5,285,603 | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv`| 121.13 | 2,206,821 | `source1_entity_id`, `matched_entity_ids` |
| `test_source1.tsv` | 166.91 | 1,732,544 | `entity_id`, `business_name`, `business_address`, `country` |
| `test_source2.tsv` | 485.86 | 4,887,273 | `entity_id`, `business_name`, `business_address`, `country` |
| `test_source3.tsv` | 482.56 | 5,082,316 | `entity_id`, `business_name`, `business_address`, `country` |

**Missing Values:**
- **Source 1 (Train):** 0 missing values across all fields.
- **Source 2 (Train):** 2 missing `business_name`s, 168,967 missing `business_address`es.
- **Source 3 (Train):** 13 missing `business_name`s, 175,916 missing `business_address`es.
- **Ground Truth:** 123,247 Source 1 entities have missing (empty) matches, meaning they are singletons without matches in Source 2/3.
- **Test:** Similar missingness patterns observed in S2/S3 for addresses.

**Unique Values & Duplicates:**
- In `train_source1`, there are 1.53M unique business names and 2.13M unique addresses out of 2.2M rows. This implies many businesses share the same name (e.g., franchises) or multiple variations map to identical addresses.
- `train_source2` and `train_source3` have roughly 4.4M-4.6M unique names/addresses for their ~5M rows.

**Sample Records (train_source1.tsv):**
- `S1-925783039` | `Orelee's Barbershop` | `1795 Westchester Drive, High Point, NC` | `US`
- `S1-755362802` | `Prabhav Business Center` | `797, Lake Town Block A, Kolkata, Howrah, West Bengal` | `India`

---

### 2. FIELD-LEVEL ANALYSIS

**`business_name`:**
- **String lengths:** Range from 2 characters up to 72+ characters (avg ~25).
- **Missing values:** Minimal (0 in S1, <100 in S2/S3).
- **Unusual characters:** Extensive presence of non-Latin characters (Devanagari, Kannada, etc.), Unicode anomalies, and symbols (e.g., `+`, `&`, `[Partners]`, `-- Holloway Peak`).
- **Capitalization:** Highly inconsistent (e.g., `BLUE DINER MIDTOWN CORP` vs `wilfordhancock.com` vs `Doit Manufacturersprivate Priafte Limited`).
- **Punctuation:** Commas, hyphens, periods, ampersands, and quotes are used arbitrarily.
- **Encoding/Unicode:** Frequent transliteration noise and potential encoding artifacts (e.g., `Léarning Center`, `Pvt.`).

**`business_address`:**
- **String lengths:** Extremely variable, from 6 characters to over 225 characters (avg ~50).
- **Missing values:** Prominent in Source 2 (~169k) and Source 3 (~176k). Source 1 has none.
- **Formatting:** Varies drastically from well-structured US addresses (`139 Walnut Street, Somerville`) to highly unstructured Indian addresses (`Shreeji Ind Park, Milatary Road, Alav, Ranpur, Botad, Bhavnagar, Gujarat`).

**`country`:**
- **Values:** The field is generally clean and consistent in length (2-6 chars). US and India are the primary countries in the training set.

---

### 3. SOURCE-SPECIFIC ANALYSIS

- **Source 1:** Generally the cleanest source. It acts as the anchor dataset. It has no missing addresses, and capitalization follows expected Title Case conventions mostly.
- **Source 2:** Highly noisy. Features excessive ALL CAPS names and addresses (e.g., `164- MORRIS LANE, TONEY, AL`). Frequently contains raw Hindi script (e.g., `राम मार्केटिंग प्राइवेट लिमिटेड`). Missing over 168k addresses. Address component order is often scrambled.
- **Source 3:** Very noisy. Contains web domain names mapped as business names (e.g., `delogistics.com`, `wilfordhancock.com`). Addresses are frequently missing or incomplete. Spelling errors are noticeable (e.g., `Priafte Limited`, `Léarning Center`).

Source 1 fields appear the most reliable, while Source 2 and 3 names/addresses must be treated with low trust and high tolerance for noise.

---

### 4. BUSINESS NAME NOISE ANALYSIS

Real dataset observations:
- **Legal Suffixes:** `Corp` vs `Corporation`, `Pvt Ltd` vs `Private Limited`, `Inc.` vs `Inc`, `LLC` vs `LLP`, `S.A.S` vs `SARL` (in test data).
- **Punctuation & Typos:** `Orelee's Barbershop`, `B+ Retail Inc`, `-- Holloway Peak Inc Seafood`, `Doit Manufacturersprivate Priafte Limited`.
- **Domain Names / DBAs:** Many Source 3 entities appear as URLs (e.g., `wilfordhancock.com`) corresponding to actual names in Source 1/2.
- **Transliteration:** E.g., `आदित्य प्रॉपर्टीज एलएलपी` (Aditya Properties LLP).

---

### 5. ADDRESS NOISE ANALYSIS

Real dataset observations:
- **Abbreviations:** `Rd` vs `Road`, `St` vs `Street`, `Fl` vs `Floor`.
- **Missing components:** Addresses often miss the PIN/postal code entirely. Sometimes the city or state is missing, or they just list `East Delhi, DL`.
- **Reordering:** `Kolkata, West Bengal, Howrah...` vs `Howrah, West Bengal, Kolkata`.
- **Granularity:** S2 and S3 addresses are sometimes just a street or a city, whereas S1 has the full door number and state.

---

### 6. COUNTRY ANALYSIS

- **Training Set Countries:** `US`, `India`
- **Test Set Countries:** `US`, `India`, `France`
- **Challenge Requirement Context:** The test set introduces a completely unseen country (`France`). This is a critical observation! We cannot build models that hardcode US/India states or ZIP logic exclusively. The system must treat the country as an *open set* and rely on generic string/token similarities rather than country-specific parsers.

---

### 7. GROUND TRUTH ANALYSIS

- **Total S1 Entities:** 2,206,821
- **Matched Relationships:** 7,638,365 total individual matches across all S1 entities.
- **Match Distribution:**
  - **Zero Matches (Singletons):** 123,247 entities
  - **One Match:** 119,157 entities
  - **Multiple Matches:** 1,964,417 entities
  - **Maximum Matches for one S1 Entity:** 11 matches
- **Source Distributions in GT:**
  - `S1 -> S2` relationships exist for ~1.91M S1 entities
  - `S1 -> S3` relationships exist for ~1.94M S1 entities
  - `S1 -> Both (S2 and S3)` overlap for ~1.77M S1 entities.

*Observation:* Ground truth indicates that a single S1 entity often resolves to multiple S2 and multiple S3 records (likely duplicates within S2 and S3).

---

### 8. ENTITY RESOLUTION OBSERVATIONS

- **What makes records likely the same:** High token overlap in name (excluding stop words and suffixes like LLC), combined with geographic proximity (same city/state) even if street addresses differ wildly.
- **What makes records likely different:** Different states/countries, or distinctly different non-suffix name tokens (e.g., `Orelee's` vs `O'Reilly's`).
- **Noisy Fields:** `business_address` in S2/S3 is missing frequently and poorly formatted. `business_name` contains domain names and translations.
- **Sufficiency:** Name alone is NOT sufficient (many identical generic names). Address alone is NOT sufficient (businesses move, or multiple businesses share an office). Combined, they provide strong signals. The `country` acts as a perfect high-level blocking key.

---

### 9. CLEANING / PREPROCESSING RECOMMENDATIONS

**Recommended Actions:**
- **Unicode/Transliteration Normalization:** Map Devanagari, Kannada, and French accented characters to ASCII equivalents where possible (e.g., `é` -> `e`, transliterate Hindi to English if libraries allow, or just rely on character unigrams/bigrams that are resilient).
- **Case Normalization:** Convert all text to lowercase.
- **Whitespace & Punctuation:** Remove redundant spaces, strip generic punctuation (`-`, `.`, `,`, `&`->`and`).
- **Legal Suffix Standardization:** Map `pvt`, `private`, `ltd`, `limited`, `inc`, `llc` to standardized empty strings or consistent tokens.
- **URL Handling:** Strip `.com`, `www.`, `http` from S3 names.

**Actions to AVOID:**
- **DO NOT** drop records with missing addresses.
- **DO NOT** use strict US ZIP code extractors, as they will fail on French or Indian records.

---

### 10. FEATURE ENGINEERING RECOMMENDATIONS

**A. Name Features:**
- Token Jaccard Similarity, TF-IDF Cosine Similarity (character n-grams to bypass typos), Jaro-Winkler (good for short strings), Levenshtein distance ratio.

**B. Address Features:**
- Token overlap, Longest Common Subsequence length, Cosine similarity. Number extraction match (e.g., do they share the same building number "1795"?).

**C. Country Features:**
- Exact match boolean. (Crucial feature).

**D. Cross-field Features:**
- Compare `business_name` tokens against `business_address` tokens (sometimes a business name is pushed into the address field in messy data).

**E. Source-Specific Features:**
- Boolean flags indicating if the pair is S1-S2 or S1-S3, as noise levels differ.

---

### 11. BLOCKING / CANDIDATE GENERATION RECOMMENDATIONS

With ~2M S1 records and ~10M S2+S3 records, computing all pairs (20 Trillion) is impossible.

**Recommended Blocking Strategies:**
1. **Country Blocking:** Only generate candidates within the same country.
2. **TF-IDF / MinHash LSH:** Vectorize character trigrams of the `business_name` and use Locality Sensitive Hashing to find approximate neighbors.
3. **Multi-pass Blocking:**
   - Pass 1: Exact normalized name match.
   - Pass 2: Exact phone/zip match (if extractable).
   - Pass 3: First 4 characters of name + State/City token match.

*Trade-off:* We need high recall (catching all 7.6M relationships) while reducing candidate size to maybe 50M-100M pairs.
*Danger:* Blocking strictly on postal codes will drop all S2/S3 missing-address records.

---

### 12. MODEL RECOMMENDATIONS

- **Gradient Boosting (XGBoost / LightGBM):** Highly recommended. Given tabular pairwise features (similarities, lengths, overlaps), tree-based models excel at finding non-linear combinations (e.g., "if name similarity > 0.9 and address is missing, still predict Match").
- **Contrastive Learning (Siamese Networks):** Using sentence-transformers could work, but tree-based models on string distance features are much faster to train and iterate on for this scale.

---

### 13. VALIDATION STRATEGY

- **Split Strategy:** Group by `source1_entity_id`. Ensure an S1 entity and all its GT relationships are entirely in the train split or entirely in the validation split to prevent data leakage.
- **Evaluation Metric:** The challenge specifies `F0.5` score (precision-heavy). The validation code must reflect this!
- **Threshold Tuning:** A standard 0.5 threshold might not maximize F0.5. We should plot precision-recall curves and pick a higher threshold (e.g., 0.7-0.8) to prioritize precision.

---

### 14. FINAL RECOMMENDED ROADMAP

1. **Dataset understanding:** (Completed - This Report)
2. **Cleaning/preprocessing:** Implement fast string normalizers (lowercase, regex stripping, transliteration).
3. **Candidate generation/blocking:** Build TF-IDF + nearest neighbors (e.g., FAISS or sparse matrix dot products) to generate candidate pairs.
4. **Candidate-pair feature engineering:** Compute Jaro-Winkler, Jaccard, lengths, and exact matches for candidates.
5. **Training data construction:** Join candidates with GT to label pairs as `1` (Match) or `0` (Non-match).
6. **Matching model:** Train a LightGBM pairwise classifier.
7. **Probability/threshold tuning:** Use the validation set to find the threshold maximizing F0.5.
8. **Full test inference:** Run the pipeline on `test_source1/2/3`.
9. **Generate candidate_pairs.tsv & matching_results.tsv.**
10. **Run validate_submission.py** to ensure formatting is correct before submission.
