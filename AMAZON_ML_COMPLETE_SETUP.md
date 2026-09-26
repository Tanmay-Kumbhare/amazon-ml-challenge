# Amazon ML Challenge --- Complete Setup Guide

This guide is for setting up the **Amazon ML Challenge Business Entity
Resolution** project on a new PC and running the same pipeline used in
this repository.

> **Important:** The GitHub repository contains the code only. Do
> **NOT** commit or upload the official Amazon challenge dataset,
> credentials, or other restricted/private challenge material to this
> public repository. Obtain the dataset through the permitted official
> challenge mechanism and place it locally as described below.

------------------------------------------------------------------------

## 1. What this project does

The pipeline solves the Business Entity Resolution task:

-   **Source 1 (S1)** contains the canonical entities.
-   **Source 2 (S2)** and **Source 3 (S3)** contain noisy records.
-   The system generates likely S1 → S2/S3 candidate matches.
-   A LightGBM matcher scores candidate pairs.
-   The final predictions are written in the required TSV format.

The current approach has three major stages:

``` text
Raw dataset
    ↓
Preprocessing / normalization
    ↓
Country-aware Top-K blocking
    ↓
Pair feature extraction
    ↓
LightGBM matcher
    ↓
Thresholding
    ↓
matching_results.tsv
candidate_pairs.tsv
```

The current blocker uses sparse character TF-IDF retrieval and keeps the
matrices sparse to avoid the memory problem caused by dense conversion /
huge vocabularies.

------------------------------------------------------------------------

# 2. Requirements

## Hardware

Recommended:

-   Windows 10/11 or Linux
-   16 GB RAM minimum
-   32 GB+ RAM preferred for large training runs
-   SSD strongly recommended
-   Multi-core CPU
-   NVIDIA GPU is **not required** for the current LightGBM pipeline

The full dataset is large, so disk space should also be kept available
for temporary/generated files.

## Software

Install:

-   Python 3.10+ (Python 3.12.x is suitable if the repository
    dependencies support it)
-   Git
-   VS Code or another IDE
-   PowerShell on Windows / Bash on Linux

------------------------------------------------------------------------

# 3. Clone the repository

Open a terminal:

``` bash
git clone https://github.com/Tanmay-Kumbhare/amazon-ml-challenge.git
cd amazon-ml-challenge
```

Check the files:

``` bash
git status
```

You should see the project code but not the official challenge dataset.

------------------------------------------------------------------------

# 4. Create a Python virtual environment

## Windows PowerShell

``` powershell
python -m venv venv
```

Activate it:

``` powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

``` powershell
.\venv\Scripts\Activate.ps1
```

You should see:

``` text
(venv)
```

at the beginning of the terminal prompt.

## Windows CMD

``` cmd
python -m venv venv
venv\Scripts\activate
```

## Linux/macOS

``` bash
python3 -m venv venv
source venv/bin/activate
```

------------------------------------------------------------------------

# 5. Install dependencies

Make sure the virtual environment is activated.

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not present in the repository yet, install the
project's required packages from the imports in the source code, or ask
the repository owner for the current requirements file.

After installation, verify:

``` bash
python --version
pip --version
```

------------------------------------------------------------------------

# 6. Add the challenge dataset

The dataset must be obtained through the allowed challenge
source/mechanism.

The expected local structure is:

``` text
amazon-ml-challenge/
│
├── dataset/
│   ├── train/
│   │   ├── train_ground_truth.tsv
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   └── train_source3.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

### Important

Do not rename the dataset files unless the code/configuration is also
changed.

Do not commit the `dataset/` directory to the public GitHub repository.

------------------------------------------------------------------------

# 7. Verify the dataset

From the project root:

``` powershell
dir dataset\train
dir dataset\test
```

You should have the expected files.

You can also run:

``` bash
python main.py --help
```

If the program displays the available arguments, the Python environment
and entry point are working.

------------------------------------------------------------------------

# 8. First run: small pilot

Do **not** start with the full dataset immediately.

First run:

``` bash
python main.py --pilot 5000
```

This is a small pipeline test.

Use it to verify:

-   dataset paths are correct
-   preprocessing works
-   blocking works
-   feature generation works
-   LightGBM training works
-   output generation works
-   there are no missing packages
-   the machine has enough memory

Let this run complete before attempting a larger run.

------------------------------------------------------------------------

# 9. Recommended larger pilot

After the 5,000-entity run succeeds:

``` bash
python main.py --pilot 50000
```

This gives a much more realistic performance test.

Monitor:

-   RAM usage
-   CPU usage
-   execution time
-   number of candidate pairs
-   blocking recall if the audit scripts are being used
-   validation Macro F0.5

Do not assume that a larger machine automatically means the pipeline is
correct. Always verify the metric and output format.

------------------------------------------------------------------------

# 10. Training sample

The current pipeline also supports a training sample argument:

``` bash
python main.py --train-sample 100000
```

This is intended for a larger training run without immediately
committing to the entire S1 training population.

Before using a large sample, confirm that:

1.  the 5,000 pilot works;
2.  the 50,000 pilot works;
3.  the machine has enough RAM/disk;
4.  the blocking stage completes;
5.  validation is being calculated correctly.

------------------------------------------------------------------------

# 11. Current matching architecture

The important pipeline components are:

### Step 1 --- Preprocessing

Names, addresses and other fields are normalized.

Typical operations include:

-   lowercasing
-   whitespace normalization
-   legal-suffix normalization
-   safe text normalization

Be careful not to remove meaningful address information such as
suite/unit numbers.

------------------------------------------------------------------------

### Step 2 --- Country-aware blocking

The audit established that the known training matches share the same
country, so country-aware blocking is used.

Within each country, the current blocker uses character-level TF-IDF
retrieval.

The blocker:

-   uses character n-grams;
-   keeps sparse matrices sparse;
-   processes S1 in chunks;
-   computes sparse similarity;
-   retrieves the Top-K target candidates;
-   avoids arbitrary candidate truncation.

The configured Top-K is currently:

``` text
30
```

Do not casually change this value without measuring its effect on
blocking recall.

------------------------------------------------------------------------

### Step 3 --- Pair features

Candidate pairs are converted into numeric matching features.

The current feature family includes similarity/distance information such
as:

-   name similarity
-   address similarity
-   Jaro-Winkler
-   Levenshtein-style distance
-   token Jaccard
-   length differences
-   other configured fields

------------------------------------------------------------------------

### Step 4 --- LightGBM matcher

The candidate-pair features are used to train a LightGBM binary
classifier.

The model predicts the probability that a candidate S2/S3 entity
corresponds to the S1 entity.

------------------------------------------------------------------------

### Step 5 --- Thresholding

The model probability is converted into a match/no-match decision using
a tuned threshold.

The current validated threshold was around:

``` text
0.65
```

However, **do not hard-code this as universally optimal**. If the
training/validation setup changes, retune the threshold using the
official evaluation definition.

------------------------------------------------------------------------

# 12. Important evaluation rule

The challenge evaluation is based on:

**Macro F0.5 per Source-1 entity**

It is NOT simply one global F0.5 over all candidate pairs.

The validation procedure therefore needs to:

1.  group predictions by S1 entity;
2.  calculate the F0.5 contribution for each S1 entity;
3.  average across S1 entities.

This matters particularly for entities with different numbers of
candidates.

Do not replace this with ordinary global `fbeta_score` without checking
the official challenge specification.

------------------------------------------------------------------------

# 13. Candidate pairs output

The required candidate output must preserve an entry for every S1
entity, including entities for which there are no candidates.

Conceptually:

``` text
source1_entity_id    candidate_entity_ids
S1-123               S2-456,S3-789
S1-124
S1-125               S2-999
```

The empty row is important.

Do not simply drop S1 entities with zero candidates.

------------------------------------------------------------------------

# 14. Matching results output

The matching results should also contain the required S1 entities.

Conceptually:

``` text
source1_entity_id    matched_entity_ids
S1-123               S2-456
S1-124
S1-125               S3-999
```

Keep the exact column names, delimiters and formatting required by the
official challenge instructions.

------------------------------------------------------------------------

# 15. Expected generated files

After a successful run, generated files may appear under:

``` text
output/
```

For example:

``` text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

These are generated artifacts and generally should not be committed to
the public repository unless there is a specific reason to version them.

------------------------------------------------------------------------

# 16. Running the audit scripts

The repository contains several audit/debug scripts.

Examples include:

``` text
src/audit_actual_blocking.py
src/audit_part1_gt_f05.py
src/audit_part2_vectorizer.py
src/audit_part3_blocking.py
src/audit_parts2_3_final.py
src/test_blocker.py
src/test_lgbm_stopping.py
src/test_split.py
```

Use these when debugging rather than immediately changing the main
pipeline.

For example:

``` bash
python src/test_blocker.py
```

or:

``` bash
python src/test_split.py
```

Check the script itself for its exact arguments before running it.

------------------------------------------------------------------------

# 17. Git workflow for two people

This repository is intended to be shared between the owner and
collaborator.

## Before starting work

Always pull the latest code:

``` bash
git pull
```

## After making a change

Check:

``` bash
git status
```

Then:

``` bash
git add .
git commit -m "Describe the change"
git push
```

Example:

``` bash
git add .
git commit -m "Optimize blocking performance"
git push
```

## If the other person pushed changes

Before starting new work:

``` bash
git pull
```

------------------------------------------------------------------------

# 18. Do not commit large/generated files

Before every push:

``` bash
git status
```

Make sure you are NOT accidentally committing:

``` text
dataset/
venv/
models/
output/
large .tsv files
large .pkl/.joblib files
credentials
API keys
```

If a file is large or generated locally, it usually belongs outside Git.

------------------------------------------------------------------------

# 19. If Git says a dataset file is already tracked

If the dataset was accidentally added before `.gitignore` was created,
adding it to `.gitignore` alone is not enough.

Remove it from Git tracking while keeping it on your PC:

``` bash
git rm -r --cached dataset/
```

Then:

``` bash
git add .gitignore
git commit -m "Stop tracking challenge dataset"
git push
```

If sensitive/restricted data has already been pushed to a public
repository, stop and clean the repository history before continuing.

------------------------------------------------------------------------

# 20. If a merge conflict happens

First:

``` bash
git status
```

Git will show the conflicting files.

Open those files and resolve the conflict.

Then:

``` bash
git add .
git commit -m "Resolve merge conflict"
git push
```

If you are unsure, do not use force-push commands such as:

``` bash
git push --force
```

without coordinating with the other person.

------------------------------------------------------------------------

# 21. Recommended collaboration rule

To avoid both people modifying the same code simultaneously:

### Person A

Works on:

``` text
blocking.py
```

### Person B

Works on:

``` text
train.py
```

Then commit and push separately.

For larger changes, use branches:

``` bash
git checkout -b feature/blocking-improvement
```

After completing the work:

``` bash
git add .
git commit -m "Improve blocking"
git push -u origin feature/blocking-improvement
```

Then create a Pull Request on GitHub.

------------------------------------------------------------------------

# 22. Performance expectations

The full dataset is very large.

The expensive part is not simply loading Python code; it is the
matching/search process over millions of records.

The current optimized blocker was introduced because the previous
approach attempted to create an excessively large TF-IDF vocabulary and
caused a memory error.

The current approach uses:

``` text
country partitioning
+
sparse character TF-IDF
+
chunked sparse dot products
+
Top-K retrieval
```

This reduces memory pressure significantly.

Still, the full run can take a long time depending on:

-   CPU
-   RAM
-   SSD speed
-   number of S1 entities
-   number of target records
-   Top-K
-   feature extraction implementation
-   LightGBM settings

**Do not interrupt a long run just because CPU usage looks uneven. Check
the logs and output first.**

------------------------------------------------------------------------

# 23. If the machine has more RAM/CPU

A stronger machine can be used for larger training/runs, but do not
blindly increase everything.

First run:

``` bash
python main.py --pilot 5000
```

Then:

``` bash
python main.py --pilot 50000
```

Then move to:

``` bash
python main.py --train-sample 100000
```

Only after these are verified should you consider a full-scale run.

------------------------------------------------------------------------

# 24. If the machine crashes with MemoryError

Do NOT immediately convert sparse matrices to dense arrays.

Look for code such as:

``` python
.toarray()
```

or:

``` python
.todense()
```

on large matrices.

The blocker is deliberately designed to remain sparse.

Also check:

-   RAM usage
-   number of candidates
-   chunk size
-   Top-K
-   pandas operations that create huge temporary arrays
-   unnecessary copies of large DataFrames

------------------------------------------------------------------------

# 25. If training is too slow

First determine which stage is slow:

``` text
Loading
↓
Preprocessing
↓
Blocking
↓
Feature extraction
↓
LightGBM training
↓
Prediction
↓
Output
```

Do not optimize a stage without measuring it.

The largest bottleneck in the current full pipeline can be candidate
generation/scoring over very large target populations.

------------------------------------------------------------------------

# 26. Important competition safety rules

This repository is for development and collaboration.

Always follow the official Amazon ML Challenge rules.

In particular:

-   Do not expose restricted challenge data publicly.
-   Do not share credentials or private access tokens.
-   Do not use prohibited external data or services.
-   Do not use another participant's solution/code if the rules prohibit
    it.
-   Do not assume that anything technically possible is
    competition-allowed.
-   Keep the final submission in the exact official format.
-   Treat the official challenge instructions as the source of truth if
    they differ from this README.

------------------------------------------------------------------------

# 27. Quick-start checklist for a new PC

A friend setting up the project should be able to follow this exact
sequence:

``` powershell
git clone https://github.com/Tanmay-Kumbhare/amazon-ml-challenge.git

cd amazon-ml-challenge

python -m venv venv

.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
```

Then obtain the permitted challenge dataset and place it under:

``` text
dataset/
    train/
    test/
```

Then test:

``` powershell
python main.py --help
```

Then:

``` powershell
python main.py --pilot 5000
```

If successful:

``` powershell
python main.py --pilot 50000
```

Then, after confirming the machine can handle it:

``` powershell
python main.py --train-sample 100000
```

------------------------------------------------------------------------

# 28. Final rule

**GitHub = code and configuration.**

**Local machine = challenge dataset, virtual environment, models and
generated outputs.**

Never upload the 1 GB challenge dataset to this public repository just
to make setup easier.
