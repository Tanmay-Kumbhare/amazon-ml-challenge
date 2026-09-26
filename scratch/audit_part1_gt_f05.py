"""
AUDIT PART 1: Corrected GT singleton statistics and F0.5 unit tests.
Runs in ~30 seconds. No TF-IDF. No blocking.
"""
import pandas as pd
import numpy as np

def f05_macro_per_s1(y_true_dict, y_pred_dict):
    """Challenge-correct macro F0.5 per S1."""
    f05_scores = []
    for s1_id, true_targets in y_true_dict.items():
        pred_targets = y_pred_dict.get(s1_id, set())
        # Case A: true singleton correctly predicted empty
        if len(true_targets) == 0 and len(pred_targets) == 0:
            f05_scores.append(1.0)
            continue
        # Case B: true singleton, false match predicted
        if len(true_targets) == 0 and len(pred_targets) > 0:
            f05_scores.append(0.0)
            continue
        # Case C: has matches, predicted none
        if len(true_targets) > 0 and len(pred_targets) == 0:
            f05_scores.append(0.0)
            continue

        tp = len(true_targets & pred_targets)
        fp = len(pred_targets - true_targets)
        fn = len(true_targets - pred_targets)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(f05)
    return np.mean(f05_scores) if f05_scores else 0.0

# ---- F0.5 UNIT TESTS ----
print("=" * 60)
print("SECTION B: MACRO F0.5 UNIT TESTS")
print("=" * 60)
tests = [
    ("A", set(),          set(),            1.0),
    ("B", set(),          {"S2-1"},         0.0),
    ("C", {"S2-1"},       set(),            0.0),
    ("D", {"S2-1"},       {"S2-1"},         1.0),
    ("E", {"S2-1","S3-1"},{"S2-1","S3-1"}, 1.0),
    # F: tp=1, fp=1, fn=1 => prec=0.5, rec=0.5 => 1.25*0.25/(0.25*0.5+0.5) = 0.3125/0.625=0.5
    ("F", {"S2-1","S3-1"},{"S2-1","S3-2"}, 0.5),
]
all_pass = True
for name, true_t, pred_t, expected in tests:
    got = f05_macro_per_s1({"s1": true_t}, {"s1": pred_t})
    ok = abs(got - expected) < 1e-6
    print(f"  Test {name}: expected={expected:.4f}  got={got:.4f}  [{'PASS' if ok else 'FAIL'}]")
    all_pass = all_pass and ok
print(f"  Overall: {'ALL PASS' if all_pass else 'HAS FAILURES'}")

# ---- GT STATS ----
print("\n" + "=" * 60)
print("SECTION A: CORRECTED GROUND TRUTH STATISTICS")
print("=" * 60)
df_gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t",
    keep_default_na=False,
    dtype=str
)
print(f"  Total rows in GT file: {len(df_gt)}")
print(f"  Columns: {list(df_gt.columns)}")

zero_match  = 0
nonzero_match = 0
total_rels  = 0
match_counts = []

for row in df_gt.itertuples():
    raw = str(row.matched_entity_ids).strip()
    if raw == "":
        matches = set()
    else:
        matches = set(m.strip() for m in raw.split(",") if m.strip())
    match_counts.append(len(matches))
    if len(matches) == 0:
        zero_match += 1
    else:
        nonzero_match += 1
        total_rels += len(matches)

match_counts = np.array(match_counts)
print(f"  Zero-match S1 (Singletons): {zero_match:,}")
print(f"  Nonzero-match S1:           {nonzero_match:,}")
print(f"  Total GT relationships:     {total_rels:,}")
print(f"  Max matches for 1 S1:       {match_counts.max()}")
print(f"  Mean matches (nonzero):     {match_counts[match_counts>0].mean():.2f}")
print(f"  Median matches (nonzero):   {np.median(match_counts[match_counts>0]):.1f}")
print("\nDone.")
