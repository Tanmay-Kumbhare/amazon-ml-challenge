import pandas as pd
from sklearn.model_selection import train_test_split

print("Testing train_test_split on PyArrow string array...")

# Create a dataframe using pyarrow string dtype (what causes the bug)
df = pd.DataFrame({
    'source1_id': ['S1-1', 'S1-2', 'S1-3', 'S1-4', 'S1-5', 'S1-1', 'S1-2']
}, dtype="string[pyarrow]")

print(f"Dtype is: {df['source1_id'].dtype}")

# 1. Attempt using list conversion (the fix)
unique_s1 = list(df['source1_id'].unique())
train_s1, val_s1 = train_test_split(unique_s1, test_size=0.2, random_state=42)

print("\nSuccess with fix!")
print("Train S1:", train_s1)
print("Val S1:", val_s1)

# Check that isin still works on pyarrow arrays using this list
train_mask = df['source1_id'].isin(train_s1)
print(f"\nMask works: {train_mask.sum()} items matched.")

