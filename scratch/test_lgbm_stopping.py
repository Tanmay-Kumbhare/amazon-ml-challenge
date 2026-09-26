import pandas as pd
import numpy as np
import lightgbm as lgb

print("Testing LightGBM Early Stopping Configuration...")

# Create dummy data
X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
y_train = np.random.randint(0, 2, 100)

X_val = pd.DataFrame(np.random.rand(20, 5), columns=[f'feature_{i}' for i in range(5)])
y_val = np.random.randint(0, 2, 20)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Params without 'metric' (to simulate the failure condition)
params = {
    'objective': 'binary',
    'verbose': -1
}

if 'metric' not in params:
    params['metric'] = 'binary_logloss'

try:
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=5)]
    )
    print("\nSuccess! LightGBM trained with early stopping enabled.")
    print(f"Best Iteration: {model.best_iteration}")
except Exception as e:
    print(f"\nFailed! Error: {e}")
