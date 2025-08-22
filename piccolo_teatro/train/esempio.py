import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ----------------------------------------
# 1) Load data (one .parquet per show)
# ----------------------------------------
data_dir = "data/"
files = glob.glob(os.path.join(data_dir, "*.parquet"))

# concatenate, assume each df has:
#  - date
#  - feature columns up to day t
#  - pct_sold_shifted_{h} for h=1..H  (so we can build iterative val)
#  - show_id
# we will predict only pct_sold_shifted_1 at train time, but eval multi-step
dfs = []
for fn in files:
    df = pd.read_parquet(fn)
    dfs.append(df)
full_df = pd.concat(dfs, ignore_index=True)

# split by show to avoid leakage
shows = full_df["show_id"].unique()
train_shows, val_shows = train_test_split(shows, test_size=0.2, random_state=42)

train_df = full_df[full_df["show_id"].isin(train_shows)].reset_index(drop=True)
val_df   = full_df[full_df["show_id"].isin(val_shows)].reset_index(drop=True)

# features used at t to predict t+1
feature_cols = [c for c in full_df.columns
                if c.startswith("feat_")]  # adjust to your real feature names

# ----------------------------------------
# 2) Build DMatrices
# ----------------------------------------
dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df["pct_sold_shifted_1"])
dval   = xgb.DMatrix(val_df[feature_cols],   label=val_df["pct_sold_shifted_1"])

# ----------------------------------------
# 3) Define multi-step aggregated MSE eval
# ----------------------------------------
H = 7  # forecast horizon

def multi_step_mse_eval(preds, dmat):
    """
    preds: flat array of shape (n_rows,) = model's 1-step predictions on val_df
    dmat: DMatrix wrapping val_df features
    
    We reconstruct an H-step forecast per show by iteratively
    feeding the model's 1-step preds back into the features
    (simply shifting the pct_sold feature), then compute
    the average MSE over h=1..H across all shows.
    """
    # rebuild a DataFrame to hold preds and features
    df = val_df.copy()
    df["pred_1"] = preds
    
    # for h=2..H, iteratively predict
    all_errors = []
    for h in range(2, H+1):
        # shift previous prediction into the pct_sold feature
        df["feat_pct_sold"] = df[f"pred_{h-1}"]
        dnext = xgb.DMatrix(df[feature_cols])
        df[f"pred_{h}"] = bst.predict(dnext, ntree_limit=bst.best_ntree_limit)
    
    # now compute MSE per step and average
    for h in range(1, H+1):
        true = df[f"pct_sold_shifted_{h}"]  # true pct at t+h
        pred = df[f"pred_{h}"]
        all_errors.append((true - pred)**2)
    # all_errors is list of length H, each a Series of squared errs
    squared = pd.concat(all_errors, axis=1)
    mse_per_step = squared.mean()       # mean over rows
    return "mse_horizon", mse_per_step.mean()  # average across h

# ----------------------------------------
# 4) Train with early stopping on multi-step MSE
# ----------------------------------------
params = {
    "objective":      "reg:squarederror", 
    "tree_method":    "hist",
    "eta":            0.05,
    "max_depth":      6,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dval, "valid")],
    feval=multi_step_mse_eval,
    maximize=False,
    early_stopping_rounds=20,
    verbose_eval=10
)

# ----------------------------------------
# 5) Final evaluation
# ----------------------------------------
# (You can now call multi_step_mse_eval manually or compute any other metrics)

