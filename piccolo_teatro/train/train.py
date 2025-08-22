import os
import pickle

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from piccolo_teatro import ts_engine
from piccolo_teatro.train.train_xgb import test_xgb, train_xgb
from . import keep_enabled_columns, separete_features_targets

# ---------------------------------------------------------------------------
# 1) LOADING DATA
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())
path = os.getenv("FOLDER_PATH")

# The training and test sets are expected to be pre‑computed and stored
# as parquet files.  They already include the engineered features.
train_df = pd.read_parquet(path + "/train_trend.gzip")
test_df = pd.read_parquet(path + "/test_trend.gzip")

print(train_df)
# Keep only the features that are enabled in ``config.Features`` so that
# the model sees a consistent input layout.
train_df = keep_enabled_columns(train_df)
test_df = keep_enabled_columns(test_df)

train_X, train_Y = separete_features_targets(train_df, sort=False, shuffle=True)
validation_X, validation_Y = separete_features_targets(test_df, sort=False, shuffle=True)

# Produce an HTML and JSON report of the training data for transparency
# and exploratory analysis.
# df_report(train_df)

# ---------------------------------------------------------------------------
# 2) TRAINING THE MODEL
# ---------------------------------------------------------------------------
# Display the configuration so that experiments are reproducible.
ts_engine.config_recap()

model = None
best_params = None
val_mae = None

model = train_xgb(train_X, train_Y, validation_X, validation_Y)
test_xgb(model)
# ---------------------------------------------------------------------------
# 3) SAVING THE MODEL
# ---------------------------------------------------------------------------
if model is not None:
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        f"{ts_engine.file_name}.pkl",
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
