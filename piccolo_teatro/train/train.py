"""Model training entry point.

This script wires together all the utilities contained in the training
package to produce a fully trained model.  The steps implemented mirror
those typically followed by a data scientist:

1. load and prepare the data;
2. perform exploratory reporting;
3. fit the model using cross‑validation and early stopping;
4. evaluate the performance on a held‑out test set; and
5. persist the trained model for later inference.
"""

import os
import pickle

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from piccolo_teatro.config import TrainConfig, config_recap
from piccolo_teatro.train.train_xgb import train_xgb
from piccolo_teatro.train.utils import df_report
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

# Keep only the features that are enabled in ``config.Features`` so that
# the model sees a consistent input layout.
train_df = keep_enabled_columns(train_df)
test_df = keep_enabled_columns(test_df)

train_X, train_Y = separete_features_targets(train_df, sort=False, shuffle=True)
validation_X, validation_Y = separete_features_targets(test_df, sort=False, shuffle=True)

# Produce an HTML and JSON report of the training data for transparency
# and exploratory analysis.
df_report(train_df)

# ---------------------------------------------------------------------------
# 2) TRAINING THE MODEL
# ---------------------------------------------------------------------------
# Display the configuration so that experiments are reproducible.
config_recap()
train_conf = TrainConfig()

model = None
best_params = None
val_mae = None
if train_conf.model == "xgb":
    model, best_params, val_mae = train_xgb(train_X, train_Y, validation_X, validation_Y)

# ---------------------------------------------------------------------------
# 3) SAVING THE MODEL
# ---------------------------------------------------------------------------
if model is not None:
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        f"{train_conf.file_name}.pkl",
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Persisting the best parameters and the validation error can be
    # extremely useful for later analysis or reproduction of the
    # training run.  They are printed here but could also be saved to a
    # log file or a database depending on the project needs.
    print("Best parameters:", best_params)
    if val_mae is not None:
        print(f"Validation MAE: {val_mae:.4f}")

