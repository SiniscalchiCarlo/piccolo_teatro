import os
import pickle
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from piccolo_teatro.config import TrainConfig, config_recap
from piccolo_teatro.train.train_xgb import train_xgb
from piccolo_teatro.trend_simulation import predict_trend
from xgboost import XGBRegressor
from . import keep_enabled_columns, separete_features_targets, abs_error

# 1) LOADING DATA
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')

# Loading full dataframes
train_df = pd.read_parquet(path+"/train_trend.gzip")
test_df = pd.read_parquet(path+"/test_trend.gzip")

# Splitting features and targets
train_df = keep_enabled_columns(train_df)
test_df = keep_enabled_columns(test_df)

train_X, train_Y = separete_features_targets(train_df, sort=False, shuffle=True)
validation_X, valdiation_Y = separete_features_targets(test_df, sort=False, shuffle=True)


# 2) TRAINING THE MODEL
# Printing Training Configurations
config_recap()
train_conf = TrainConfig()

model = None
if train_conf.model == "xgb":
    model = train_xgb(train_X, train_Y)

if model != None:
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", f"${train_conf.file_name}.pkl")
    pickle.dump(model, open(model_path, "wb"))

