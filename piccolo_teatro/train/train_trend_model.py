import os

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from config import TrainConfig
from train.model_preparation import Features, ModelData, TrainModel


pd.set_option('display.max_columns', None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')

train_df = pd.read_parquet(path+f"\\train_trend.gzip")
validation_df = pd.read_parquet(path+f"\\validation_trend.gzip")

train_config = TrainConfig()
features_config = Features()

train_data = ModelData(df=train_df)
validation_data = ModelData(df=validation_df)


train_data.separete_features_targets(sort=False, shuffle=True)
validation_data.separete_features_targets(sort=False, shuffle=True)

train_xgb = TrainModel(train_data=train_data, validation_data=validation_data)
train_xgb.train_XGBRegressor({
    "n_estimators": 200,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
})
train_xgb.abs_error("train")
train_xgb.abs_error("validation")
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "XGB_trend.pkl")
train_xgb.save_model(model_path)
