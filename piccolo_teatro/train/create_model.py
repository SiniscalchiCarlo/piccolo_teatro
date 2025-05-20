import os

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from piccolo_teatro.train.model_preparation import Features, ModelData, TrainModel

from ..config import TrainConfig
from .utils import keep_enabled_columns, separete_features_targets, train_XGBRegressor, save_model, abs_error

pd.set_option('display.max_columns', None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')

train_df = pd.read_parquet(path+f"\\train_trend.gzip")
validation_df = pd.read_parquet(path+f"\\validation_trend.gzip")

train_df = keep_enabled_columns(train_df)
validation_df = keep_enabled_columns(validation_df)

train_X, train_Y = separete_features_targets(train_df, sort=False, shuffle=True)
valdiation_X, valdiation_Y = separete_features_targets(validation_df, sort=False, shuffle=True)

model = train_XGBRegressor(train_X, train_Y, 
                   parameters = {
                                "n_estimators": 200,
                                "learning_rate": 0.1,
                                "objective": "reg:squarederror",
                            })

abs_error(model, valdiation_X, valdiation_Y)

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "XGB_trend2.pkl")
save_model(model, model_path)