import os
import pickle
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from xgboost import XGBRegressor
from . import keep_enabled_columns, separete_features_targets, abs_error

pd.set_option('display.max_columns', None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')

# Loading full dataframes
train_df = pd.read_parquet(path+f"\\train_trend.gzip")
test_df = pd.read_parquet(path+f"\\test_trend.gzip")

# Splitting features and targets
train_df = keep_enabled_columns(train_df)
test_df = keep_enabled_columns(test_df)

train_X, train_Y = separete_features_targets(train_df, sort=False, shuffle=True)
validation_X, valdiation_Y = separete_features_targets(test_df, sort=False, shuffle=True)

# Training
model = XGBRegressor(n_estimators=200, 
                                learning_rate=0.1, 
                                objective="reg:squarederror")
model.fit(train_X, train_Y)

# Calculating error
abs_error(model, validation_X, valdiation_Y)

# Saving model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "XGB_trend2.pkl")
pickle.dump(model, open(model_path, "wb"))