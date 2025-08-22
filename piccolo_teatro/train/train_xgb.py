import os
import glob
from piccolo_teatro.config import XGBConfig
from piccolo_teatro.trend_simulation import predict_trend
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

xgb_config =  XGBConfig()

def train_xgb(
    train_X,
    train_Y,
    validation_X,
    validation_Y,
):
    dtrain = xgb.DMatrix(train_X, label=train_Y)
    dval   = xgb.DMatrix(validation_X, label=validation_Y)

    bst = xgb.train(
        xgb_config.parameters, 
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dval, "valid")],
        #feval=multi_step_mse_eval,
        maximize=False,
        #early_stopping_rounds=20,
        #verbose_eval=10
    )

    return bst 

def test_xgb(model, offset=0.4):
    path = os.environ.get("FOLDER_PATH")
    # 1. List all gzip files in the folder
    folder = os.path.join(path, "shows", "test")
    parquet_files = glob.glob(os.path.join(folder, "*.gzip"))

    # 1. List all gzip files in the folder
    folder = os.path.join(path, "shows", "test")
    parquet_files = glob.glob(os.path.join(folder, "*.gzip"))
    for file_name in parquet_files:
        print(file_name)
        show_df = pd.read_parquet(file_name)
        
        print("show_df")
        print(show_df)
        sales_duration = show_df["sales_duration"].iloc[0]
        
        # 1. Select only the two columns and make a copy
        target = show_df[['date', 'percentage_bought']].copy()
        # 2. Parse your dates
        target['date'] = pd.to_datetime(target['date'], format='%d/%m/%Y')
        # 3. Move the date column into the index
        target.set_index('date', inplace=True)


        known_df = show_df.head(int(sales_duration * offset)).copy()


        predictions = predict_trend(known_df, model, show_df["last_date"][0])
        print(show_df["last_date"][0], show_df["sales_duration"][0], type(show_df["last_date"][0]))
        plt.plot(target , color="blue")
        plt.plot(predictions, color="orange")
        plt.show()

