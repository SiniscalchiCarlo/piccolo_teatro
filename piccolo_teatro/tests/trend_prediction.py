import os
import pickle
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from ..train.model_preparation import ModelData
from ..use_cases.trend_prediction import TrendPrediction

pd.set_option("display.max_columns", None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')
test_df = pd.read_parquet(path+f"\\test_trend.gzip")
model = pickle.load(open(path+f"\\models\\XGB_trend.pkl", "rb"))

test_data = ModelData(df=test_df)

performances = test_data.df['show_id'].unique()
i = 0
len_performances = len(performances)
for show_id in performances:
    i+=1
    print(show_id, i/len_performances)
    df = pd.read_parquet(path+f"\\shows\\{show_id}_target.gzip")

    performance_df = pd.read_parquet(path+f"\\shows\\{show_id}.gzip")    
    performance_data = ModelData(df=performance_df)
    performance_prediction = TrendPrediction(model=model, data=performance_data)
    end_date = performance_data.df["last_date"].iloc[0]
    performance_prediction.trend_prediction(last_date=end_date, plot=True, offset=0.4)
