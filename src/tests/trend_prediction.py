import os

from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from train.model_preparation import ModelData
from use_cases.trend_prediction import TrendPrediction




load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')
test_path = path+f"\\test_trend.gzip"

test_data = ModelData(df_path=test_path)
test_data.load_df()
performances = test_data.df['show_id'].unique()
i = 0
len_performances = len(performances)
for show_id in performances:
    i+=1
    print(show_id, i/len_performances)
    df = pd.read_parquet(path+f"\\shows\\{show_id}_target.gzip")
    
    performance_data = ModelData(df_path=path+f"\\shows\\{show_id}.gzip")
    performance_data.load_df()
    performance_prediction = TrendPrediction(model_path=path+f"\\models\\XGB_trend.pkl", data=performance_data)
    end_date = performance_data.df["last_date"].iloc[0]
    performance_prediction.trend_prediction(last_date=end_date, plot=True, offset=0.4)
