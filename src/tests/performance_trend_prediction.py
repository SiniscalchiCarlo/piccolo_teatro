import os
from tkinter import N

from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from train.model_preparation import ModelData
from use_cases.trend_prediction import TrendPrediction




load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')
test_path = path+f"\\test_trend.csv"

test_data = ModelData(df_path=test_path)
test_data.load_df()
performances = test_data.df['performance_id'].unique()
i = 0
len_performances = len(performances)
for performance_id in performances:
    i+=1
    print(performance_id, i/len_performances)
    df = pd.read_csv(path+f"\\performances\\{performance_id}.csv")
    
    performance_data = ModelData(df_path=path+f"\\performances\\{performance_id}.csv")
    performance_data.load_df()
    performance_prediction = TrendPrediction(model_path=path+f"\\models\\XGB_trend.pkl", data=performance_data)
    end_date = performance_data.df["performance_date"].iloc[0]
    print(type(end_date), end_date)
    performance_prediction.trend_prediction(performance_date=end_date, plot=True, offset=0.4)

