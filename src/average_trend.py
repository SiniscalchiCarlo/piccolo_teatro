import os
import random
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt

from utils.df_operations import add_cumulative_sum, add_moving_avarages, add_shifted_values, print_unique_values 
from config.train_config import TrainConfig

def get_longest_duration(df):

    performances = df['performance_id'].unique()
    longest_trend = 0
    path = "C:\\Users\\te7carsinisc\\Downloads\\dati_piccolo_teatro\\performances"
    performances = os.listdir(path)
    n_performances = len(performances)
    i = 0
    for name in performances:
        file_path = path + "\\" + name
        df_ = pd.read_csv(file_path)
        i += 1
        print(i/n_performances)
        len_ = len(df_)
        if len_ > longest_trend:
            longest_trend = len_
    return longest_trend

def get_average_trend(df, max_duration):
    performances = df['performance_id'].unique()
    longest_trend = 0
    path = "C:\\Users\\te7carsinisc\\Downloads\\dati_piccolo_teatro\\performances"
    performances = os.listdir(path)

    for name in performances[:10]:
        file_path = path + "\\" + name
        df_ = pd.read_csv(file_path)
        trend = df_['percentage_bought'].copy()
        trend_duration = len(trend)
        indexes = list(range(1, len(trend)+1))
        k = max_duration/(trend_duration+2)
        new_indexes = [i * k for i in indexes]
        trend.index = new_indexes
        print(trend)
        plt.plot(new_indexes, trend)
    plt.show()

        
        



pd.set_option('display.max_columns', None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')

train_config = TrainConfig()

train_df = pd.read_csv(path+f"\\train_trend.csv", index_col=False)
validation_df = pd.read_csv(path+f"\\validation_trend.csv", index_col=False)
test_df = pd.read_csv(path+f"\\test_trend.csv", index_col=False)

#longest_trend = get_longest_duration(train_df)
#print(longest_trend)
get_average_trend(train_df, 400)
