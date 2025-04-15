import os

from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from train.model_preparation import ModelData
from use_cases.trend_prediction import TrendPrediction
from utils import add_cumulative_sum



pd.set_option('display.max_columns', None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')
path = os.environ.get("FOLDER_PATH")
PERFORMANCES = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
test_path = path+f"\\test_trend.csv"

test_data = ModelData(df_path=test_path)
test_data.load_df()
shows_ids = test_data.df['show_id'].unique()
len_shows = len(shows_ids)
for show_id in shows_ids:
    show_id = 10228587005392
    performances_same_show = PERFORMANCES[PERFORMANCES["D_CONFIG_PROD_LIST_T_PRODUCT_ID"]==show_id]
    n_performances = len(performances_same_show)
    print("N performances", n_performances)
    performances_predictions = []
    for performance_id in performances_same_show["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"]:
        df = pd.read_csv(path+f"\\performances\\{performance_id}.csv")
        
        performance_data = ModelData(df_path=path+f"\\performances\\{performance_id}.csv")
        performance_data.load_df()
        performance_prediction = TrendPrediction(model_path=path+f"\\models\\XGB_trend.pkl", data=performance_data)
        end_date = performance_data.df["performance_date"].iloc[0]
        predictions_df = performance_prediction.trend_prediction(performance_date=end_date, plot=False, offset=0.4)
        performances_predictions.append(predictions_df)
    all_predictions = pd.concat(performances_predictions)
    all_predictions["delta"] = all_predictions["delta"]/n_performances
    all_predictions = all_predictions.groupby(['date']).sum()
    all_predictions = add_cumulative_sum(all_predictions, ["delta"])
    print(all_predictions)

    real_show_trend = pd.read_parquet(path+f"\\shows\\{show_id}.gzip")
    plt.plot(real_show_trend["date"], real_show_trend["percentage_bought"], color="blue")
    plt.plot(all_predictions["delta_cum_sum"], color="orange")
    plt.show()
