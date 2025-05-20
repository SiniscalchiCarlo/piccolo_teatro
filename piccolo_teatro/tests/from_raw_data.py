import os
import pickle
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from ..train.model_preparation import ModelData
from ..implementations.trend_predictions import get_trend_prediction
from ..data_ingestion.feature_engineering import FeatureEngineering
from ..use_cases.trend_prediction import TrendPrediction
from ..models import get_trend_model

def plot_static(real_time_prediction, fixed_prediction, k):
    plt.figure(figsize=(12, 6))
    plt.plot(real_time_prediction['date'], real_time_prediction['predictions'] * k, label='Predizione', linewidth=2)
    plt.plot(fixed_prediction['date'], fixed_prediction['scaled_predictions'], label='Obiettivo', linewidth=2, linestyle='--')

    # Layout and labels
    plt.title(f'Previsione incassi', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


pd.set_option("display.max_columns", None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')
test_df = pd.read_parquet(path+f"\\test_trend.gzip")
model = pickle.load(open(path+f"\\models\\XGB_trend.pkl", "rb"))

test_data = ModelData(df=test_df)

performances = test_data.df['show_id'].unique()

SALES = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv", index_col=False)

PERFORMANCES = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
SEASONS = pd.read_csv(path+"\\stagioni.csv", index_col=False)
i = 0
len_performances = len(performances)
for show_id in performances:
    i+=1

    model = get_trend_model("XGB_trend")
    offset = 0.9
    estimated_sales = None

    show_sales = SALES[SALES["D_SALES_LIST_SALES_T_PRODUCT_ID"]==show_id]
    
    feat_eng = FeatureEngineering(show_sales, PERFORMANCES, SEASONS)
    feat_eng.ingest_sales()

    feat_eng.sales_df = feat_eng.sales_df.sort_values(by="date")
    feat_eng.sales_df = feat_eng.sales_df.set_index("date").reset_index(drop=False)
    feat_eng.sales_df = feat_eng.sales_df.head(int(len(feat_eng.sales_df)*0.5))
    print("FEAT ENG SALES", feat_eng.sales_df)
    
    df_all_features = feat_eng.add_features(feat_eng.sales_df, fill_to_show_date=False)
    print("ALL FEATURES",df_all_features)
    show_data = ModelData(df=df_all_features.copy())
    performance_prediction = TrendPrediction(model=model, data=show_data)
    end_date = show_data.df["last_date"].iloc[0]
    output = performance_prediction.trend_prediction(last_date=end_date, offset=offset)

    if estimated_sales is not None:
        start = output["predictions"].iloc[0]
        end = output["predictions"].iloc[-1]
        new_end = estimated_sales
        scale_factor = (new_end - start) / (end - start)

        scaled_predictions = [start + scale_factor * (p - start) for p in output["predictions"]] 
        output['scaled_predictions'] = scaled_predictions


    plt.plot(output['date'], output['predictions'])
    plt.plot(df_all_features["percentage_bought"])
    plt.show()