import os
import pickle
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from ..data_ingestion.feature_engineering import FeatureEngineering
from ..train.model_preparation import ModelData
from ..use_cases.trend_prediction import TrendPrediction

def get_trend_prediction(show_id, estimated_sales, SALES:pd.DataFrame, PERFORMANCES:pd.DataFrame, SEASONS:pd.DataFrame):
    print("loading model...")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "XGB_trend.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("ingesting data...")
    feat_eng = FeatureEngineering(SALES, PERFORMANCES, SEASONS)
    feat_eng.ingest_sales()
    print("adding features...")
    show_df = feat_eng.SALES[feat_eng.SALES["show_id"]==show_id]
    show_df = feat_eng.add_features(show_df)

    print("creating model data...")
    show_data = ModelData(df=show_df)
    print("predicting...")
    performance_prediction = TrendPrediction(model=model, data=show_data)
    end_date = show_data.df["last_date"].iloc[0]
    predictions = performance_prediction.trend_prediction(last_date=end_date)
    return predictions

