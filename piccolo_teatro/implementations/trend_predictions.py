import os
import pickle
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

from ..data_ingestion.feature_engineering import FeatureEngineering
from ..train.model_preparation import ModelData
from ..use_cases.trend_prediction import TrendPrediction
from ..models import get_trend_model

def get_trend_prediction(show_id, SALES: pd.DataFrame, PERFORMANCES:pd.DataFrame, SEASONS:pd.DataFrame, estimated_sales: float = None, offset:float=None):
    print("loading model...")
    model = get_trend_model("XGB_trend")
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
    output = performance_prediction.trend_prediction(last_date=end_date, offset=offset)

    if estimated_sales is not None:
        start = output["predictions"].iloc[0]
        end = output["predictions"].iloc[-1]
        new_end = estimated_sales
        scale_factor = (new_end - start) / (end - start)

        scaled_predictions = [start + scale_factor * (p - start) for p in output["predictions"]] 
        output['scaled_predictions'] = scaled_predictions

    return output

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    load_dotenv(find_dotenv())
    path = os.environ.get("FOLDER_PATH")
    SALES = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv", index_col=False)
    PERFORMANCES = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
    SEASONS = pd.read_csv(path+"\\stagioni.csv", index_col=False)
    predictions = get_trend_prediction(
        show_id=10228607990643,
        estimated_sales=1000,
        SALES=SALES,
        PERFORMANCES=PERFORMANCES,
        SEASONS=SEASONS,
        offset = 0.4,
    )
    print(predictions)
