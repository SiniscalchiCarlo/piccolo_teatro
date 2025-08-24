import os
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv

from ..data_pipeline import run_data_pipeline
from . import predict_trend
from piccolo_teatro.models import get_model

pd.set_option("display.max_columns", None)
load_dotenv(find_dotenv())
path = os.environ.get("FOLDER_PATH")
model = get_model("$xgb_trend")

offset = 0.4
SALES = pd.read_csv(path+"/D_SALES_LIST_SALES.csv", index_col=False)
PRODUCTS = pd.read_csv(path+"/D_CONFIG_PROD_LIST.csv", index_col=False)
SEASONS = pd.read_csv(path+"/stagioni.csv", index_col=False)
test_df = pd.read_parquet(path+f"/test_trend.gzip")

ids = test_df['show_id'].unique().tolist()
print(test_df)
ids = random.sample(ids, k=len(ids))

print("len ids", len(ids))
for show_id in ids:
    df = run_data_pipeline(SALES, PRODUCTS, SEASONS, show_id)

    print(df)
    print("len df", len(df))
    # We are simulationg we don't know all the data
    unknown_trend = df["percentage_bought"].copy()
    known_df = df.head(int(len(df) * offset)).copy()

    predicted_trend = predict_trend(known_df, model)

    # Plotting
    plt.plot(predicted_trend)
    plt.plot(unknown_trend)
    plt.show()
