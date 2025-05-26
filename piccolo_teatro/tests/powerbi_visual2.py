# Prolog - Auto Generated #
import os, uuid, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import pandas

import sys

os.chdir(u'C:/Users/39370/PythonEditorWrapper_9926174e-931f-4b65-98a3-b8328749f53d')
dataset = pandas.read_csv('input_df_a6a786ea-4685-45ea-809b-c22118abcfa1.csv')

# from .trend_prediction2 import trend_prediction
# from ..data_ingestion.raw_data import Sales, Products, Seasons
from piccolo_teatro.tests.trend_prediction2 import trend_prediction
from piccolo_teatro.data_ingestion.raw_data import Sales, Products, Seasons
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def add_cumulative_sum(df, column_names: list[str], time_col: str):
    output = df.copy()
    # Ensure time column is datetime (optional, but safer)
    output[time_col] = pd.to_datetime(output[time_col], errors='coerce')

    # Sort by time column
    output = output.sort_values(by=time_col).reset_index(drop=True)
    output = output.set_index(time_col)
    
    # Compute cumulative sums
    for col_name in column_names:
        output[col_name + "_cum_sum"] = output[col_name].cumsum()

    return output

OFFSET = 0.3
def powerbi_visual(dataset, static = True):

    sales_cols = [
        "Individuali/Gruppi",
        "Tipologia canale",
        "TOTAL_CURRENT_AMT_ITX",
        "CURRENT_QUANTITY",
        "REFERENCE_DATE",
        "T_PRODUCT_ID",
        "T_SEASON_ID",
        "SEASON",
        "T_PERFORMANCE_ID",
        "OPERATION_TYPE",
        "T_OPERATION_KIND",
    ]

    performances_cols = [
        "T_PRODUCT_ID",
        "T_PERFORMANCE_ID",
        "PERFORMANCE_STATE",
        "Tipologia spettacolo",
        "SEASON",
        "SPACE",
        "PERFORMANCE_QUOTA",
        "PRODUCT_DATE_TIME",
    ]
    seasons_cols = [
        "FINE STAGIONE",
        "FINE VENDITE",
        "INIZIO STAGIONE",
        "INIZIO VENDITE",
        "SEASON.1",
        "T_SEASON_ID",
    ]
    print("DATASET",dataset["T_PERFORMANCE_ID"])
    SALES = dataset[sales_cols].copy()
    SALES = SALES.rename(columns={'SEASON.1': 'SEASON', "Individuali/Gruppi": "Individuali_Gruppi", "Tipologia canale": "Tipologia_canale"})
    SALES.columns = ["D_SALES_LIST_SALES_"+col for col in SALES.columns]
    REAL_SALES = add_cumulative_sum(SALES, column_names=["D_SALES_LIST_SALES_TOTAL_CURRENT_AMT_ITX"], time_col="D_SALES_LIST_SALES_REFERENCE_DATE")

    pd.set_option("display.max_columns", None)
    PRODUCTS = dataset[performances_cols].copy()
    PRODUCTS = PRODUCTS.rename(columns={'Tipologia spettacolo': 'Tipologia_spettacolo'})

    PRODUCTS.columns = ["D_CONFIG_PROD_LIST_"+col for col in PRODUCTS.columns]

    SEASONS = dataset[seasons_cols].copy()
    SEASONS = SEASONS.rename(columns={
        "FINE STAGIONE": "fine_stagione",
        "FINE VENDITE": "fine_vendite",
        "INIZIO STAGIONE": "inizio_stagione",
        "INIZIO VENDITE": "inizio_vendite",
        "SEASON.1": "season_name",
        "T_SEASON_ID": "season_id",
    })
    show_id = int(dataset["T_PRODUCT_ID"].iloc[0])

    sales = Sales(SALES)
    products = Products(PRODUCTS)
    seasons = Seasons(SEASONS)

    sales.get_same_show(show_id=show_id)

    sales.clean()
    products.clean()
    seasons.clean()

    prediction_df, all_trend = trend_prediction(sales, products, seasons, offset=0.3)
    print(all_trend)
    print("SHOW ID", show_id)
    plt.plot(prediction_df["predictions"])
    plt.plot(all_trend)
    plt.show()

powerbi_visual(dataset, static=True)