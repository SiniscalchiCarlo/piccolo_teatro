
from ..data_pipeline import run_data_pipeline
from ..trend_simulation import predict_trend
from ..models import get_model
from .. import powerbi_visual
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(u'C:/Users/39370/PythonEditorWrapper_2bb9f65f-f4ae-43d4-a20d-b947cf6e1c96')
dataset = pd.read_csv('input_df_e115a812-5f0a-452a-bc57-676ebe7071ff.csv')


#pd.set_option("display.max_columns", None)

def powerbi_visual2(dataset, static = True):

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
    SALES = dataset[sales_cols].copy()
    SALES = SALES.rename(columns={'SEASON.1': 'SEASON', "Individuali/Gruppi": "Individuali_Gruppi", "Tipologia canale": "Tipologia_canale"})
    SALES.columns = ["D_SALES_LIST_SALES_"+col for col in SALES.columns]
   
    PRODUCTS = dataset[performances_cols].copy()
    PRODUCTS = PRODUCTS.drop_duplicates()
    PRODUCTS = PRODUCTS.rename(columns={'Tipologia spettacolo': 'Tipologia_spettacolo'})

    PRODUCTS.columns = ["D_CONFIG_PROD_LIST_"+col for col in PRODUCTS.columns]

    SEASONS = dataset[seasons_cols].copy()
    SEASONS = SEASONS.drop_duplicates()
    SEASONS = SEASONS.rename(columns={
        "FINE STAGIONE": "fine_stagione",
        "FINE VENDITE": "fine_vendite",
        "INIZIO STAGIONE": "inizio_stagione",
        "INIZIO VENDITE": "inizio_vendite",
        "SEASON.1": "season_name",
        "T_SEASON_ID": "season_id",
    })
    show_id = int(dataset["T_PRODUCT_ID"].iloc[0])

    target_perc = float(dataset["OBIETTIVO RIEMPIMENTO"].iloc[0])
    target_incasso = float(dataset["OBIETTIVO INCASSO"].iloc[0])
    k = (target_incasso/target_perc)*100
    product_name = dataset["PRODUCT_EXTERNAL_NAME"].iloc[-1]
    
    df = run_data_pipeline(SALES, PRODUCTS, SEASONS, show_id)

    sales_duration = df["sales_duration"].iloc[0]

    pd.set_option("display.max_columns", None)

    offset = 0.3
    known_trend = df["percentage_bought"].copy()

    model = get_model("XGB_trend2")
    future_prediction = predict_trend(df, model)
    
    if int(sales_duration * offset)<= len(df):
        fixed_df = df.head(int(sales_duration * offset)).copy()
        predicted_fixed_trend = predict_trend(fixed_df, model)
        plt.plot(predicted_fixed_trend, color="green")


    # Plotting
    
    plt.plot(known_trend, color="blue")
    plt.plot(future_prediction, color="orange")
    plt.show()


powerbi_visual(dataset, static=False)
