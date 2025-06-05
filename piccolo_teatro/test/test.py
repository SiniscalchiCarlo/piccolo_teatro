
from ..data_pipeline import run_data_pipeline
from ..trend_simulation import predict_trend
from ..models import get_model
import pandas as pd
import matplotlib.pyplot as plt
import os



#pd.set_option("display.max_columns", None)

def powerbi_visual(dataset, n):
    #c = dataset["REFERENCE_DATE"].copy()
    #c = c.sort_values()
    #print(c.tail(1))

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

    offset = 0.1
    known_trend = df["percentage_bought"].copy()


    model = get_model("XGB_trend2")
    #future_prediction = predict_trend(df, model, fixed=False)


    fixed_df = df.head(len(df)-1).copy()
    fixed_df = df.loc[:'2024-11-20'].copy()

    print(len(fixed_df))
    fixed_df.to_csv(rf"C:\Users\39370\Downloads\dati_piccolo_teatro2\debugging\fixed_{n}.csv")

    predicted_fixed_trend = predict_trend(fixed_df, model, fixed=True)

    return known_trend, predicted_fixed_trend, fixed_df

os.chdir(u'C:/Users/39370/PythonEditorWrapper_2bb9f65f-f4ae-43d4-a20d-b947cf6e1c96')
dataset = pd.read_csv('input_df_e115a812-5f0a-452a-bc57-676ebe7071ff.csv')

known_trend1, predicted_fixed_trend1, fixed_df1 = powerbi_visual(dataset, 1)
print("\n====\n")

os.chdir(u'C:/Users/39370/PythonEditorWrapper_273aa73c-8b39-4e1e-bd08-69d1bc175c4d')
dataset = pd.read_csv('input_df_4f4b286c-7f79-46e6-8fbd-d3f81b703b41.csv')
known_trend2, predicted_fixed_trend2, fixed_df2 = powerbi_visual(dataset, 2)

print(fixed_df1.equals(fixed_df1))
plt.plot(known_trend1, color="blue")
plt.plot(known_trend2, color="purple")

plt.plot(predicted_fixed_trend1, color="red")
plt.plot(predicted_fixed_trend2, color="green")
plt.show()

