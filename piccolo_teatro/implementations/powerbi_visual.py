from trend_predictions import get_trend_prediction
import pandas as pd
import matplotlib.pyplot as plt

def powerbi_visual(dataset):
    sales_cols = [
        "Individuali/Gruppi",\
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

    pd.set_option("display.max_columns", None)
    PERFORMANCES = dataset[performances_cols].copy()
    PERFORMANCES = PERFORMANCES.rename(columns={'Tipologia spettacolo': 'Tipologia_spettacolo'})

    PERFORMANCES.columns = ["D_CONFIG_PROD_LIST_"+col for col in PERFORMANCES.columns]
    print(PERFORMANCES)
    SEASONS = dataset[seasons_cols].copy()
    SEASONS = SEASONS.rename(columns={
        "FINE STAGIONE": "fine_stagione",
        "FINE VENDITE": "fine_vendite",
        "INIZIO STAGIONE": "inizio_stagione",
        "INIZIO VENDITE": "inizio_vendite",
        "SEASON.1": "season_name",
        "T_SEASON_ID": "season_id",
    })

    prediction_df = get_trend_prediction(show_id=10228587005389, 
                                        estimated_sales=0.8, 
                                        SALES=SALES, 
                                        PERFORMANCES=PERFORMANCES, 
                                        SEASONS=SEASONS,
                                        offset=0.4,)

    plt.plot(prediction_df['date'], prediction_df['predictions'])
    plt.plot(prediction_df['date'], prediction_df['scaled_predictions'])
    plt.show()