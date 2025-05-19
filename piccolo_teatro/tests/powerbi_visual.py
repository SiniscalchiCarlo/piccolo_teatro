# Prolog - Auto Generated #
import os, uuid, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import pandas

import sys

os.chdir(u'C:/Users/39370/PythonEditorWrapper_6b21d22e-9446-479a-861b-4298711a8dee')
dataset = pandas.read_csv('input_df_26387f20-2361-4a83-a773-b106ea28deb0.csv')

matplotlib.pyplot.figure(figsize=(5.55555555555556,4.16666666666667), dpi=72)
matplotlib.pyplot.show = lambda args=None,kw=None: matplotlib.pyplot.savefig(str(uuid.uuid1()))
# Original Script. Please update your script content here and once completed copy below section back to the original editing window #
# Il codice seguente consente di creare un dataframe e rimuovere righe duplicate e viene sempre eseguito e funge da preambolo per lo script: 

# dataset = pandas.DataFrame(Individuali/Gruppi, T_SEASON_ID, T_PRODUCT_ID, T_PERFORMANCE_ID, Tipologia canale, TOTAL_CURRENT_AMT_ITX, Anno, Trimestre, Mese, Giorno, Anno.1, Trimestre.1, Mese.1, Giorno.1, Anno.2, Trimestre.2, Mese.2, Giorno.2, CURRENT_QUANTITY)
# dataset = dataset.drop_duplicates()

# Incollare o digitare qui il codice dello script:
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from ..implementations.trend_predictions import get_trend_prediction


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
    SALES = dataset[sales_cols].copy()
    SALES = SALES.rename(columns={'SEASON.1': 'SEASON', "Individuali/Gruppi": "Individuali_Gruppi", "Tipologia canale": "Tipologia_canale"})
    SALES.columns = ["D_SALES_LIST_SALES_"+col for col in SALES.columns]

    pd.set_option("display.max_columns", None)
    PERFORMANCES = dataset[performances_cols].copy()
    PERFORMANCES = PERFORMANCES.rename(columns={'Tipologia spettacolo': 'Tipologia_spettacolo'})

    PERFORMANCES.columns = ["D_CONFIG_PROD_LIST_"+col for col in PERFORMANCES.columns]

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

    target_perc = float(dataset["OBIETTIVO RIEMPIMENTO"].iloc[0])
    target_incasso = float(dataset["OBIETTIVO INCASSO"].iloc[0])
    k = (target_incasso/target_perc)*100
    product_name = dataset["PRODUCT_EXTERNAL_NAME"].iloc[-1]
    prediction_df, show_df = get_trend_prediction(show_id=show_id, 
                                        estimated_sales=target_incasso, 
                                        SALES=SALES, 
                                        PERFORMANCES=PERFORMANCES, 
                                        SEASONS=SEASONS)
    print("SHOW DF", show_df)
    plt.plot(show_df["percentage_sales_day"])
    plt.show()
    k=target_incasso/target_perc

    if static:
        plt.figure(figsize=(12, 6))
        plt.plot(prediction_df['date'], prediction_df['predictions'] * k, label='Predizione', linewidth=2)
        plt.plot(prediction_df['date'], prediction_df['scaled_predictions'], label='Obiettivo', linewidth=2, linestyle='--')

        # Layout and labels
        plt.title(f'Previsione incassi {product_name}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=prediction_df['date'], y=prediction_df['predictions']*k, mode='lines', name='Predizione'))
        fig.add_trace(go.Scatter(x=prediction_df['date'], y=prediction_df['scaled_predictions'], mode='lines', name='Obiettivo'))

        fig.update_layout(
            title=f'Previsione incassi {product_name}',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )

        fig.show()

powerbi_visual(dataset, static=True)

# Epilog - Auto Generated #
os.chdir(u'C:/Users/39370/PythonEditorWrapper_6b21d22e-9446-479a-861b-4298711a8dee')
