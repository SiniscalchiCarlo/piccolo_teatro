from piccolo_teatro.implementations.trend_predictions import get_trend_prediction
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    prediction_df,_ = get_trend_prediction(show_id=show_id, 
                                        estimated_sales=target_incasso, 
                                        SALES=SALES, 
                                        PERFORMANCES=PERFORMANCES, 
                                        SEASONS=SEASONS,
                                        offset=0.4)
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