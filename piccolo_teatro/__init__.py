from .data_pipeline import run_data_pipeline
from .trend_simulation import predict_trend
from .models import get_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.graph_objects as go

def powerbi_visual(dataset, offset, static = True):

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

    known_trend = df["percentage_bought"].copy()

    fig = go.Figure()

    model = get_model("XGB_trend2")
    future_prediction = predict_trend(df, model)

    if not static:
        fig = go.Figure()

    # Generate fixed trend prediction if within bounds
    if int(sales_duration * offset) <= len(df):
        fixed_df = df.head(int(sales_duration * offset)).copy()
        predicted_fixed_trend = predict_trend(fixed_df, model)
        if static:
            style.use('seaborn-v0_8-pastel')
            plt.plot(predicted_fixed_trend, color="green", label="Fixed Trend Prediction")
        else:
            print(predicted_fixed_trend.index, predicted_fixed_trend)
            fig.add_trace(go.Scatter(x=predicted_fixed_trend.index, y=predicted_fixed_trend["predictions"], mode='lines', name='Fixed Trend Prediction', line=dict(width=4)))

    

    if static:
        plt.plot(known_trend, color="blue", label="Known Trend")
        plt.plot(future_prediction, color="orange", label="Future Prediction")

        plt.title("Sales Trend Analysis", fontsize=14)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Sales", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(known_trend)
        print(future_prediction)
        fig.add_trace(go.Scatter(x=known_trend.index, y=known_trend, mode='lines', name='Known Trend', line=dict(width=4)))
        
        fig.add_trace(go.Scatter(x=future_prediction.index, y=future_prediction["predictions"], mode='lines', name='Future Prediction', line=dict(width=4)))

        
        fig.update_layout(
            title='Sales Trend Analysis',
            title_font=dict(size=28),
            xaxis_title='Date',
            xaxis_title_font=dict(size=24),
            yaxis_title='Value',
            yaxis_title_font=dict(size=24),
            legend=dict(
            font=dict(size=20),  # Legend label font size
        ))

        fig.show()

