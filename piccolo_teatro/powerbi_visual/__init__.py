from piccolo_teatro.config import TimeSeriesEngine
from .data_pipeline import run_data_pipeline
from .trend_simulation import predict_trend
from .models import get_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import plotly.graph_objects as go

ts_engine = TimeSeriesEngine()

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

def rename_cols(dataset):

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
    return SALES, PRODUCTS, SEASONS

def plot_trends(static, product_name, known_gain, known_trend, future_prediction, predicted_fixed_trend, scale_factor):
    if static:
        
        plt.plot(future_prediction*scale_factor, color="orange", label="Future Prediction")
        plt.plot(known_trend*scale_factor, color="blue", label="Known Trend")
        if scale_factor != 1:
            plt.plot(known_gain, color="black", label="Real Trend")

        if len(predicted_fixed_trend) > 0:
            style.use('seaborn-v0_8-pastel')
            plt.plot(predicted_fixed_trend*scale_factor, color="green", label="Fixed Trend Prediction")

        plt.title(f'Trend Prediction for "{product_name}"', fontsize=14)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel('Show gain', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_prediction.index, y=future_prediction["predictions"]*scale_factor, mode='lines', name='Future Prediction', line=dict(width=4)))
        fig.add_trace(go.Scatter(x=known_trend.index, y=known_trend*scale_factor, mode='lines', name='Known Trend', line=dict(width=4)))
        
        if scale_factor != 1:
            fig.add_trace(go.Scatter(x=future_prediction.index, y=known_gain, mode='lines', name='Future Prediction', line=dict(width=4)))

        if len(predicted_fixed_trend) > 0:
            fig.add_trace(go.Scatter(x=predicted_fixed_trend.index, y=predicted_fixed_trend*scale_factor, mode='lines', name='Fixed Trend Prediction', line=dict(width=4)))
        
        fig.update_layout(
            title=f'Trend Prediction for "{product_name}"',
            title_font=dict(size=28),
            xaxis_title='Date',
            xaxis_title_font=dict(size=24),
            yaxis_title='Show gain',
            yaxis_title_font=dict(size=24),
            legend=dict(
            font=dict(size=20),  # Legend label font size
        ))

        fig.show()

def scale_trend(trend, new_end):
    start = trend.iloc[0]
    end = trend.iloc[-1]
    scale_factor = (new_end - start) / (end - start)

    # Apply scaling while preserving the original index
    scaled_values = trend.apply(lambda p: start + scale_factor * (p - start))
    return scaled_values

def powerbi_visual(dataset, offset, static = True, gain_trend = False):

    SALES, PRODUCTS, SEASONS = rename_cols(dataset)

    # Get show info
    show_id = int(dataset["T_PRODUCT_ID"].iloc[0])
    target_perc = float(dataset["OBIETTIVO RIEMPIMENTO"].iloc[0])
    target_gain = float(dataset["OBIETTIVO INCASSO"].iloc[0])
    target_tickets = float(dataset["OBIETTIVO VENDITE"].iloc[0])
    avg_ticket_price = target_gain/target_tickets
    product_name = dataset["PRODUCT_EXTERNAL_NAME"].iloc[-1]

    
    # Run a pipeline to clean the raw data and add features.
    df = run_data_pipeline(SALES, PRODUCTS, SEASONS, show_id)
    sales_duration = df["sales_duration"].iloc[0]
    known_trend = df["percentage_bought"].copy()
    known_gain = df["gain_cum_sum"].copy()
    capacity = df["show_capacity"].iloc[0]

    scale_factor = 1
    if gain_trend:
        scale_factor = capacity * (avg_ticket_price)


    # Make prediction using all the known data
    model = get_model("XGB_trend2")
    future_prediction = predict_trend(df, model)
    known_trend = known_trend.rename("predictions")
    known_and_predicted = pd.concat([known_trend, future_prediction])    

    # Generate a fixed trend prediction if there is enough data.
    predicted_fixed_trend = []
    if int(sales_duration * offset) <= len(df):
        fixed_df = df.head(int(sales_duration * offset)).copy()
        predicted_fixed_trend = predict_trend(fixed_df, model)
        predicted_fixed_trend = scale_trend(predicted_fixed_trend["predictions"],target_perc)
        

    plot_trends(static, product_name, 
                known_gain,
                known_trend, 
                known_and_predicted, 
                predicted_fixed_trend,
                scale_factor)

    


