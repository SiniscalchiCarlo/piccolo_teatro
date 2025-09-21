
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from ..train import keep_enabled_columns, separete_features_targets
from ..config import TimeSeriesEngine 
import xgboost as xgb

pd.set_option("display.max_columns", None)

def predict_trend(known_df, target, model, last_date):
    """
    Predict the sales trend of a show.

    Given known features up to day ``t``, forecast the cumulative percentage of sales for day ``t+1``.
    The features are then updated (assuming the prediction equals the true value) before proceeding to
    predict subsequent days. This loop continues until the full sales trend has been generated.
    """
    start_time = time()
    current_day = known_df["date"].iloc[-1]

    df = keep_enabled_columns(known_df.copy())
    X, Y = separete_features_targets(df, sort=False, shuffle=False)
    ts_engine = TimeSeriesEngine(df=X)
    predictions = []
    predicitons_days = []
    
    # Keep iterating until we predict the last day (day of the final performance of the show).
    n_predictions = df["sales_duration"].iloc[0]-len(known_df)   

    last_date = last_date-pd.Timedelta(days=1)
    while(current_day<last_date):
        current_day += pd.Timedelta(days=1)
        
        input_row = ts_engine.df.iloc[[-1]]
        if isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(input_row)
            prediction = model.predict(dmat)[0]
        else:
            prediction = model.predict(input_row)[0]
        
        # Ensure the prediction is never below the last observed percentage.
        prediction = max(prediction, input_row.iloc[0][target])

        predictions.append(prediction)
        predicitons_days.append(current_day)

        # Update all features with the new prediction.
        ts_engine.update_features(prediction)

    prediction_df = pd.DataFrame({"predictions": predictions}, index=predicitons_days)
    return prediction_df
