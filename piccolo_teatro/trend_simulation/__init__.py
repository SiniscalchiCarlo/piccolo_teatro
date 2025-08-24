
import pandas as pd
import matplotlib.pyplot as plt

from ..train import keep_enabled_columns, separete_features_targets
from ..config import Features

pd.set_option("display.max_columns", None)

def predict_trend(known_df, model):
    '''
    Predicts the sales trend of a show:
    Given known features up to day t, this function predicts the cumulative percentage of sales for day t+1.
    It then updates the features (assuming the prediction is equal to the real value) and proceeds to predict the next day's value.
    This process is repeated iteratively until the full sales trend is generated.
    '''

    current_day = known_df["date"].iloc[-1]

    df = keep_enabled_columns(known_df.copy())
    X, Y = separete_features_targets(df, sort=False, shuffle=False)
    features = Features(df=X)
    predictions = []
    predicitons_days = []
    
    # Keep iterating until we predict the last day (day of the last performance of the show)
    n_predictions = df["sales_duration"].iloc[0]-len(known_df)   

    for i in range(n_predictions):
        current_day += pd.Timedelta(days=1)
        
        input_row = features.df.iloc[[-1]]
        prediction = model.predict(input_row)[0]
        
        # Prediciton need to be always highes than the last percentage 
        prediction = max(prediction, input_row.iloc[0]["percentage_bought"])

        predictions.append(prediction)
        predicitons_days.append(current_day)

        # Updating all the features with the new prediction
        features.update_features(prediction)

    prediction_df = pd.DataFrame({"predictions": predictions}, index=predicitons_days)
    return prediction_df
