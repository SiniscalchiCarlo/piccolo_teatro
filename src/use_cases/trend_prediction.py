

import pickle
from typing import Any
import pandas as pd
import datetime
from pydantic import BaseModel
import matplotlib.pyplot as plt

from config import TrainConfig
from train.model_preparation import Features, ModelData

class TrendPrediction(BaseModel):
    model_path: str
    data: ModelData
    train_config: TrainConfig = TrainConfig()
    features_config: Features = Features()
    model:  Any = None

    class Config:
        arbitrary_types_allowed = True

    
    def load_model(self):
        self.model = pickle.load(open(self.model_path, "rb"))

    def trend_prediction(self, last_date: datetime, plot: bool, offset: float=None):

        #get only the part of known data to do the simulation
        
        if self.model is None:
            self.load_model()
        
        # Calculating the predictions span
        if offset is not None:
            day_offset = int(len(self.data.df)*offset)
            start_day = self.data.df.head(day_offset)["date"].iloc[-1]

        predictions_range = pd.date_range(start=pd.to_datetime(start_day, format="%d/%m/%Y"),
                           end=pd.to_datetime(last_date, format="%d/%m/%Y"))
        num_predictions = len(predictions_range)

        if self.data.X is None and self.data.Y is None:
            # Keep only enabled features, separeta features from target,
            self.data.separete_features_targets(sort=True)

        if offset is not None:
            self.data.X = self.data.X.head(day_offset)

        predictions = []
        for i in range(num_predictions):
            #1. MODEL PREDICTION
            input_row = self.data.X.iloc[[-1]] # Take the last known row of known data/data generated from predictions
            input_row = input_row[self.model.feature_names_in_]  # Ordering features in the order the model was trained
            prediction = self.model.predict(input_row)[0]
            predictions.append(prediction)

            #2. UPDATING FEATURES
            self.features_config.update_features(self.data.X, prediction)
            self.data.X = self.features_config.updated_df.copy()
        
        if plot:
            plt.plot(self.data.Y)
            plt.plot(predictions_range, predictions, color="red")
            plt.show()
