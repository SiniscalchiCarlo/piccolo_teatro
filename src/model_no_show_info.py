import os
import random

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt

from feature_engineering import add_moving_avarages, add_past_values


def prepare_data(df):
    # split data
    df = df.sort_values(by="date")
    df = df.set_index("date")
    Y_df = df["target"]
    X_df = df.drop(columns=["target",
                   "season_id", "performance_id"])
    return X_df, Y_df


def trend_error(df, model, start_offset, prediction_period, plot):
    performances = df['performance_id'].unique()
    abs_errors=[]
    i=0
    for performance_id in performances:
        performance_id = random.choice(performances)
        i+=1
        print(i/len(performances))
        df_ = df.copy()
        df_ = df_[df_["performance_id"] == performance_id]
        X_test, Y_test = prepare_data(df_)

        start_day = X_test.index[start_offset]
        end_date = X_test.index[-1]

        # Initial data needed to make the first prediction
        # The rows up to and including start_day are known data used for making the first prediction.
        input_gain = Y_test.head(start_offset).tolist()
        first_row = X_test.iloc[0]
        start = int(X_test.iloc[0]["start_season_distance"])
        end = int(X_test.iloc[0]["end_season_distance"])
        start_distances = list(range(start, start + start_offset))
        end_distances = list(range(end, end - start_offset, -1))
        #is_internazionale = first_row["internazionale"]
        #is_ospitalita = first_row["ospitalità"]
        #is_collaborazione = first_row["collaborazione"]
        #is_produzione = first_row["produzione"]
        #is_festival = first_row["festival"]
        #performance_capacity = first_row["performance_capacity"]

        # So the first prediction will made for the start_day+1day
        predictions_rows = Y_test.loc[start_day+pd.Timedelta(days=1):end_date]
        predictions = []
        for index, y in predictions_rows.items():
            # Creating a DataFrame with all the features necessary for the model to make predictions
            input_df = pd.DataFrame({"gain_cum_sum": input_gain,
                                     "start_season_distance": start_distances,
                                     "end_season_distance": end_distances,
                                    # "internazionale":is_internazionale,
                                    # "ospitalità":is_ospitalita,
                                    # "collaborazione":is_collaborazione,
                                    # "produzione":is_produzione, 
                                    # "festival":is_festival,
                                    # "performance_capacity":performance_capacity,
                                     })
            input_df = add_moving_avarages(input_df, ["gain_cum_sum"], [2, 4, 6, 8, 10, 15, 20, 30])
            input_df = add_past_values(input_df, ["gain_cum_sum"], [2, 4, 6, 8, 10, 15, 20, 30])
            input_row = input_df.iloc[[-prediction_period]]

            # Model predictions
            input_row = input_row[model.feature_names_in_]
            prediction = model.predict(input_row)[0]
            predictions.append(prediction)
            abs_errors.append(abs(prediction-y))

            # Updading data used to create features
            input_gain.append(prediction)
            start_distances.append(start_distances[-1]+1)
            end_distances.append(end_distances[-1]-11)
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.title(f"model{prediction_period} on {performance_id}")
            plt.plot(Y_test, label="Historical Sales")
            plt.plot(predictions_rows.index, predictions, label="Predicted Sales by day 0",
                    linestyle="dashed", color="red")
            plt.show()
    print("AVG_ERROR", sum(abs_errors)/len(abs_errors))


def train_model(prediction_period, path):
    # get and shuffle data
    train_df = pd.read_csv(path+f"\\train_trend_{prediction_period}_target.csv", index_col=False)
    validation_df = pd.read_csv(path+f"\\validation_trend_{prediction_period}_target.csv", index_col=False)
    test_df = pd.read_csv(path+f"\\test_trend_{prediction_period}_target.csv", index_col=False)
    train_df = train_df.sample(frac=1)
    validation_df = validation_df.sample(frac=1)
    test_df = test_df.sample(frac=1)
    train_df["date"] = pd.to_datetime(train_df["date"], format='%d/%m/%Y')
    validation_df["date"] = pd.to_datetime(validation_df["date"], format='%d/%m/%Y')
    test_df["date"] = pd.to_datetime(test_df["date"], format='%d/%m/%Y')

    # split data in X and Y
    X_train, Y_train = prepare_data(train_df)
    X_validation, Y_validation = prepare_data(validation_df)

    # training
    model = XGBRegressor(n_estimators=50, learning_rate=0.1,
                        objective="reg:squarederror")
    model.fit(X_train, Y_train)

    # Predict on the training and validation sets
    train_predictions = model.predict(X_train)
    validation_predictions = model.predict(X_validation)

    # Calculate and print metrics
    train_mae = mean_absolute_error(Y_train, train_predictions)
    validation_mae = mean_absolute_error(Y_validation, validation_predictions)
    print(f"Training Mean Absolute Error: {train_mae}")
    print(f"Validation Mean Absolute Error: {validation_mae}")
    trend_error(test_df, model, start_offset=30, prediction_period=prediction_period, plot=True)
    return model

pd.set_option('display.max_columns', None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')
#model1 = train_model(1,path)
model7 = train_model(1,path)




