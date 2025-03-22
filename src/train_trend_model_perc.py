import os
import random
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt

from utils.df_operations import add_cumulative_sum, add_moving_avarages, add_shifted_values, print_unique_values 
from config.train_config import TrainConfig

def trend_error(df, model, start_offset, train_config, plot=True):

    prediction_period = train_config.prediction_period
    target_type = train_config.target
    target_col = f"TARGET_{target_type}_{prediction_period}"
    features_periods = train_config.periods


    performances = df['performance_id'].unique()
    abs_errors = []
    for performance_id in performances:
        performance_id = random.choice(performances)
        df_ = df.copy()
        df_ = df_[df_["performance_id"] == performance_id]
        X_test, Y_test = prepare_data(df_, sort=True)
        plt.title(str(performance_id))
        plt.plot(Y_test, label="Historical Sales", color="blue")
        
        

        start_offset = int(len(X_test)*0.3)

        last_known_date = X_test.index[start_offset-1] #se voglio la start_offset riga es = 10 voglio la rica di indice start_offset-1=9 perché inizio a contare da 0
        end_date = X_test.index[-1]

        # Initial data needed to make the first prediction
        # The rows up to and including start_day are known data used for making the first prediction.
        first_row = X_test.iloc[0]
        
        
        # Initial known features
        input_df = pd.DataFrame()
        known_features = X_test.head(start_offset)
        #print(known_features)
        # known variable features
        for feature in train_config.features:
            if feature.enabled and not feature.const:
                for col in feature.columns:

                    input_df[col] = known_features[col]

        # known const features
        const_values = {}
        for feature in train_config.features:
            if feature.enabled and feature.const:
                for col in feature.columns:
                    const_values[col] = first_row[col]
                    input_df[col] = first_row[col]

        # So the first prediction will made for the start_day+1day
        print(last_known_date, last_known_date + pd.Timedelta(days=1))
        target_df = Y_test.loc[last_known_date + pd.Timedelta(days=1):end_date]
        predictions = []
        plt.plot(target_df, color="blue")
        print("INPUT ROWS:")
        for index, y in target_df.items():

            # MODEL PREDICTION
            #print("input_df", input_df)
            input_row = input_df.iloc[[-prediction_period]]
            input_row = input_row[model.feature_names_in_]  # Ordering features in the order the model was trained
            #print(index)
            #print(input_df["percentage_bought"].tail(10))
            #print_unique_values(input_row)
            #print("\n")
            prediction = model.predict(input_row)[0]
            #print("prediction",prediction)
            predictions.append(prediction)
            abs_errors.append(abs(prediction-y)) #calculate error

            # UPDATING INPUT DATA BASED 
            updated_input_df = pd.DataFrame()
            targets_and_predictions = input_df[target_type].tolist() + [prediction]
            updated_input_df[target_type] = targets_and_predictions
            # Updating data that need the prediction
            # (se il target_type = "tickets_cum_sum" allora non devo calcolare la cum_sum perchè è il valore che ho predetto)


            # Add moving avarages and shifted values
            if train_config.percentage_bought_avg.enabled:
                updated_input_df = add_moving_avarages(updated_input_df, [target_type], features_periods)

            if train_config.percentage_bought_shifted.enabled:
                updated_input_df = add_shifted_values(updated_input_df, [target_type], features_periods)
            
            # Adding the const values
            for col in const_values:
                updated_input_df[col] = const_values[col]

            # Updating features that don't need the value of the prediction
            # print(input_row)
            updated_input_df = updated_input_df.reset_index(drop=True)
            
            if train_config.start_sales_distance.enabled:
                new_value = float(input_row["start_sales_distance"].iloc[0]) + 1
                values = input_df["start_sales_distance"].tolist()
                values.append(new_value)
                updated_input_df["start_sales_distance"] = values

            if train_config.end_sales_distance.enabled:
                new_value = float(input_row["end_sales_distance"].iloc[0]) -1
                values = input_df["end_sales_distance"].tolist()
                values.append(new_value)
                updated_input_df["end_sales_distance"] = values
                

            input_df = updated_input_df.copy()
            #print("input_df2", input_df)
            #time.sleep(1000)

        if plot:
            plt.plot(target_df.index, predictions, color="red")
        #print(input_df)
        plt.show()
    print("AVG_ERROR", sum(abs_errors)/len(abs_errors))


def keep_needed_columns(df):
    # Removing not needed features
    features = train_config.features
    target_col = f"TARGET_{train_config.target}_{train_config.prediction_period}"
    columns_to_keep = ["date", target_col]
    for feature in features:
        if feature.enabled:
            columns_to_keep += feature.columns


    df = df[columns_to_keep]
    return df


def prepare_data(df, shuffle=True, sort=False):
    df = keep_needed_columns(df)
    df.loc[:, "date"] = pd.to_datetime(df["date"], format='%d/%m/%Y')
    df = df.set_index("date")

    if shuffle:
        df = df.sample(frac=1)
    if sort:
        df = df.sort_values(by="date")
    target_col = f"TARGET_{train_config.target}_{train_config.prediction_period}"
    Y_df = df[target_col]
    X_df = df.drop(columns=[target_col])
    return X_df, Y_df


pd.set_option('display.max_columns', None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')

train_config = TrainConfig()

train_df = pd.read_csv(path+f"\\train_trend.csv", index_col=False)
validation_df = pd.read_csv(path+f"\\validation_trend.csv", index_col=False)
test_df = pd.read_csv(path+f"\\test_trend.csv", index_col=False)

X_train, Y_train = prepare_data(train_df)
X_validation, Y_validation = prepare_data(validation_df)
print("FEATURES:")
print_unique_values(X_train)


# training
model = XGBRegressor(n_estimators=200, learning_rate=0.1,
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

trend_error(df=validation_df, model=model,
            start_offset=30, train_config=train_config)
