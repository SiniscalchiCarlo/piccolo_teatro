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

from feature_engineering import print_unique_values




def prepare_data(df):
    # split data
    Y_df = df["tomorrow_gain"]
    X_df = df.drop(columns=["tomorrow_gain", "date",
                   "season_id", "performance_id"])
    return X_df, Y_df


pd.set_option('display.max_columns', None)
load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')



# get and shuffle data
train_df = pd.read_csv(path+"\\train_trend_prediciton.csv", index_col=False)
validation_df = pd.read_csv(path+"\\validation_trend_prediciton.csv", index_col=False)
test_df = pd.read_csv(path+"\\test_trend_prediciton.csv", index_col=False)

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
model = XGBRegressor(n_estimators=300, learning_rate=0.1,
                     objective="reg:squarederror")
model.fit(X_train, Y_train)

# Predict on the training and validation sets
train_predictions = model.predict(X_train)
validation_predictions = model.predict(X_validation)

# Calculate and print metrics
train_mse = mean_squared_error(Y_train, train_predictions)
validation_mse = mean_squared_error(Y_validation, validation_predictions)
train_mae = mean_absolute_error(Y_train, train_predictions)
validation_mae = mean_absolute_error(Y_validation, validation_predictions)


print(f"Training Mean Absolute Error: {train_mae}")
print(f"Validation Mean Absolute Error: {validation_mae}")

# TEST:
# pick performances in test set that are not in train or validation

performance_id = test_df['performance_id'].sample().iloc[0]
test_df = test_df[test_df["performance_id"] == performance_id]
test_df = test_df.sort_values(by="date")

X_test, Y_test = prepare_data(test_df)
X_test.reset_index(drop=True, inplace=True)
Y_test.reset_index(drop=True, inplace=True)

start_day = 30
predicted_gains = Y_test.head(start_day).tolist()
start = int(X_test.iloc[0]["start_season_distance"])
end = int(X_test.iloc[0]["end_season_distance"])
start_distances = list(range(start, start + start_day))
end_distances = list(range(end, end - start_day, -1))


def add_moving_avarages(df, column_names: list[str], periods: list[int]):
    for period in periods:
        for col_name in column_names:
            col_name = col_name+"_cum_sum"
            df[col_name+f"{period}d_avg"] = df[col_name].rolling(period).mean()
            df = df.fillna(df[col_name].iloc[0])
    return df

for i in range(len(Y_test)-start_day):
    prediction_df = pd.DataFrame({"gain_cum_sum":predicted_gains,
                                  "start_season_distance":start_distances,
                                  "end_season_distance":end_distances})

    prediction_df = add_moving_avarages(prediction_df, ["gain"], [5,10,15,20,30,40,50])
    prediction_df = prediction_df.drop(columns=["gain_cum_sum"])
    last_row = prediction_df.iloc[[-1]]    
    #print(last_row)
    prediction = model.predict(last_row)[0]
    predicted_gains.append(prediction)
    start_distances.append(start_distances[-1]+1)
    end_distances.append(end_distances[-1]-11)

plt.figure(figsize=(12, 6))
plt.plot(Y_test, label="Historical Sales")
plt.plot(predicted_gains, label="Predicted Sales by day 0",
         linestyle="dashed", color="red")
plt.show()

