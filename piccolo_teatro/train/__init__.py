import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from ..config import TrainConfig, Features, TrainConfig

features_config: Features = Features()
train_config: TrainConfig = TrainConfig()

def separete_features_targets(df, sort=False, shuffle=False):
    """
    splits the df into features and target
    sort: if true sorts the df, if false shuffles it
    """
    if df is None:
        raise Exception("Please load the df first")

    if sort and shuffle:
        raise Exception("You can't both shuffle and sort, please choose one")
    
    # Convert date column to datetime"
    df.loc[:, "date"] = pd.to_datetime(df["date"], format='%d/%m/%Y')
    df = df.set_index("date")

    
    if sort:
        df = df.sort_values(by="date")
    if shuffle:
        df = df.sample(frac=1)

    target_col = f"TARGET_{train_config.target}_{train_config.prediction_period}"
    if target_col in df:
        Y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, Y


def keep_enabled_columns(df):
    """
    Keeps only the enabled features and the target column
    """
    # Removing not needed features
    features = features_config.enabled_features
    target_col = f"TARGET_{train_config.target}_{train_config.prediction_period}"
    columns_to_keep = ["date", target_col]
    for feature in features:
        if feature.enabled:
            columns_to_keep += feature.columns

    df = df[columns_to_keep]
    return df

def abs_error(model, X, Y):
        train_prediciton = model.predict(X)
        train_mae = mean_absolute_error(Y, train_prediciton)
        plt.plot(Y-X["percentage_bought"], "ro")
        plt.show()
        print(f"Mean Absolute Error: {train_mae}")