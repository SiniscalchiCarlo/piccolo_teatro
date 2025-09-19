import pandas as pd
import numpy as np
from piccolo_teatro import ts_engine

def one_hot_encode(df, encoding_dict):
    # Expand categorical columns into one-hot encoded indicator columns based
    # on the provided dictionary of possible values.
    for col_name in encoding_dict:
        values = encoding_dict[col_name]
        for value in values:
            if str(value)!="nan":
                df[value.lower()] = (df[col_name] == value).astype(int)
    df = df.drop(columns=list(encoding_dict.keys()))
    return df


def add_log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # Apply a log1p transform to stabilise growth trends and store it as a
    # separate feature column.
    new_column = f"{column}_log1p"
    df[new_column] = np.log1p(df[column])
    return df


def add_deltas(df, columns, lags):
    # Create finite-difference features that measure change over each
    # specified lag.
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_delta_{lag}"] = df[col].diff(periods=lag)
    return df

def add_cumulative_sum(df, column_names: list[str]):
    # Track cumulative totals for the provided columns.
    for col_name in column_names:
        df[col_name+"_cum_sum"] = df[col_name].cumsum()
    return df

def add_moving_avarages(df, column_names: list[str], periods: list[int]):
    # Compute moving averages for each column across every requested window.
    for period in periods:
        for col_name in column_names:
            df[col_name+f"_avg_{period}"] = df[col_name].rolling(period).mean()
    return df

def add_shifted_values(df, column_names: list[str], periods: list[int]):
    # Append lagged versions of each column so models can access prior values
    # directly.
    for period in periods:
        for col_name in column_names:
            df[col_name+f"_shifted_{period}"] = df[col_name].shift(period).fillna(df[col_name].iloc[0])
    return df

def print_unique_values(df):
    # Helper routine to inspect the first few unique values of every column.
    for col_name in df.columns:
        print(f"{col_name}: {df[col_name].unique()[:10]}")

def add_targets(df,columns):
    # Shift each potential target column forward one step so the next day's
    # value becomes the supervised learning target.
    df = df.copy()
    for col in columns:
        df.loc[:, f"TARGET_{col}"] = df[col].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)
    return df
