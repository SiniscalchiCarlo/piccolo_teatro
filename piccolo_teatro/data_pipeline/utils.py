import pandas as pd
from piccolo_teatro import ts_engine

def one_hot_encode(df, encoding_dict):
    for col_name in encoding_dict:
        values = encoding_dict[col_name]
        for value in values:
            if str(value)!="nan":
                df[value.lower()] = (df[col_name] == value).astype(int)
    df = df.drop(columns=list(encoding_dict.keys()))
    return df

def add_cumulative_sum(df, column_names: list[str]):
    for col_name in column_names:
        df[col_name+"_cum_sum"] = df[col_name].cumsum()
    return df

def add_moving_avarages(df, column_names: list[str], periods: list[int]):
    for period in periods:
        for col_name in column_names:
            df[col_name+f"_avg_{period}"] = df[col_name].rolling(period).mean()
            df = df.fillna(df[col_name].iloc[0])
    return df

def add_shifted_values(df, column_names: list[str], periods: list[int]):
    for period in periods:
        for col_name in column_names:
            df[col_name+f"_shifted_{period}"] = df[col_name].shift(period).fillna(df[col_name].iloc[0])
            df = df.fillna(df[col_name].iloc[0])
    return df

def print_unique_values(df):
    for col_name in df.columns:
        print(f"{col_name}: {df[col_name].unique()[:10]}")

def add_targets(df):
    col_name = ts_engine.target
    df[f"TARGET_{col_name}"] = df[col_name].shift(-1)
    df = df.iloc[:-1]
    return df
