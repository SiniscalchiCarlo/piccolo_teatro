import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
import logging


from . import clean_data
from .ingestion import Sales, Products, Seasons
from .transformation import add_features
from piccolo_teatro import ts_engine

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
path = os.environ.get("FOLDER_PATH")

encoding_dict = ts_engine.encoding_dict

pd.set_option("display.max_columns", None)
load_dotenv(find_dotenv())



def save_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, path: str, save_csv=False) -> None:
    """Saves the train, validation, and test datasets to disk."""

    train.to_parquet(f"{path}/train_trend.gzip", index=False)
    validation.to_parquet(f"{path}/validation_trend.gzip", index=False)
    test.to_parquet(f"{path}/test_trend.gzip", index=False)

    if save_csv:
        train.to_csv(f"{path}/train_trend.csv", index=False)
        validation.to_csv(f"{path}/validation_trend.csv", index=False)
        test.to_csv(f"{path}/test_trend.csv", index=False)



def create_full_datasets(sales: Sales, products: Products, seasons: Seasons, path: str, save_csv=False, train_dim=0.6, val_dim=0.2):
    '''
    Divide shows in train, validation and test folders.
    A file is created for each show containing 
    all the data of that show to prevent data leakage across
    different folder.
    '''
    save_folder = f"{path}/shows/"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
        os.makedirs(save_folder+"/train/")
        os.makedirs(save_folder+"/validation/")
        os.makedirs(save_folder+"/test/")

    groups = sales.get_groups()

    total_shows = len(groups)
    train_shows = 0
    val_shows = 0
    train_limit = total_shows*train_dim
    val_limit = total_shows*val_dim
    groups = [(show, g) 
          for show, g in groups]

    np.random.seed(42)
    np.random.shuffle(groups)
    for i, (show_id, group) in enumerate(groups):
        print(f"{(i+1)/total_shows:.2%} processed")
        
        group = add_features(seasons=seasons,
                             products=products,
                             df=group,
                             live_data=False)

        if train_shows < train_limit:
            train_shows +=1
            folder = "train"
        elif val_shows < val_limit and val_dim!=0:
            val_shows+=1
            folder = "validation"
        else:
            folder = "test"
        
        # We save the data only if has at least 10 days of sales
        if len(group.index.unique().tolist())>10:
            group.to_parquet(f"{path}/shows/{folder}/{show_id}.gzip", index=False)
            if save_csv:
                group.to_csv(f"{path}/shows/{folder}/{show_id}.csv", index=False)


# Loading data
path = os.environ.get("FOLDER_PATH")
SALES = pd.read_csv(path+"/D_SALES_LIST_SALES.csv", index_col=False)
PRODUCTS = pd.read_csv(path+"/D_CONFIG_PROD_LIST.csv", index_col=False)
SEASONS = pd.read_csv(path+"/stagioni.csv", index_col=False)

sales, products, seasons = clean_data(SALES, PRODUCTS, SEASONS)
print("Sales Df:")
print(f"Shape: {sales.df.shape[0]} rows × {sales.df.shape[1]} cols")
print(sales.df.dtypes)
print("Products Df:")
print(f"Shape: {products.df.shape[0]} rows × {products.df.shape[1]} cols")
print(products.df.dtypes)
print("Seasons Df:")
print(f"Shape: {seasons.df.shape[0]} rows × {seasons.df.shape[1]} cols")
print(seasons.df.dtypes)
create_full_datasets(sales, products, seasons, path, save_csv=True)

