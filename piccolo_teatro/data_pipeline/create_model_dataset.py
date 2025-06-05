import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import logging


from ..config import FeatEngConf
from . import clean_data
from .ingestion import Sales, Products, Seasons
from .transformation import add_features

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
path = os.environ.get("FOLDER_PATH")

feat_eng_conf = FeatEngConf()
encoding_dict = feat_eng_conf.encoding_dict
targets_dict = feat_eng_conf.targets_dict

pd.set_option("display.max_columns", None)
load_dotenv(find_dotenv())



def save_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, path: str) -> None:
    """Saves the train, validation, and test datasets to disk."""

    train.to_parquet(f"{path}/train_trend.gzip", index=False)
    validation.to_parquet(f"{path}/validation_trend.gzip", index=False)
    test.to_parquet(f"{path}/test_trend.gzip", index=False)


def create_full_datasets(sales: Sales, products: Products, seasons: Seasons, path: str, save_csv=False, train_dim=0.6, val_dim=0):
    '''
    Splits the input sales data into training, validation, and test datasets,
    while maintaining time series continuity for each show. Each show's data
    is added in full to only one dataset to prevent leakage across splits.

    Parameters:
        sales (Sales): Sales data object containing a DataFrame.
        products (Products): Product information used to enrich the dataset.
        seasons (Seasons): Season information for time-based feature engineering.
        path (str): Directory path to save the output files.
        train_dim (float): Proportion of shows to include in the training set.
        validation_dim (float): Proportion of shows to include in the validation set.

    Output:
        Saves the resulting datasets (train, validation, test) and individual show files
        as Parquet files in the specified path.
    '''
    TRAIN = pd.DataFrame()
    VALIDATION = pd.DataFrame()
    TEST = pd.DataFrame()

    groups = sales.get_groups()

    total_shows = len(groups)
    train_shows = 0
    val_shows = 0
    train_limit = total_shows*train_dim
    val_limit = total_shows*val_dim

    for i, (show_id, group) in enumerate(groups):
        print(f"{(i+1)/total_shows:.2%} processed")

        group = add_features(seasons=seasons,
                             products=products,
                             df=group,
                             live_data=False)
        
            
        group.to_parquet(f"{path}/shows/{show_id}.gzip", index=False)

        if save_csv:
            group.to_csv(f"{path}/shows/csv/{show_id}.csv", index=False)

        if train_shows < train_limit:
            TRAIN = pd.concat([TRAIN, group], ignore_index=True)
            train_shows +=1
        elif val_shows < val_limit and val_dim!=0:
            VALIDATION = pd.concat([VALIDATION, group], ignore_index=True)
            val_shows+=1
        else:
            TEST = pd.concat([TEST, group], ignore_index=True)

    return TRAIN, VALIDATION, TEST

# Loading data
path = os.environ.get("FOLDER_PATH")
SALES = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv", index_col=False)
PRODUCTS = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
SEASONS = pd.read_csv(path+"\\stagioni.csv", index_col=False)

sales, products, seasons = clean_data(SALES, PRODUCTS, SEASONS)
TRAIN, VALIDATION, TEST = create_full_datasets(sales, products, seasons, path, save_csv=True)
save_splits(TRAIN, VALIDATION, TEST, path)

