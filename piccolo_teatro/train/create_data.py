import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
import logging


from ..config import FeatEngConf
from ..data_ingestion.raw_data import Sales, Products, Seasons
from ..data_ingestion.feat import add_features

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
path = os.environ.get("FOLDER_PATH")

feat_eng_conf = FeatEngConf()
encoding_dict = feat_eng_conf.encoding_dict
targets_dict = feat_eng_conf.targets_dict

pd.set_option("display.max_columns", None)
load_dotenv(find_dotenv())

# Loading data
path = os.environ.get("FOLDER_PATH")
SALES = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv", index_col=False)
PRODUCTS = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
SEASONS = pd.read_csv(path+"\\stagioni.csv", index_col=False)

# Ingesting and cleaning data
sales = Sales(SALES)
products = Products(PRODUCTS)
seasons = Seasons(SEASONS)

sales.clean()
products.clean()
seasons.clean()

# Creating dataset needed to train and test the model
def create_model_input(sales: Sales, products:Products, seasons: Seasons, train_dim = 0.6, validation_dim = 0.2):
    TRAIN = pd.DataFrame()
    VALIDATION = pd.DataFrame()
    TEST = pd.DataFrame()

    sales_df = sales.df
    groups = sales_df.groupby('show_id')
    

    i=0
    counter=0
    len_=len(groups)
    tr=False
    val=False
    for show_id, group in groups:
        i+=1
        counter+=1
        print(i/len_)
        sales = Sales(group)
        group = add_features(seasons=seasons,
                             products=products,
                             sales=group,
                             fill_to_show_date=True)
        
        if len(group)>30:
            group.to_parquet(path+f"\\shows\\{show_id}.gzip", index=False)
            
            if not tr:
                TRAIN = pd.concat([TRAIN, group], ignore_index=True)
                if counter/len_>train_dim:
                    counter=0
                    tr=True

            elif not val:
                VALIDATION = pd.concat([VALIDATION, group], ignore_index=True)
                if counter/len_>validation_dim:
                    counter=0
                    val=True

            elif tr and val:
                TEST = pd.concat([TEST, group], ignore_index=True)

    TRAIN.to_parquet(path+f"\\train_trend.gzip", index=False)
    VALIDATION.to_parquet(path+f"\\validation_trend.gzip", index=False)
    TEST.to_parquet(path+f"\\test_trend.gzip", index=False)

create_model_input(sales, products, seasons)
