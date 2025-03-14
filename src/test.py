import pandas as pd
from utils.logger import setup_logger
import os
from dotenv import load_dotenv, find_dotenv



load_dotenv(find_dotenv())
path = os.getenv('FOLDER_PATH')

sales_db = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv")
sales_db = sales_db.rename(columns={
        "D_SALES_LIST_SALES_Individuali_Gruppi": "individuali_gruppi",
        "D_SALES_LIST_SALES_Tipologia_canale": "online_offline",
        "D_SALES_LIST_SALES_TOTAL_CURRENT_AMT_ITX": "gain",
        "D_SALES_LIST_SALES_CURRENT_QUANTITY": "numero_biglietti",
        "D_SALES_LIST_SALES_REFERENCE_DATE": "date",
        "D_SALES_LIST_SALES_T_SEASON_ID": "season_id",
        "D_SALES_LIST_SALES_T_PERFORMANCE_ID": "performance_id",
    })
sales_db["date"] = pd.to_datetime(sales_db["date"], format='%d/%m/%Y')
# Group by season_id and get the min and max dates
db_start_end_season = sales_db.groupby('season_id')["date"].agg(['min', 'max']).reset_index()

# Get the row numbers of the min and max dates for each season_id
min_indices = sales_db.groupby('season_id')['date'].idxmin().reset_index(name='min_index')
max_indices = sales_db.groupby('season_id')['date'].idxmax().reset_index(name='max_index')

# Merge to get the min and max indices with the original min and max dates
result = db_start_end_season.merge(min_indices, on='season_id').merge(max_indices, on='season_id')

print(result)