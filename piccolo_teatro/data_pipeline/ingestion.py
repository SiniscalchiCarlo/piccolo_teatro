from pydantic import BaseModel
from typing import List, Dict, Callable
import pandas as pd

class Sales():
    def __init__(self, initial_df):
        # Dictionary mapping original column names to desired names
        self.sales_cols = {
            "D_SALES_LIST_SALES_Individuali_Gruppi": "individuali_gruppi",
            "D_SALES_LIST_SALES_Tipologia_canale": "online_offline",
            "D_SALES_LIST_SALES_TOTAL_CURRENT_AMT_ITX": "gain",
            "D_SALES_LIST_SALES_CURRENT_QUANTITY": "tickets",
            "D_SALES_LIST_SALES_REFERENCE_DATE": "date",
            "D_SALES_LIST_SALES_T_PRODUCT_ID": "show_id",
            "D_SALES_LIST_SALES_T_SEASON_ID": "season_id",
            "D_SALES_LIST_SALES_T_PERFORMANCE_ID": "performance_id",
            "D_SALES_LIST_SALES_SEASON": "season_name",
            "D_SALES_LIST_SALES_T_OPERATION_KIND": "operation_kind",
            "D_SALES_LIST_SALES_OPERATION_TYPE": "operation_type",
            "D_CONFIG_PROD_LIST_T_PERFORMANCE_ID": "performance_id",
        }

        # Seasons to exclude (covid-seasons+season with missing sales)
        self.seasons_to_remove = [1346872739,
                            2070233463,
                            1112493566,
                            1098863756]

        # Keep only show ticket sales:
        # - PRODUCT_COMPOSITION: sales via subscription
        # - SINGLE_ENTRY: single‐ticket purchases
        self.operations_to_keep = [
                    "PRODUCT_COMPOSITION",
                    "SINGLE_ENTRY",
                ]
        self.df = initial_df
        self.rename_cols()

    def rename_cols(self):
        # Some columns have a space before the name, we need to remove it:
        self.df = self.df.rename(columns=lambda x: x.lstrip())
        self.df = self.df.rename(columns=self.sales_cols)

    def assign_types(self):
        # Ensure 'gain' values use dot as decimal separator if stored as strings
        if type(self.df["gain"].iloc[0]) == str:
            self.df["gain"] = self.df["gain"].str.replace(",",".")
        
        self.df = self.df.dropna(subset=list(self.sales_cols.values()))

        # Cast columns to appropriate types
        self.df = self.df.astype({
            "individuali_gruppi": str,
            "online_offline": str,
            "gain": float,
            "tickets": int,
            "show_id": int,
            "season_id": int,
            "performance_id": int,
            "season_name": str,
            "operation_kind": str,
            "operation_type": str,})
        
        self.cols_too_keep = ["individuali_gruppi",
                                "online_offline",
                                "gain",
                                "tickets",
                                "date",
                                "season_id",
                                "show_id",
                                "performance_id",
                                ]
        
        self.df["date"] = pd.to_datetime(self.df["date"], errors='coerce')

    def clean_cols(self):
        # Keep only the columns needed.
        self.df = self.df[list(self.cols_too_keep)]

    def clean_rows(self):
        
        self.df['season_id'] = self.df['season_id'].astype(int)

        # Exclude sales from specified seasons
        self.df = self.df[~self.df['season_id'].isin(self.seasons_to_remove)]

        # Keep only desired operation kinds (shows)
        self.df = self.df[self.df['operation_kind'].isin(
            self.operations_to_keep)]
        
        # Consider only sales operations.
        self.df = self.df[self.df['operation_type'] == "Venduti"]
    
    def assign_index(self):
        self.df = self.df.set_index('date', drop=False) 

    def sort_values(self):
        self.df = self.df.sort_values('date')

    def get_head(self, offset):
        self.sort_values()
        self.df = self.df.iloc[:int(len(self.df) * offset)]

    def get_same_show(self, show_id):
        self.df = self.df[self.df["show_id"] == show_id]

    def get_groups(self):
        return self.df.groupby('show_id')
    
    def clean(self):
        self.assign_types()
        self.clean_rows()
        self.clean_cols()


class Products():
    def __init__(self, initial_df):
        self.products_cols = {
            "D_CONFIG_PROD_LIST_T_PRODUCT_ID": "show_id",
            "D_CONFIG_PROD_LIST_PERFORMANCE_STATE": "performance_state",
            "D_CONFIG_PROD_LIST_Tipologia_spettacolo": "show_type",
            "D_CONFIG_PROD_LIST_SEASON": "season_name",
            "D_CONFIG_PROD_LIST_SPACE": "space",
            "D_CONFIG_PROD_LIST_PERFORMANCE_QUOTA": "max_tickets",
            "D_CONFIG_PROD_LIST_PRODUCT_DATE_TIME": "performance_day"

        }

        self.df = initial_df
        self.rename_cols()


    def rename_cols(self):
        self.df = self.df.rename(columns=self.products_cols)

    def assign_types(self):
        self.df = self.df.astype({
            "show_id": int,
            "performance_state": str,
            "performance_day": str,
            "show_type": str,
            "season_name": str,
            "space": str,
            "max_tickets":int,})
        
    def clean(self):
        self.assign_types()

class Seasons():

    def __init__(self, initial_df):
        self.seasons_cols = {
            "season_id": "season_id",
            "season_name": "season_name",
            "inizio_vendite": "inizio_vendite",
            "fine_vendite": "fine_vendite",
            "inizio_stagione": "inizio_stagione",
            "fine_stagione": "fine_stagione"

        }

        self.df = initial_df
        self.rename_cols()


    def rename_cols(self):
        self.df = self.df.rename(columns=self.seasons_cols)

    def assign_types(self):
        self.seasons_cols = {
            "season_id": int,
            "season_name": str,
        }

        self.df["inizio_vendite"] = pd.to_datetime(self.df["inizio_vendite"], errors='coerce')
        self.df["fine_vendite"] = pd.to_datetime(self.df["fine_vendite"], errors='coerce')
        self.df["inizio_stagione"] = pd.to_datetime(self.df["inizio_stagione"], errors='coerce')
        self.df["fine_stagione"] = pd.to_datetime(self.df["fine_stagione"], errors='coerce')
        
    def clean(self):
        self.assign_types()
