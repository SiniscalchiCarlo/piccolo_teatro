import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

from piccolo_teatro import ts_engine
from .utils import add_deltas, add_log_transform, one_hot_encode, add_cumulative_sum, add_moving_avarages, add_shifted_values, add_targets
from .ingestion import Sales, Products, Seasons
import logging
import time

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
path = os.environ.get("FOLDER_PATH")
encoding_dict = ts_engine.encoding_dict

def get_season_dates(seasons_df, season_id: str):
    '''
    From season id to start and end date of the season
    '''
    season_row = seasons_df[seasons_df["season_id"] == season_id]
    start_date = season_row["inizio_vendite"].iloc[0]
    end_date = season_row["fine_vendite"].iloc[0]
    return start_date, end_date

def add_show_info(products_df, df, show_id):
    '''
    This functions enriches the dataframe by adding informations about shows (hour, number of tickes...)
    '''
    show_id=int(show_id)
    performances_same_show = products_df[products_df["show_id"]==show_id]



    performance_state = performances_same_show["performance_state"].iloc[0]
    show_type = performances_same_show["show_type"].iloc[0]
    preformance_season = performances_same_show["season_name"].iloc[0]
    performance_space = performances_same_show["space"].iloc[0]
    show_capacity = sum(performances_same_show["max_tickets"])
    


    def get_day_month(date, year1, year2):
        date = date.split(" ")
        day = int(date[1].split("/")[0])
        month = int(date[1].split("/")[1])
        if month<=12 and month>=9:
            year = year1
        else:   
            year = year2
            
        return f"{day}/{month}/{year}"
    

    year1 = preformance_season.split(' ')[1].split('/')[0]
    year2 = "20" + preformance_season.split(' ')[1].split('/')[1]
    performances_same_show = performances_same_show.copy()
    
    if performances_same_show['performance_day'].iloc[0].split(" ")[0] in ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]:
        performances_same_show.loc[:, 'performance_date'] = performances_same_show['performance_day'].apply(lambda x: get_day_month(x, year1, year2))
        performances_same_show.loc[:, "performance_date"] = pd.to_datetime(performances_same_show["performance_date"], format='%d/%m/%Y')
    else:
        performances_same_show.loc[:, 'performance_date'] = pd.to_datetime(performances_same_show["performance_day"], errors='coerce')


    last_date = performances_same_show["performance_date"].max()

    # Get all the performances of the same show
    performances_same_show = products_df[products_df["show_id"]==show_id].copy()

    performances_to_not_consider = [
        "Evento collaterale",
        "Altro",
        "Spettacolo per bambini e ragazzi",
    ]
    spaces_to_consider = [
        "Teatro Studio Melato",
        "Teatro Strehler",
        "Teatro Grassi"
    ]

    if performance_state=="In esecuzione" and show_type not in performances_to_not_consider and performance_space in spaces_to_consider:
        df["show_type"] = show_type 
        df["show_capacity"] = show_capacity 
        df["num_performances"] = len(performances_same_show)
        df["last_date"] = last_date
        df = one_hot_encode(df, encoding_dict)
        return df
    else:
        return pd.DataFrame()
        
def add_features(seasons: Seasons, products: Products, df: pd.DataFrame, live_data=False):
        '''
        This function is called to enrich the sales dataframe with new features, 
        ensuring it contains all the features that might be needed.
        '''
        
    
        if df['show_id'].nunique() != 1:
            raise Exception("Error: passed to 'add_features()' sales that are not all from the same show")
        
        df = df.sort_values(by="date")

        # Get start and end season date
        start_date, end_date = get_season_dates(seasons.df, season_id=df["season_id"].iloc[0])
        # Remove not needed features
        df = df.drop(columns=["individuali_gruppi","online_offline"])

        # Daily sum of the sales
        df = df.groupby(['date', 'season_id', 'show_id'], as_index=False).sum()
        show_id = df["show_id"].iloc[0]
        
        # Calculating cumulative sum of gains and tickets
        df = add_cumulative_sum(df, column_names=["gain", "tickets"])
        df["avg_ticket_price"] = df["gain_cum_sum"]/df["tickets_cum_sum"]
        
        # Add informations about the performance, and checks the product is one of the one we are interested in 
        df = add_show_info(products.df, df, show_id)  
        if not df.empty:
            if live_data:
                fill_date = df["date"].iloc[-1]
            else:
                fill_date = df["last_date"].iloc[0]

            # Aggiungo i dati dei giorni mancanti (giorni senza vendite), li riempio mettendo l'ultimo valore noto
            date_range = pd.date_range(start=df["date"].min(), end=fill_date)
            df = df.set_index("date").reindex(date_range, method="ffill")
            df["date"] = df.index

        
            # Distanza della transazione dall'inizio e dalla fine della stagione
            df["start_sales_distance"] = (df["date"]-start_date).dt.days.abs()
            df["end_season_distance"] = (df["date"]-end_date).dt.days.abs()
            df["sales_duration"] = (df["last_date"]-start_date).dt.days
            df["end_sales_distance"] = (df["last_date"]-df["date"]).dt.days
            df["percentage_sales_day"] = df["start_sales_distance"]/(df["last_date"]-start_date).dt.days
            df["percentage_sales_day"] = df["percentage_sales_day"]
        
        
            # Numero biglietti rimanenti per raggiungere capienza massima
            df["remaining_tickets"] = df["show_capacity"]-df["tickets_cum_sum"]
            df["percentage_bought"] = df["tickets_cum_sum"]/df["show_capacity"]
            df["percentage_bought_delta"] = df["percentage_bought"].diff(periods=1)

            
            df = add_log_transform(df, "percentage_bought")

            df = add_deltas(df, ["percentage_bought", "percentage_bought_log1p"], [2,4,6,8,10,15,20,30])
            # Aggiungo medie mobili con differenti periodi
            df = add_moving_avarages(df, ["percentage_bought_delta", "gain_cum_sum", "tickets_cum_sum", "percentage_bought","percentage_bought_log1p"], [2,4,6,8,10,15,20,30])

            # Aggiungo valori shiftati
            df = add_shifted_values(df, ["percentage_bought_delta", "gain_cum_sum", "tickets_cum_sum", "percentage_bought","percentage_bought_log1p"], [2,4,6,8,10,15,20,30])
            
            # Aggiungo i possibili target da prevedere:
            df = add_targets(df, ["percentage_bought", "percentage_bought_log1p", "percentage_bought_delta"])
            
        else:
            df = pd.DataFrame()
        return df
