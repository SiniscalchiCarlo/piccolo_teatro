import pandas as pd
from utils.logger import setup_logger
import os
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt
import time
logger = setup_logger(__name__, level="INFO")


def convert_data(df):
    df["gain"] = df["gain"].str.replace(",",".")
    df = df.astype({
        "individuali_gruppi": str,
        "online_offline": str,
        "gain": float,
        "numero_biglietti": int,
        "season_id": int,
        "performance_id": int,
    })
    df["date"] = pd.to_datetime(df["date"], format='%d/%m/%Y')
    return df

def one_hot_encode(df, encoding_dict):
    for col_name in encoding_dict:
        values = encoding_dict[col_name]
        for value in values:
            df[value.lower()] = (df[col_name] == value)
    return df

def add_cumulative_sum(df, column_names: list[str]):
    for col_name in column_names:
        df[col_name+"_cum_sum"] = df[col_name].cumsum()
    df = df.drop(columns=column_names)

    return df

def add_moving_avarages(df, column_names: list[str], periods: list[int]):
    for period in periods:
        for col_name in column_names:
            df[col_name+f"{period}d_avg"] = df[col_name].rolling(period).mean()
            df = df.fillna(df[col_name].iloc[0])
    return df

def add_past_values(df, column_names: list[str], periods: list[int]):
    for period in periods:
        for col_name in column_names:
            df[col_name+f"{period}d_ago"] = df[col_name].shift(period).fillna(df[col_name].iloc[0])
            df = df.fillna(df[col_name].iloc[0])
    return df

def print_unique_values(df):
    for col_name in df.columns:
        print(f"{col_name}: {df[col_name].unique()[:10]}")

def get_sales(path: str):
    sales_db = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv", index_col=False)
    print("\nSTART DB:")
    print_unique_values(sales_db)
    # some columns have a space before the name, to remove it:
    sales_db = sales_db.rename(columns=lambda x: x.lstrip())

    sales_db = sales_db.rename(columns={
        "D_SALES_LIST_SALES_Individuali_Gruppi": "individuali_gruppi",
        "D_SALES_LIST_SALES_Tipologia_canale": "online_offline",
        "D_SALES_LIST_SALES_TOTAL_CURRENT_AMT_ITX": "gain",
        "D_SALES_LIST_SALES_CURRENT_QUANTITY": "numero_biglietti",
        "D_SALES_LIST_SALES_REFERENCE_DATE": "date",
        "D_SALES_LIST_SALES_T_SEASON_ID": "season_id",
        "D_SALES_LIST_SALES_T_PERFORMANCE_ID": "performance_id",
    })

    colonne_da_mantenere = ["individuali_gruppi",
                            "online_offline",
                            "gain",
                            "numero_biglietti",
                            "date",
                            "season_id",
                            "performance_id",
                            ]
    
    seasons_to_remove = ["Stagione 2014/15",
                         "Stagione 2019/20",
                         "Stagione 2020/21",
                         "Stagione 2021/22"]
                         
    operations_to_remove = [
        "PRODUCT_COMPOSITION",
        "SINGLE_ENTRY",
    ]
    # remoniving covid seasons
    sales_db = sales_db[~sales_db['season_id'].isin(
        seasons_to_remove)]

    # removing sales of products that are not shows
    # sales_db = sales_db[sales_db['D_SALES_LIST_SALES_T_OPERATION_KIND'] == "SIMPLE_PRODUCT"]
    sales_db = sales_db[sales_db['D_SALES_LIST_SALES_T_OPERATION_KIND'].isin(
        operations_to_remove)]
    
    # considering only sales operations
    sales_db = sales_db[sales_db['D_SALES_LIST_SALES_OPERATION_TYPE'] == "Venduti"]

    # keep only the coulmns needed
    sales_db = sales_db[colonne_da_mantenere]

    return sales_db

def adding_performance_info(df, performances_db, performance_id):
    performance_row = performances_db[performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"]==performance_id]

    performance_state = performance_row["D_CONFIG_PROD_LIST_PERFORMANCE_STATE"].iloc[0]
    performance_type = performance_row["D_CONFIG_PROD_LIST_Tipologia_spettacolo"].iloc[0]
    performance_space = performance_row["D_CONFIG_PROD_LIST_SPACE"].iloc[0]
    performance_capacity = performance_row["D_CONFIG_PROD_LIST_PERFORMANCE_QUOTA"].iloc[0]
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
    if performance_state=="In esecuzione" and performance_type not in performances_to_not_consider and performance_space in spaces_to_consider:
        df["performance_type"] = performance_type
        df["perfmormance_capacity"] = performance_capacity
        return df
    else:
        return pd.DataFrame()

def create_model_input(cleaned_sales, season_df, prediction_period, train_dim = 0.6, validation_dim = 0.2):
    train_df = pd.DataFrame()
    validation_df = pd.DataFrame()
    test_df = pd.DataFrame()

    performances_db = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
    performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"] = performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"].astype(int)
    encoding_dict = {
            "performance_type": performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"].unique()
        }
    
    cleaned_sales = cleaned_sales.sort_values(by="date")
    groups = cleaned_sales.groupby('performance_id')
    

    i=0
    counter=0
    len_=len(groups)
    tr=False
    val=False
    for performance_id, group in groups:
        i+=1
        counter+=1
        print(i/len_)
        #get start and end season date
        group_season = group["season_id"].iloc[0]
        season_row = season_df[(season_df["season_id"]) == group_season]
        start_date = season_row["inizio_vendite"].iloc[0]
        end_date = season_row["fine_vendite"].iloc[0]

        group = group.sort_values(by="date")

        #rimuovo features che non mi servono
        group = group.drop(columns=["individuali_gruppi","online_offline", "numero_biglietti"])
        group = group.groupby(['date', 'season_id', 'performance_id'], as_index=False).sum()
        
        
        #FEATURES:
        # Distanza della transazione dall'inizio e dalla fine della stagione
        group["start_season_distance"] = (group["date"]-start_date).dt.days.abs()
        group["end_season_distance"] = (group["date"]-end_date).dt.days.abs()

        # Calcolo la cumulata del guadagno e del numero di biglietti venduti
        group = add_cumulative_sum(group, column_names=["gain"])
        
        # Aggiungo il guadagno cumulato dei giorni mancanti
        date_range = pd.date_range(start=group["date"].min(), end=group["date"].max())
        group = group.set_index("date").reindex(date_range, method="ffill")
        group["date"] = group.index
        # Aggiungo medie mobili con differenti periodi
        group = add_moving_avarages(group, ["gain_cum_sum"], [2,4,6,8,10,15,20,30])
        
        #Aggiungo valori shiftati
        group = add_past_values(group, ["gain_cum_sum"], [2,4,6,8,10,15,20,30])
        
        # Aggiungo colonna target che il modello deve prevedere
        group["target"] = group["gain_cum_sum"].shift(-prediction_period)
        group = group.iloc[:-prediction_period]

        # Rimuovo feature che possono fare leakage
        group = group.drop(columns=["gain_cum_sum"])
        group = adding_performance_info(group,performance_id)  
        if not group.empty:
            group = one_hot_encode(group, encoding_dict)

        if len(group)>30:
            group.to_csv(path+f"\\performances\\{performance_id}_{prediction_period}_target.csv", date_format='%d/%m/%Y', index=False)
            
            #print(len(train_df)/total_data_points,len(validation_df)/total_data_points,len(test_df)/total_data_points)
            #print(counter/len_,train_dim,tr,val,)
            if not tr:
                train_df = pd.concat([train_df, group], ignore_index=True)
                if counter/len_>train_dim:
                    counter=0
                    tr=True
                #print("tr")

            elif not val:
                validation_df = pd.concat([validation_df, group], ignore_index=True)
                if counter/len_>validation_dim:
                    counter=0
                    val=True
                #print("val")

            elif tr and val:
                test_df = pd.concat([test_df, group], ignore_index=True)
                #print("test")
    train_df.to_csv(path+f"\\train_trend_{prediction_period}_target.csv", date_format='%d/%m/%Y', index=False)
    validation_df.to_csv(path+f"\\validation_trend_{prediction_period}_target.csv", date_format='%d/%m/%Y', index=False)
    test_df.to_csv(path+f"\\test_trend_{prediction_period}_target.csv", date_format='%d/%m/%Y', index=False)

    return train_df, validation_df, test_df

def prepare_data(path):
    
    # Eemove transaction (rows) and columns I don't need
    sales_df = get_sales(path)

    #Convert columns to the right data type
    sales_df = convert_data(sales_df)
    
    # Ordering rows by date
    sales_df = sales_df.sort_values(by="date")

    # get seasons db
    seasons_df = pd.read_csv(path+"\\stagioni.csv", index_col=False)
    seasons_df["inizio_vendite"] = pd.to_datetime(seasons_df["inizio_vendite"], format='%d/%m/%Y')
    seasons_df["fine_vendite"] = pd.to_datetime(seasons_df["fine_vendite"], format='%d/%m/%Y')

    # Create Model Input
    train_df, validation_df, test_df = create_model_input(sales_df, seasons_df, prediction_period=7)

if __name__ == "__main__":
    
    load_dotenv(find_dotenv())
    path = os.getenv('FOLDER_PATH')

    prepare_data(path)
