import pandas as pd
from config import FeatEngConf
import os
from dotenv import load_dotenv, find_dotenv
from utils import one_hot_encode, add_cumulative_sum, add_moving_avarages, add_shifted_values, add_targets
from datetime import datetime




def create_model_input(cleaned_sales, season_df, prediction_periods, train_dim = 0.6, validation_dim = 0.2):
    train_df = pd.DataFrame()
    validation_df = pd.DataFrame()
    test_df = pd.DataFrame()

    performances_db = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
    performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"] = performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"].astype(int)
    


    groups = cleaned_sales.groupby('performance_id')
    

    i=0
    counter=0
    len_=len(groups)
    tr=False
    val=False
    for performance_id, group in groups:
        #print(group)
        i+=1
        counter+=1
        print(i/len_)
        # Get start and end season date
        group_season = group["season_id"].iloc[0]
        season_row = season_df[(season_df["season_id"]) == group_season]
        start_date = season_row["inizio_vendite"].iloc[0]
        end_date = season_row["fine_vendite"].iloc[0]

        group = group.sort_values(by="date")

        # Rimuovo features che non mi servono
        group = group.drop(columns=["individuali_gruppi","online_offline"])

        # Sommo giorno i dati delle vendite
        group = group.groupby(['date', 'season_id', 'performance_id', 'show_id'], as_index=False).sum()
        
        # ADDING FEATURES:
        
        
        # Calcolo la cumulata del guadagno e del numero di biglietti venduti
        group = add_cumulative_sum(group, column_names=["gain", "tickets"])
        group["avg_ticket_price"] = group["gain_cum_sum"]/group["tickets_cum_sum"]
        
        # Add informations about the performance, is in another db
        group = add_performance_info(group,performances_db,performance_id)  

        if not group.empty:
            performance_date = group["performance_date"].iloc[0]
            # Aggiungo i dati dei giorni mancanti (giorni senza vendite), li riempio mettendo l'ultimo valore noto
            date_range = pd.date_range(start=group["date"].min(), end=performance_date)
            group = group.set_index("date").reindex(date_range, method="ffill")
            group["date"] = group.index

        
            # Distanza della transazione dall'inizio e dalla fine della stagione
            group["start_sales_distance"] = (group["date"]-start_date).dt.days.abs()
            group["end_season_distance"] = (group["date"]-end_date).dt.days.abs()
            group["sales_duration"] = (group["date"].max()-start_date).days
            group["end_sales_distance"] = (group["date"].max()-group["date"]).dt.days
            group["percentage_sales_day"] = group["start_sales_distance"]/(group["date"].max()-start_date).days

        
        
            # Numero bigliettirimanenti per raggiungere capienza massima
            group["remaining_tickets"] = group["performance_capacity"]-group["tickets_cum_sum"]
            group["percentage_bought"] = group["tickets_cum_sum"]/group["performance_capacity"]

            # Aggiungo medie mobili con differenti periodi
            group = add_moving_avarages(group, ["gain_cum_sum", "tickets_cum_sum", "percentage_bought"], [2,4,6,8,10,15,20,30])
            
            # Aggiungo valori shiftati
            group = add_shifted_values(group, ["gain_cum_sum", "tickets_cum_sum", "percentage_bought"], [2,4,6,8,10,15,20,30])

            # Aggiungo i possibili target da prevedere:
            group = add_targets(group, targets_dict)
        
        if len(group)>30:
            group.to_csv(path+f"\\performances\\{performance_id}.csv", date_format='%d/%m/%Y', index=False)
            
            if not tr:
                train_df = pd.concat([train_df, group], ignore_index=True)
                if counter/len_>train_dim:
                    counter=0
                    tr=True

            elif not val:
                validation_df = pd.concat([validation_df, group], ignore_index=True)
                if counter/len_>validation_dim:
                    counter=0
                    val=True

            elif tr and val:
                test_df = pd.concat([test_df, group], ignore_index=True)

    train_df.to_csv(path+f"\\train_trend.csv", date_format='%d/%m/%Y', index=False)
    validation_df.to_csv(path+f"\\validation_trend.csv", date_format='%d/%m/%Y', index=False)
    test_df.to_csv(path+f"\\test_trend.csv", date_format='%d/%m/%Y', index=False)

    return train_df, validation_df, test_df
    
def convert_data(df):
    df["gain"] = df["gain"].str.replace(",",".")
    df = df.astype({
        "individuali_gruppi": str,
        "online_offline": str,
        "gain": float,
        "tickets": int,
        "season_id": int,
        "performance_id": int,
    })
    df["date"] = pd.to_datetime(df["date"], format='%d/%m/%Y')
    return df

def get_sales(path: str):
    print(path)
    sales_db = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv", index_col=False)

    # Some columns have a space before the name, to remove it:
    sales_db = sales_db.rename(columns=lambda x: x.lstrip())

    sales_db = sales_db.rename(columns={
        "D_SALES_LIST_SALES_Individuali_Gruppi": "individuali_gruppi",
        "D_SALES_LIST_SALES_Tipologia_canale": "online_offline",
        "D_SALES_LIST_SALES_TOTAL_CURRENT_AMT_ITX": "gain",
        "D_SALES_LIST_SALES_CURRENT_QUANTITY": "tickets",
        "D_SALES_LIST_SALES_REFERENCE_DATE": "date",
        "D_SALES_LIST_SALES_T_SEASON_ID": "season_id",
        "D_SALES_LIST_SALES_T_PRODUCT_ID": "show_id",
        "D_SALES_LIST_SALES_T_PERFORMANCE_ID": "performance_id",
    })

    colonne_da_mantenere = ["individuali_gruppi",
                            "online_offline",
                            "gain",
                            "tickets",
                            "date",
                            "season_id",
                            "show_id",
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
    sales_db = sales_db[~sales_db['D_SALES_LIST_SALES_SEASON'].isin(
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

def add_performance_info(df, performances_db, performance_id):
    performance_row = performances_db[performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"]==performance_id]

    performance_state = performance_row["D_CONFIG_PROD_LIST_PERFORMANCE_STATE"].iloc[0]
    performance_type = performance_row["D_CONFIG_PROD_LIST_Tipologia_spettacolo"].iloc[0]
    performance_space = performance_row["D_CONFIG_PROD_LIST_SPACE"].iloc[0]
    performance_capacity = performance_row["D_CONFIG_PROD_LIST_PERFORMANCE_QUOTA"].iloc[0]
    performance_date_info = performance_row["D_CONFIG_PROD_LIST_PRODUCT_DATE_TIME"].iloc[0]
    preformance_season = performance_row["D_CONFIG_PROD_LIST_SEASON"].iloc[0]
    show_id = performance_row["D_CONFIG_PROD_LIST_T_PRODUCT_ID"].iloc[0]

    # Get all the performances of the same show
    performances_same_show = performances_db[performances_db["D_CONFIG_PROD_LIST_T_PRODUCT_ID"]==show_id].copy()

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
    performance_date = get_day_month(performance_date_info, year1, year2)
    performance_date = datetime.strptime(performance_date, "%d/%m/%Y")
    performances_same_show.loc[:, 'performance_date'] = performances_same_show['D_CONFIG_PROD_LIST_PRODUCT_DATE_TIME'].apply(lambda x: get_day_month(x, year1, year2))
    performances_same_show["performance_date"] = pd.to_datetime(performances_same_show["performance_date"], format='%d/%m/%Y')
    performances_same_show.sort_values(by='performance_date', inplace=True)
    performances_same_show = performances_same_show.reset_index(drop=True)

    performance_number = performances_same_show[performances_same_show["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"]==performance_id].index[0]

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
        df["performance_capacity"] = performance_capacity
        df["performance_day"] = performance_date_info[0]
        df["performance_hour"] = performance_date_info[2].split(":")[0]
        df["num_performances"] = len(performances_same_show)
        df["performance_number"] = performance_number
        df["performance_date"] = performance_date
        df = one_hot_encode(df, encoding_dict)
        return df
    else:
        return pd.DataFrame()

def prepare_data(path):
    print("path",path)
    # Remove transaction (rows) and columns I don't need
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
    train_df, validation_df, test_df = create_model_input(sales_df, seasons_df, prediction_periods=[1,5])

if __name__ == "__main__":
    
    load_dotenv(find_dotenv())
    path = os.environ.get("FOLDER_PATH")
    feat_eng_conf = FeatEngConf()
    encoding_dict = feat_eng_conf.encoding_dict
    targets_dict = feat_eng_conf.targets_dict
    prepare_data(path)


#git stash push
#git stash pop