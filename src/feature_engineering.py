import pandas as pd
from utils.logger import setup_logger
logger = setup_logger(__name__, level="INFO")


def convert_data(sales_db):
    sales_db["gain"] = sales_db["gain"].str.replace(",",".")
    sales_db = sales_db.astype({
        "individuali_gruppi": str,
        "online_offline": str,
        "gain": float,
        "numero_biglietti": int,
        "season_id": int,
        "performance_id": int,
    })
    sales_db['data'] = pd.to_datetime(sales_db['data'], format='%Y-%m-%d')
    return sales_db

def one_hot_encode(df, column_names: list[str]):
    encoded_col_names = []
    for col_name in column_names:
        # Perform one hot encoding
        df_encoded = pd.get_dummies(df[col_name], prefix='category', dtype=int)

        # Collect new column names
        encoded_col_names.extend(df_encoded.columns.tolist())

        # Concatenate the new one hot encoded columns to the original DataFrame
        df = pd.concat([df, df_encoded], axis=1)
    df = df.drop(columns=column_names)
    return df, encoded_col_names

def add_cumulative_sum(df, column_names: list[str]):
    for col_name in column_names:
        df[col_name+"_cum_sum"] = df[col_name].cumsum()
    df = df.drop(columns=column_names)

    return df

def add_moving_avarages(df, column_names: list[str], periods: list[int]):
    for period in periods:
        for col_name in column_names:
            col_name = col_name+"_cum_sum"
            print(col_name+"_cum_sum")
            df[col_name+f"{period}d_avg"] = df[col_name].rolling(period).mean()
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
        "D_SALES_LIST_SALES_REFERENCE_DATE": "data",
        "D_SALES_LIST_SALES_T_SEASON_ID": "season_id",
        "D_SALES_LIST_SALES_T_PERFORMANCE_ID": "performance_id",
    })

    colonne_da_mantenere = ["individuali_gruppi",
                            "online_offline",
                            "gain",
                            "numero_biglietti",
                            "data",
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

    #converting each column to the right value
    sales_db['data'] = pd.to_datetime(sales_db['data'], format='%d/%m/%Y')
    return sales_db

def adding_performance_info(df, performance_id):
    performances_db = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
    performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"] = performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"].astype(int)
    print("ID",performance_id)
    print(performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"])
    performance_row = performances_db[performances_db["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"]==performance_id]
    print("PERFORMANCE_ROW", performance_row)
    performance_state = performance_row["D_CONFIG_PROD_LIST_PERFORMANCE_STATE"].iloc[0]
    performance_type = performance_row["D_CONFIG_PROD_LIST_Tipologia_spettacolo"].iloc[0]
    performance_space = performance_row["D_CONFIG_PROD_LIST_SPACE"].iloc[0]
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
        df["performance_type"] = performance_row["D_CONFIG_PROD_LIST_Tipologia_spettacolo"].iloc[0]
        return df
    else:
        return None

def create_dataset(cleaned_sales, db_start_end_season):
    # raggruppo le 
    start_date = None
    end_date = None
    for performance_id, group in cleaned_sales.groupby('performance_id'):

        #get start and end season date
        if start_date == None:
            group_season = group["season_id"].iloc[0]
            season_row = db_start_end_season[(db_start_end_season["season_id"]) == group_season]
            start_date = season_row["first_ticket_sold"].iloc[0]
            end_date = season_row["last_ticket_sold"].iloc[0]


        #PER ORA IPOTIZIAMO CHE LE VENDITE APRANO IL GIORNO DELLA PRIMA TRANSAZIONE
        #cacolo la distanza in giorni dal primo giorno d'acquisto e dall'ultimo
        group["start_season_distance"] = (group["data"]-start_date).dt.days
        group["end_season_distance"] = -(group["data"]-end_date).dt.days
        group = group.drop(columns=["data", 
                                    "season_id", 
                                    "performance_id"])
        group = adding_performance_info(group, performance_id)
        #faccio one hot encoding delle seguenti colonne:
        columns_to_encode = [
            "individuali_gruppi", #se chi ha acquistato è individuale o gruppo
            "online_offline", #se comprato offline o online
            "performance_type"
        ]
        group, encoded_col_names = one_hot_encode(group, column_names=columns_to_encode)
        columns_to_cum_sum = ["gain", #prezzi 
                             "numero_biglietti", #numero biglietti
                            ]
        
        # calcolo la somma cumulata di: 
        #   numero di persone che hanno comprato offline
        #   numero di gruppi che hanno comprato
        #   numero biglietti acquistati
        #   gain
        group = add_cumulative_sum(group, column_names=encoded_col_names+columns_to_cum_sum)

        group = group.rename(columns={
        "category_Gruppi_cum_sum:": "category_gruppi_cum_sum",
        "category_Individuali_cum_sum": "category_individuali_cum_sum",
        "category_Offline_cum_sum:": "category_offline_cum_sum",
        "category_Online_cum_sum": "category_online_cum_sum",
    })


        group = add_moving_avarages(group, ["gain"], [5,10,15,20,30,40,50])
        group["tomorrow_gain"] = group["gain_cum_sum"].shift(-1)
        group = group.iloc[:-1]
        #calcolo medie mobili
        print("\nFINAL DB:")
        print_unique_values(group)

        break


path = "C:\\Users\\te7carsinisc\\Downloads\\dati_piccolo_teatro"
create_clean_sales = True
create_sales_by_perfomance = True

performances_db = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)

if create_clean_sales:
    cleaned_sales = get_sales(path)
    cleaned_sales = convert_data(cleaned_sales)
    cleaned_sales.to_csv(path+"\\CLEANED_SALES.csv", index=False)
else:
    cleaned_sales = pd.read_csv(path+"\\CLEANED_SALES.csv", index_col=False)

    cleaned_sales = convert_data(cleaned_sales)

print("\nCLEANED DB:")
print_unique_values(cleaned_sales)


if create_sales_by_perfomance:
    #get the first and last ticket sold for each season
    db_start_end_season = cleaned_sales.groupby('season_id')['data'].agg(['min', 'max']).reset_index()
    db_start_end_season.columns = ['season_id', 'first_ticket_sold', 'last_ticket_sold']
    db_start_end_season['season_id'] = db_start_end_season['season_id'].astype(int)

    create_dataset(cleaned_sales, db_start_end_season)
