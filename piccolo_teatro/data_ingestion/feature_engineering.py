import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

from ..config import FeatEngConf
from ..utils import one_hot_encode, add_cumulative_sum, add_moving_avarages, add_shifted_values, add_targets

class FeatureEngineering:
    def __init__(self, SALES: pd.DataFrame, PERFORMANCES: pd.DataFrame, SEASONS: pd.DataFrame):
        load_dotenv(find_dotenv())
        self.path = os.environ.get("FOLDER_PATH")

        self.feat_eng_conf = FeatEngConf()
        self.encoding_dict = self.feat_eng_conf.encoding_dict
        self.targets_dict = self.feat_eng_conf.targets_dict

        self.SALES = SALES
        self.PERFORMANCES = PERFORMANCES
        self.SEASONS = SEASONS
        self.TRAIN = pd.DataFrame()
        self.VALIDATION = pd.DataFrame()
        self.TEST = pd.DataFrame()

    def get_season_dates(self, season_id: str):
        season_row = self.SEASONS[self.SEASONS["season_id"] == season_id]
        start_date = season_row["inizio_vendite"].iloc[0]
        end_date = season_row["fine_vendite"].iloc[0]
        return start_date, end_date

    def add_features(self, group):
        # Get start and end season date
        start_date, end_date =self.get_season_dates(season_id=group["season_id"].iloc[0])

        group = group.sort_values(by="date")

        # Rimuovo features che non mi servono
        group = group.drop(columns=["individuali_gruppi","online_offline"])

        # Somma giornaliera dei dati delle vendite
        group = group.groupby(['date', 'season_id', 'show_id'], as_index=False).sum()
        show_id = group["show_id"].iloc[0]
        
        # Calcolo la cumulata del guadagno e del numero di biglietti venduti
        group = add_cumulative_sum(group, column_names=["gain", "tickets"])
        group["avg_ticket_price"] = group["gain_cum_sum"]/group["tickets_cum_sum"]
        
        # Add informations about the performance, and checks the product is one of the one we are interested in 
        group = self.add_show_info(group, show_id)  
        if not group.empty:
            last_date = group["last_date"].iloc[0]
            # Aggiungo i dati dei giorni mancanti (giorni senza vendite), li riempio mettendo l'ultimo valore noto
            date_range = pd.date_range(start=group["date"].min(), end=last_date)
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
            group = add_targets(group, self.targets_dict)
        else:
            group = pd.DataFrame()

        return group

    def create_model_input(self, train_dim = 0.6, validation_dim = 0.2):
        groups = self.SALES.groupby('show_id')
        

        i=0
        counter=0
        len_=len(groups)
        tr=False
        val=False
        for show_id, group in groups:
            i+=1
            counter+=1
            print(i/len_)
            group = self.add_features(group)
            
            if len(group)>30:
                group.to_parquet(self.path+f"\\shows\\{show_id}.gzip", index=False)
                
                if not tr:
                    self.TRAIN = pd.concat([self.TRAIN, group], ignore_index=True)
                    if counter/len_>train_dim:
                        counter=0
                        tr=True

                elif not val:
                    self.VALIDATION = pd.concat([self.VALIDATION, group], ignore_index=True)
                    if counter/len_>validation_dim:
                        counter=0
                        val=True

                elif tr and val:
                    self.TEST = pd.concat([self.TEST, group], ignore_index=True)

        self.TRAIN.to_parquet(self.path+f"\\train_trend.gzip", index=False)
        self.VALIDATION.to_parquet(self.path+f"\\validation_trend.gzip", index=False)
        self.TEST.to_parquet(self.path+f"\\test_trend.gzip", index=False)
    

    def assign_types(self):
        self.SALES["gain"] = self.SALES["gain"].str.replace(",",".")
        self.SALES = self.SALES.astype({
            "individuali_gruppi": str,
            "online_offline": str,
            "gain": float,
            "tickets": int,
            "season_id": int,
            "performance_id": int,
        })
        self.SALES["date"] = pd.to_datetime(self.SALES["date"], format='%d/%m/%Y')

        self.SEASONS["inizio_vendite"] = pd.to_datetime(self.SEASONS["inizio_vendite"], format='%d/%m/%Y')
        self.SEASONS["fine_vendite"] = pd.to_datetime(self.SEASONS["fine_vendite"], format='%d/%m/%Y')

        self.PERFORMANCES["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"] = self.PERFORMANCES["D_CONFIG_PROD_LIST_T_PERFORMANCE_ID"].astype(int)

    def clean_sales(self):
        # Some columns have a space before the name, to remove it:
        self.SALES = self.SALES.rename(columns=lambda x: x.lstrip())

        self.SALES = self.SALES.rename(columns={
            "D_SALES_LIST_SALES_Individuali_Gruppi": "individuali_gruppi",
            "D_SALES_LIST_SALES_Tipologia_canale": "online_offline",
            "D_SALES_LIST_SALES_TOTAL_CURRENT_AMT_ITX": "gain",
            "D_SALES_LIST_SALES_CURRENT_QUANTITY": "tickets",
            "D_SALES_LIST_SALES_REFERENCE_DATE": "date",
            "D_SALES_LIST_SALES_T_PRODUCT_ID": "show_id",
            "D_SALES_LIST_SALES_T_SEASON_ID": "season_id",
            "D_SALES_LIST_SALES_T_PERFORMANCE_ID": "performance_id",
        })

        cols_too_keep = ["individuali_gruppi",
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
        self.SALES = self.SALES[~self.SALES['D_SALES_LIST_SALES_SEASON'].isin(
            seasons_to_remove)]

        # removing sales of products that are not shows
        # SALES = SALES[SALES['D_SALES_LIST_SALES_T_OPERATION_KIND'] == "SIMPLE_PRODUCT"]
        self.SALES = self.SALES[self.SALES['D_SALES_LIST_SALES_T_OPERATION_KIND'].isin(
            operations_to_remove)]
        
        # considering only sales operations
        self.SALES = self.SALES[self.SALES['D_SALES_LIST_SALES_OPERATION_TYPE'] == "Venduti"]

        # keep only the coulmns needed
        self.SALES = self.SALES[cols_too_keep]


    def add_show_info(self, group, show_id):
        performances_same_show = self.PERFORMANCES[self.PERFORMANCES["D_CONFIG_PROD_LIST_T_PRODUCT_ID"]==show_id]
        performance_state = performances_same_show["D_CONFIG_PROD_LIST_PERFORMANCE_STATE"].iloc[0]
        performance_type = performances_same_show["D_CONFIG_PROD_LIST_Tipologia_spettacolo"].iloc[0]
        preformance_season = performances_same_show["D_CONFIG_PROD_LIST_SEASON"].iloc[0]
        performance_space = performances_same_show["D_CONFIG_PROD_LIST_SPACE"].iloc[0]
        performance_capacity = sum(performances_same_show["D_CONFIG_PROD_LIST_PERFORMANCE_QUOTA"])
        


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
        performances_same_show.loc[:, 'performance_date'] = performances_same_show['D_CONFIG_PROD_LIST_PRODUCT_DATE_TIME'].apply(lambda x: get_day_month(x, year1, year2))
        performances_same_show.loc[:, "performance_date"] = pd.to_datetime(performances_same_show["performance_date"], format='%d/%m/%Y')


        last_date = performances_same_show["performance_date"].max()

        # Get all the performances of the same show
        performances_same_show = self.PERFORMANCES[self.PERFORMANCES["D_CONFIG_PROD_LIST_T_PRODUCT_ID"]==show_id].copy()

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
            group["performance_type"] = performance_type
            group["performance_capacity"] = performance_capacity
            group["num_performances"] = len(performances_same_show)
            group["last_date"] = last_date
            group = one_hot_encode(group, self.encoding_dict)
            return group
        else:
            return pd.DataFrame()

    def ingest_sales(self):
        # Remove transaction (rows) and columns I don't need
        self.clean_sales()

        #Convert columns to the right data type
        self.assign_types()


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    load_dotenv(find_dotenv())
    path = os.environ.get("FOLDER_PATH")
    SALES = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv", index_col=False)
    PERFORMANCES = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
    SEASONS = pd.read_csv(path+"\\stagioni.csv", index_col=False)
    feat_eng = FeatureEngineering(SALES, PERFORMANCES, SEASONS)
    feat_eng.ingest_sales()
    feat_eng.create_model_input()
