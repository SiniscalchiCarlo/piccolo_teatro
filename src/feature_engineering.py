import pandas as pd

def get_sales(path: str):
    sales_db = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv")

    seasons_to_remove = ["Stagione 2014/15",
                         "Stagione 2019/20",
                         "Stagione 2020/21",
                         "Stagione 2021/22"]
                         
    operations_to_remove = [
        "PRODUCT_COMPOSITION",
        "SINGLE_ENTRY",
    ]

    colonne_da_mantenere = ["D_SALES_LIST_SALES_Individuali_Gruppi",
                            "D_SALES_LIST_SALES_Tipologia_canale",
                            "D_SALES_LIST_SALES_TOTAL_CURRENT_AMT_ITX",
                            "D_SALES_LIST_SALES_CURRENT_QUANTITY",
                            "D_SALES_LIST_SALES_REFERENCE_DATE",
                            "D_SALES_LIST_SALES_T_AUDIENCE_SUB_CAT_ID",
                            "D_SALES_LIST_SALES_T_PERFORMANCE_ID",
                            "D_SALES_LIST_SALES_T_SEASON_ID",
                            ]
    
    # some columns have a space before the name, to remove it:
    sales_db = sales_db.rename(columns=lambda x: x.lstrip())

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

    #converting id to int

    sales_db['D_SALES_LIST_SALES_T_PERFORMANCE_ID'] = sales_db['D_SALES_LIST_SALES_T_PERFORMANCE_ID'].astype(int)
    return sales_db


def group_by_performance(sales_db):
    # Utilizzo di groupby per creare sottotabelle
    for id_, group in sales_db.groupby('D_SALES_LIST_SALES_T_PERFORMANCE_ID'):
        print(f"Sottotabella per ID: {id_}")
        print(group)
        print("\n")



path = "C:\\Users\\te7carsinisc\\Downloads\\dati_piccolo_teatro"
sales_db = get_sales(path)
group_by_performance(sales_db)
