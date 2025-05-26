import os
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv

from ..data_pipeline.ingestion import Sales, Products, Seasons
from ..data_pipeline.transformation import add_features
from ..train.utils import keep_enabled_columns, separete_features_targets, train_XGBRegressor, save_model, abs_error
from ..train.model_preparation import Features, ModelData

load_dotenv(find_dotenv())
path = os.environ.get("FOLDER_PATH")
model = pickle.load(open(r"C:\Users\39370\Desktop\piccolo_teatro\piccolo_teatro\models\XGB_trend2.pkl", "rb"))

def trend_prediction(sales: Sales, products: Products, seasons: Seasons, offset=None):
    
    # Check all transaction are from the same show
    if sales.df['show_id'].nunique() == 1:
        all_data = add_features(seasons=seasons,
                    products=products,
                    sales=sales,
                    fill_to_show_date=False)
    else:
        raise Exception("Sales transactions are not all from the same show")
    
    all_trend = all_data["percentage_bought"].copy()
    if offset != None:
        known_data = all_data.head(int(len(all_data) * offset)).copy()
    else:
        known_data = all_data.copy()

    known_trend = known_data["percentage_bought"].copy()
    last_day = known_data["last_date"].iloc[0]
    current_day = known_data["date"].iloc[-1]

    df = keep_enabled_columns(known_data.copy())
    X, Y = separete_features_targets(df, sort=False, shuffle=False)

    features = Features(df=X)

    model_input = X.iloc[[-1]]
    predictions = []
    predicitons_days = []
    
    input_row = features.df.iloc[[-1]]
    while(input_row["sales_duration"].iloc[0]-input_row["start_sales_distance"].iloc[0]>0):
        #print(current_day, last_day)
        
        current_day += pd.Timedelta(days=1)
        predicitons_days.append(current_day)

        input_row = features.df.iloc[[-1]]
        prediction = model.predict(input_row)[0]
        #print(input_row["percentage_sales_day"].iloc[0], input_row["percentage_bought"].iloc[0], input_row["percentage_bought_avg_2"].iloc[0])
        #print("PREDICTION", prediction)
        predictions.append(prediction)

         #2. UPDATING FEATURES
        features.update_features(prediction)

    prediction_df = pd.DataFrame({"predictions": predictions}, index=predicitons_days)
    return prediction_df, all_trend



if __name__ == "__main__":
    pd.set_option("display.max_columns", None)

    SALES = pd.read_csv(path+"\\D_SALES_LIST_SALES.csv", index_col=False)
    PRODUCTS = pd.read_csv(path+"\\D_CONFIG_PROD_LIST.csv", index_col=False)
    SEASONS = pd.read_csv(path+"\\stagioni.csv", index_col=False)
    
    test_df = pd.read_parquet(path+f"\\validation_trend.gzip")
    print(test_df)
    ids = test_df['show_id'].unique().tolist()
    ids = random.sample(ids, k=len(ids))
    
    for show_id in ids:
        # Ingesting and cleaning data
        sales = Sales(SALES)
        products = Products(PRODUCTS)
        seasons = Seasons(SEASONS)

        sales.get_same_show(show_id=show_id)

        sales.clean()
        products.clean()
        seasons.clean()

        trend_prediction(model, sales, products, seasons, offset=0.3)

