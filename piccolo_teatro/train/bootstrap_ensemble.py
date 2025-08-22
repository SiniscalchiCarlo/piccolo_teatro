import os
import json
import glob
from dotenv import load_dotenv, find_dotenv
from piccolo_teatro import train
from piccolo_teatro.config import XGBConfig
from piccolo_teatro.models import load_ensemble, load_parameters, save_ensemble, save_parameters
from piccolo_teatro.train import keep_enabled_columns, separete_features_targets
from piccolo_teatro.train.metrics import TimeSeriesMetrics
from piccolo_teatro.trend_simulation import predict_trend
import xgboost as xgb
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

xgb_config =  XGBConfig()
load_dotenv(find_dotenv())
path = os.getenv("FOLDER_PATH")


def save_predicted_trends(index, target, mean, low, up, file_path):
    df = pd.DataFrame({
        'lower': low,
        'upper': up,
        'mean': mean,
        'target': target
    })
    df.index = index

    folder_path = os.path.dirname(file_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    df.to_csv(file_path, index=False)


def get_bootstrap_df(grouped, keys):
    """
    Select N random groups (with replacement) from a pandas GroupBy object
    and concatenate them into a single DataFrame.

    Parameters:
    - grouped: pandas GroupBy object (e.g., train_df.groupby('category'))
    - keys: list of group keys (e.g., list(grouped.groups.keys()))

    Returns:
    - random_keys: numpy array of selected keys
    - df_concat: single DataFrame concatenating all selected groups
    """
    # Sample N keys with replacement
    random_keys = np.random.choice(keys, size=len(keys), replace=True)
    random_groups = [grouped.get_group(key) for key in random_keys]
    # Concatenate into one DataFrame
    df_concat = pd.concat(random_groups, ignore_index=True)
    return df_concat

def bootstrap_models(train_df, validation_df, best_params, n_bootstraps=100):
    print("Training and ensemble of models:")

    num_boost_round = best_params.pop("n_estimators", 100)

    # First we have to separate the different shows
    df = pd.concat([train_df, validation_df], axis=0)

    grouped = train_df.groupby('show_id')
    keys = list(grouped.groups.keys())
    models = []
    for _ in range(n_bootstraps):
        print("%trained models:",_/n_bootstraps)
        df = get_bootstrap_df(grouped, keys)        
        df = keep_enabled_columns(df)

        train_X, train_Y = separete_features_targets(df, sort=False, shuffle=True)
        dtrain = xgb.DMatrix(train_X, label=train_Y)
        # 3. Train model
        model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            #feval=multi_step_mse_eval,
            maximize=False,
            #early_stopping_rounds=20,
            #verbose_eval=10
        )
        models.append(model)
    return models

def get_best_params(train_df, n_splits):
    groups = train_df["show_id"].copy()
    train_df = keep_enabled_columns(train_df)
    train_X, train_Y = separete_features_targets(train_df, sort=False, shuffle=True)

    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist',
        verbosity=0
    )

    cv = GroupKFold(n_splits=n_splits)

    stopper = DeltaYStopper(delta=0.001, n_best=5)
    opt = BayesSearchCV(
        estimator=xgb,
        search_spaces=xgb_config.param_space,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_iter=100,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )


    opt.fit(train_X, train_Y, groups=groups,callback=[stopper])
    best_params = opt.best_params_
    return best_params

def test_xgb(shows_folder, models, lower_q, upper_q, offset=0.4, plot=False, folder=None):
    parquet_files = glob.glob(os.path.join(shows_folder, "*.gzip"))
    saved_trends_folder = os.path.dirname(folder)
    already_saved = glob.glob(os.path.join(saved_trends_folder, "*.csv"))
    done_ids = []
    for f in already_saved:
        done_ids.append(f.split("/")[-1].replace(".csv",""))

    # Iterating all the the test shows
    targets = []
    n_show = len(parquet_files) 
    i_show = 0

    for file_name in parquet_files:
        i_show +=1
        show_id = file_name.split("/")[-1].replace(".gzip","")
        if show_id in done_ids:
            continue
        print("Predicting shows:",i_show/n_show, i_show, n_show)

        show_df = pd.read_parquet(file_name)

        # Calculating target df which is the real trend line we want to predict
        target = show_df[['date', 'percentage_bought']].copy()
        target['date'] = pd.to_datetime(target['date'], format='%d/%m/%Y')
        target.set_index('date', inplace=True)


        # getting only the first "offset" part of the dataset so we can predict the rest of the trend line
        known_df = show_df.head(int(len(show_df)* offset)).copy()

        # Predicting the trend for each model in the ensemble
        boot_preds = []
        m = 0
        for model in models:
            m+=1
            print("n model", m)
            predicted_trend = predict_trend(known_df, model, show_df["last_date"][0])
            boot_preds.append(predicted_trend["predictions"].values)
            index = predicted_trend.index

        targets.append(target["percentage_bought"].loc[index].values)
        lower = np.percentile(boot_preds, lower_q*100, axis=0)
        upper = np.percentile(boot_preds, upper_q*100, axis=0)
        mean  = np.mean(boot_preds, axis=0)

        if folder!=None:
            save_predicted_trends(index, target["percentage_bought"].loc[index].values, mean, lower, upper, folder+f"{show_id}.csv")


        if plot:
            plt.plot(target, color="black")
            plt.plot(index,lower, color="orange")
            plt.plot(index,mean, color="red")
            plt.plot(index,upper, color="blue")
            plt.plot(index, target["percentage_bought"].loc[index], color="purple")
            plt.show()

if __name__ == "__main__":
    train_df = pd.read_parquet(path + "/train_trend.gzip")
    validation_df = pd.read_parquet(path + "/validation_trend.gzip")

    best_params = load_parameters(xgb_config.file_name)
    ensemble = load_ensemble(xgb_config.file_name)

    if best_params == None:
        best_params = get_best_params(train_df, 5)
        save_parameters(xgb_config.file_name, best_params)

    if ensemble == None:
        ensemble = bootstrap_models(train_df, validation_df, best_params, 70)
        save_ensemble(xgb_config.file_name, ensemble)

    
    lower_q = (1-xgb_config.ic_dim)/2
    upper_q = 1-lower_q 
 
    # Creating a folder for storing predictions and metrics about the model
    save_folder = path+f"/{xgb_config.file_name}/"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
        os.makedirs(save_folder+"/train/")
        os.makedirs(save_folder+"/validation/")
        os.makedirs(save_folder+"/test/")


    for set_name in ["test", "validation", "train"]:
        # Getting predictions for test shows
        shows_folder = os.path.join(path, "shows", set_name)
        test_xgb(shows_folder, ensemble, lower_q, upper_q, 
                 plot=False, folder=save_folder+f"/{set_name}/")

        # Evaluating Metrics    
        ts_metrics = TimeSeriesMetrics(save_folder+f"/{set_name}/", lower_q, upper_q)
        res = ts_metrics.evaluate([5,10,15,20,25,30], save_folder, set_name)
        print(f"\nMetrics summart of {set_name} set:")
        print(res)
