import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import xgboost as xgb
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper
from xgboost import XGBRegressor

from piccolo_teatro import ts_engine
from piccolo_teatro.config import XGBConfig
from piccolo_teatro.models import (
    load_ensemble,
    load_parameters,
    save_ensemble,
    save_parameters,
)
from piccolo_teatro.train import keep_enabled_columns, separete_features_targets
from piccolo_teatro.train.metrics import TimeSeriesMetrics
from piccolo_teatro.trend_simulation import predict_trend

xgb_config = XGBConfig()
load_dotenv(find_dotenv())
path = os.getenv("FOLDER_PATH")


def save_predicted_trends(index, target, mean, low, up, file_path):
    df = pd.DataFrame({"lower": low, "upper": up, "mean": mean, "target": target})
    df.index = index

    folder_path = os.path.dirname(file_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    df.to_csv(file_path, index=False)


def get_bootstrap_df(shows_df):
    '''
    show_df: list of N pandas df
    Returns df_concat concatentation of N randoomly choosed dfs from show_df
    '''
    sampled_dfs = random.choices(shows_df, k=len(shows_df))
    df_concat = pd.concat(sampled_dfs, ignore_index=True)
    return df_concat


def bootstrap_models(shows_df, best_params, n_bootstraps=100):
    '''
    Function used to train a ensemble on n_bootstraps XGBoost models
    Returns a list with the trained models.
    '''
    print("Training and ensemble of models:")
    print("best parameters:", best_params)

    num_boost_round = best_params["n_estimators"]
    del best_params["n_estimators"]

    models = []
    for _ in range(n_bootstraps):
        print("%trained models:", _ / n_bootstraps)
        df = get_bootstrap_df(shows_df)
        df = keep_enabled_columns(df)

        train_X, train_Y = separete_features_targets(df, sort=False, shuffle=False)
        dtrain = xgb.DMatrix(train_X, label=train_Y)

        # 3. Train model
        model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            maximize=False,
        )

        models.append(model)
    return models


def get_best_params(train_shows, n_splits):
    '''
    Performs BayesSearch and return the optimal paramenters.
    '''
    train_df = pd.concat(train_shows, axis=0, ignore_index=True)
    groups = train_df["show_id"].copy()
    train_df = keep_enabled_columns(train_df)
    train_X, train_Y = separete_features_targets(train_df, sort=False, shuffle=True)

    xgb = XGBRegressor(
        objective="reg:squarederror", random_state=42, tree_method="hist", verbosity=0
    )

    cv = GroupKFold(n_splits=n_splits)

    stopper = DeltaYStopper(delta=0.001, n_best=5)
    opt = BayesSearchCV(
        estimator=xgb,
        search_spaces=xgb_config.param_space,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_iter=100,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    opt.fit(train_X, train_Y, groups=groups, callback=[stopper])
    best_params = opt.best_params_
    return best_params


def test_xgb(test_shows, models, lower_q, upper_q, folder, offset=0.4, plot=False):
    '''
    Function used to test the performances of the ensemble of models.
    Performs for each show in the test_shows folder iterative forecasting to predict 
    the sales trend. For each predicted day the mean prediction and the confidence interval
    for the mean of are computed given all the predictions of the ensemble.
    The predicted mean, lower bound and upper bound are saved to later compute metrics to
    evaluate the model ensemble.
    '''
    # Getting already predicted and saved shows
    saved_trends_folder = os.path.dirname(folder)
    already_saved = glob.glob(os.path.join(saved_trends_folder, "*.csv"))
    done_ids = []
    for f in already_saved:
        done_ids.append(f.split("/")[-1].replace(".csv", ""))

    targets = []
    n_show = len(test_shows)
    i_show = 0
    for show_df in test_shows:
        i_show += 1
        show_id = show_df["show_id"].values[0]

        # Skipping already predicted shows/too small
        if str(show_id) in done_ids or int(len(show_df) * offset) < 30:
            print(show_id, "already_done or too small")
            continue
        print("Predicting shows:", i_show / n_show, i_show, n_show)

        # Calculating target df which is the real trend line we want to predict
        target = show_df[["date", "percentage_bought"]].copy()
        target["date"] = pd.to_datetime(target["date"], format="%d/%m/%Y")
        target.set_index("date", inplace=True)

        # getting only the first "offset" part of the dataset so we can predict the rest of the trend line
        known_df = show_df.head(int(len(show_df) * offset)).copy()

        # Predicting the trend for each model in the ensemble
        boot_preds = []
        for model in models:
            predicted_trend = predict_trend(
                known_df, ts_engine.target, model, show_df["last_date"][0]
            )
            if ts_engine.target == "percentage_bought_log1p":
                boot_preds.append(np.expm1(predicted_trend["predictions"].values))
            else:
                boot_preds.append(predicted_trend["predictions"].values)
            index = predicted_trend.index
        targets.append(target["percentage_bought"].loc[index].values)
        if ts_engine.quantile_regression is False:

            boot_preds = np.stack(boot_preds, axis=0)
            # numero di bootstrap
            B = boot_preds.shape[0]

            # stima della media predetta
            mean = np.mean(boot_preds, axis=0)
            # errore standard della media
            se = np.std(boot_preds, axis=0, ddof=1)
            # livello di confidenza (es. lower_q=0.025 e upper_q=0.975 → confidenza 95%)
            alpha = 1 - (upper_q - lower_q)
            df = B - 1
            # quantile t critico
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            # calcolo dei limiti dell’intervallo di confidenza
            lower = mean - t_crit * se
            upper = mean + t_crit * se
        else:
            lower = boot_preds[0]
            mean = boot_preds[1]
            upper = boot_preds[2]

        save_predicted_trends(
            index,
            target["percentage_bought"].loc[index].values,
            mean,
            lower,
            upper,
            folder + f"{show_id}.csv",
        )

        if plot:
            plt.plot(target, color="black")
            plt.plot(index, lower, color="orange")
            plt.plot(index, mean, color="red")
            plt.plot(index, upper, color="blue")
            plt.plot(index, target["percentage_bought"].loc[index], color="purple")
            plt.show()


if __name__ == "__main__":

    ts_engine.config_recap()
    
    # Reading input data and creating list of shows divided in train, validation and test
    train_files = glob.glob(os.path.join(path + f"/shows/train", "*.gzip"))
    validation_files = glob.glob(os.path.join(path + f"/shows/validation", "*.gzip"))
    test_files = glob.glob(os.path.join(path + f"/shows/test", "*.gzip"))

    train_shows = []
    for file in train_files:
        train_shows.append(pd.read_parquet(file))

    validation_shows = []
    for file in validation_files:
        validation_shows.append(pd.read_parquet(file))

    test_shows = []
    for file in test_files:
        test_shows.append(pd.read_parquet(file))

    lower_q = (1 - xgb_config.ic_dim) / 2
    upper_q = 1 - lower_q
    
    # Loading best parameters and trained models for the configured problem
    # if they where already calculated.
    best_params = load_parameters(ts_engine.pb_name)
    ensemble = load_ensemble(ts_engine.pb_name)

    if best_params == None:
        best_params = get_best_params(train_shows, 5)
        save_parameters(ts_engine.pb_name, best_params)

    if ensemble == None:
        ensemble = bootstrap_models(train_shows + validation_shows, best_params, 70)
        save_ensemble(ts_engine.pb_name, ensemble)


    # Creating a folder for storing predictions and metrics about the model
    save_folder = path + f"/{ts_engine.pb_name}/"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
        os.makedirs(save_folder + "/train/")
        os.makedirs(save_folder + "/validation/")
        os.makedirs(save_folder + "/test/")

    # Computing metrics for test and train+validation sets
    for set_name in ["test", "train_validation"]:
        if set_name == "test":
            shows = test_shows
        if set_name == "train_validation":
            shows = train_shows + validation_shows

        # Getting predictions for test shows
        test_xgb(
            shows,
            ensemble,
            lower_q,
            upper_q,
            folder=save_folder + f"/{set_name}/",
            plot=False,
        )

        # Evaluating Metrics
        ts_metrics = TimeSeriesMetrics(save_folder + f"/{set_name}/", lower_q, upper_q)
        res = ts_metrics.evaluate([5, 10, 15, 20, 25, 30], save_folder, set_name)
        print(f"\nMetrics summart of {set_name} set:")
        print(res)
