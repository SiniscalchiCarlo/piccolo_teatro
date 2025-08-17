import os
import pickle
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from piccolo_teatro.config import TrainConfig, config_recap
from xgboost import XGBRegressor
from . import keep_enabled_columns, separete_features_targets, abs_error

train_conf = TrainConfig()
def train_xgb(train_X, train_Y):
    model = XGBRegressor(n_estimators=train_conf.parameters["n_estimators"], 
                         learning_rate=train_conf.parameters["learning_rate"], 
                         objective=train_conf.parameters["objective"])
    model.fit(train_X, train_Y)
    return model
