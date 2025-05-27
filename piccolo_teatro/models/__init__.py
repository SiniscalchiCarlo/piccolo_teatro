import pickle
import os
import importlib.resources as resources

def get_model(name:str):
    with open(f"{name}.pkl", "rb") as f:
        model = pickle.load(f)

    return model

