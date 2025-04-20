import pickle
import os
import importlib.resources as resources
def get_trend_model(name:str):

    with resources.files(__package__).joinpath(f"{name}.pkl").open("rb") as file:
        model = pickle.load(file)
    return model

