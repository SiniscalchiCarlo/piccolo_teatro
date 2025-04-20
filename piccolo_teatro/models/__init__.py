import pickle
import os

def get_trend_model(name:str):
    print(os.getcwd())
    with open(f"{os.getcwd()}\piccolo_teatro\models\{name}.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

