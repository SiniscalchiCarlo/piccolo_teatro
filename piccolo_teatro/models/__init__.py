import pickle
def get_trend_model(name:str):
    with open(f"piccolo_teatro/models/{name}.pkl", 'rb') as file:
        model = pickle.load(file)
    return model