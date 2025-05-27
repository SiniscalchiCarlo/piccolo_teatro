import os
import pickle

def get_model(name):
    # Use __file__ to get the directory of the current script
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, f"{name}.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

