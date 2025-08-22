import os
import json
import pickle
import glob

current_dir = os.path.dirname(__file__)

def load_parameters(model_name: str) -> dict | None:
    """
    Load JSON parameters for a given model name.
    Returns the dict if exists, else None.
    """
    params_path = os.path.join(current_dir, f"{model_name}_parameters.json")
    if not os.path.isfile(params_path):
        return None
    with open(params_path, 'r') as f:
        return json.load(f)


def save_parameters(model_name: str, params: dict) -> str:
    """
    Save a dict of parameters to a JSON file named {model_name}_parameters.json.
    Returns the filepath.
    """
    params_path = os.path.join(current_dir, f"{model_name}_parameters.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    return params_path


def load_model(model_name: str) -> object | None:
    """
    Load a single pickled model from {model_name}.pkl.
    Returns the model object if exists, else None.
    """
    model_path = os.path.join(current_dir, f"{model_name}.pkl")
    if not os.path.isfile(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_model(model_name: str, model: object) -> str:
    """
    Save a single model object to {model_name}.pkl via pickle.
    Returns the filepath.
    """
    model_path = os.path.join(current_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path


def load_ensemble(model_name: str) -> list[object] | None:
    """
    Load all pickled models from the folder {model_name}_ensemble/.
    Returns a list of model objects if any, else None.
    """
    folder = os.path.join(current_dir, f"{model_name}_ensemble")
    if not os.path.isdir(folder):
        return None
    paths = glob.glob(os.path.join(folder, "*.pkl"))
    if not paths:
        return None
    models = []
    for path in sorted(paths):
        with open(path, "rb") as f:
            models.append(pickle.load(f))
    return models


def save_ensemble(model_name: str, models: list[object]) -> str:
    """
    Save a list of model objects into {model_name}_ensemble/ as numbered .pkl files.
    Returns the folder path.
    """
    folder = os.path.join(current_dir, f"{model_name}_ensemble")
    os.makedirs(folder, exist_ok=True)
    for idx, model in enumerate(models):
        path = os.path.join(folder, f"{model_name}_{idx:03d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return folder

