import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

folder = "/home/carlo/Data/dati_piccolo_teatro2/xgb_log_ic_1/train_validation"
files = glob.glob(os.path.join(folder, "*.csv"))

for file in files:
    show_id = file.split("/")[-1].replace(".csv","")

    df = pd.read_csv(file)
    new_df = df.copy()

    k = df["mean"]*(-np.sqrt(70)+1)
    new_df["lower"] = new_df["lower"]*np.sqrt(70)+k
    new_df["upper"] = new_df["upper"]*np.sqrt(70)+k
    new_df.to_csv(folder+f"/{show_id}.csv")



