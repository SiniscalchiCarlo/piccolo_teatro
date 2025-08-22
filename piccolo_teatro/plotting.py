import random
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

files = glob.glob(os.path.join("/home/carlo/Data/dati_piccolo_teatro2/shows/test", "*.csv"))

while(True):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes_flat = axes.flatten()

    for ax in axes_flat:
        # Select a random file and load its data
        f = random.choice(files)
        df = pd.read_csv(f)

        # Plot the 'percentage_bought' column
        ax.plot(df["percentage_bought"])
        ax.set_xlabel("n days")
        ax.set_ylabel("percentage bought tickets")

    plt.tight_layout()
    plt.show()

