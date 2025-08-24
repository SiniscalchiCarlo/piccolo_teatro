import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def plot_one(files):
    for f in files:
        f = random.choice(files)
        show_id = f.split("/")[-1].replace(".csv","")
        print(show_id)
        df = pd.read_csv(f)
        if 0.4*len(df)>45:
            plt.plot(df["mean"],  label="Mean prediction",          color="blue")
            plt.plot(df["lower"], label="Lower bound (5th %ile)",    color="green")
            plt.plot(df["upper"], label="Upper bound (95th %ile)",   color="red")
            plt.plot(df["target"],label="True values",               color="black")

            plt.show()

def plot_grid(files):
    while True:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes_flat = axes.flatten()
       
        selected = 0
        print(len(axes_flat))
        while selected != len(axes_flat):
            f = random.choice(files)
            show_id = f.split("/")[-1].replace(".csv","")
            features = pd.read_csv(f"/home/carlo/Data/dati_piccolo_teatro2/shows/test/{show_id}.csv")

            df = pd.read_csv(f)
            if 0.4*len(df)>45:
                predictions_index = list(range(len(features)-len(df),len(features)))
                ax = axes_flat[selected]
                selected +=1
                # Plot each series with a label (only the first plot needs to set the labels)
                # ax.plot(features["percentage_bought"][:len(features)-len(df)+n_predictions])
                # ax.plot(predictions_index[:n_predictions], 
                #         df["mean"][:n_predictions],  label="Mean prediction",          color="blue")
                # ax.plot(predictions_index[:n_predictions], 
                #         df["lower"][:n_predictions], label="Lower bound (5th %ile)",    color="green")
                # ax.plot(predictions_index[:n_predictions], 
                #         df["upper"][:n_predictions], label="Upper bound (95th %ile)",   color="red")
                # ax.plot(predictions_index[:n_predictions], 
                #         df["target"][:n_predictions],label="True values",               color="black")
                ax.plot(df["mean"],  label="Mean prediction",          color="blue")
                ax.plot(df["lower"], label="Lower bound (5th %ile)",    color="green")
                ax.plot(df["upper"], label="Upper bound (95th %ile)",   color="red")
                ax.plot(df["target"],label="True values",               color="black")

                ax.set_title(show_id)
                ax.set_xlabel("n days")
                ax.set_ylabel("Percentage of tickets sold")

        # Extract one set of handles & labels and create a single legend
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize="small")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at top for the legend
        plt.show()


files = glob.glob(os.path.join("/home/carlo/Data/dati_piccolo_teatro2/xgb_log/test", "*.csv"))
plot_one(files)
