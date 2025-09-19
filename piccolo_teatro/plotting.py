import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Interactive helper to visualise random prediction intervals alongside the
# true target curves for quick qualitative inspection.
files = glob.glob(os.path.join("/home/carlo/Data/dati_piccolo_teatro2/xgb_log_ic_1/test", "*.csv"))
metrics = pd.read_csv("/home/carlo/Data/dati_piccolo_teatro2/xgb_log/test_metrics.csv")
while True:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes_flat = axes.flatten()

    selected = 0
    plotted = []
    print(len(axes_flat))
    while selected != len(axes_flat):
        f = random.choice(files)
        show_id = f.split("/")[-1].replace(".csv","")
        show_metrics = metrics[metrics["id"]==int(show_id)]
        features = pd.read_csv(f"/home/carlo/Data/dati_piccolo_teatro2/shows/test/{show_id}.csv")

        df = pd.read_csv(f)
        if 0.4*len(features)>30 and show_metrics["MAE_tot"].values[0]<0.5 and show_id not in plotted:
            plotted.append(show_id)
            print(show_id,len(features),0.4*len(features))
            predictions_index = list(range(len(features)-len(df),len(features)))
            ax = axes_flat[selected]
            selected +=1
            # Plot each series with a label (only the first plot needs to set the labels)
            ax.plot(features["percentage_bought"])
            ax.plot(predictions_index, df["mean"],  label="Mean prediction",          color="blue")
            ax.plot(predictions_index, df["lower"], label="Lower bound (5th %ile)",    color="green")
            ax.plot(predictions_index, df["upper"], label="Upper bound (95th %ile)",   color="red")
            ax.plot(predictions_index, df["target"],label="True values",               color="black")

            ax.set_title(show_id)
            ax.set_xlabel("n days")
            ax.set_ylabel("Percentage of tickets sold")

    # Extract one set of handles & labels and create a single legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize="small")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at top for the legend
    plt.show()

