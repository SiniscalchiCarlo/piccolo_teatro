import os
import glob
import numpy as np
import pandas as pd

class TimeSeriesMetrics:
    """
    Class to evaluate forecast trends against true trends using various metrics:
      - MSE
      - MAE
      - PICP (Prediction Interval Coverage Probability)
      - DTW Shape-Based Distance
      - Shift-Invariant Shape-Based Distance
      - Interval Score

    Inputs:
      ids_list            : list of identifiers for each series (e.g., show IDs)
      y_true_list         : list of array-like, true trends
      y_pred_list         : list of array-like, mean predictions
      lower_bounds_list   : list of array-like, lower quantile bounds
      upper_bounds_list   : list of array-like, upper quantile bounds
      lower_q, upper_q    : floats, quantile levels (e.g. 0.05, 0.95)
    """
    def __init__(self, folder, lower_q, upper_q):
        self.folder = folder
        self.lower_q = lower_q
        self.upper_q = upper_q
        # alpha for interval score
        self.alpha = 1.0 - (upper_q - lower_q)

    def mse(self, y_true, y_pred):
        """Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)

    def mae(self, y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))

    def picp(self, y_true, lower, upper):
        """Prediction Interval Coverage Probability"""
        inside = (y_true >= lower) & (y_true <= upper)
        return np.mean(inside)

    def interval_score(self, y_true, lower, upper):
        """
        Compute interval score for given true values and prediction interval [lower, upper].
        """
        width = upper - lower
        below = lower - y_true
        above = y_true - upper
        score = width.copy()
        # penalty when true is below lower
        mask_below = y_true < lower
        score[mask_below] += (2.0 / self.alpha) * below[mask_below]
        # penalty when true is above upper
        mask_above = y_true > upper
        score[mask_above] += (2.0 / self.alpha) * above[mask_above]
        return np.mean(score)

    def evaluate(self, periods=[5,10,15,20,30], save_folder=None, name=None):
        """
        Evaluate all series and return a DataFrame with metrics and ids.
        Optionally save metrics to a single CSV at save_folder.

        Returns:
          pandas.DataFrame with columns ['id', 'MSE', 'MAE', 'PICP', 'IS']
        """
        records = []
        show_files = glob.glob(os.path.join(self.folder, "*.csv"))
        
        for show_file in show_files:
            df = pd.read_csv(show_file)
            y_pred = df["mean"].values
            y_true = df["target"].values
            low = df["lower"].values
            up = df["upper"].values
            
            rec={'id': show_file.split("/")[-1].replace(".csv","")}
            for p in periods:
                if p<len(y_true):
                    rec = {
                        **rec,
                        f'MSE_{p}': self.mse(y_true[:p], y_pred[:p]),
                        f'MAE_{p}': self.mae(y_true[:p], y_pred[:p]),
                        f'PICP_{p}': self.picp(y_true[:p], low[:p], up[:p]),
                        f'IS_{p}': self.interval_score(y_true[:p], low[:p], up[:p])
                    }
                else:
                    rec = {
                        **rec,
                        f'MSE_{p}': np.nan,
                        f'MAE_{p}': np.nan,
                        f'PICP_{p}': np.nan,
                        f'IS_{p}': np.nan,
                    }

            rec = {
                **rec,
                f'MSE_tot': self.mse(y_true, y_pred),
                f'MAE_tot': self.mae(y_true, y_pred),
                f'PICP_tot': self.picp(y_true, low, up),
                f'IS_tot': self.interval_score(y_true, low, up),
                "len": len(y_true),
            }
            records.append(rec)
        df = pd.DataFrame.from_records(records)
        if save_folder:
            os.makedirs(os.path.dirname(save_folder), exist_ok=True)
        df.to_csv(save_folder+f"/{name}_metrics.csv", index=False)

        df = df[df["len"]>45]
        df = df.drop(columns=["id", "len"]).mean().to_frame(name='mean').T 
        df.to_csv(save_folder+f"/{name}_summary.csv", index=False)
        return df

