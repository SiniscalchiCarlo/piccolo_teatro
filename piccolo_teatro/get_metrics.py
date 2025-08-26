
from piccolo_teatro.train.metrics import TimeSeriesMetrics

folder = "/home/carlo/Data/dati_piccolo_teatro2/xgb_log_ic_1"
for set_name in ["test", "train_validation"]:
    ts_metrics = TimeSeriesMetrics(folder+f"/{set_name}/", 0.5, 0.95)
    res = ts_metrics.evaluate([5,10,15,20,25,30], folder, set_name)

