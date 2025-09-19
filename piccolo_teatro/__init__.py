from piccolo_teatro.config import TimeSeriesEngine

# Provide a shared TimeSeriesEngine instance configured with the default
# feature set so that other modules can import and reuse it directly.
ts_engine = TimeSeriesEngine()
