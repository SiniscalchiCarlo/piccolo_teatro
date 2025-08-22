
from ..data_pipeline import run_data_pipeline
from ..trend_simulation import predict_trend
from ..models import get_model
from .. import powerbi_visual
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(u'C:/Users/39370/PythonEditorWrapper_eb3663a9-df77-4981-a463-cf73c8cc62b7')
dataset = pd.read_csv('input_df_1475bfb3-7cbb-4ba0-9c1e-8285dbab6262.csv')

powerbi_visual(dataset, offset=0.1, static=True, gain_trend=True)
