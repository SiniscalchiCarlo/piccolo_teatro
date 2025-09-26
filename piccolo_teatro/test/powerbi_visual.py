
from ..data_pipeline import run_data_pipeline
from ..trend_simulation import predict_trend
from ..models import get_model
from .. import powerbi_visual
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(u'C:/Users/39370/PythonEditorWrapper_0484369b-1f36-4949-b751-d4dcc5e1e656')
dataset = pd.read_csv('input_df_be21c7ba-35f1-40b4-b7f3-f2179c604951.csv')

# trend_type = ["perc", "gain", "tickets"]
# offset: (numero fra 0 e 1) indica quanta percentuale di dati e' necessaria (e utilizzata) per fare la previsione (fissa che non varia anche se ci sono piu' dati a disposizione)
# static_plot: True crea un grafico statico visualizzabile direttamente dentro powerbi, False uno dinamico visualizzabile sul browser
# trend_type = "perc" mostra la percentuale di biglietti venduti, "gain" il guadagno cumulato, "tickets" il numero cumulato di biglietti venduti

powerbi_visual(dataset, offset=0.1, static_plot=True, trend_type="gain")
