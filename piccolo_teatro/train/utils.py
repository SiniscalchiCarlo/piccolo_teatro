
from ydata_profiling import ProfileReport

def df_report(df):
    profile = ProfileReport(df, title="Profilo Dataset", explorative=True)
    profile.to_file("report.html")   # genera un file HTML
    json_report = profile.to_json()
    print(json_report)            


