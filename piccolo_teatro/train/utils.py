
from ydata_profiling import ProfileReport

def df_report(df):
    profile = ProfileReport(df, title="Profilo Dataset", explorative=True)

    # Export to HTML
    profile.to_file("report.html")

    # Generate JSON representation
    json_report = profile.to_json()

    # Save JSON to file
    with open("report.json", "w", encoding="utf-8") as f:
        f.write(json_report)

