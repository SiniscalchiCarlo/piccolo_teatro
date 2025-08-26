import pandas as pd
import re

# Load the CSV
df = pd.read_csv('/home/carlo/Data/dati_piccolo_teatro2/xgb_log_ic_1/train_validation_summary.csv')
df.rename(columns=lambda col: re.sub(r'IntervalScore_(\w+)', r'IS_\1', col), inplace=True)
# Split columns into chunks of 8, padding the last if necessary
cols = df.columns.tolist()
chunks = [cols[i:i+8] for i in range(0, len(cols), 8)]

# Generate LaTeX tables
for idx, chunk in enumerate(chunks, start=1):
    print(f"% Table {idx}")
    print("\\begin{table}[H]")
    print("  \\centering")
    col_format = " | ".join(["c"] * 8)
    print(f"  \\begin{{tabular}}{{| {col_format} |}}")
    print("    \\hline")
    # Header row
    header = " & ".join([col.replace("_","\_") if col else "" for col in chunk])
    print(f"    {header} \\\\ \\hline")
    # Data rows
    for _, row in df.iterrows():
        row_vals = [str(round(row[col],3)) if col else "" for col in chunk]
        print("    " + " & ".join(row_vals) + " \\\\ \\hline")
    print("  \\end{tabular}")
    print("\\end{table}\n")
