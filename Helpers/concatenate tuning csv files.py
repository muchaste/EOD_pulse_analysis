# python script for concatenating all parameter tuning CSVs into one master CSV for analysis in 05_7

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os

root = tk.Tk()
root.withdraw()
csv_folder = filedialog.askdirectory(title="Select folder containing parameter_tuning_results_*.csv files")
root.destroy()
all_csvs = [f for f in os.listdir(csv_folder) if f.startswith('parameter_tuning_results_iter') and f.endswith('.csv')]

print(f"Found {len(all_csvs)} tuning result CSVs. Concatenating into master CSV...")

all_dfs = []
for csv in all_csvs:
    df = pd.read_csv(os.path.join(csv_folder, csv))
    iter_tag = csv.replace('parameter_tuning_results_', '').replace('.csv', '')  # e.g. iter01
    df['iteration'] = iter_tag
    all_dfs.append(df)


master_df = pd.concat(all_dfs, ignore_index=True)
master_csv_path = os.path.join(csv_folder, 'parameter_tuning_results_master.csv')
master_df.to_csv(master_csv_path, index=False)
print(f"Master CSV saved to {master_csv_path}")