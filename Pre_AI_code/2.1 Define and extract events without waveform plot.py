# -*- coding: utf-8 -*-
"""
Extract EOD events from logger recordings and re-analyze with lower amp threshold
Input: 
    csv files with results from EOD extraction
    
Output:
    1 .csv file with all pulse parameters, event number etc.
    1 .csv file with waveforms of all pulses that are included in the events
    1 .csv file with summary of all events

Created on Tue May 28 11:36:24 2024

@author: Stefan Mucha
"""

# 


import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
import datetime as dt
import glob
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.ioff()

# Set directories
root = tkinter.Tk()
root.withdraw()
classification_dir = filedialog.askdirectory(title="Select Folder with classified EOD Tables")
event_results = filedialog.askdirectory(title="Select Folder to Store Analysis Results")

print('Define amplitude threshold (0 = no threshold)')
amp_threshold = float(input())

# List .csv files
eod_files = glob.glob(classification_dir + '/*eod_table.csv', recursive=True)

# Extract timestamps and sort by it
def extract_and_sort_files(files):
    timecol = [pd.to_datetime(fname.split('-')[-1].split('_')[0], format='%Y%m%dT%H%M%S') for fname in files]
    files_set = pd.DataFrame({'timestamp': timecol, 'filename': files})
    return files_set.sort_values(by=['timestamp'], ignore_index=True)

eod_files_set = extract_and_sort_files(eod_files)

# Load metadata
dat_list = []
for fname in eod_files_set['filename']:
    temp_df = pd.read_csv(fname)
    temp_df['row_index'] = np.arange(0, temp_df.shape[0], 1)
    temp_df['filename'] = fname
    temp_df['file_timestamp'] = fname.split('_')[-3].split('-')[-1]

    dat_list.append(temp_df)
    
dat = pd.concat(dat_list, axis=0, ignore_index=True)

# 
dat = dat[dat['h2t_indicator'] == 1].reset_index(drop=True)
dat['timestamp'] = pd.to_datetime(dat['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

if amp_threshold != 0:
    dat = dat[dat['amplitude'] >= amp_threshold].reset_index(drop=True)

# Calculate IPIs for the entire dataset
dat['IPI_lag'] = dat['timestamp'].diff().dt.total_seconds() * 1000
dat['IPI_lead'] = dat['timestamp'].diff(-1).dt.total_seconds() * 1000

# Find start and end indices of events
idx_lag = dat[(dat['IPI_lag'] >= 5000) | dat['IPI_lag'].isna()].index
idx_lead = dat[(dat['IPI_lead'] <= -5000) | dat['IPI_lead'].isna()].index
    
# Initialize lists for events
event_data = []
event_summary = []

# Create color map once for all species
species_unique = dat['species_pred'].unique()
colors = mpl.colormaps.get_cmap('tab10')
species_color_map = {species: colors(i / len(species_unique)) for i, species in enumerate(species_unique)}

for start_idx, end_idx in zip(idx_lag, idx_lead):
    sub = dat.loc[start_idx:end_idx].copy()
    
    # Only include events with more than 10 pulses
    if len(sub) > 10:
        event_idx = len(event_summary) + 1
        print('Event: '+str(event_idx))
        sub.loc[:, 'event'] = event_idx

        event_data.append(sub)
        
        # Summarize event metrics
        t_start = sub['timestamp'].min()
        t_end = sub['timestamp'].max()
        t_mean = sub['timestamp'].mean()
        duration = (t_end - t_start).total_seconds()
        num_pulses = len(sub)
        mean_frequency = num_pulses / duration
        IPI_cv = sub['IPI_lag'].std() / sub['IPI_lag'].mean()
        amp_cv = sub['amplitude'].std() / sub['amplitude'].mean()
        pp_ratio_cv = sub['pp_ratio'].std() / sub['pp_ratio'].mean()
        mean_fft = sub['fft_freq'].mean()
        fft_cv = sub['fft_freq'].std() / sub['fft_freq'].mean()
        
        event_summary.append({
            'event': event_idx, 't_start': t_start, 't_end': t_end, 't_mean': t_mean, 
            'dur': duration, 'pulses': num_pulses, 'mean_f': mean_frequency, 'IPI_cv': IPI_cv, 
            'amp_cv': amp_cv, 'pp_ratio_cv': pp_ratio_cv, 'mean_fft': mean_fft, 'fft_cv': fft_cv
        })

        # Plot event
        gc.collect()

# Convert lists to DataFrames
event_pulses_df = pd.concat(event_data, ignore_index=True)
event_summary_df = pd.DataFrame(event_summary)

# Save to CSV
event_pulses_df.to_csv(f'{event_results}/eods_by_event.csv', index=False)
event_summary_df.to_csv(f'{event_results}/event_list.csv', index=False)
