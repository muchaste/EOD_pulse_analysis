#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 00_Gain_Correction_Diel_Analysis.py
# Gain correction and diel analysis of multi-channel recordings
This script identifies recordings with few EODs, calculates noise levels, merges with temperature data, and computes gain correction factors.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog
import glob
import audioio as aio


# 1. Identify recordings with few EODs
extraction_dir = filedialog.askdirectory(title='Select directory with pulse extraction files')

eod_tables = glob.glob(extraction_dir + '/*_eod_table.csv')

sparse_eod_tables = []
for table in eod_tables:
    df = pd.read_csv(table)
    if df.shape[0] < 300:  # Arbitrary threshold for "sparsity"
        sparse_eod_tables.append(table)
        print(f"Sparse EOD table found: {table} with {df.shape[0]} EODs")

# 2. Extract recording id (eod table name before first '_')
recording_ids = [table.split('\\')[-1].split('_')[0] for table in sparse_eod_tables]
print("Recording IDs with sparse EODs:", recording_ids)

# 3. Find corresponding *.wav recordings
wav_dir = filedialog.askdirectory(title='Select directory with .wav files')
wav_files = glob.glob(wav_dir + '/*.wav')

sparse_wav_files = []
for wav in wav_files:
    wav_id = wav.split('\\')[-1].split('.')[0]
    if wav_id in recording_ids:
        sparse_wav_files.append(wav)
        print(f"Corresponding .wav file found: {wav}")

# 4. Calculate noise levels and extract timestamps from filenames (L5-20231011T091504 = 2023-10-11 09:15:04)
# Noise levels are calculated and stored for each of the 8 channels
results = []

for wav in sparse_wav_files:
    wav_id = wav.split('/')[-1].split('.')[0]
    timestamp_str = wav_id.split('-')[1]  # Extract timestamp part
    timestamp = pd.to_datetime(timestamp_str, format='%Y%m%dT%H%M%S')
    
    # Load audio file and calculate noise level
    audio_data, sample_rate = aio.load_audio(wav)
    n_channels = audio_data.shape[1]
    rms_values = [np.sqrt(np.mean(audio_data[:, chan]**2)) for chan in range(n_channels)]
    results.append([wav_id, timestamp] + rms_values)

    print(f"Processed {wav_id} at {timestamp}, RMS values: {rms_values}")
    
# Convert results to DataFrame
columns = ['Recording_ID', 'Timestamp'] + [f'Channel_{i+1}_RMS' for i in range(n_channels)]
rms_df = pd.DataFrame(results, columns=columns)

# 5. Read in .xslx table with time stamps and temperature data
temp_file = filedialog.askopenfilename(title='Select temperature data .xlsx file', filetypes=[("Excel files", "*.xlsx")])
temp_df = pd.read_excel(temp_file)
temp_df['Datetime'] = pd.to_datetime(temp_df['Datetime'])  # Ensure timestamp is in datetime format

# Average temperature across columns
n_cols = temp_df.shape[1] - 1  # Exclude 'Datetime' column
temp_df['Avg_Temperature'] = temp_df.iloc[:, 1:n_cols+1].mean(axis=1)

# 6. Merge RMS data with temperature data based on nearest timestamp
# First, there is one rms value per 10 min, and two temperature values per 10 min
# We will average the two temperature values to get one per 10 min
# Then we will merge based on nearest timestamp
av_temp_df = temp_df.resample('10T', on='Datetime').mean().reset_index()

merged_df = pd.merge_asof(rms_df.sort_values('Timestamp'), temp_df[['Datetime', 'Avg_Temperature']].sort_values('Datetime'), left_on='Timestamp', right_on='Datetime', direction='nearest')

# 7. Plot RMS values against temperature for each channel
for i in range(n_channels):
    plt.figure()
    plt.scatter(merged_df['Avg_Temperature'], merged_df[f'Channel_{i+1}_RMS'])
    plt.title(f'Channel {i+1} RMS vs Temperature')
    plt.xlabel('Average Temperature (°C)')
    plt.ylabel('RMS Value')
    plt.grid(True)
    plt.show()

# 8. Do a correlation analysis
correlation_results = {}
for i in range(n_channels):
    corr = merged_df['Avg_Temperature'].corr(merged_df[f'Channel_{i+1}_RMS'])
    correlation_results[f'Channel_{i+1}'] = corr
    print(f'Correlation between Channel {i+1} RMS and Temperature: {corr}')

# 9. Plot RMS values over time for each channel
plt.figure()
for i in range(n_channels):
    plt.plot(merged_df['Timestamp'], merged_df[f'Channel_{i+1}_RMS'], label=f'Channel {i+1}')

plt.title(f'Channel {i+1} RMS over Time')
plt.xlabel('Time')
plt.ylabel('RMS Value')
plt.grid(True)
plt.show()

# 10. Calculate correction factors based on median RMS values
channel_medians = []
for i in range(n_channels):
    channel_medians.append(merged_df[f'Channel_{i+1}_RMS'].median())

# Use overall median as target
target_level = np.median(channel_medians)

# Alternatively, use max. medan as target
target_level = np.max(channel_medians)

# Calculate correction factors
correction_factors = []
for i in range(n_channels):
    correction_factor = target_level / channel_medians[i]
    correction_factors.append(correction_factor)
    print(f'Channel {i+1} correction factor: {correction_factor:.3f}')

# Apply corrections
for i in range(n_channels):
    merged_df[f'Channel_{i+1}_RMS_corrected'] = merged_df[f'Channel_{i+1}_RMS'] * correction_factors[i]