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

# TODOS:
    # - correct channel numbers (+1?)


import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
import datetime as dt
import glob
import gc
import matplotlib.pyplot as plt
# import matplotlib as mpl

plt.ioff()

# Set directories
root = tkinter.Tk()
root.withdraw()
data_dir = filedialog.askdirectory(title="Select Folder with extraction results")
event_results = filedialog.askdirectory(title="Select Folder to Store Analysis Results")

print('Define amplitude threshold (0 = no threshold)')
amp_threshold = float(input())

# List .csv files
eod_files = glob.glob(data_dir + '/*eod_table.csv', recursive=True)
waveform_files = glob.glob(data_dir + '/*waveforms.csv', recursive=True)

# Extract timestamps and sort by it
if len(eod_files) > 1:
    def extract_and_sort_files(files):
        timecol = [pd.to_datetime(fname.split('-')[-1].split('_')[0], format='%Y%m%dT%H%M%S') for fname in files]
        files_set = pd.DataFrame({'timestamp': timecol, 'filename': files})
        return files_set.sort_values(by=['timestamp'], ignore_index=True)
    
    eod_files_set = extract_and_sort_files(eod_files)
    waveform_files_set = extract_and_sort_files(waveform_files)
    
# Load data
def load_data(eod_files, waveform_files):
    dat_list = [pd.read_csv(fname) for fname in eod_files]
    wf_list = [pd.read_csv(fname) for fname in waveform_files]
    dat = pd.concat(dat_list, axis=0, ignore_index=True)
    waveforms = pd.concat(wf_list, axis=0, ignore_index=True)
    dat['timestamp'] = pd.to_datetime(dat['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    return dat, waveforms

if len(eod_files) > 1:
    dat, waveforms = load_data(eod_files_set['filename'], waveform_files_set['filename'])
else:
    dat, waveforms = load_data(eod_files, waveform_files)

    
# dat['pulse_orientation'] = 'HP'
# dat.loc[dat['trough_idx'] < dat['peak_idx'], 'pulse_orientation'] = 'HN'

if amp_threshold != 0:
    keep_idc = np.where(dat['amplitude'] >= amp_threshold)[0]
    dat = dat.loc[keep_idc]
    waveforms = waveforms.loc[keep_idc]

#%%
# Function to plot each event
def plot_eel_event(ev_dat, ev_wf, ev_idx, output_dir):
    channels = ev_dat['channel'].unique()
    num_channels = len(channels)
    time_us = np.arange(0, ev_wf.shape[1])/96000 * 1000000
    
    # Sort channels in descending order
    channels = sorted(channels, reverse=True)
    
    fig, axs = plt.subplots(num_channels, 2, figsize=(18, 6 * num_channels), 
                            gridspec_kw={'width_ratios': [5, 1], 'height_ratios': [1] * num_channels})
    
    # Ensure axs is 2D array for consistent indexing
    if num_channels == 1:
        axs = np.expand_dims(axs, axis=0)

    buffer_sec = (ev_dat['timestamp'].max() - ev_dat['timestamp'].min()).total_seconds() / 20
    base_time = ev_dat['timestamp'].min()

    for i, chan in enumerate(channels):
        chan_data = ev_dat[ev_dat['channel'] == chan]
        chan_waveforms = ev_wf[ev_dat['channel'] == chan]
        

        # Plot horizontal line at 0
        axs[i, 0].hlines(y=0, 
                         xmin=ev_dat['timestamp'].min() - dt.timedelta(seconds=buffer_sec),
                         xmax=ev_dat['timestamp'].max() + dt.timedelta(seconds=buffer_sec),
                         color='black')

        for idx, row in chan_data.iterrows():
            waveform = chan_waveforms.loc[idx]
            time_offset = row['timestamp']
            amplitude = row['amplitude']
            
            axs[i, 1].plot(time_us, waveform, alpha=0.1, color = "black")

            
            if row['pulse_orientation'] == 'HP':
                waveform = waveform * amplitude / max(waveform)
            else:
                waveform = -waveform * amplitude / max(waveform)
                # waveform = np.roll(waveform, (row['peak_idx'] - row['trough_idx']))

            time_seconds = (time_offset - base_time).total_seconds()
            waveform_time = np.linspace(time_seconds - buffer_sec / 2, time_seconds + buffer_sec / 2, len(waveform))
            axs[i, 0].plot(base_time + pd.to_timedelta(waveform_time, unit='s'), waveform, alpha=0.7, color = "black")

        axs[i, 0].set_title(f'Channel {chan} - Event {ev_idx}')
        axs[i, 0].set_xlim(ev_dat['timestamp'].min() - dt.timedelta(seconds=buffer_sec),
                           ev_dat['timestamp'].max() + dt.timedelta(seconds=buffer_sec))
        axs[i, 1].set_title('EOD Waveforms - Channel {chan}')
        axs[i, 1].set_xlabel('Time (uS)')
    


    # Set common labels
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/event_{ev_idx}.png')
    plt.close()
    
#%%
# Initialize lists for events
event_data = []
event_waveforms = []
event_summary = []

# Calculate IPIs for the entire dataset
dat['IPI_lag'] = dat['timestamp'].diff().dt.total_seconds() * 1000
dat['IPI_lead'] = dat['timestamp'].diff(-1).dt.total_seconds() * 1000

# Find start and end indices of events
idx_lag = dat[(dat['IPI_lag'] >= 5000) | dat['IPI_lag'].isna()].index
idx_lead = dat[(dat['IPI_lead'] <= -5000) | dat['IPI_lead'].isna()].index



for start_idx, end_idx in zip(idx_lag, idx_lead):
    sub = dat.loc[start_idx:end_idx].copy()
    sub_waveform = waveforms.loc[start_idx:end_idx].copy()
    
    # Only include events with more than 10 pulses
    if len(sub) > 10:
        event_idx = len(event_summary) + 1
        print('Event: '+str(event_idx))
        sub.loc[:, 'event'] = event_idx

        event_data.append(sub)
        event_waveforms.append(sub_waveform)
        
        # Summarize event metrics
        t_start = sub['timestamp'].min()
        t_end = sub['timestamp'].max()
        t_mean = sub['timestamp'].mean()
        duration = (t_end - t_start).total_seconds()
        num_pulses = len(sub)
        mean_frequency = num_pulses / duration
        IPI_cv = sub['IPI_lag'].std() / sub['IPI_lag'].mean()
        amp_cv = sub['amplitude'].std() / sub['amplitude'].mean()
        # pp_ratio_cv = sub['pp_ratio'].std() / sub['pp_ratio'].mean()
        mean_fft = sub['fft_freq'].mean()
        fft_cv = sub['fft_freq'].std() / sub['fft_freq'].mean()
        
        event_summary.append({
            'event': event_idx, 't_start': t_start, 't_end': t_end, 't_mean': t_mean, 
            'dur': duration, 'pulses': num_pulses, 'mean_f': mean_frequency, 'IPI_cv': IPI_cv, 
            'amp_cv': amp_cv, 'mean_fft': mean_fft, 'fft_cv': fft_cv
        })

        # Plot event
        plot_eel_event(sub, sub_waveform, event_idx, event_results)
        gc.collect()

# Convert lists to DataFrames
event_pulses_df = pd.concat(event_data, ignore_index=True)
event_waveforms_df = pd.concat(event_waveforms, ignore_index=True)
event_summary_df = pd.DataFrame(event_summary)

# Save to CSV
event_pulses_df.to_csv(f'{event_results}/eods_by_event.csv', index=False)
event_waveforms_df.to_csv(f'{event_results}/waveforms_by_event.csv', index=False)
event_summary_df.to_csv(f'{event_results}/event_list.csv', index=False)
