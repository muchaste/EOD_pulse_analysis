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

# - last change: switched between waveform and DWT classification -> changed 'species_pred' to 'predicted_species_mda' and back
# - only use EODs where the h2t_indicator = 1


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
waveform_dir = filedialog.askdirectory(title="Select Folder with waveforms")
event_results = filedialog.askdirectory(title="Select Folder to Store Analysis Results")

print('Define amplitude threshold (0 = no threshold)')
amp_threshold = float(input())

# List .csv files
eod_files = glob.glob(classification_dir + '/*eod_table.csv', recursive=True)
waveform_files = glob.glob(waveform_dir + '/*waveforms.csv', recursive=True)

# Extract timestamps and sort by it
def extract_and_sort_files(files):
    timecol = [pd.to_datetime(fname.split('-')[-1].split('_')[0], format='%Y%m%dT%H%M%S') for fname in files]
    files_set = pd.DataFrame({'timestamp': timecol, 'filename': files})
    return files_set.sort_values(by=['timestamp'], ignore_index=True)

eod_files_set = extract_and_sort_files(eod_files)
waveform_files_set = extract_and_sort_files(waveform_files)

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
dat = dat[dat['h2t_indicator'] == 1]
dat['timestamp'] = pd.to_datetime(dat['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

if amp_threshold != 0:
    dat = dat[dat['amplitude'] >= amp_threshold].reset_index(drop=True)

# Calculate IPIs for the entire dataset
dat['IPI_lag'] = dat['timestamp'].diff().dt.total_seconds() * 1000
dat['IPI_lead'] = dat['timestamp'].diff(-1).dt.total_seconds() * 1000

# Find start and end indices of events
idx_lag = dat[(dat['IPI_lag'] >= 5000) | dat['IPI_lag'].isna()].index
idx_lead = dat[(dat['IPI_lead'] <= -5000) | dat['IPI_lead'].isna()].index

# Function to load waveform data for a specific event
def load_event_waveforms(event_dat, waveform_files_set):
    event_start_time = event_dat['timestamp'].min()
    event_end_time = event_dat['timestamp'].max()
    
    first_file_idx = waveform_files_set[waveform_files_set['timestamp'] <= event_start_time].index[-1]
    last_file_idx = waveform_files_set[waveform_files_set['timestamp'] <= event_end_time].index[-1]

    fitting_files = waveform_files_set.loc[first_file_idx:last_file_idx, 'filename'].values

    waveforms_list = []
    for fname in fitting_files:
        wf = pd.read_csv(fname)
        wf['row_index'] = np.arange(0, wf.shape[0], 1)
        wf['file_timestamp'] = fname.split('_')[-3].split('-')[-1]
        # ftime = pd.to_datetime(fname.split('-')[-1].split('_')[0], format='%Y%m%dT%H%M%S')
        # wf['timestamp'] = ftime

        waveforms_list.append(wf)
    waveforms = pd.concat(waveforms_list, axis=0, ignore_index=True)
    
    event_waveforms = []
    for _, row in event_dat.iterrows():
        waveform_series = waveforms[(waveforms['file_timestamp'] == row['file_timestamp'])].iloc[row['row_index']].drop(labels=['row_index', 'file_timestamp'])
        waveform_df = pd.DataFrame(waveform_series).transpose()
        event_waveforms.append(waveform_df)
        
    event_waveforms = pd.concat(event_waveforms, axis=0, ignore_index=True)
    
    return event_waveforms

# # Function to plot each event
# def plot_event(ev_dat, ev_wf, ev_idx, color_map, output_dir):
#     channels = ev_dat['channel'].unique()
#     num_channels = len(channels)
    
#     species_unique = ev_dat['species_pred'].unique()
#     num_species = len(species_unique)
    
#     time_us = np.arange(0, ev_wf.shape[1])/96000 * 1000000
    
#     # Sort channels in descending order
#     channels = sorted(channels, reverse=True)
    
#     fig, axs = plt.subplots(num_channels, 1 + num_species, figsize=(15+3*num_species, 6 * num_channels), 
#                             gridspec_kw={'width_ratios': [5] + [1] * num_species, 'height_ratios': [1] * num_channels})
    
#     # Ensure axs is 2D array for consistent indexing
#     if num_channels == 1:
#         axs = np.expand_dims(axs, axis=0)

#     buffer_sec = (ev_dat['timestamp'].max() - ev_dat['timestamp'].min()).total_seconds() / 20
#     base_time = ev_dat['timestamp'].min()

#     for i, chan in enumerate(channels):
#         chan_data = ev_dat[ev_dat['channel'] == chan]
#         chan_waveforms = ev_wf[ev_dat['channel'] == chan]

#         # Plot horizontal line at 0
#         axs[i, 0].hlines(y=0, 
#                          xmin=ev_dat['timestamp'].min() - dt.timedelta(seconds=buffer_sec),
#                          xmax=ev_dat['timestamp'].max() + dt.timedelta(seconds=buffer_sec),
#                          color='black')

#         for idx, row in chan_data.iterrows():
#             waveform = chan_waveforms.loc[idx]
#             time_offset = row['timestamp']
#             amplitude = row['amplitude']
#             color = color_map[row['species_pred']]
            
#             if row['pulse_orientation'] == 'HP':
#                 waveform = waveform * amplitude / max(waveform)
#             else:
#                 waveform = -waveform * amplitude / max(waveform)
#                 waveform = np.roll(waveform, (row['peak_idx'] - row['trough_idx']))

#             time_seconds = (time_offset - base_time).total_seconds()
#             waveform_time = np.linspace(time_seconds - buffer_sec / 2, time_seconds + buffer_sec / 2, len(waveform))
#             axs[i, 0].plot(base_time + pd.to_timedelta(waveform_time, unit='s'), waveform, color=color, alpha=0.7)

#         axs[i, 0].set_title(f'Channel {chan} - Event {ev_idx}')
#         axs[i, 0].set_xlim(ev_dat['timestamp'].min() - dt.timedelta(seconds=buffer_sec),
#                            ev_dat['timestamp'].max() + dt.timedelta(seconds=buffer_sec))
    
#         # Plot waveforms for each species in separate columns
#         for s, species in enumerate(species_unique):
#             species_waveforms = chan_waveforms[chan_data['species_pred'] == species]
#             for j in range(len(species_waveforms)):
#                 color = color_map[species]
#                 axs[i, s + 1].plot(time_us, species_waveforms.iloc[j, :], color=color, alpha=0.7)
#             axs[i, s + 1].set_title(f'{species} - Channel {chan}')
#             axs[i, s + 1].set_xlabel('Time (uS)')
#             if s > 0:
#                 axs[i, s + 1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

#     # Set common labels
#     fig.text(0.5, 0.04, 'Time', ha='center')
#     fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

#     plt.tight_layout()
#     plt.savefig(f'{output_dir}/event_{ev_idx}.png')
#     plt.close()

# Function to plot each event
def plot_event(ev_dat, ev_wf, ev_idx, color_map, output_dir):
    channels = ev_dat['channel'].unique()
    num_channels = len(channels)
    
    species_unique = ev_dat['species_pred'].unique()
    num_species = len(species_unique)
    
    time_us = np.arange(0, ev_wf.shape[1]) / 96000 * 1000000
    
    # Sort channels in descending order
    channels = sorted(channels, reverse=True)
    
    fig, axs = plt.subplots(num_channels, 1 + num_species, figsize=(15 + 3 * num_species, 6 * num_channels), 
                            gridspec_kw={'width_ratios': [5] + [1] * num_species, 'height_ratios': [1] * num_channels})
    
    # Ensure axs is 2D array for consistent indexing
    if num_channels == 1:
        axs = np.expand_dims(axs, axis=0)

    buffer_sec = (ev_dat['timestamp'].max() - ev_dat['timestamp'].min()).total_seconds() / 20
    base_time = ev_dat['timestamp'].min()

    for i, chan in enumerate(channels):
        chan_data = ev_dat[ev_dat['channel'] == chan].reset_index(drop=True)
        chan_waveforms = ev_wf.iloc[np.where(ev_dat['channel'] == chan)[0]].reset_index(drop=True)

        # Plot horizontal line at 0
        axs[i, 0].hlines(y=0, 
                         xmin=ev_dat['timestamp'].min() - dt.timedelta(seconds=buffer_sec),
                         xmax=ev_dat['timestamp'].max() + dt.timedelta(seconds=buffer_sec),
                         color='black')

        for idx, row in chan_data.iterrows():
            waveform = chan_waveforms.iloc[idx]
            time_offset = row['timestamp']
            amplitude = row['amplitude']
            color = color_map[row['species_pred']]
            
            if row['pulse_orientation'] == 'HP':
                waveform = waveform * amplitude / max(waveform)
            else:
                waveform = -waveform * amplitude / max(waveform)
                waveform = np.roll(waveform, (row['peak_idx'] - row['trough_idx']))

            time_seconds = (time_offset - base_time).total_seconds()
            waveform_time = np.linspace(time_seconds - buffer_sec / 2, time_seconds + buffer_sec / 2, len(waveform))
            axs[i, 0].plot(base_time + pd.to_timedelta(waveform_time, unit='s'), waveform, color=color, alpha=0.7)

        axs[i, 0].set_title(f'Channel {chan} - Event {ev_idx}')
        axs[i, 0].set_xlim(ev_dat['timestamp'].min() - dt.timedelta(seconds=buffer_sec),
                           ev_dat['timestamp'].max() + dt.timedelta(seconds=buffer_sec))
    
        # Plot waveforms for each species in separate columns
        for s, species in enumerate(species_unique):
            species_waveforms = chan_waveforms[chan_data['species_pred'] == species]
            for j in range(len(species_waveforms)):
                color = color_map[species]
                axs[i, s + 1].plot(time_us, species_waveforms.iloc[j, :], color=color, alpha=0.7)
            axs[i, s + 1].set_title(f'{species} - Channel {chan}')
            axs[i, s + 1].set_xlabel('Time (uS)')
            if s > 0:
                axs[i, s + 1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    # Set common labels
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/event_{ev_idx}.png')
    plt.close()
    
# Initialize lists for events
event_data = []
event_waveforms = []
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

        # Load corresponding waveforms for the event
        sub_waveform = load_event_waveforms(sub, waveform_files_set)

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
        pp_ratio_cv = sub['pp_ratio'].std() / sub['pp_ratio'].mean()
        mean_fft = sub['fft_freq'].mean()
        fft_cv = sub['fft_freq'].std() / sub['fft_freq'].mean()
        
        event_summary.append({
            'event': event_idx, 't_start': t_start, 't_end': t_end, 't_mean': t_mean, 
            'dur': duration, 'pulses': num_pulses, 'mean_f': mean_frequency, 'IPI_cv': IPI_cv, 
            'amp_cv': amp_cv, 'pp_ratio_cv': pp_ratio_cv, 'mean_fft': mean_fft, 'fft_cv': fft_cv
        })

        # Plot event
        plot_event(sub, sub_waveform, event_idx, species_color_map, event_results)
        gc.collect()

# Convert lists to DataFrames
event_pulses_df = pd.concat(event_data, ignore_index=True)
event_waveforms_df = pd.concat(event_waveforms, ignore_index=True)
event_summary_df = pd.DataFrame(event_summary)

# Save to CSV
event_pulses_df.to_csv(f'{event_results}/eods_by_event.csv', index=False)
event_waveforms_df.to_csv(f'{event_results}/waveforms_by_event.csv', index=False)
event_summary_df.to_csv(f'{event_results}/event_list.csv', index=False)
