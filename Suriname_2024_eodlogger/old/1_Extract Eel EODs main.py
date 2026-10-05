"""Load and cluster pulsefish recordings."""

import matplotlib.pyplot as plt
import audioio as aio
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
import glob
import datetime as dt
from sklearn.preprocessing import StandardScaler
from eel_eodlogger_functions import analyze_r4_snippets, calc_fft_peak
import joblib
import matplotlib as mpl

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Logger Files")
output_path = filedialog.askdirectory(title="Select Folder to Store Analysis Results")
cal_file = filedialog.askopenfilename(title="Select File with Calibration Data")
cor_factors = np.array(pd.read_csv(cal_file))

# List all .wav files
filelist = glob.glob(input_path + '/*.wav', recursive=True)

# Sort filelist by time
timecol = [pd.to_datetime(fname.split('-')[3][0:-4], format='%Y%m%dT%H%M%S') for fname in filelist]
file_set = pd.DataFrame({'timestamp': timecol, 'filename': filelist})
file_set = file_set.sort_values(by=['timestamp'], ignore_index=True)
#%%
# Load 60 sec of first file
tmin, tmax = 0, 60
with aio.AudioLoader(file_set['filename'][0], 60) as sf:
    rate = sf.samplerate
    data = sf[int(tmin * rate):int(tmax * rate), :]
n_channels = data.shape[1]

# Calibrate with correction factor from .csv
for i in range(n_channels):
    data[:, i] *= cor_factors[i, 1]
    sd = np.std(data[:, i])

parameters = {'peak_window_us':10000,
      'peak_threshold':0.001,
      'interpolation_factor':1,
      'peak_dur_min':625,
      'peak_dur_max':1250,
      'peak_fft_freq_min':40,
      'peak_fft_freq_max':400}

# Plot raw data
offset = np.max(abs(data))
plt.figure(figsize=(40, 12))
for i in range(n_channels):
    plt.plot(data[0:int(60 * rate - 1), i] + i * offset, label=str(i + 1))
    plt.hlines(y=parameters['peak_threshold'] + i * offset, xmin=0, xmax=int(60 * rate - 1))
plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
plt.savefig('%s\\%s_one_minute_raw.png' % (output_path, file_set['filename'][0].split('\\')[-1][:-4]))
plt.show(block=False)




print(parameters)
print("change parameters? (1/0)")
change_params = int(input())
while change_params:
    print("input parameter name")
    ch_par = input()
    print("input parameter value")
    ch_par_value = float(input())
    parameters[ch_par] = ch_par_value
    print(parameters)
    print("done? (1/0)")
    done = int(input())
    if done:
        change_params = 0
parameters = pd.DataFrame({k: [v] for k, v in parameters.items()})
parameters.to_csv('%s\\analysis_parameters.csv' % output_path, index=False)
peak_window = int(parameters['peak_window_us'][0] * rate / 1e6)
plt.close()

#%%
# Process each file
for n, filepath in enumerate(file_set['filename']):
    fname = filepath.split('\\')[-1]
    print(fname)
    
    # Load file
    data, rate = aio.load_audio(filepath)
    n_channels = data.shape[1]
    
    # Calibrate with correction factor
    for i in range(n_channels):
        data[:, i] *= cor_factors[i, 1]
    
    # Find peaks in all channels
    peaks = []
    durs = []
    for i in range(n_channels):
        
        p_temp, peak_params = find_peaks(data[:, i], height=parameters['peak_threshold'][0], 
                                   width = (parameters['peak_dur_min'][0]*1e3/rate, parameters['peak_dur_max'][0]*1e3/rate * 1e6),
                                   rel_height=0.5)
        t_temp, trough_params = find_peaks(-data[:, i], height=parameters['peak_threshold'][0], 
                                   width = (parameters['peak_dur_min'][0]*1e3/rate, parameters['peak_dur_max'][0]*1e3/rate * 1e6),
                                   rel_height=0.5)
        
        p_temp = p_temp.astype(np.int64)
        t_temp = t_temp.astype(np.int64)
        p_durs = peak_params['widths']*rate/1e3
        t_durs = trough_params['widths']*rate/1e3
        
        peaks_temp = np.concatenate((p_temp, t_temp))
        durs_temp = np.concatenate((p_durs, t_durs))

        # Use only the absolute peaks and troughs per peak window
        for j, p in enumerate(peaks_temp):
            indexer = np.arange(p - peak_window // 2, p + peak_window // 2)
            if np.max(indexer) >= data.shape[0]:
                to_pad = int(np.max(indexer) - data.shape[0] + 1)
                peak = np.zeros(len(indexer))
                peak[:int(peak_window - to_pad)] = data[int(np.min(indexer)):, i]
            elif np.min(indexer) < 0:
                to_pad = int(abs(np.min(indexer)))
                peak = np.zeros(len(indexer))
                peak[to_pad:] = data[:int(np.max(indexer) + 1), i]
            else:
                peak = data[indexer, i]
            if data[p, i] < np.max(abs(peak)):
                continue
            else:
                peaks.append(p)
                durs.append(durs_temp[j])
    
    peaks.sort()
    
    # Get unique peaks - round to nearest 20 samples because of long peaks
    peaks_rounded = [20 * round(p/20) for p in peaks]
    _ , idc = np.unique(peaks_rounded, return_index=True)
    peaks_unique = np.array(peaks)[idc]
    durs_unique = np.array(durs)[idc]
  
    print("Peaks found: " + str(len(peaks_unique)))
    
    if len(peaks_unique) == 0:
        continue
    
    # Extract peak snippets
    snippets = []
    for p in peaks_unique:
        indexer = np.arange(p - peak_window // 2, p + peak_window // 2)
        if np.max(indexer) >= data.shape[0]:
            to_pad = int(np.max(indexer) - data.shape[0] + 1)
            peak = np.zeros((len(indexer), n_channels))
            peak[:int(peak_window - to_pad) - 1, :] = data[int(np.min(indexer)), :]
        elif np.min(indexer) < 0:
            to_pad = int(abs(np.min(indexer)))
            peak = np.zeros((len(indexer), n_channels))
            peak[to_pad:, :] = data[:int(np.max(indexer) + 1), :]
        else:
            peak = data[indexer]
        snippets.append(peak)
    
    n_snippets = len(snippets)
    
    # Extract head-to-tail waveforms, channels, amplitudes, and indices
    h2t_waveforms, amps, h2t_amp, cor_coeffs, h2t_chan, h2t_found, peak_idc, trough_idc = analyze_r4_snippets(snippets, peaks_unique, parameters['interpolation_factor'][0])
    
    # Create differential data (for plotting only)
    data_diff = np.diff(data)
    offset_diff = np.max(h2t_amp) * 1.5
    n_channels_diff = data_diff.shape[1]
    
    gc.collect()
    
    fft_freqs = np.zeros(n_snippets)
    for i in range(n_snippets):
        fft_freqs[i] = calc_fft_peak(h2t_waveforms[i], rate, zero_padding_factor=100)

   
    # Apply filters
    keep_mask = (
        (fft_freqs > parameters['peak_fft_freq_min'][0]) & (fft_freqs <parameters['peak_fft_freq_max'][0]) &
        (durs_unique > parameters['peak_dur_min'][0]) & (durs_unique < parameters['peak_dur_max'][0])
    )

    keep_indices = np.where(keep_mask)[0]
    filtered_h2t_waveforms = h2t_waveforms[keep_indices]
    filtered_durs = durs_unique[keep_indices]
    filtered_ffts = fft_freqs[keep_indices]

    n_eods = filtered_h2t_waveforms.shape[0]
    
    print('EODs after freq/dur/ratio filter: ' + str(n_eods))
    
    if n_eods != 0:
        # Filter the other variables
        filtered_amps = amps[keep_indices]
        filtered_h2t_amp = h2t_amp[keep_indices]
        filtered_cor_coeffs = cor_coeffs[keep_indices]
        filtered_h2t_chan = h2t_chan[keep_indices]
        filtered_h2t_found = h2t_found[keep_indices]
        filtered_peak_idc = peak_idc[keep_indices]
        filtered_trough_idc = trough_idc[keep_indices]
        filtered_pulse_orientation = np.array(['HP'] * n_eods)
        filtered_pulse_orientation[np.where(filtered_trough_idc < filtered_peak_idc)[0]] = 'HN'
        
  
        plt.figure(figsize=(30, 12))
        offset_diff = np.max(filtered_h2t_amp) * 1.5
        
        for i in range(n_channels_diff):
            plt.plot(data_diff[:, i] + i * offset_diff, linewidth=0.5)
            h2t_idc = np.where(filtered_h2t_chan == i) 
            plt.plot(filtered_peak_idc[h2t_idc], data_diff[filtered_peak_idc[h2t_idc], i] + i * offset_diff, 'o', markersize=2)
            plt.plot(filtered_trough_idc[h2t_idc], data_diff[filtered_trough_idc[h2t_idc], i] + i * offset_diff, 'o', markersize=2)
    
        plt.ylim(bottom=None, top=(n_channels_diff - 0.5) * offset_diff)
        plt.title(fname)
        plt.xlabel('Sample')
        plt.ylabel('Voltage')
        plt.savefig('%s\\%s.png' % (output_path, fname[:-4]))
        plt.close()
        
        gc.collect()
        
        # Compile results and save
        filtered_h2t_timestamps = [file_set['timestamp'][n] + dt.timedelta(seconds=filtered_peak_idc[i] / rate) for i in range(n_eods)]
        
        eod_table = pd.DataFrame({
            'timestamp': filtered_h2t_timestamps,
            'channel': filtered_h2t_chan+1,
            'amplitude': filtered_h2t_amp,
            'peak_idx': filtered_peak_idc,
            'trough_idx': filtered_trough_idc,
            'pulse_orientation': filtered_pulse_orientation,
            'h2t_indicator': filtered_h2t_found,
            'duration_us': filtered_durs,
            'fft_freq': filtered_ffts
        })
        
        eod_table.to_csv('%s\\%s_eod_table.csv' % (output_path, fname[:-4]), index=False)
        
        waveform_table = pd.DataFrame(filtered_h2t_waveforms)
        waveform_table.to_csv('%s\\%s_h2t_waveforms.csv' % (output_path, fname[:-4]), index=False)
        
        gc.collect()
