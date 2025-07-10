"""Load and cluster pulsefish recordings."""

import matplotlib.pyplot as plt
import audioio as aio
from scipy.signal import find_peaks, correlate
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
import glob
import datetime as dt
from sklearn.preprocessing import StandardScaler
from Pre_AI_code.R4_eodlogger_functions import analyze_r4_snippets, filter_waveforms
import joblib
import matplotlib as mpl

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Logger Files")
output_path = filedialog.askdirectory(title="Select Folder to Store Analysis Results")
cal_file = filedialog.askopenfilename(title="Select File with Calibration Data")
cor_factors = np.array(pd.read_csv(cal_file))

# Pick directory with models for noise and species classification
noise_model_dir = filedialog.askdirectory(title="Select Folder with Noise/Fish GMMs")
species_model_dir = filedialog.askdirectory(title="Select Folder with Species GMMs")
gmm_noise_file = glob.glob(noise_model_dir + '/gmm_noise.pkl')[0]
gmm_fish_file = glob.glob(noise_model_dir + '/gmm_fish.pkl')[0]
gmm_species_list = glob.glob(species_model_dir + '/gmm*.pkl')

# Load the GMM models
gmm_fish = joblib.load(gmm_fish_file)
gmm_noise = joblib.load(gmm_noise_file)
species_labs = pd.read_csv(species_model_dir + '/species_labels.csv')

gmm_species_models = {i: joblib.load(gmm) for i, gmm in enumerate(gmm_species_list)}

# List all .wav files
filelist = glob.glob(input_path + '/*.wav', recursive=True)

# Sort filelist by time
timecol = [pd.to_datetime(fname.split('-')[1][0:-4], format='%Y%m%dT%H%M%S') for fname in filelist]
timecol = pd.to_datetime(timecol)
#%%

# If the logger lost the time, 
# if timecol[0].year == 2019:
#     print("Logger seems to have lost the time. Select another file with correct timestamp for the first recording")
#     anchor_file = filedialog.askopenfilename()
#     anchor_timestamp = pd.to_datetime(anchor_file.split('/')[-1].split('-')[-1].split('.')[0], format='%Y%m%dT%H%M%S')
#     anchor_data,rate = aio.load_audio(anchor_file)
#     first_file, _ = aio.load_audio(filelist[0])
    
#     # calculate offset
#     # Compute cross-correlation
#     correlation = correlate(anchor_data[:,0], first_file[:,0])
    
#     # Plot the cross-correlation
#     plt.figure(figsize=(10, 5))
#     plt.plot(correlation)
#     plt.title('Cross-Correlation between Recordings')
#     plt.xlabel('Lag')
#     plt.ylabel('Correlation')
#     plt.savefig('%s\\cross_correlation.png' % (output_path))
#     plt.show(block=False)
#     # Find the index of the peak
#     offset_index = np.argmax(correlation) - (len(first_file[:,0]) - 1)
#     offset_seconds = offset_index/rate
#     first_timestamp = anchor_timestamp + dt.timedelta(seconds = offset_seconds)
#     first_timedelta = first_timestamp - timecol[0]
#     timecol = timecol + first_timedelta
    
#     # Create a DataFrame with the offset information
#     df = pd.DataFrame({'Offset_samples': [offset_index], 'Offset_seconds': [offset_seconds],\
#                        'File': [filelist[0]], 'Anchor': [anchor_file],\
#                            'New_first_timestamp':[first_timestamp]})
#     # Save the DataFrame to a CSV file
#     df.to_csv('%s\\cross_correlation_offset.csv' %(output_path), index=False)
    
#     del anchor_data, first_file, correlation


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
thresh = 0
for i in range(n_channels):
    data[:, i] *= cor_factors[i, 1]
    sd = np.std(data[:, i])
    thresh = max(thresh, sd)
thresh *= 10

# Plot raw data
offset = np.max(abs(data))
plt.figure(figsize=(40, 12))
for i in range(n_channels):
    plt.plot(data[0:int(60 * rate - 1), i] + i * offset, label=str(i + 1))
    plt.hlines(y=thresh + i * offset, xmin=0, xmax=int(60 * rate - 1))
plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
plt.savefig('%s\\%s_one_minute_raw.png' % (output_path, file_set['filename'][0].split('\\')[-1][:-4]))
plt.show(block=False)

# parameters = {
#     'peak_window_us': 1200,
#     'peak_threshold': thresh,
#     'interpolation_factor': 1,
#     'peak_dur_min': 20,
#     'peak_dur_max': 500,
#     'peak_pp_r_min': 0.1,
#     'peak_pp_r_max': 3,
#     'peak_fft_freq_min': 500,
#     'peak_fft_freq_max': 30000
# }

parameters = {'peak_window_us':5000,
      'peak_threshold':0.002,
      'interpolation_factor':1,
      'peak_dur_min':20,
      'peak_dur_max':1000,
      'peak_pp_r_min':0.1,
      'peak_pp_r_max':3,
      'peak_fft_freq_min':150,
      'peak_fft_freq_max':30000}

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
    for i in range(n_channels):
        peaks_temp, _ = find_peaks(data[:, i], height=parameters['peak_threshold'][0])
        peaks_temp = peaks_temp.astype(np.int64)
        
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
            if data[p, i] < np.max(peak):
                continue
            else:
                peaks.append(p)
    
    peaks.sort()
    peaks_unique = np.unique(peaks)
    print("Peaks found: " + str(len(peaks)))
    
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
    
    # Filter through waveforms with FFT, duration, and ratio filter
    filtered_h2t_waveforms, keep_indices, features = filter_waveforms(
        h2t_waveforms, peak_idc, trough_idc, rate,
        dur_min=parameters['peak_dur_min'][0],
        dur_max=parameters['peak_dur_max'][0],
        pp_r_min=parameters['peak_pp_r_min'][0],
        pp_r_max=parameters['peak_pp_r_max'][0],
        fft_freq_min=parameters['peak_fft_freq_min'][0],
        fft_freq_max=parameters['peak_fft_freq_max'][0],
        interp_factor=parameters['interpolation_factor'][0],
        return_features=True, return_params=False
    )
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
        
        # Step 1: Classify as Fish or Noise using the first GMM
        sc = StandardScaler()
        X = sc.fit_transform(filtered_h2t_waveforms)
        
        fish_probs = gmm_fish.score_samples(X)
        noise_probs = gmm_noise.score_samples(X)
        
        fish_pred_mda = (fish_probs > noise_probs).astype(int)
        fish_indices = np.where(fish_pred_mda == 1)[0]
        
        if len(fish_indices) == 0:
            continue
        
        print('EODs after noise filtering with GMM: ' + str(len(fish_indices)))
        
        # Filter out the noise signals
        filtered_h2t_waveforms = filtered_h2t_waveforms[fish_indices]
        filtered_amps = filtered_amps[fish_indices]
        filtered_h2t_amp = filtered_h2t_amp[fish_indices]
        filtered_cor_coeffs = filtered_cor_coeffs[fish_indices]
        filtered_h2t_chan = filtered_h2t_chan[fish_indices]
        filtered_h2t_found = filtered_h2t_found[fish_indices]
        filtered_peak_idc = filtered_peak_idc[fish_indices]
        filtered_trough_idc = filtered_trough_idc[fish_indices]
        filtered_pulse_orientation = filtered_pulse_orientation[fish_indices]
        features = features.iloc[fish_indices].reset_index(drop=True)
        n_eods = len(fish_indices)
        
        # Step 2: Classify Species using the species GMMs or LDAs
        # LDA
        # X = features
        
        # GMM: Standardscaling needs to be done! But GMM works poorer when features are used 
        # X = filtered_h2t_waveforms
        # sc = StandardScaler()
        # X = sc.fit_transform(X)
        
        species_preds = np.zeros(n_eods)
        species_probs = np.zeros((n_eods, len(species_labs['0'])))
        for i in range(len(species_labs['0'])):
            species_probs[:, i] = gmm_species_models[i].score_samples(filtered_h2t_waveforms)
        species_preds = np.argmax(species_probs, axis=1)
        species_preds_labels = [species_labs['0'][pred] for pred in species_preds]
        
        # Control plot
        species_unique = species_labs['0'].unique()
        colors = mpl.colormaps.get_cmap('Set2')
        species_color_map = {species: colors(i / len(species_unique)) for i, species in enumerate(species_unique)}
        
        plt.figure(figsize=(30, 12))
        offset_diff = np.max(filtered_h2t_amp) * 1.5
        
        for i in range(n_channels_diff):
            plt.plot(data_diff[:, i] + i * offset_diff, linewidth=0.5)
        
        for species in species_unique:
            species_idc = np.where(np.array(species_preds_labels) == species)[0]
            for i in range(n_channels_diff):
                # h2t_idc = np.where((filtered_h2t_chan == i) & (np.isin(filtered_peak_idc, species_indices)))[0]
                h2t_idc = np.where(filtered_h2t_chan == i) 
                plot_idc = np.intersect1d(h2t_idc, species_idc)
                plt.plot(filtered_peak_idc[plot_idc], data_diff[filtered_peak_idc[plot_idc], i] + i * offset_diff, 'o', markersize=2, color=species_color_map[species], label=species if i == 0 else "")
                plt.plot(filtered_trough_idc[plot_idc], data_diff[filtered_trough_idc[plot_idc], i] + i * offset_diff, 'o', markersize=2, color=species_color_map[species])
        
        plt.ylim(bottom=None, top=(n_channels_diff - 0.5) * offset_diff)
        plt.title(fname)
        plt.legend(loc='upper right')
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
            'species_pred': species_preds_labels
        })
        eod_table_complete = eod_table.join(features)
        
        eod_table_complete.to_csv('%s\\%s_eod_table.csv' % (output_path, fname[:-4]), index=False)
        
        waveform_table = pd.DataFrame(filtered_h2t_waveforms)
        waveform_table.to_csv('%s\\%s_h2t_waveforms.csv' % (output_path, fname[:-4]), index=False)
        
        gc.collect()
