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
from R4_eodlogger_functions import analyze_r4_snippets, filter_waveforms
import joblib

#%%
plt.ioff()

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title = "Select Folder with Logger Files")
output_path = filedialog.askdirectory(title = "Select Folder to Store Analysis Results")

# Pick file for amplitude calibration
cal_file = filedialog.askopenfilename(title = "Select File with Calibration Data")
cor_factors = np.array(pd.read_csv(cal_file))

# List all .wav files
filelist = glob.glob(input_path+'/*.wav', recursive=True)

# Sort filelist by time
timecol = []
for j, fname in enumerate(filelist):
    time_str = fname.split('-')[1][0:-4]
    time_start = pd.to_datetime(time_str, format='%Y%m%dT%H%M%S')
    timecol.append(time_start)

file_set = pd.DataFrame({'timestamp':timecol, 'filename':filelist})
file_set = file_set.sort_values(by = ['timestamp'], ignore_index=True)


#%% Find peaks and extract head-to-tail waveform

# Load 60 sec of first file
tmin = 0
tmax = 60
with aio.AudioLoader(file_set['filename'][0], 60) as sf:
    rate = sf.rate
    data = sf[int(tmin*rate):int(tmax*rate), :]
n_channels = data.shape[1]

# calibrate with correction factor from .csv
cor_factors = np.array(pd.read_csv(cal_file))
thresh = 0
for i in range(n_channels):
    data[:,i] *= cor_factors[i,1]
    sd = np.std(data[:,i])
    print(sd)
    if sd > thresh:
        thresh = sd

thresh *= 10

# Plot raw data
offset = np.max(abs(data))#*1.5
plt.figure(figsize=(40,12))
for i in range(n_channels):
    plt.plot(data[0:int(60*rate-1), i] + i*offset, label=str(i+1))
    plt.hlines(y = thresh + i*offset, xmin = 0, xmax = int(60*rate-1))

plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
plt.savefig('%s\\%s_one_minute_raw.png'%(output_path, file_set['filename'][0].split('\\')[-1][:-4]))
plt.show(block=False)

parameters = {'peak_window_us':1200,
          'peak_threshold':thresh,
          'interpolation_factor':1,
          'peak_dur_min':20,
          'peak_dur_max':500,
          'peak_pp_r_min':0.1,
          'peak_pp_r_max':3,
          'peak_fft_freq_min':500,
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
    else:
        change_params = 1

parameters = {k:[v] for k,v in parameters.items()}  # WORKAROUND
parameters = pd.DataFrame(parameters)
parameters.to_csv('%s\\analysis_parameters.csv'%(output_path), index=False )

peak_window = int(parameters['peak_window_us'][0]*rate/1e6)

#%%
for n, filepath in enumerate(file_set['filename']):

    fname = filepath.split('\\')[-1]
    print(fname)
    # Load file
    data, rate = aio.load_audio(filepath)
    n_channels = data.shape[1]

    # calibrate with correction factor from .csv
    for i in range(n_channels):
        data[:,i] *= cor_factors[i,1]

    # # detrend
    # for i in range(n_channels):
    #     data[:,i] = detrend(data[:,i])

    # determine threshold for peak detection
    # peak_threshold = np.sqrt(np.std(data))/5
    # print("Peak threshold: " + str(peak_threshold))

    # Find peaks in all channels
    peaks = []
    for i in range(n_channels):
        peaks_temp, _ = find_peaks(data[:,i], height=parameters['peak_threshold'][0])
        peaks_temp = peaks_temp.astype(np.int64)
        
        # For each channel: use only the absolute peaks and troughs per peak window
        for j,p in enumerate(peaks_temp):
            indexer = np.arange(p-peak_window//2, p+peak_window//2)
            if np.max(indexer) >= data.shape[0]:
                to_pad = int(np.max(indexer) - data.shape[0] +1)
                peak = np.zeros(len(indexer))
                peak[:int(peak_window-to_pad)-1] = data[int(np.min(indexer)):, i]
            elif np.min(indexer) < 0:
                to_pad = int(abs(np.min(indexer)))
                peak = np.zeros(len(indexer))
                peak[to_pad:] = data[:int(np.max(indexer)+1),i]
            else:
                peak = data[indexer,i]
            if data[p,i] < np.max(peak):
                continue
            else:
                peaks.append(p)
            
    # peaks = np.concatenate((peaks, peaks_found), axis = None)
    # peaks = peaks.astype(np.int64)

    # remove duplicates
    peaks.sort()
    peaks_unique = np.unique(peaks)
    
    # peak_idx_diff = np.diff(peaks)
    # peaks have to be at least peak_window//2 samples apart
    # peaks_unique = np.array([p_idx for i, p_idx in enumerate(peaks[0:-1]) if peak_idx_diff[i] > peak_window//2 ])
    print("Peaks found: " +str(len(peaks)))
    
    if len(peaks_unique) == 0:
        continue
    else:
        # extract peak snippets
        snippets = []
        for i, p in enumerate(peaks_unique):
            indexer = np.arange(p-peak_window//2, p+peak_window//2)

            # if snippet is too close to beginning or end of recording, pad with zeros
            if np.max(indexer) >= data.shape[0]:
                to_pad = int(np.max(indexer) - data.shape[0] +1)
                peak = np.zeros((len(indexer), n_channels))
                peak[:int(peak_window-to_pad)-1,:] = data[int(np.min(indexer)),:]
                snippets.append(peak)
            elif np.min(indexer) < 0:
                to_pad = int(abs(np.min(indexer)))
                peak = np.zeros((len(indexer), n_channels))
                peak[to_pad:,:] = data[:int(np.max(indexer)+1),:]
                snippets.append(peak)
            else:
                snippets.append(data[indexer])
    
        n_snippets = len(snippets)
    
        # extract head-to-tail waveforms, channels, amplitudes and indices:
        h2t_waveforms, amps, h2t_amp, cor_coeffs, h2t_chan, h2t_found, peak_idc, trough_idc = analyze_r4_snippets(snippets, peaks_unique, parameters['interpolation_factor'][0])
    
        # create differential data (for plotting only)
        data_diff = np.diff(data)
        offset_diff = np.max(h2t_amp)*1.5
        n_channels_diff = data_diff.shape[1]
        
        # control plot with differential data and peaks/troughs where the h2t waveform was found
        # plt.figure()
        # for i in range(n_channels-1):
        #     h2t_idc = np.where(h2t_chan == i)
        #     plt.plot(data_diff[:,i]+i*offset_diff, linewidth = 0.5)
        #     plt.plot(peak_idc[h2t_idc], data_diff[peak_idc[h2t_idc],i]+i*offset_diff, 'o', markersize = 1)
        #     plt.plot(trough_idc[h2t_idc], data_diff[trough_idc[h2t_idc],i]+i*offset_diff, 'o', markersize = 1)
    
        # plt.ylim(bottom=None, top=(n_channels_diff-1)*offset_diff)
        # plt.title(fname)
        # plt.show()
        # plt.savefig(output_path+'/'+fname.split('.')[0]+'.png', dpi=300)
        # plt.close()
    
        gc.collect()
    
        # Filter through waveforms with FFT, duration and ratio filter
        filtered_h2t_waveforms, keep_indices, features  = filter_waveforms(h2t_waveforms, peak_idc, trough_idc, rate,
                                                            dur_min=parameters['peak_dur_min'][0],
                                                            dur_max = parameters['peak_dur_max'][0],
                                                            pp_r_min = parameters['peak_pp_r_min'][0],
                                                            pp_r_max=parameters['peak_pp_r_max'][0],
                                                            fft_freq_min=parameters['peak_fft_freq_min'][0],
                                                            fft_freq_max = parameters['peak_fft_freq_max'][0],
                                                            interp_factor=parameters['interpolation_factor'][0],
                                                            return_features=True, return_params=False)
        n_eods = filtered_h2t_waveforms.shape[0]
        print('EODs after freq/dur/ratio filter: '+str(n_eods))
        
        if n_eods != 0:
            

            # filter the other variables
            filtered_amps = amps[keep_indices]
            filtered_h2t_amp = h2t_amp[keep_indices]
            filtered_cor_coeffs = cor_coeffs[keep_indices]
            filtered_h2t_chan = h2t_chan[keep_indices]
            filtered_h2t_found = h2t_found[keep_indices]
            filtered_peak_idc = peak_idc[keep_indices]
            filtered_trough_idc = trough_idc[keep_indices]
            filtered_pulse_orientation = np.array(['HP']*n_eods)
            filtered_pulse_orientation[np.where(filtered_trough_idc < filtered_peak_idc)[0]] = 'HN'
        
            # del data
        
            # Control plot
            offset_diff = np.max(filtered_h2t_amp)*1.5
    
            plt.figure(figsize=(100,12))
            # Iterate over each channel and plot the data
            for i in range(n_channels_diff):
                h2t_idc = np.where(filtered_h2t_chan == i)[0]  # Filter h2t_chan to get relevant indices for channel i
                plt.plot(data_diff[:, i] + i * offset_diff, linewidth = 0.5, label=str(i+1))
                plt.plot(filtered_peak_idc[h2t_idc], data_diff[filtered_peak_idc[h2t_idc], i] + i * offset_diff, 'o', markersize = 1)
                plt.plot(filtered_trough_idc[h2t_idc], data_diff[filtered_trough_idc[h2t_idc], i] + i * offset_diff, 'o', markersize = 1)
        
        
            plt.ylim(bottom=None, top=(n_channels_diff-0.5)*offset_diff)
            plt.title(fname)
            # plt.show()
            plt.legend(loc='upper right')
            plt.xlabel('Sample')
            plt.ylabel('Voltage')
            plt.savefig('%s\\%s.png'%(output_path, fname[:-4]))
        
            # plt.savefig(output_path+'/'+fname.split('.')[0]+'.png', dpi=300)
            plt.close()
            gc.collect()
        
            # compile results and save
            filtered_h2t_timestamps = []
            for i in range(n_eods):
                filtered_h2t_timestamps.append(file_set['timestamp'][n] + dt.timedelta(seconds=filtered_peak_idc[i]/rate))
        
            eod_table = pd.DataFrame({'timestamp':filtered_h2t_timestamps, \
                                      'channel':filtered_h2t_chan, \
                                      'amplitude': filtered_h2t_amp, \
                                      'peak_idx': filtered_peak_idc, \
                                      'trough_idx': filtered_trough_idc, \
                                      'pulse_orientation': filtered_pulse_orientation,\
                                      'h2t_indicator': filtered_h2t_found})
            eod_table_complete = eod_table.join(features)
        
            eod_table_complete.to_csv('%s\\%s_eod_table.csv'%(output_path, fname[:-4]), index=False )
        
            waveform_table = pd.DataFrame(filtered_h2t_waveforms)
            waveform_table.to_csv('%s\\%s_h2t_waveforms.csv'%(output_path, fname[:-4]), index=False )