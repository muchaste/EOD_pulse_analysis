"""Load and cluster pulsefish recordings."""

import matplotlib.pyplot as plt
import audioio as aio
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
# import math
import gc
import glob
from Pre_AI_code.R4_eodlogger_functions import filter_waveforms

# repeat = 1

#%%
plt.ioff()

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title = "Select Folder with Logger Files")

# Pick file for parameter estimation
output_path = filedialog.askdirectory(title = "Select Folder to Store Analysis Results")

# List all .wav files
filelist = glob.glob(input_path+'/*/*.wav', recursive=True)
filelist.sort()


# parameters = {'peak_window_us':1200,
#           'peak_threshold':0.02,
#           'interpolation_factor':1,
#           'peak_dur_min':20,
#           'peak_dur_max':500,
#           'peak_pp_r_min':0.1,
#           'peak_pp_r_max':3,
#           'peak_fft_freq_min':500,
#           'peak_fft_freq_max':30000}

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
    else:
        change_params = 1

parameters = {k:[v] for k,v in parameters.items()}  # WORKAROUND
parameters = pd.DataFrame(parameters)
parameters.to_csv('%s\\analysis_parameters.csv'%(output_path), index=False )

#%%
for n, filepath in enumerate(filelist):
    print(str(n) + ' - ' + filepath)
    
print("Set Start File Index")
start_file_idx = int(input())
filelist = filelist[start_file_idx:]
for n, filepath in enumerate(filelist):
    fname = filepath.split('\\')[-1]
    fish_id = fname.split('_')[1]
    species = fname.split('_')[0]
    
    print(str(n) + ' - ' + filepath)
    # Load file
    data, rate = aio.load_audio(filepath)
    peak_window = int(parameters['peak_window_us'][0]*rate/1e6)
   
    data = data[:,0]
    
    # Find peaks
    peaks_found, _ = find_peaks(data, height=parameters['peak_threshold'][0], )
    peaks_found = peaks_found.astype(np.int64)
    
    peaks_found.sort()
    print("Peaks found: " +str(len(peaks_found)))

    if len(peaks_found) != 0:
        # extract peak snippets
        snippets = []
        peaks = []
        for i, p in enumerate(peaks_found):

            indexer = np.arange(p-peak_window//2, p+peak_window//2)
            if np.max(indexer) >= data.shape[0]:
                to_pad = int(np.max(indexer) - data.shape[0] +1)
                peak = np.zeros(len(indexer))
                peak[:int(peak_window-to_pad)] = data[int(np.min(indexer)):]
            elif np.min(indexer) < 0:
                to_pad = int(abs(np.min(indexer)))
                peak = np.zeros(len(indexer))
                peak[to_pad:] = data[:int(np.max(indexer)+1)]
            else:
                peak = data[indexer]
            if data[p] < np.max(peak):
                continue
            else:
                peaks.append(p)
                snippets.append(peak)
            
            
            # indexer = np.arange(p - peak_window // 2, p + peak_window // 2)
            # if np.max(indexer) >= data.shape[0]:
            #     to_pad = int(np.max(indexer) - data.shape[0] + 1)
            #     peak = np.zeros(len(indexer))
            #     peak[:int(peak_window - to_pad) - 1] = data[int(np.min(indexer)):, i]
            # elif np.min(indexer) < 0:
            #     to_pad = int(abs(np.min(indexer)))
            #     peak = np.zeros(len(indexer))
            #     peak[to_pad:] = data[:int(np.max(indexer) + 1), i]
            # else:
            #     peak = data[indexer, i]
            # if data[p, i] < np.max(peak):
            #     continue
            # else:
            #     peaks.append(p)
            
                
        n_snippets = len(snippets)
        n_samples = snippets[0].shape[0]
        interp_samples = parameters['interpolation_factor'][0] * n_samples

        # Preallocate arrays
        h2t_waveforms = np.zeros((n_snippets, interp_samples))
        h2t_amp = np.zeros(n_snippets)
        peak_idc = np.zeros(n_snippets, dtype=int)
        trough_idc = np.zeros(n_snippets, dtype=int)
        # h2t_orientation = np.ones(n_snippets, dtype=int)
    
        # Iterate through snippets, extract data
        for i, s in enumerate(snippets):
            # 1. Find peaks and amplitudes across channels
            h2t_peak_idx = np.argmax(s)
            h2t_trough_idx = np.argmin(s)
            h2t_amp[i] = abs(s[h2t_peak_idx] - s[h2t_trough_idx])

            # 2 interpolate, rectify and normalize amplitude of h2t waveform
            if h2t_trough_idx < h2t_peak_idx:
                # h2t_orientation[i] = -1
                s *= -1
                s = np.roll(s, (h2t_peak_idx-h2t_trough_idx))

            interper = interp1d(np.arange(len(s)), s, kind='quadratic')
            s = interper(np.linspace(0, len(s)-1, interp_samples))
            s = np.roll(s, interp_samples//2-np.argmax(s))
            s /= np.max(np.abs(s))
            h2t_waveforms[i, :] = s

            # Store again the indices of peak and trough
            peak_idc[i] = peaks[i] - n_samples // 2 + h2t_peak_idx
            trough_idc[i] = peaks[i] - n_samples // 2 + h2t_trough_idx

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
        print('EODs: '+str(n_eods))
        
        if n_eods != 0:
            # filter the other variables
            filtered_h2t_amp = h2t_amp[keep_indices]
            filtered_peak_idc = peak_idc[keep_indices]
            filtered_trough_idc = trough_idc[keep_indices]
            filtered_pulse_orientation = np.array(['HP']*n_eods)
            filtered_pulse_orientation[np.where(filtered_trough_idc < filtered_peak_idc)[0]] = 'HN'
            
           
            eod_table = pd.DataFrame({'amplitude': filtered_h2t_amp, \
                                      'peak_idx': filtered_peak_idc, \
                                      'trough_idx': filtered_trough_idc, \
                                      'pulse_orientation': filtered_pulse_orientation}
                                     )
            eod_table['fish_id'] = fish_id
            eod_table_complete = eod_table.join(features)
    
        
                
            # print("Input Species")
            # species = input()
            eod_table_complete['species'] = species
                
            time = np.arange(0, len(data)) / rate
            
            # Control plot
            fig, axs = plt.subplots(1,2, width_ratios=(5,1), figsize=(20,6))
            
            # Iterate over each channel and plot the data
            axs[0].plot(time, data, linewidth = 0.5)
            axs[0].plot(time[filtered_peak_idc], data[filtered_peak_idc], 'o', markersize = 1)
            axs[0].plot(time[filtered_trough_idc], data[filtered_trough_idc], 'o', markersize = 1)
            
            axs[0].set_title(fname)
            # plt.show()
            axs[0].set_xlabel('Sample')
            axs[0].set_ylabel('Voltage')
            for i, wf in enumerate(filtered_h2t_waveforms):
                axs[1].plot(wf, color = 'black', alpha = 0.1)
           
            # plt.show(block=False)
        
        
            plt.savefig('%s\\%s.png'%(output_path, fname[:-4]))
            
            # aio.write_audio('%s\\%s.wav'%(output_path, fname[:-4]), data, rate)
            eod_table_complete.to_csv('%s\\%s_eod_table.csv'%(output_path, fname[:-4]), index=False )
            waveform_table = pd.DataFrame(filtered_h2t_waveforms)
            waveform_table.to_csv('%s\\%s_h2t_waveforms.csv'%(output_path, fname[:-4]), index=False )
            

    plt.close()
    gc.collect()