"""Load and cluster pulsefish recordings."""

import matplotlib.pyplot as plt
import audioio as aio
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
from Pre_AI_code.R4_eodlogger_functions import filter_waveforms


#%%
plt.ioff()

# Set directories
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename(title = "Select Logger Files")
output_path = filedialog.askdirectory(title = "Select Folder to Store Analysis Results")
cal_file = filedialog.askopenfilename(title="Select File with Calibration Data")
cor_factors = np.array(pd.read_csv(cal_file))

counter = 1

parameters = {'peak_window_us':5000,
      'peak_threshold':0.002,
      'interpolation_factor':1,
      'peak_dur_min':20,
      'peak_dur_max':1000,
      'peak_pp_r_min':0.1,
      'peak_pp_r_max':3,
      'peak_fft_freq_min':150,
      'peak_fft_freq_max':30000}



fname = filepath.split('/')[-1]
fish_id = fname.split('-')[0]

print(filepath)
# Load file
data, rate = aio.load_audio(filepath)
n_channels = data.shape[1]

for i in range(n_channels):
    data[:, i] *= cor_factors[i, 1]

# Create differential data
data_diff = np.diff(data)
offset_diff = np.max(abs(data_diff)) * 1.5
n_channels_diff = data_diff.shape[1]
 
# species = fish_id[0:2]
while True:
    change_params = 1   

    while change_params:
        plt.close()
        gc.collect()
        
        # Plot data
        plt.figure(figsize=(40, 12))
        for i in range(n_channels_diff):
            plt.plot(data_diff[:,i] + i * offset_diff, label=str(i + 1))
            plt.hlines(y=parameters['peak_threshold'] + i * offset_diff, xmin=0, xmax=data_diff.shape[0]-1)
        plt.legend(loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('Voltage')
        plt.show(block=False)
        
        print(parameters)
        print("change parameters? (1/0)")
        change_params = int(input())
        
        if change_params:
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
    
    print('Choose channel (1-7):')
    chan_select = int(input())-1        
    
    plt.close()
    gc.collect()

    parameters_df = {k:[v] for k,v in parameters.items()}  # WORKAROUND
    parameters_df = pd.DataFrame(parameters_df)
    parameters_df.to_csv('%s\\analysis_parameters%s.csv'%(output_path, counter), index=False )
    
    peak_window = int(parameters['peak_window_us']*rate/1e6)
    
    # only use selected channel 
    dat = data_diff[:,chan_select]
    
    
    time = np.arange(0, len(dat)) / rate
    # Find peaks
    peaks_found, _ = find_peaks(dat, height=parameters['peak_threshold'], )
    peaks_found = peaks_found.astype(np.int64)
    
    peaks_found.sort()
    print("Peaks found: " +str(len(peaks_found)))

    if len(peaks_found) != 0:
        # extract peak snippets
        snippets = []
        peaks = []
        for i, p in enumerate(peaks_found):

            indexer = np.arange(p-peak_window//2, p+peak_window//2)
            if np.max(indexer) >= dat.shape[0]:
                to_pad = int(np.max(indexer) - dat.shape[0] +1)
                peak = np.zeros(len(indexer))
                peak[:int(peak_window-to_pad)-1] = dat[int(np.min(indexer)):]
            elif np.min(indexer) < 0:
                to_pad = int(abs(np.min(indexer)))
                peak = np.zeros(len(indexer))
                peak[to_pad:] = dat[:int(np.max(indexer)+1)]
            else:
                peak = dat[indexer]
            if dat[p] < np.max(peak):
                continue
            else:
                peaks.append(p)
                snippets.append(peak)
            
                
        n_snippets = len(snippets)
        n_samples = snippets[0].shape[0]
        interp_samples = parameters['interpolation_factor'] * n_samples

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
                                                            dur_min=parameters['peak_dur_min'],
                                                            dur_max = parameters['peak_dur_max'],
                                                            pp_r_min = parameters['peak_pp_r_min'],
                                                            pp_r_max=parameters['peak_pp_r_max'],
                                                            fft_freq_min=parameters['peak_fft_freq_min'],
                                                            fft_freq_max = parameters['peak_fft_freq_max'],
                                                            interp_factor=parameters['interpolation_factor'],
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
    
        
            # Control plot
            fig, axs = plt.subplots(1,2, width_ratios=(5,1), figsize=(20,6))
            
            # Iterate over each channel and plot the data
            axs[0].plot(time, dat, linewidth = 0.5)
            axs[0].plot(time[filtered_peak_idc], dat[filtered_peak_idc], 'o', markersize = 1)
            axs[0].plot(time[filtered_trough_idc], dat[filtered_trough_idc], 'o', markersize = 1)
            
            axs[0].set_title(fname)
            # plt.show()
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Voltage')
            for i, wf in enumerate(filtered_h2t_waveforms):
                axs[1].plot(wf, color = 'black', alpha = 0.1)
           
            plt.show(block=False)
            
            # plt.figure()
            # plt.plot(data)
            # plt.title(fish_id + '_' + fname)
            # plt.show(block=False)
            
            
            # Define Start and Stop and Enter Species Information
            print("Input Start Time")
            start_s = int(float(input())*rate)
            print("Input End Time (0 for end of file)")
            stop_s = int(float(input())*rate)
            
            # if start_s != 0 and stop_s == 0:
            #     keep_idc = np.where(filtered_peak_idc >= start_s)
            #     eod_table_complete = eod_table_complete.iloc[keep_idc]
            #     filtered_h2t_waveforms = filtered_h2t_waveforms[keep_idc]
            #     data = data[start_s:]
            # elif start_s != 0 and stop_s != 0:
            #     keep_idc = np.where((filtered_peak_idc >= start_s) & (filtered_peak_idc < stop_s))
            #     eod_table_complete = eod_table_complete.iloc[keep_idc]
            #     filtered_h2t_waveforms = filtered_h2t_waveforms[keep_idc]
            #     data = data[start_s:stop_s+1]
            # elif start_s == 0 and stop_s != 0:
            #     keep_idc = np.where(filtered_peak_idc < stop_s)
            #     eod_table_complete = eod_table_complete.iloc[keep_idc]
            #     filtered_h2t_waveforms = filtered_h2t_waveforms[keep_idc]
            #     data = dat[start_s:stop_s+1]
            
            keep_idc = None
            if start_s != 0 and stop_s == 0:
                keep_idc = np.where(filtered_peak_idc >= start_s)
                dat = dat[start_s:]
                time = time[start_s:]
            elif start_s != 0 and stop_s != 0:
                keep_idc = np.where((filtered_peak_idc >= start_s) & (filtered_peak_idc < stop_s))
                dat = dat[start_s:stop_s+1]
                time = time[start_s:stop_s+1]

            elif start_s == 0 and stop_s != 0:
                keep_idc = np.where(filtered_peak_idc < stop_s)
                dat = dat[:stop_s+1]
                time = time[:stop_s+1]
            
            if keep_idc is not None:
                eod_table_complete = eod_table_complete.iloc[keep_idc]
                filtered_h2t_waveforms = filtered_h2t_waveforms[keep_idc]
                filtered_peak_idc = filtered_peak_idc[keep_idc] - start_s
                filtered_trough_idc = filtered_trough_idc[keep_idc] - start_s
                
            print("Input Species")
            species = input()
            eod_table_complete['species'] = species
            
            if keep_idc is not None:
                plt.close()
                
                time = np.arange(0, len(dat)) / rate
                
                # Control plot
                fig, axs = plt.subplots(1,2, width_ratios=(5,1), figsize=(20,6))
                
                # Iterate over each channel and plot the dat
                axs[0].plot(time, dat, linewidth = 0.5)
                axs[0].plot(time[filtered_peak_idc], dat[filtered_peak_idc], 'o', markersize = 1)
                axs[0].plot(time[filtered_trough_idc], dat[filtered_trough_idc], 'o', markersize = 1)
                
                axs[0].set_title(fish_id + '_' + fname)
                # plt.show()
                axs[0].set_xlabel('Sample')
                axs[0].set_ylabel('Voltage')
                for i, wf in enumerate(filtered_h2t_waveforms):
                    axs[1].plot(wf, color = 'black', alpha = 0.1)
                    
               
                plt.show(block=False)
            
            
            plt.savefig('%s\\%s_%s.png'%(output_path, species, fname[:-4]))
            aio.write_audio('%s\\%s_%s.wav'%(output_path, species, fname[:-4]), dat, rate)
            
            eod_table_complete.to_csv('%s\\%s_%s_eod_table.csv'%(output_path, species, fname[:-4]), index=False )
            waveform_table = pd.DataFrame(filtered_h2t_waveforms)
            waveform_table.to_csv('%s\\%s_%s_h2t_waveforms.csv'%(output_path, species, fname[:-4]), index=False )
            
    print("Repeat? (1/0)")
    repeat = int(input())
    plt.close()
    gc.collect()
    counter+=1
    if repeat == 0:
        break
   
            
            
