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
# import glob
from Pre_AI_code.R4_eodlogger_functions import filter_waveforms

#%%
plt.ioff()

# Set directories
root = tk.Tk()
root.withdraw()
# input_path = filedialog.askdirectory(title = "Select Folder with Logger Files")


# # List all .wav files
# filelist = glob.glob(input_path+'/*/*.wav', recursive=True)
# filelist.sort()

filepath = filedialog.askopenfilename(title = "Select .wav file")
output_path = filedialog.askdirectory(title = "Select Folder to Store Analysis Results")
cal_file = filedialog.askopenfilename(title = "Select File with Calibration Data")
cor_factors = np.array(pd.read_csv(cal_file))

fname = filepath.split('/')[-1]
logger_id = fname.split('-')[0]
species = "Noise"
cor_factors = np.array(pd.read_csv(cal_file))
print(fname)

#%%
# Load file
data, rate = aio.load_audio(filepath)
n_channels = data.shape[1]

# calibrate with correction factor from .csv
for i in range(n_channels):
    data[:,i] *= cor_factors[i,1]
    
# create diff data
data_diff = np.diff(data, axis = 1)

# Plot diff data
offset = np.max(abs(data_diff))#*1.5
plt.figure(figsize=(40,12))
for i in range(n_channels-1):
    plt.plot(data_diff[:, i] + i*offset, label=str(i+1))

plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
plt.title('Differential data')
plt.show(block=False)

# Use channel that was selected
print('Select diff. channel')
diff_chan = int(input())-1

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
parameters.to_csv('%s\\%s_analysis_parameters.csv'%(output_path, logger_id), index=False )

#%%

data = data[:,diff_chan:diff_chan+2]
n_channels = data.shape[1]
peak_window = int(parameters['peak_window_us'][0]*rate/1e6)

# Extract peaks the same way as in the EOD-extraction
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

# remove duplicates
peaks.sort()
peaks_unique = np.unique(peaks)

# peak_idx_diff = np.diff(peaks)
# peaks have to be at least peak_window//2 samples apart
# peaks_unique = np.array([p_idx for i, p_idx in enumerate(peaks[0:-1]) if peak_idx_diff[i] > peak_window//2 ])
print("Peaks found: " +str(len(peaks)))

if len(peaks_unique) != 0:
    # extract peak snippets
    snippets = []
    for i, p in enumerate(peaks_unique):
        indexer = np.arange(p-peak_window//2, p+peak_window//2)

        # if snippet is too close to beginning or end of recording, pad with zeros
        if np.max(indexer) >= data.shape[0]:
            to_pad = int(np.max(indexer) - data.shape[0] +1)
            peak = np.zeros((len(indexer), n_channels))
            peak[:int(peak_window-to_pad)-1,:] = data[int(np.min(indexer)),:]
            peak = np.diff(peak, axis=1)[:,0]
        elif np.min(indexer) < 0:
            to_pad = int(abs(np.min(indexer)))
            peak = np.zeros((len(indexer), n_channels))
            peak[to_pad:,:] = data[:int(np.max(indexer)+1),:]
            peak = np.diff(peak, axis=1)[:,0]
        else:
            peak = np.diff(data[indexer], axis=1)[:,0]
        
        snippets.append(peak)

    n_snippets = len(snippets)

    # extract head-to-tail waveforms, channels, amplitudes and indices:
    n_samples = snippets[0].shape[0]
    interp_samples = parameters['interpolation_factor'][0] * n_samples
   
    # Preallocate arrays
    h2t_waveforms = np.zeros((n_snippets, interp_samples))
    h2t_amp = np.zeros(n_snippets)
    peak_idc = np.zeros(n_snippets, dtype=int)
    trough_idc = np.zeros(n_snippets, dtype=int)
   
    # Iterate through snippets, extract data
    for i, s in enumerate(snippets):
        # 1. Find peaks and amplitudes across channels
        # s_diff = np.diff(s, axis = 1)[:,0]
        h2t_peak_idx = np.argmax(s)
        h2t_trough_idx = np.argmin(s)
        h2t_amp[i] = abs(s[h2t_peak_idx] - s[h2t_trough_idx])
   
   
        # 4.2 interpolate, rectify and normalize amplitude of h2t waveform
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
        peak_idc[i] = peaks_unique[i] - n_samples // 2 + h2t_peak_idx
        trough_idc[i] = peaks_unique[i] - n_samples // 2 + h2t_trough_idx
   
    gc.collect()
    
    # create differential data (for plotting only)
    data_diff = np.diff(data)
    offset_diff = np.max(h2t_amp)*1.5
    n_channels_diff = data_diff.shape[1]
        
    # # control plot with differential data and peaks/troughs where the h2t waveform was found
    # plt.figure()
    # for i in range(n_channels_diff):
    #     plt.plot(data_diff[:,i]+i*offset_diff, linewidth = 0.5)
    #     plt.plot(peak_idc, data_diff[peak_idc,i]+i*offset_diff, 'o', markersize = 1)
    #     plt.plot(trough_idc, data_diff[trough_idc,i]+i*offset_diff, 'o', markersize = 1)

    # plt.ylim(bottom=None, top=(n_channels_diff)*offset_diff)
    # plt.title(fname)
    # plt.show()
    # plt.savefig(output_path+'/'+fname.split('.')[0]+'.png', dpi=300)
    # plt.close()

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
        

        # compile results and save
        eod_table = pd.DataFrame({'amplitude': filtered_h2t_amp, \
                                  'peak_idx': filtered_peak_idc, \
                                  'trough_idx': filtered_trough_idc, \
                                  'pulse_orientation': filtered_pulse_orientation}
                                 )
        eod_table['species'] = species
        eod_table['fish_id'] = logger_id
        eod_table_complete = eod_table.join(features)

    
        # Control plot
        fig, axs = plt.subplots(1,2, width_ratios=(5,1), figsize=(20,6))
        
        # Iterate over each channel and plot the data
        axs[0].plot(data_diff, linewidth = 0.5)
        axs[0].plot(filtered_peak_idc, data_diff[filtered_peak_idc], 'o', markersize = 1)
        axs[0].plot(filtered_trough_idc, data_diff[filtered_trough_idc], 'o', markersize = 1)

    
        axs[0].set_title(fname)
        # plt.show()
        axs[0].set_xlabel('Sample')
        axs[0].set_ylabel('Voltage')
        for i, wf in enumerate(filtered_h2t_waveforms):
            axs[1].plot(wf, color = 'black', alpha = 0.1)
        
        
        plt.savefig('%s\\%s_%s.png'%(output_path, logger_id, fname[:-4]))
        
        plt.close()
        gc.collect()
        
        aio.write_audio('%s\\%s.wav'%(output_path, fname[:-4]), data, rate)
        eod_table_complete.to_csv('%s\\%s_eod_table.csv'%(output_path, fname[:-4]), index=False )
    
        waveform_table = pd.DataFrame(filtered_h2t_waveforms)
        waveform_table.to_csv('%s\\%s_h2t_waveforms.csv'%(output_path, fname[:-4]), index=False )



