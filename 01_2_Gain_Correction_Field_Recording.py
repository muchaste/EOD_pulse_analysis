# -*- coding: utf-8 -*-
"""
Calibrate EOD logger 4.x recordings based on rms noise level during field recordings. Needs "empty" recordings without EODs.
"""
import matplotlib.pyplot as plt
import audioio as aio
# from audioio import load_audio
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
import glob

# Pick a folder
root = tk.Tk()
root.withdraw()
plt.ioff()


# Pick multiple files
file_paths = filedialog.askopenfilenames(title = "Select Audio Files", filetypes = (("WAV files", "*.wav"), ("All files", "*.*")))
file_list = list(file_paths)
plot_data = False
output_path = filedialog.askdirectory(title = "Select Folder to Store Analysis Results")


# starting with the 1st file, take every 7th file (10 min per file, 1h intervals)
# folder_path = filedialog.askdirectory(title = "Select Folder Containing Audio Files")
# list all .wav files in the folder
# file_list = glob.glob(folder_path + "/*.wav")
# file_list = file_list[::7]
logger_id = file_list[0].split('/')[-1].split('_')[0]
print("Logger ID: %s"%(logger_id))

# The correction factors from all files should be stored with an index and name of the file they belong to
cf_all_files = []

for fidx, fname in enumerate(file_list):
    print("Processing file %d of %d: %s"%(fidx+1, len(file_list), fname))
    data, rate = aio.load_audio(fname)
    channels = len(data[0,:])
    file_id = fname.split('/')[-1].split('.')[0]   
    
    rms_stim = []

    for i in range(channels):
        # rms = np.std(data[:,i])*np.sqrt(2)
        rms_stim.append(np.std(data[:, i]) * np.sqrt(2))

    rms_stim = np.array(rms_stim)
    rms_stim /= np.max(abs(rms_stim))

    cor_factors = 1 / rms_stim
    cor_factors /= np.max(cor_factors)

    cf_all_files.append(cor_factors)

    if plot_data:
        data_cal = np.array(data)
        for i in range(channels):
            data_cal[:,i] *= cor_factors[i]

        offset = 1.5*np.max(abs(data[:,:]))
        offset_cal = 1.5*np.max(abs(data_cal[:,:]))

        # Plot only every nth sample for large datasets to save memory
        step = max(1, len(data[:,0]) // 10000000)  # Limit to ~10 mio points per channel
        x_coords = np.arange(0, len(data), step)

        # create figure with two subplots and size 10x6 inches
        fig, ax = plt.subplots(2,1, figsize=(10,6))
        for i in range(channels):
            ax[0].plot(x_coords, data[::step,i]+i*offset, linewidth=.5)
            ax[1].plot(x_coords, data_cal[::step,i]+i*offset_cal, linewidth=.5)

        ax[1].set_xlabel('Time (s)')
        ax[0].set_ylabel('Voltage Raw (V)')
        ax[1].set_ylabel('Voltage Calibrated (V)')
        ax[0].set_title(fname.split('/')[-1])
        plt.savefig('%s\\%s_recording.png'%(output_path, file_id))
        plt.show(block=False)

        # cf_dict = {'channel': np.arange(1,channels+1), 'cor_factor': cor_factors}
        # cf_df = pd.DataFrame(cf_dict)
        # cf_df.to_csv('%s\\%s_correction_factors.csv'%(output_path, file_id), index=False)

    gc.collect()

# Flatten the list of correction factors and save to a CSV file
cf_all_files = np.array(cf_all_files)
cf_df = pd.DataFrame(cf_all_files)

# Add column names
col_names = ['ch_%d'%(i+1) for i in range(channels)]
cf_df.columns = col_names

# Add file names as index
file_ids = [fname.split('/')[-1].split('.')[0] for fname in file_list]
cf_df['file_id'] = file_ids

cf_df.to_csv('%s\\%s_correction_factors_all_files.csv'%(output_path, logger_id), index=False)

# Calculate medians per channel and save to a separate CSV file
medians = np.median(cf_all_files, axis=0)
med_df = pd.DataFrame({'channel': np.arange(1,channels+1), 'median_correction_factor': medians})
med_df.to_csv('%s\\%s_correction_factors_median_per_channel.csv'%(output_path, logger_id), index=False)

# Plot all correction factors and median per channel
x_vals = np.arange(1, len(file_list)+1)
plt.figure(figsize=(10,6))
for i in range(channels):
    plt.plot(x_vals, cf_all_files[:,i], marker='o', linestyle='-', label='Channel %d'%(i+1))
    # plot median of channel as dashed horizontal line
    plt.hlines(np.median(cf_all_files[:,i]), xmin=1, xmax=len(file_list), colors='k', linestyles='--', linewidth=2)
    # add label with numerical value of median at the end of the line
    plt.text(len(file_list)+0.1, np.median(cf_all_files[:,i]), '%.2f'%(np.median(cf_all_files[:,i])), fontsize=10, verticalalignment='center')

# plt.plot(np.arange(1,channels+1), np.median(cf_all_files, axis=0), marker='s', color='k', linestyle='-', linewidth=2, markersize=10, label='Median')
plt.xlabel('Channel')
plt.ylabel('Correction Factor')
plt.title('Correction Factors per Channel and File')
plt.legend()
plt.show(block=False)
plt.savefig('%s\\%s_correction_factors_all_files.png'%(output_path, logger_id))

# # Extra plot
# offset = 1.5*np.max(abs(data[:,:]))

# plt.figure()
# for i in range(channels):
#     plt.plot(xvals, data[:,i]+i*offset, linewidth = 1) # eods_float, linewidth = 1)
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.title('Raw Data')
# plt.show(block=False)


# plt.figure()
# for i in range(channels):
#     plt.plot(xvals, data[:,i])
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.title('Raw Data')
# plt.show(block=False)

# plt.figure()
# for i in range(channels):
#     plt.plot(xvals, data_cal[:,i])
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (V)')
# plt.title('Calibrated Data')
# plt.show()
