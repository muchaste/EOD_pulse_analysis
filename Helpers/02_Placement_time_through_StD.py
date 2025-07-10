# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:46:07 2024

@author: ShuttleBox
"""
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import glob
import audioio as aio
import matplotlib.pyplot as plt

#%%
# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title = "Select Folder with Logger Files")

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

#%% Read first Minute of the first few files to find out when the logger was
# placed in the water... (hint: Std. dev. increases in water)
print('Read out SD from first 20 files')

sds = []
n_files = 20
tmin = 0
tmax = 60

print('Plotting std. deviation of the first %s files to identify when logger was placed in water...'%(str(n_files)))

for recording in file_set['filename'][0:n_files-1]:
    print(recording)

    with aio.AudioLoader(recording, 60) as sf:
        rate = sf.samplerate
        data = sf[int(tmin*rate):int(tmax*rate), :]
        # extract number of channels
        n_channels = len(data[0,:])
        # data = np.diff(data, axis=1)
    for channel in range(n_channels):
        sds.append(data[:,channel].std())

plt.figure()
for channel in range(n_channels):
    plt.plot(sds[channel::n_channels])
plt.title('Standard deviations for the first %s files'%(str(n_files)))
plt.savefig('%s\\SDs absolute.png'%(input_path))
plt.show(block=False)

plt.figure()
for channel in range(n_channels):
    sds_relative = [i / j for i, j in zip(sds[channel::n_channels], sds[0::n_channels])]
    plt.plot(sds_relative)
plt.title('Standard deviations for the first %s files Relative to channel 1'%(str(n_files)))
plt.savefig('%s\\SDs relative.png'%(input_path))
plt.show(block=False)

print('Input file number to display filename')
fnum = int(input())
print(file_set['filename'][fnum])
