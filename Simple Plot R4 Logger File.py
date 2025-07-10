# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:19:21 2023

@author: Stefan Mucha
"""

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
# import wave
# from EOD_analysis_functions import load_wav, se_to_diff, calibrate_amps
# import pandas as pd
import audioio as aio


# Pick a file
root = tk.Tk()
root.withdraw()
plt.ioff()
fname = filedialog.askopenfilename()
output_path = '.'.join(fname.split('.')[:-1]) + '.csv'

# v_range = 3.3                           # input voltage range (absolute)
res = 16                                # bit resolution of recording
# fs = 96000                             # recording sample rate
print("No of channels to plot (0 reads all channels): ")
to_plot = int(input())
print("Duration to read (s, 0 reads the whole file): ")
to_read = int(input())
calibrate = False
one_plot = True
plot_spec = True
plot_diff = False

if to_read == 0:
    eod_frame, fs = aio.load_audio(fname)
else:
    with aio.AudioLoader(fname, to_read) as sf:
        fs = sf.rate
        eod_frame = sf[:int(to_read * fs), :]

n_chan = eod_frame.shape[1]
if to_plot == 0:
    to_plot = n_chan

# eod_frame, n_chan = load_wav(fname, fs, to_read*fs)
xvals = np.arange(0,(len(eod_frame[:,0])/fs),1/fs)


# if calibrate:
#     # Pick file for parameter estimation
#     calib_file = filedialog.askopenfilename(title = "Select File with Calibration Data")
#     calibration_data = pd.read_csv(calib_file)
#     calibration_channel_amps = np.array(calibration_data['mean_amp'][0::n_chan+1])
#     eod_frame = calibrate_amps(eod_frame, calibration_channel_amps,
#                                gain=10, cal_gain=10, cal_amp=1)


offset = 1*np.max(abs(eod_frame))

if one_plot:
    plt.figure()
    for i in range(to_plot):
        plt.plot(xvals, eod_frame[:, i]+i*offset, linewidth = 1) # eods_float, linewidth = 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Raw Data')
    plt.show()

else:
    for i in range(to_plot):
        plt.figure()
        plt.plot(xvals,  eod_frame[:,i], linewidth = 1)# eods_float, linewidth = 1)#
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.show()


# Spectrogram?
if plot_spec:
    for i in range(to_plot):
        plt.figure()
        plt.specgram(eod_frame[:,i], NFFT = 2**13, Fs = fs)
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.ylim(1,2000)
        plt.show()

# Differential Data?
if plot_diff:
    eod_frame_diff = np.diff(eod_frame)
    n_chan_diff = n_chan-1
    # eod_frame_diff, n_chan_diff = se_to_diff(eod_frame, paired=False)
    offset = 1.5*np.max(abs(eod_frame_diff))
    plt.figure()
    for i in range(n_chan_diff):
        plt.plot(xvals, eod_frame_diff[:,i]+i*offset, linewidth = 1) # eods_float, linewidth = 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Differential Raw Data')
    plt.show()
