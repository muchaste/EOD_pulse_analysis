
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import tkinter as tk
from tkinter import filedialog
import os
import gc
import json
import audioio as aio
# from scipy.signal import find_peaks
from eod_functions_backup import load_variable_length_waveforms, save_variable_length_waveforms


# Pick waveform file
wf_file = filedialog.askopenfilename(
    title="Select File with Waveform Data"
)

base_path = wf_file.split('_concatenated.npz')[0]

waveforms = load_variable_length_waveforms(base_path)

# Plot waveform overlay with time axis and matplotlib
wf_lengths = []
for waveform in waveforms:
    wf_lengths.append(len(waveform))

max_length = max(wf_lengths)

fs = 96000
plt.figure(figsize=(10, 6))
for i, waveform in enumerate(waveforms):
    # pad if necessary
    if len(waveform) < max_length:
        pad_width = max_length - len(waveform)
        waveform = np.pad(waveform, (pad_width // 2, pad_width - pad_width // 2), mode='constant')

    time = np.arange(0, len(waveform)) / fs
    # # center horizontally by zero-crossing between peak and trough
    # center = len(waveform) // 2
    # zero_crossing = np.where(np.diff(np.sign(waveform[center-10:center+10])))[0]
    # if len(zero_crossing) > 0:
    #     center_idx = center + zero_crossing[0] - 10
    plt.plot(time, waveform, alpha=0.1, color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform Overlay')
# plt.legend()
plt.show()