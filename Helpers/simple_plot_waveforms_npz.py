
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
import sys
# from scipy.signal import find_peaks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulse_functions import (
    load_variable_length_waveforms, 
    save_variable_length_waveforms,
    normalize_waveforms
)


# Pick waveform file
wf_file = filedialog.askopenfilename(
    title="Select File with Waveform Data"
)

eod_table = filedialog.askopenfilename(
    title="Select EOD Table File"
)

base_path = wf_file.split('_concatenated.npz')[0]

waveforms = load_variable_length_waveforms(base_path)
eod_data = pd.read_csv(eod_table)

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

# Plot normalized waveforms
# Slope subtraction: remove linear DC drift across each raw snippet window.
# bg_ratio: boundary slope relative to peak amplitude — per-pulse noise quality metric.
waveforms_detrended = []
bg_ratio_arr = np.zeros(len(waveforms))
for i, wf in enumerate(waveforms):
    slope = np.linspace(float(wf[0]), float(wf[-1]), len(wf))
    wf_d = wf - slope
    waveforms_detrended.append(wf_d)
    peak_amp = np.max(np.abs(wf_d))
    bg_ratio_arr[i] = abs(float(wf[0]) - float(wf[-1])) / peak_amp if peak_amp > 0 else 0.0
eod_data['bg_ratio'] = bg_ratio_arr

# -------------------------------------------------------------------------
# Normalize waveforms: P1 to +1, baseline subtracted, resampled to fixed length
# -------------------------------------------------------------------------
print("\nNormalizing waveforms...")
normalized_waveforms = normalize_waveforms(
    waveforms_detrended,
    snippet_p1_idc=eod_data['snippet_p1_idx'].values,
    snippet_p2_idc=eod_data['snippet_p2_idx'].values,
    method='p1_unity',
    crop_and_interpolate=True,
    crop_factor=10,
    target_length=300
)
normalized_waveforms = np.array(normalized_waveforms)  # shape: (n_pulses, waveform_target_length)


plt.figure(figsize=(10, 6))
for i, waveform in enumerate(normalized_waveforms):
    time = np.arange(0, len(waveform)) / fs
    plt.plot(time, waveform, alpha=0.1, color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Normalized Waveform Overlay')
plt.show()

