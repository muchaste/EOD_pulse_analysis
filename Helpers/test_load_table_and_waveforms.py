# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 16:49:00 2025

@author: ShuttleBox
"""

from pulse_functions import load_fixed_length_waveforms
import pandas as pd
import matplotlib.pyplot as plt
import audioio as aio
import numpy as np

dat = pd.read_csv('E:\\Stefan Mucha\\pulse_extraction_tests\\L2_20231011\\events_merged_across\\L2-20231011T191303_event_10_20231011T191831.csv')

wfs = load_fixed_length_waveforms('E:\\Stefan Mucha\\pulse_extraction_tests\\L2_20231011\\events_merged_across\\L2-20231011T191303_event_10_20231011T191831_waveforms.npz')

audio_dat, rate = aio.load_audio('E:\\Stefan Mucha\\pulse_extraction_tests\\L2_20231011\\events_merged\\L2-20231011T191303_event_10_20231011T191831.wav')

audio_dat_diff = np.diff(audio_dat)


# Plot raw data
offset = np.max(abs(audio_dat_diff))
plt.figure()
for ch in range(audio_dat_diff.shape[1]):
    plt.plot(audio_dat_diff[:,ch]+offset*ch)
    plt.plot(dat['p1_idx'][dat['eod_channel'] == ch], audio_dat_diff[dat['p1_idx'][dat['eod_channel'] == ch],ch]+offset*ch, 'o')
    plt.plot(dat['p2_idx'][dat['eod_channel'] == ch], audio_dat_diff[dat['p2_idx'][dat['eod_channel'] == ch],ch]+offset*ch, 'o')
    
# Test validation: re-extract a waveform and plot it together with waveform that was saved
wf_idx = 14
wf_len = len(wfs[0])
wf_snippet = audio_dat_diff[dat['midpoint_idx'][wf_idx]-(wf_len//2):dat['midpoint_idx'][wf_idx]+(wf_len//2), dat['eod_channel'][wf_idx]]

cors = []
for wf in wfs:
    cors.append(np.corrcoef(wf_snippet, wf)[0,1])
    
plt.figure()
plt.plot(cors)

plt.figure()
plt.plot(wfs[wf_idx])
plt.plot(wf_snippet)



# Double test: do the same for all wfs
all_wfs_cors = []
plt.figure()
for i,wf in enumerate(wfs[:-1]):
    wf_len = len(wf)
    wf_snippet = audio_dat_diff[dat['midpoint_idx'][i]-(wf_len//2):dat['midpoint_idx'][i]+(wf_len//2), dat['eod_channel'][i]]
    all_wfs_cors.append(np.corrcoef(wf_snippet, wf)[0,1])
    plt.plot(wf_snippet - wf)
    
plt.figure()
plt.plot(all_wfs_cors)


# Plot waveforms
plt.figure()
for wf in wfs[14:20]:
    plt.plot(wf)
    
    

# Old extraction with variable lengths: check wf lengths from table against actual length
l_ratios = []

for i, wf in enumerate(wfs):
    l_ratios.append(dat['waveform_length'][i] // len(wf))
    

plt.figure()
plt.plot(l_ratios)
