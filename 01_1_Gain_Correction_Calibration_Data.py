# -*- coding: utf-8 -*-
"""
Calibrate EOD logger 4.x recordings with calibration recordings (1 kHz sine wave)
"""
import matplotlib.pyplot as plt
import audioio as aio
# from audioio import load_audio
from scipy import signal
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import sys
# from scipy.signal import detrend
# from EOD_analysis_functions import extract_eod, load_wav
# import gc
# import re

def find_sine_wave_onset_offset(audio_data, target_freq=1000, sample_rate=96000, bandwidth=50,
                                threshold=0.3):
    """
    Extracts the onset and offset of a target sine wave from a given audio signal.
    
    Parameters
    ----------
    audio_data: array-like
        The audio data to be analyzed.
    sample_rate: int
        The sample rate of the audio recording. Default is 96000 Hz.
    target_freq: float
        The frequency of the sine wave to detect. Default is 1000 Hz.
    bandwidth: float
        The bandwidth around the target frequency to consider for filtering.
        
    Returns
    -------
    onset: float
        The onset time of the sine wave in seconds.
    offset: float
        The offset time of the sine wave in seconds.
    """
    
    # Design a bandpass filter to isolate the target frequency
    nyquist = 0.5 * sample_rate
    low = (target_freq - bandwidth) / nyquist
    high = (target_freq + bandwidth) / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_audio = signal.lfilter(b, a, audio_data)
    
    # Compute the envelope of the filtered signal
    envelope = np.abs(signal.hilbert(filtered_audio))
    
    # Normalize the envelope
    envelope = envelope / np.max(envelope)
    
    # Find the points where the envelope exceeds a threshold
    # threshold = 0.1
    above_threshold = envelope > threshold
    
    # Find the onset and offset indices
    onset_idx = np.argmax(above_threshold)
    offset_idx = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])
    
    # # Convert indices to time
    # onset_time = onset_idx / sample_rate
    # offset_time = offset_idx / sample_rate
    
    return onset_idx, offset_idx

#%%
def main():
    if len(sys.argv) > 1:
        amp_threshold = float(sys.argv[1])
    else:
        amp_threshold = 0.5
    # Pick a file
    root = tk.Tk()
    root.withdraw()
    plt.ioff()
    fname = filedialog.askopenfilename()
    output_path = filedialog.askdirectory(title = "Select Folder to Store Analysis Results")
    
    data, rate = aio.load_audio(fname)
    channels = len(data[0,:])
    logger_id = fname.split('/')[-1].split('-')[0]
    xvals = np.arange(0,(len(data[:,0])/rate),1/rate)
    
    rms_stim = []
    
    for i in range(channels):
        # rms = np.std(data[:,i])*np.sqrt(2)
        stim_onset, stim_offset = find_sine_wave_onset_offset(data[:,i], threshold = amp_threshold)
        rms_stim.append(np.std(data[stim_onset:stim_offset, i])*np.sqrt(2))
    
    rms_stim = np.array(rms_stim)
    rms_stim /= np.max(abs(rms_stim))
    
    cor_factors = 1/rms_stim
    cor_factors /= np.max(cor_factors)
    
    
    data_cal = np.array(data)
    
    fig, ax = plt.subplots(2,1)
    for i in range(channels):
        data_cal[:,i] *= cor_factors[i]
        ax[0].plot(xvals, data[:,i])
        ax[1].plot(xvals, data_cal[:,i])
    
    ax[1].set_xlabel('Time (s)')
    ax[0].set_ylabel('Voltage Raw (V)')
    ax[1].set_ylabel('Voltage Calibrated (V)')
    ax[0].set_title(fname.split('/')[-1])
    plt.savefig('%s\\%s_recording.png'%(output_path, logger_id))
    plt.show(block=False)
    
    cf_dict = {'channel': np.arange(1,channels+1), 'cor_factor': cor_factors}
    cf_df = pd.DataFrame(cf_dict)
    cf_df.to_csv('%s\\%s_correction_factors.csv'%(output_path, logger_id), index=False)
    
    
    # Extra plot
    offset = 1.5*np.max(abs(data[:,:]))
    
    plt.figure()
    for i in range(channels):
        plt.plot(xvals, data[:,i]+i*offset, linewidth = 1) # eods_float, linewidth = 1)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Raw Data')
    plt.show(block=False)
    
    
    plt.figure()
    for i in range(channels):
        plt.plot(xvals, data[:,i])
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Raw Data')
    plt.show(block=False)
    
    plt.figure()
    for i in range(channels):
        plt.plot(xvals, data_cal[:,i])
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Calibrated Data')
    plt.show()


if __name__ == "__main__":
    main()
