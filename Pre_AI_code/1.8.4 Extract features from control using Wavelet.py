import numpy as np
import pandas as pd
import pywt
import scipy.signal
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import glob
import os

# CHANGELOG:
    # - reactivated trunkation of signals, however: noisy signals -> trunkated at different samples -> diferent

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Logger Files")

def find_crossings(signal, threshold):
    abs_signal = np.abs(signal)
    crossings = np.where(abs_signal >= threshold)[0]
    return crossings[0], crossings[-1]

def align_signal(signal, peak_index, target_index, total_length):
    shift = target_index - peak_index
    aligned_signal = np.zeros(total_length)
    start = max(0, shift)
    end = min(total_length, shift + len(signal))
    signal_start = max(0, -shift)
    signal_end = signal_start + (end - start)
    aligned_signal[start:end] = signal[signal_start:signal_end]
    return aligned_signal

def plot_species_characteristics(processed_signals, species_labels, resample_factor, original_sample_rate=96000):
    unique_species = np.unique(species_labels)
    sample_rate = original_sample_rate / resample_factor
    n_samples = len(processed_signals[0])
    time_axis = np.linspace(0, len(processed_signals[0]) / sample_rate, n_samples)
    
    fig, axes = plt.subplots(len(unique_species), 4, figsize=(20, 5 * len(unique_species)))
    
    for i, species in enumerate(unique_species):
        indices = np.where(species_labels == species)[0]
        species_processed_signals = processed_signals[indices]
        mean_processed_signal = np.mean(species_processed_signals, axis=0)
        
        # Processed signals plot
        for signal in species_processed_signals:
            axes[i, 0].plot(time_axis, signal, color='grey', alpha=0.1)
        axes[i, 0].plot(time_axis, mean_processed_signal, color='blue', lw=2)
        axes[i, 0].set_title(f'{species} Processed Signals')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        
        # Power spectral density plot
        freqs, psd = scipy.signal.welch(species_processed_signals, fs=sample_rate, axis=1, nperseg=n_samples)
        psd_mean = np.mean(psd, axis=0)
        psd_mean_db = 10 * np.log10(psd_mean / np.max(psd_mean))
        for signal_psd in psd:
            signal_psd_db = 10 * np.log10(signal_psd / np.max(signal_psd))
            axes[i, 1].plot(freqs, signal_psd_db, color='grey', alpha=0.1)
        axes[i, 1].plot(freqs, psd_mean_db, color='blue', lw=2)
        axes[i, 1].set_xlim(0, 20000)  # Limit frequency to 20 kHz
        axes[i, 1].set_ylim(-50,5)  # Limit frequency to 20 kHz
        axes[i, 1].set_title(f'{species} Power Spectral Density')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Power/Frequency (dB/Hz)')
        
        # Spectrogram plot
        f, t, Sxx = scipy.signal.spectrogram(mean_processed_signal, fs=sample_rate, window = ('gaussian', 0.25), nperseg=32, noverlap=4)
        axes[i, 2].pcolormesh(t, f, Sxx, shading='gouraud')
        axes[i, 2].set_ylim(0, 20000)  # Limit frequency to 20 kHz
        axes[i, 2].set_title(f'{species} Spectrogram')
        axes[i, 2].set_xlabel('Time (s)')
        axes[i, 2].set_ylabel('Frequency (Hz)')
        
        # DWT transform plot
        coeffs = pywt.wavedec(mean_processed_signal, 'sym4', level=4)
        for j, coeff in enumerate(coeffs):
            axes[i, 3].plot(coeff, label=f'Level {j}') 
        axes[i, 3].set_title(f'{species} DWT (Symmlet-4)')
        axes[i, 3].legend()
    
    plt.tight_layout()
    plt.show()

def plot_unprocessed_waveforms(unprocessed_signals, species_labels):
    unique_species = np.unique(species_labels)
    fig, axes = plt.subplots(len(unique_species), 1, figsize=(10, 5 * len(unique_species)))
    
    if len(unique_species) == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one species
    
    for i, species in enumerate(unique_species):
        indices = np.where(species_labels == species)[0]
        species_signals = unprocessed_signals[indices]
        mean_signal = np.mean(species_signals, axis=0)
        
        for signal in species_signals:
            axes[i].plot(signal, color='grey', alpha=0.3)
        axes[i].plot(mean_signal, color='blue', lw=2)
        axes[i].set_title(f'{species} Unprocessed Signals')
    
    plt.tight_layout()
    plt.show()

# Load the combined data and metadata
filtered_waveforms = pd.read_csv(glob.glob(os.path.join(input_path, 'filtered_waveforms.csv'))[0])
filtered_metadata = pd.read_csv(glob.glob(os.path.join(input_path, 'filtered_metadata.csv'))[0])

# Plot the unprocessed waveforms
plot_unprocessed_waveforms(filtered_waveforms.values, filtered_metadata['species'].values)

# Process each signal in the DataFrame
processed_signals = []
aligned_signals = []
resample_factors = []
wavelet_bases = ['haar', 'sym4', 'db4', 'db10']
dwt_coefficients = {wavelet: [] for wavelet in wavelet_bases}
dwt_level = 4  # Adjusted DWT level to avoid boundary effects

max_len = 0

# First pass: find the maximum length of the truncated signals
for index, row in filtered_waveforms.iterrows():
    signal = row.values
    max_amplitude = np.max(np.abs(signal))
    threshold = 0.03 * max_amplitude

    # Find crossings
    B, E = find_crossings(signal, threshold)
    truncated_signal = signal[B:E+1]
    max_len = max(max_len, len(truncated_signal))

# Second pass: align signals and process them
target_index = max_len // 2
for index, row in filtered_waveforms.iterrows():
    signal = row.values
    max_amplitude = np.max(np.abs(signal))
    threshold = 0.03 * max_amplitude

    # Find crossings
    B, E = find_crossings(signal, threshold)
    truncated_signal = signal[B:E+1]

    # Find the major peak
    peak_index = np.argmax(truncated_signal)
    
    # Align signal
    aligned_signal = align_signal(truncated_signal, peak_index, target_index, max_len)
    
    # No TRUNKATION; CHANGE THIS BACK IF NOT GOOD
    # aligned_signal = signal
    aligned_signals.append(aligned_signal)
    
    
    
    # Force baseline to zero
    baseline = np.linspace(aligned_signal[0], aligned_signal[-1], len(aligned_signal))
    centered_signal = aligned_signal - baseline

    # Resample to 512 points
    resampled_signal = scipy.signal.resample(centered_signal, 512)
    resample_factors.append(len(centered_signal) / 512)
    processed_signals.append(resampled_signal)

    # Conduct DWT for each wavelet basis
    for wavelet in wavelet_bases:
        coeffs = pywt.wavedec(resampled_signal, wavelet, level=dwt_level)
        dwt_coefficients[wavelet].append(coeffs)

aligned_signals = np.array(aligned_signals)
processed_signals = np.array(processed_signals)
resample_factors = np.array(resample_factors)

# Save processed signals and resample factors
np.save(input_path + '/aligned_signals.npy', aligned_signals)
np.save(input_path + '/processed_signals.npy', processed_signals)
np.save(input_path + '/resample_factors.npy', resample_factors)

# Save DWT coefficients with varying lengths for each wavelet
for wavelet, coeffs_list in dwt_coefficients.items():
    coeffs_array = np.array(coeffs_list, dtype=object)  # Use object dtype to handle varying lengths
    np.save(input_path + f'/dwt_coefficients_{wavelet}.npy', coeffs_array)

# Plot species characteristics
plot_species_characteristics(processed_signals, filtered_metadata['species'].values, resample_factors[0])
