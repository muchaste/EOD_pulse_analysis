import numpy as np
import pandas as pd
import pywt
import scipy.signal
import os
import glob
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# CHANGELOG:
    # - removed trunkation

# Function to process and align signals
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

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Field Recordings")
output_path = filedialog.askdirectory(title="Select Output Folder")
model_path = filedialog.askdirectory(title="Select Folder with Models")

# Set wavelet
print('Input wavelet')
wavelet_base = input()

# Load the best combination parameters
best_combination_df = pd.read_csv(os.path.join(model_path, 'best_combination.csv'), index_col=0)
best_combination = best_combination_df.loc[wavelet_base].to_dict()
best_combination['indices'] = np.fromstring(best_combination['indices'][1:-1], sep=' ', dtype=int)

# Load the species labels
species_labels_df = pd.read_csv(os.path.join(model_path, 'species_labels.csv'))
le = LabelEncoder()
le.classes_ = species_labels_df['species'].values

# species_labs = pd.read_csv(model_path + '/species_labels.csv')
# Low-pass filter function
def low_pass_filter(signal, cutoff=10000, fs=96000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return scipy.signal.filtfilt(b, a, signal)

# Load models
# with open(os.path.join(model_path, 'best_lda_model_' + wavelet_base + '.pkl'), 'rb') as f:
#     lda_model = pickle.load(f)
with open(os.path.join(model_path, 'best_mda_model_' + wavelet_base + '.pkl'), 'rb') as f:
    mda_models = pickle.load(f)

# Process field recordings
waveform_files = glob.glob(os.path.join(input_path, '*_h2t_waveforms.csv'))
metadata_files = glob.glob(os.path.join(input_path, '*_eod_table.csv'))


#%%
for waveform_file, metadata_file in zip(waveform_files, metadata_files):
    print(waveform_file)
    waveforms = pd.read_csv(waveform_file)
    metadata = pd.read_csv(metadata_file)

    processed_signals = []
    aligned_signals = []
    max_len = 0

    # # Low-pass filter
    # filtered_signals = []
    # for wf in range(0,waveforms.shape[0]):
    #     f_sig = low_pass_filter(waveforms.iloc[wf,:])
    #     filtered_signals.append(f_sig)
        
    # First pass: find the maximum length of the truncated signals
    for index, row in waveforms.iterrows():
        signal = row.values
        max_amplitude = np.max(np.abs(signal))
        threshold = 0.03 * max_amplitude

        # Find crossings
        B, E = find_crossings(signal, threshold)
        truncated_signal = signal[B:E+1]
        max_len = max(max_len, len(truncated_signal))

    # Second pass: align signals and process them
    target_index = max_len // 2
    for index, row in waveforms.iterrows():
        signal = row.values
        max_amplitude = np.max(np.abs(signal))
        threshold = 0.03 * max_amplitude

        # Find crossings
        B, E = find_crossings(signal, threshold)
        truncated_signal = signal[B:E+1]

        # Find the major peak
        peak_index = np.argmax(truncated_signal)

        # # Align signal
        aligned_signal = align_signal(truncated_signal, peak_index, target_index, max_len)
        # aligned_signal = signal
        aligned_signals.append(aligned_signal)

        # Force baseline to zero
        baseline = np.linspace(aligned_signal[0], aligned_signal[-1], len(aligned_signal))
        centered_signal = aligned_signal - baseline

        # Resample to 512 points
        resampled_signal = scipy.signal.resample(centered_signal, 512)
        processed_signals.append(resampled_signal)
        
        

    # DWT for each signal
    wavelet = wavelet_base
    dwt_coefficients = []
    for signal in processed_signals:
        coeffs = pywt.wavedec(signal, wavelet, level=4)
        dwt_coefficients.append(np.hstack(coeffs))

    # Convert to numpy array
    dwt_coefficients = np.array(dwt_coefficients)

    # Extract top coefficients
    best_indices = best_combination['indices']
    X = dwt_coefficients[:, best_indices]

    # # Classify with LDA
    # predicted_species_lda = lda_model.predict(X)

    # Classify with MDA
    probs = np.array([mda_models[spec].score_samples(X) for spec in mda_models]).T
    predicted_species_mda = np.argmax(probs, axis=1)
    predicted_species = [species_labels_df['species'][pred] for pred in predicted_species_mda]


    # predicted_species_lda = le.inverse_transform(predicted_species_lda)
    # predicted_species_mda = le.inverse_transform(predicted_species_mda)

    # Add predicted species to metadata
    # metadata['predicted_species_lda'] = predicted_species_lda
    metadata['predicted_species_mda'] = predicted_species

    # Save updated metadata
    output_file = os.path.join(output_path, os.path.basename(metadata_file))
    metadata.to_csv(output_file, index=False)
