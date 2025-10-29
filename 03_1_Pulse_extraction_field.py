"""
# 03_Pulse_extraction_field.py
# Field Recording EOD Pulse Extraction (Script 03)

This script extracts EOD pulses from multi-channel field recordings

"""

import thunderfish.pulses as pulses
import matplotlib.pyplot as plt
import audioio as aio
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import gc
import glob
import datetime as dt
import os
import json
from pathlib import Path
import pickle

# Import EOD functions
from eod_functions_backup import (
    save_variable_length_waveforms,
    extract_pulse_snippets,
    filter_waveforms,
    filter_waveforms_with_classifier
)

# ML-related imports for classifier loading
try:
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. ML classification will be disabled.")
    ML_AVAILABLE = False

use_param_file = False
create_plots = True

# Option to import parameters from diagnostic tool
import_diagnostic_params = messagebox.askyesno("Import Diagnostic Parameters", 
                                           "Do you want to import parameters from a diagnostic tool JSON file?")

diagnostic_parameters = None
if import_diagnostic_params:
    param_file = filedialog.askopenfilename(
        title="Select Diagnostic Parameter File", 
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    try:
        with open(param_file, 'r') as f:
            diagnostic_parameters = json.load(f)
        print(f"Imported diagnostic parameters from: {param_file}")
    except Exception as e:
        messagebox.showerror("Import Error", f"Failed to import parameters:\n{str(e)}")
        print(f"Parameter import failed: {e}")
        diagnostic_parameters = None

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title = "Select Folder with Logger Files")
output_path = filedialog.askdirectory(title = "Select Folder to Store Analysis Results")
# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Option to use enhanced ML-based filtering
use_ml_filtering = messagebox.askyesno("Enhanced Filtering", 
                                      "Do you want to use machine learning-based fish vs noise classification?\n\n"
                                      "This requires a trained classifier from the annotation analysis.")

classifier_path = None
fish_probability_threshold = 0.5  # Default threshold
if use_ml_filtering:
    classifier_file = filedialog.askopenfilename(
        title="Select Trained Classifier File", 
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        initialdir=output_path  # Look in the output folder first
    )
    if classifier_file and os.path.exists(classifier_file):
        classifier_path = classifier_file
        print(f"Using ML classifier: {classifier_path}")
        
        # Ask for probability threshold
        threshold_input = simpledialog.askfloat(
            "Classifier Threshold",
            "Enter the minimum probability threshold for classifying as fish:\n"
            "(0.5 = balanced, higher = more conservative, lower = more inclusive)",
            initialvalue=0.5,
            minvalue=0.1,
            maxvalue=0.9
        )
        if threshold_input:
            fish_probability_threshold = threshold_input
            print(f"Using fish probability threshold: {fish_probability_threshold}")
        
    else:
        print("No valid classifier file selected - using basic filtering only")
        use_ml_filtering = False

# Pick file for amplitude calibration
cal_file = filedialog.askopenfilename(title = "Select File with Calibration Data")
cor_factors = np.array(pd.read_csv(cal_file))

# Pick file with control parameters
if use_param_file:
    param_file = filedialog.askopenfilename(title = "Select File with Control Parameters")
    parameters_imported = pd.read_csv(param_file).to_dict(orient='list')

# List all .wav files
filelist = glob.glob(input_path+'/*.wav', recursive=True)

if not filelist:
    print(f"No WAV files found in {input_path}")
    exit()

print(f"Found {len(filelist)} WAV files")

# Load ML classifier if selected
loaded_classifier = None
loaded_scaler = None
classifier_name = None
if use_ml_filtering and classifier_path and ML_AVAILABLE:
    try:
        print(f"\nLoading ML classifier from: {classifier_path}")
        with open(classifier_path, 'rb') as f:
            classifier_data = pickle.load(f)
        
        loaded_classifier = classifier_data['classifier']
        loaded_scaler = classifier_data['scaler']
        classifier_name = classifier_data['classifier_name']
        training_accuracy = classifier_data['accuracy']
        
        print(f"Loaded classifier: {classifier_name}")
        print(f"Training accuracy: {training_accuracy:.3f}")
        print(f"Fish probability threshold: {fish_probability_threshold}")
        print("ML-enhanced filtering will be used\n")
        
    except Exception as e:
        print(f"Error loading classifier: {e}")
        print("Falling back to basic filtering\n")
        use_ml_filtering = False
        loaded_classifier = None
        loaded_scaler = None

# Sort files by timestamp extracted from filename
try:
    timecol = [pd.to_datetime(os.path.basename(fname).split('-')[1][0:-4], format='%Y%m%dT%H%M%S') for fname in filelist]
    timecol = pd.to_datetime(timecol)
    file_set = pd.DataFrame({'timestamp': timecol, 'filename': filelist})
    file_set = file_set.sort_values(by=['timestamp'], ignore_index=True)
except Exception as e:
    print(f"Warning: Could not parse timestamps from filenames: {e}")
    file_set = pd.DataFrame({'timestamp': pd.NaT, 'filename': sorted(filelist)})

# Test file loading and setup detection threshold
tmin, tmax = 0, 60
try:
    with aio.AudioLoader(file_set['filename'][0], 60) as sf:
        rate = sf.rate
        data = sf[int(tmin * rate):int(tmax * rate), :]
    n_channels = data.shape[1]
    
except Exception as e:
    print(f"Error loading test file: {e}")
    exit()

# Apply calibration factors
# thresh = 0
for i in range(n_channels):
    data[:, i] *= cor_factors[i, 1]
    sd = np.std(data[:, i])
    # thresh = max(thresh, sd)
# thresh *= 10

thresh = 0.01


# Setup parameters

if use_param_file:
    parameters = parameters_imported
elif diagnostic_parameters is not None:
    # Use diagnostic parameters but ensure all required parameters are present
    parameters = diagnostic_parameters.copy()
    # Update threshold if needed
    if 'thresh' in parameters:
        thresh = parameters['thresh']
    print("Using diagnostic tool parameters")
else:
    # Default field recording parameters
    parameters = {
        'thresh': thresh,
        'min_rel_slope_diff': 0.25,
        'min_width_us': 30,
        'max_width_us': 1000,  # in microseconds
        'width_fac_detection': 7.0,
        'interp_factor': 3,  # Interpolation factor for waveform extraction
        'amplitude_ratio_min': 0.2,
        'amplitude_ratio_max': 4,
        'save_filtered_out': False,
        'noise_removal': False,
        'peak_fft_freq_min': 50,
        'peak_fft_freq_max': 10000,
        'return_diff': True  # Return differential data
    }

# Plot raw data
offset = np.max(abs(data))#*1.5
plt.figure(figsize=(40,12))
for i in range(n_channels):
    plt.plot(data[0:int(60*rate-1), i] + i*offset, label=str(i+1))
    plt.hlines(y = thresh + i*offset, xmin = 0, xmax = int(60*rate-1))

plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
plt.savefig('%s\\%s_one_minute_raw.png'%(output_path, file_set['filename'][0].split('\\')[-1][:-4]))
plt.show(block=False)

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

# Helper function to access parameters (works for both dict and DataFrame)
def get_param(param_name):
    if isinstance(parameters, dict):
        return parameters[param_name]
    else:  # DataFrame
        return parameters[param_name][0]

# Convert to DataFrame for further processing
parameters_df = pd.DataFrame({k: [v] for k, v in parameters.items()})
parameters_df.to_csv('%s\\analysis_parameters.csv' % output_path, index=False)
plt.close()
gc.collect()


# Process each file
for n, filepath in enumerate(file_set['filename']):
    fname = os.path.basename(filepath)
    print(f"Processing {n+1}/{len(file_set)}: {fname}")
    
    file_start_time = dt.datetime.now()
    gc.collect()
    
    try:
        # Load file
        data, rate = aio.load_audio(filepath)
        n_channels = data.shape[1]
        file_duration = len(data) / rate
        
        # Calibrate with correction factor
        for i in range(n_channels):
            data[:, i] *= cor_factors[i, 1]
    
        # Detect pulses across all channels
        peaks = []
        troughs = []
        pulse_widths = []
        
        for i in range(n_channels):
            ch_peaks, ch_troughs, _, ch_pulse_widths = \
                pulses.detect_pulses(data[:, i], rate, 
                                     thresh=get_param('thresh'), 
                                     min_rel_slope_diff=get_param('min_rel_slope_diff'),
                                     min_width=get_param('min_width_us') / 1e6,
                                     max_width=get_param('max_width_us') / 1e6,
                                     width_fac=get_param('width_fac_detection'),
                                     verbose=0,
                                     return_data=False)
            peaks.append(ch_peaks)
            troughs.append(ch_troughs)
            pulse_widths.append(ch_pulse_widths)

        # Unify pulses across channels
        all_pulses = []
        for ch_peaks, ch_troughs, ch_widths in zip(peaks, troughs, pulse_widths):
            n_pairs = min(len(ch_peaks), len(ch_troughs))
            for j in range(n_pairs):
                mp = (ch_peaks[j] + ch_troughs[j]) // 2
                all_pulses.append((mp, ch_peaks[j], ch_troughs[j], ch_widths[j]))

        # Sort by midpoint and remove duplicates
        all_pulses.sort(key=lambda x: x[0])
        unique_pulses = []
        
        if all_pulses:
            last_mp = all_pulses[0][0]
            unique_pulses.append(all_pulses[0])
            for event in all_pulses[1:]:
                if event[0] - last_mp > 3:
                    unique_pulses.append(event)
                    last_mp = event[0]
            del all_pulses
            gc.collect()
        
        # Unpack to arrays for further analysis
        if unique_pulses:
            unique_midpoints = np.array([e[0] for e in unique_pulses])
            unique_peaks = np.array([e[1] for e in unique_pulses])
            unique_troughs = np.array([e[2] for e in unique_pulses])
            unique_widths = np.array([e[3] for e in unique_pulses])
        else:
            unique_midpoints = np.array([])
            unique_peaks = np.array([])
            unique_troughs = np.array([])
            unique_widths = np.array([])

        print(f"  Found {len(unique_midpoints)} unique pulses")
        del peaks, troughs, pulse_widths
        gc.collect()

        # Extract and analyze waveforms
        if len(unique_midpoints) > 0:
            # Extract variable-width snippets and analyze them
            try:
                # (
                #     eod_waveforms, eod_amps, eod_widths, ch_amps, ch_cor_coeffs, eod_chan, is_differential,
                #     snippet_peak_idc, snippet_trough_idc, snippet_midpoint_idc,
                #     final_peak_idc, final_trough_idc, final_midpoint_idc,
                #     original_pulse_orientation, amplitude_ratios, waveform_lengths, fft_peak_freqs
                # ) = extract_pulse_snippets_legacy(
                #     data, parameters_df, rate, unique_midpoints, unique_peaks, unique_troughs, unique_widths,
                #     center_on_zero_crossing=False
                #     )
                
                # # Development version with faster extraction
                # # (
                # #     eod_waveforms_d, eod_amps_d, eod_widths_d, ch_amps_d, ch_cor_coeffs_d, eod_chan_d, is_differential_d,
                # #     snippet_peak_idc_d, snippet_trough_idc_d, snippet_midpoint_idc_d,
                # #     final_peak_idc_d, final_trough_idc_d, final_midpoint_idc_d,
                # #     original_pulse_orientation_d, amplitude_ratios_d, waveform_lengths_d, fft_peak_freqs_d
                # # )
                (
                    eod_waveforms, eod_amps, eod_widths, ch_amps, ch_cor_coeffs, eod_chan, is_differential,
                    snippet_peak_idc, snippet_trough_idc, snippet_midpoint_idc,
                    final_peak_idc, final_trough_idc, final_midpoint_idc,
                    original_pulse_orientation, amplitude_ratios, waveform_lengths, fft_peak_freqs
                ) = extract_pulse_snippets(
                    data, unique_midpoints, unique_peaks, unique_troughs, thresh = get_param('thresh'),
                    width_factor = get_param('width_fac_detection'), interp_factor = get_param('interp_factor'), rate = rate,
                    source = 'multich_linear', return_differential = get_param('return_diff')
                )

                # Remove duplicates: same channel and midpoint within 3 samples
                unique_mask = np.ones(len(final_midpoint_idc), dtype=bool)
                for i in range(len(final_midpoint_idc)):
                    if unique_mask[i]:
                        for j in range(i+1, len(final_midpoint_idc)):
                            if (eod_chan[i] == eod_chan[j] and 
                                abs(final_midpoint_idc[i] - final_midpoint_idc[j]) <= 3):
                                unique_mask[j] = False
                
                print(f"    Removed {np.sum(~unique_mask)} proximity-based duplicate pulses")
                
                # Filter all arrays by proximity-based unique_mask first
                eod_waveforms = [eod_waveforms[i] for i in range(len(eod_waveforms)) if unique_mask[i]]
                eod_amps = eod_amps[unique_mask]
                eod_widths = eod_widths[unique_mask]
                ch_amps = ch_amps[unique_mask]
                ch_cor_coeffs = ch_cor_coeffs[unique_mask]
                eod_chan = eod_chan[unique_mask]
                is_differential = is_differential[unique_mask]
                snippet_peak_idc = snippet_peak_idc[unique_mask]
                snippet_trough_idc = snippet_trough_idc[unique_mask]
                snippet_midpoint_idc = snippet_midpoint_idc[unique_mask]
                final_peak_idc = final_peak_idc[unique_mask]
                final_trough_idc = final_trough_idc[unique_mask]
                final_midpoint_idc = final_midpoint_idc[unique_mask]
                original_pulse_orientation = original_pulse_orientation[unique_mask]
                amplitude_ratios = amplitude_ratios[unique_mask]
                waveform_lengths = waveform_lengths[unique_mask]
                fft_peak_freqs = fft_peak_freqs[unique_mask]
                
                # ENHANCED DUPLICATE DETECTION: Remove events with identical peak/trough indices
                # This handles multi-phasic EODs where low threshold detects both p1-t1 and p2-t2,
                # but _process_waveform_common re-detects the same dominant p2-t2 in both snippets
                print(f"    Before exact duplicate removal: {len(eod_waveforms)} events")
                
                # Create unique identifier for each event based on peak and trough indices
                # Use tuple of (peak_idx, trough_idx) as the key
                unique_events = {}
                duplicate_indices = []
                
                for i in range(len(final_peak_idc)):
                    event_key = (final_peak_idc[i], final_trough_idc[i])
                    
                    if event_key in unique_events:
                        # Found duplicate - keep the one with better characteristics
                        existing_idx = unique_events[event_key]
                        
                        # Comparison criteria (in order of importance):
                        # 1. Smaller waveform length (more precise extraction)
                        # 2. Higher amplitude (better signal quality)
                        # 3. Better amplitude ratio (closer to expected range)
                        
                        current_length = waveform_lengths[i]
                        existing_length = waveform_lengths[existing_idx]
                        current_amp = eod_amps[i]
                        existing_amp = eod_amps[existing_idx]
                        current_ratio = amplitude_ratios[i]
                        existing_ratio = amplitude_ratios[existing_idx]
                        
                        # Decide which one to keep
                        keep_current = False
                        
                        if current_length != existing_length:
                            # Prefer smaller length (more precise extraction)
                            keep_current = current_length < existing_length
                        elif current_amp != existing_amp:
                            # If lengths are equal, prefer higher amplitude
                            keep_current = current_amp > existing_amp
                        else:
                            # If both length and amplitude are equal, prefer better amplitude ratio
                            # (closer to the middle of acceptable range)
                            ratio_min = get_param('amplitude_ratio_min')
                            ratio_max = get_param('amplitude_ratio_max')
                            ratio_mid = (ratio_min + ratio_max) / 2
                            
                            current_ratio_distance = abs(current_ratio - ratio_mid)
                            existing_ratio_distance = abs(existing_ratio - ratio_mid)
                            keep_current = current_ratio_distance < existing_ratio_distance
                        
                        if keep_current:
                            # Mark the existing one for removal and update the dictionary
                            duplicate_indices.append(existing_idx)
                            unique_events[event_key] = i
                        else:
                            # Mark the current one for removal
                            duplicate_indices.append(i)
                    else:
                        # First occurrence of this peak/trough pair
                        unique_events[event_key] = i
                
                # Remove exact duplicates from all arrays and lists
                if duplicate_indices:
                    print(f"      Removing {len(duplicate_indices)} exact duplicate events")
                    
                    # Create mask for non-duplicate indices
                    non_duplicate_mask = np.ones(len(eod_waveforms), dtype=bool)
                    non_duplicate_mask[duplicate_indices] = False
                    
                    # Filter all arrays and lists
                    eod_waveforms = [eod_waveforms[i] for i in range(len(eod_waveforms)) if non_duplicate_mask[i]]
                    eod_amps = eod_amps[non_duplicate_mask]
                    eod_widths = eod_widths[non_duplicate_mask]
                    ch_amps = ch_amps[non_duplicate_mask]
                    ch_cor_coeffs = ch_cor_coeffs[non_duplicate_mask]
                    eod_chan = eod_chan[non_duplicate_mask]
                    is_differential = is_differential[non_duplicate_mask]
                    snippet_peak_idc = snippet_peak_idc[non_duplicate_mask]
                    snippet_trough_idc = snippet_trough_idc[non_duplicate_mask]
                    snippet_midpoint_idc = snippet_midpoint_idc[non_duplicate_mask]
                    final_peak_idc = final_peak_idc[non_duplicate_mask]
                    final_trough_idc = final_trough_idc[non_duplicate_mask]
                    final_midpoint_idc = final_midpoint_idc[non_duplicate_mask]
                    original_pulse_orientation = original_pulse_orientation[non_duplicate_mask]
                    amplitude_ratios = amplitude_ratios[non_duplicate_mask]
                    waveform_lengths = waveform_lengths[non_duplicate_mask]
                    fft_peak_freqs = fft_peak_freqs[non_duplicate_mask]
                    
                    print(f"      After exact duplicate removal: {len(eod_waveforms)} events")
                else:
                    print(f"      No exact duplicates found")
                
            except Exception as e:
                print(f"    Error in waveform extraction: {e}")
                continue
            
            # Apply filtering pipeline
            try:
                if use_ml_filtering and loaded_classifier is not None and loaded_scaler is not None:
                    # Use enhanced ML-based filtering
                    keep_indices, filtered_features, filteredout_features = filter_waveforms_with_classifier(
                        eod_waveforms, eod_widths, amplitude_ratios, fft_peak_freqs, rate,
                        classifier=loaded_classifier,
                        scaler=loaded_scaler,
                        dur_min=get_param('min_width_us'), 
                        dur_max=get_param('max_width_us'),
                        pp_r_min=get_param('amplitude_ratio_min'), 
                        pp_r_max=get_param('amplitude_ratio_max'),
                        fft_freq_min=get_param('peak_fft_freq_min'), 
                        fft_freq_max=get_param('peak_fft_freq_max'),
                        fish_probability_threshold=fish_probability_threshold,
                        use_basic_filtering=True,
                        return_features=True, 
                        return_filteredout_features=True
                    )
                else:
                    # Use basic threshold filtering
                    keep_indices, filtered_features, filteredout_features = filter_waveforms(
                        eod_waveforms, eod_widths, amplitude_ratios, fft_peak_freqs, rate,
                        dur_min=get_param('min_width_us'), 
                        dur_max=get_param('max_width_us'),
                        pp_r_min=get_param('amplitude_ratio_min'), 
                        pp_r_max=get_param('amplitude_ratio_max'),
                        fft_freq_min=get_param('peak_fft_freq_min'), 
                        fft_freq_max=get_param('peak_fft_freq_max'),
                        return_features=True, return_filteredout_features=True
                    )
                
                print(f"    Filtered {len(eod_waveforms) - len(keep_indices)} out of {len(eod_waveforms)} pulses")
                
            except Exception as e:
                print(f"    Error in filtering: {e}")
                keep_indices = np.arange(len(eod_waveforms))

            # Get indices of filtered-out pulses for QC
            all_indices = np.arange(len(eod_waveforms))
            filtered_out_indices = np.setdiff1d(all_indices, keep_indices)
            
            # Create results table
            eod_timestamps = []
            for i in range(len(final_midpoint_idc)):
                if pd.isna(file_set['timestamp'][n]):
                    base_time = file_start_time
                    eod_timestamps.append(base_time + dt.timedelta(seconds=final_midpoint_idc[i]/rate))
                else:
                    eod_timestamps.append(file_set['timestamp'][n] + dt.timedelta(seconds=final_midpoint_idc[i]/rate))
            
            complete_eod_table = pd.DataFrame({
                'timestamp': eod_timestamps,
                'file_timestamp': [file_set['timestamp'][n]] * len(final_midpoint_idc),
                'filename': [fname] * len(final_midpoint_idc),
                'midpoint_idx': final_midpoint_idc,
                'relative_time_s': final_midpoint_idc / rate,
                'peak_idx': final_peak_idc,
                'trough_idx': final_trough_idc,
                'eod_channel': eod_chan,
                'snippet_peak_idx': snippet_peak_idc,
                'snippet_trough_idx': snippet_trough_idc,
                'snippet_midpoint_idx': snippet_midpoint_idc,
                'eod_amplitude': eod_amps,
                'eod_width_us': eod_widths,
                'eod_amplitude_ratio': amplitude_ratios,
                'pulse_orientation': original_pulse_orientation,
                'waveform_length': waveform_lengths,
                'fft_freq_max': fft_peak_freqs
                })

            # Add multi-channel amplitude information
            for ch in range(n_channels):
                complete_eod_table[f'amplitude_ch{ch}'] = ch_amps[:, ch] if len(ch_amps) > 0 else []

            # Add cross-channel correlation coefficients
            for ch in range(n_channels - 1):
                complete_eod_table[f'correlation_ch{ch}_{ch+1}'] = ch_cor_coeffs[:, ch] if len(ch_cor_coeffs) > 0 else []

            # Filter by the initial criteria (duration, ratio, frequency)
            eod_table = complete_eod_table.iloc[keep_indices].copy().reset_index(drop=True)
            filtered_eod_waveforms = [eod_waveforms[i] for i in keep_indices]
            filteredout_eod_table = complete_eod_table.iloc[filtered_out_indices].copy().reset_index(drop=True)

        # Save individual file results
        output_file = os.path.join(output_path, f'{fname[:-4]}_eod_table.csv')
        eod_table.to_csv(output_file, index=False)
        
        # Save waveforms
        if len(keep_indices) > 0:
            try:
                waveform_base = os.path.join(output_path, f'{fname[:-4]}_eod_waveforms')
                waveform_metadata = save_variable_length_waveforms(
                    filtered_eod_waveforms, waveform_base
                )
                
            except Exception as e:
                print(f"  Warning: Waveform saving failed: {e}")
        else:
            # Create empty waveform files for consistency
            empty_file = os.path.join(output_path, f'{fname[:-4]}_eod_waveforms_concatenated.npz')
            np.savez_compressed(empty_file, data=np.array([]))
            
            metadata_file = os.path.join(output_path, f'{fname[:-4]}_eod_waveforms_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump({'lengths': [], 'start_indices': [], 'total_waveforms': 0, 'total_samples': 0}, 
                         f, separators=(',', ':'))
        
        # Control plot
        if create_plots:
            # Control plots
            if len(keep_indices) > 0:
                # Plot: Differential detections with differential data
                eod_idc = np.arange(len(filtered_eod_waveforms))
                # data_diff = np.diff(data, axis=1)
                offset_diff = np.max(eod_table['eod_amplitude']) * 1.5

                plt.figure(figsize=(20, 8))
                for i in range(data.shape[1]-1):
                    # Find pulses detected on this differential channel
                    data_diff = np.diff(data[:,i:i+2])
                    ch_diff_idc = np.where(eod_table['eod_channel'] == i)[0]
                    actual_diff_idc = eod_idc[ch_diff_idc]
                    
                    # Plot only every nth sample for large datasets to save memory
                    step = max(1, len(data_diff) // 15000000)  # Limit to ~15 mio points per channel
                    x_coords = np.arange(0, len(data_diff), step)
                    plt.plot(x_coords, data_diff[::step] + i * offset_diff, linewidth=0.5, label=f'Ch{i}-{i+1}')
                    
                    # Plot filtered (accepted) pulses
                    if len(actual_diff_idc) > 0:
                        plt.plot(eod_table['peak_idx'].iloc[actual_diff_idc], 
                                data_diff[eod_table['peak_idx'].iloc[actual_diff_idc]] + i * offset_diff, 
                                'o', markersize=1, color='red')
                        plt.plot(eod_table['trough_idx'].iloc[actual_diff_idc], 
                                data_diff[eod_table['trough_idx'].iloc[actual_diff_idc]] + i * offset_diff, 
                                'o', markersize=1, color='blue')
                    
                    # Plot filtered-out pulses in grey
                    if len(filtered_out_indices) > 0:
                        # Find filtered-out pulses detected on this differential channel
                        filteredout_eod_chan = filteredout_eod_table['eod_channel']
                        filteredout_final_peak_idc = filteredout_eod_table['peak_idx']
                        filteredout_final_trough_idc = filteredout_eod_table['trough_idx']
                        
                        filteredout_ch_diff_idc = np.where(filteredout_eod_chan == i)[0]
                        if len(filteredout_ch_diff_idc) > 0:
                            plt.plot(filteredout_final_peak_idc.iloc[filteredout_ch_diff_idc], 
                                    data_diff[filteredout_final_peak_idc.iloc[filteredout_ch_diff_idc]] + i * offset_diff, 
                                    'o', markersize=1, color='grey', alpha=0.6)
                            plt.plot(filteredout_final_trough_idc.iloc[filteredout_ch_diff_idc], 
                                    data_diff[filteredout_final_trough_idc.iloc[filteredout_ch_diff_idc]] + i * offset_diff, 
                                    'o', markersize=1, color='grey', alpha=0.6)
                
                plt.ylim(bottom=None, top=(data.shape[1]-1.5)*offset_diff)
                plt.title(f'{fname} - Differential EOD Detections - Red=Peaks, Blue=Troughs, Grey=Filtered Out (n={len(eod_idc)} kept, {len(filtered_out_indices)} filtered)')
                plt.legend(loc='upper right')
                plt.xlabel('Sample')
                plt.ylabel('Voltage')
                plt.savefig(f'{output_path}\\{fname[:-4]}_differential_detection_plot.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                # Clear differential data from memory
                del data_diff
                gc.collect()
                
                # Summary analysis plots
                plt.figure(figsize=(12, 8))  # Reduced size
                
                # Check if we have any data to plot
                if len(filtered_eod_waveforms) == 0:
                    plt.text(0.5, 0.5, 'No filtered pulses to display', ha='center', va='center', 
                            transform=plt.gcf().transFigure, fontsize=16)
                    plt.savefig(f'{output_path}\\{fname[:-4]}_analysis_summary.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    continue
                
                # Plot some example waveforms (limit to 25 for memory)
                plt.subplot(1, 3, 1)
                n_examples = min(25, len(filtered_eod_waveforms))
                for i in range(n_examples):
                    plt.plot(filtered_eod_waveforms[i], alpha=0.4, linewidth=0.5)
                plt.title(f'Example EOD Waveforms (n={n_examples})')
                plt.xlabel('Sample')
                plt.ylabel('Normalized Amplitude')
                
                # Plot width distribution
                plt.subplot(1, 3, 2)
                plt.hist(eod_table['eod_width_us'], bins=20, alpha=0.7)  # Fewer bins
                plt.title(f'EOD Width Distribution (n={len(filtered_eod_waveforms)})')
                plt.xlabel('Peak-Trough Width (uS)')
                plt.ylabel('Count')
                
                # Plot fft distribution
                plt.subplot(1, 3, 3)
                plt.hist(eod_table['fft_freq_max'], bins=20, alpha=0.7)  # Fewer bins
                plt.title(f'EOD FFT-Peak Distribution (n={len(filtered_eod_waveforms)})')
                plt.xlabel('Peak-FFT Frequency (Hz)')
                plt.ylabel('Count')
                
                plt.tight_layout()
                plt.savefig(f'{output_path}\\{fname[:-4]}_analysis_summary.png', dpi=150, bbox_inches='tight')  # Reduced DPI
                plt.close()
                gc.collect()

        # Record processing statistics
        file_stats = {
            'filename': fname,
            'file_duration_s': file_duration,
            'n_channels': n_channels,
            'raw_detections': len(unique_pulses) if 'unique_pulses' in locals() else 0,
            'filtered_pulses': len(filtered_eod_waveforms),
            'filtering_efficiency': len(filtered_eod_waveforms) / len(unique_pulses) * 100 if len(unique_pulses) > 0 else 0,
            'pulses_per_second': len(filtered_eod_waveforms) / file_duration if file_duration > 0 else 0,
        }
        # Save processing stats
        stats_file = os.path.join(output_path, 'processing_stats.csv')
        stats_df = pd.DataFrame([file_stats])
        if not os.path.exists(stats_file):
            stats_df.to_csv(stats_file, index=False)
        
        # Optional: Save filtered-out pulses for quality control
        if len(filtered_out_indices) > 0 and get_param('save_filtered_out'):
            try:
                # Create filtered-out data structures  
                filteredout_eod_waveforms = [eod_waveforms[i] for i in filtered_out_indices]

                # filteredout_ch_amps = ch_amps[filtered_out_indices]
                # filteredout_eod_amp = eod_amps[filtered_out_indices]
                # filteredout_eod_widths = eod_widths[filtered_out_indices]
                # filteredout_amplitude_ratios = amplitude_ratios[filtered_out_indices]
                # filteredout_ch_cor_coeffs = ch_cor_coeffs[filtered_out_indices]
                # filteredout_eod_chan = eod_chan[filtered_out_indices]
                # filteredout_snippet_peak_idc = snippet_peak_idc[filtered_out_indices]
                # filteredout_snippet_trough_idc = snippet_trough_idc[filtered_out_indices]
                # filteredout_snippet_midpoint_idc = snippet_midpoint_idc[filtered_out_indices]
                # filteredout_final_peak_idc = final_peak_idc[filtered_out_indices]
                # filteredout_final_trough_idc = final_trough_idc[filtered_out_indices]
                # filteredout_unique_midpoints = unique_midpoints[filtered_out_indices]
                # filteredout_original_pulse_orientation = original_pulse_orientation[filtered_out_indices]
                # filteredout_waveform_lengths = waveform_lengths[filtered_out_indices]
                # filteredout_fft_peaks = filteredout_features['fft_freq'][filtered_out_indices] if 'fft_freq' in filteredout_features else np.empty(0)

                # # Create filtered-out table
                # filteredout_durations_us = np.abs(filteredout_final_peak_idc - filteredout_final_trough_idc) / rate * 1e6
                # filteredout_features = pd.DataFrame({
                #     'pp_dur_us': filteredout_durations_us,
                #     'pp_ratio': filteredout_amplitude_ratios,
                # })

                # # Generate timestamps for filtered-out pulses
                # filteredout_eod_timestamps = []
                # for i in range(len(filteredout_eod_waveforms)):
                #     if pd.isna(file_set['timestamp'][n]):
                #         base_time = file_start_time
                #         filteredout_eod_timestamps.append(base_time + dt.timedelta(seconds=filteredout_unique_midpoints[i]/rate))
                #     else:
                #         filteredout_eod_timestamps.append(file_set['timestamp'][n] + dt.timedelta(seconds=filteredout_unique_midpoints[i]/rate))

                # filteredout_eod_table = pd.DataFrame({
                #     'timestamp': filteredout_eod_timestamps,
                #     'filename': [fname] * len(filteredout_unique_midpoints),
                #     'midpoint_idx': filteredout_unique_midpoints,
                #     'peak_idx': filteredout_final_peak_idc,
                #     'trough_idx': filteredout_final_trough_idc,
                #     'eod_channel': filteredout_eod_chan,
                #     'snippet_peak_idx': filteredout_snippet_peak_idc,
                #     'snippet_trough_idx': filteredout_snippet_trough_idc,
                #     'snippet_midpoint_idx': filteredout_snippet_midpoint_idc,
                #     'eod_amplitude': filteredout_eod_amp,
                #     'eod_width_us': filteredout_eod_widths,
                #     'eod_amplitude_ratio': filteredout_amplitude_ratios,
                #     'pulse_orientation': filteredout_original_pulse_orientation,
                #     'waveform_length': filteredout_waveform_lengths,
                #     'fft_peak_freq': filteredout_fft_peaks
                # })

                # # Add multi-channel data
                # for ch in range(n_channels):
                #     filteredout_eod_table[f'amplitude_ch{ch}'] = filteredout_ch_amps[:, ch] if len(filteredout_ch_amps) > 0 else []
                # for ch in range(n_channels - 1):
                #     filteredout_eod_table[f'correlation_ch{ch}_{ch+1}'] = filteredout_ch_cor_coeffs[:, ch] if len(filteredout_ch_cor_coeffs) > 0 else []

                # filteredout_eod_table = filteredout_eod_table.join(filteredout_features)

                # Save filtered-out data
                filteredout_file = os.path.join(output_path, f'{fname[:-4]}_eod_table_filteredout.csv')
                filteredout_eod_table.to_csv(filteredout_file, index=False)

                # Save filtered-out waveforms
                if len(filteredout_eod_waveforms) > 0:
                    filteredout_waveform_base = os.path.join(output_path, f'{fname[:-4]}_eod_waveforms_filteredout')
                    save_variable_length_waveforms(filteredout_eod_waveforms, filteredout_waveform_base)

            except Exception as e:
                print(f"    Warning: QC file creation failed: {e}")
        
        # Clear memory 
        del data
        if 'eod_waveforms' in locals():
            del eod_waveforms, eod_amps, eod_widths
        gc.collect()
        
    except Exception as e:
        print(f"  ERROR: File processing failed: {e}")
        continue

# Save session metadata
session_metadata = {
    'session_id': f"field_session_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    'processing_date': dt.datetime.now().isoformat(),
    'script_version': "03_enhanced",
    'input_directory': input_path,
    'output_directory': output_path,
    'calibration_file': cal_file,
    'files_processed': len(file_set),
    'parameters_used': parameters
}

metadata_file = os.path.join(output_path, 'session_metadata.json')
with open(metadata_file, 'w') as f:
    json.dump(session_metadata, f, indent=2, default=str)

print("Field recording pulse extraction completed!")