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
import gc
import glob
import datetime as dt
import os
import json
import pickle

# Import EOD functions
from pulse_functions import (
    bandpass_filter,
    unify_across_channels,
    extract_pulse_snippets,
    remove_duplicates,
    filter_waveforms,
    filter_waveforms_with_classifier,
    save_waveforms,
    # save_fixed_length_waveforms,
    create_channel_events,
    merge_channel_events,
    filter_events,
    create_event_plots
)

# Import parameter configuration GUI
from parameter_gui import ParameterConfigGUI


# ============================================================================
# INITIALIZE CONFIGURATION GUI
# ============================================================================
print("Starting Parameter Configuration GUI...")

root = tk.Tk()
config_gui = ParameterConfigGUI(root)
root.mainloop()

# Check if user cancelled
if config_gui.result is None:
    print("Configuration cancelled by user")
    exit()

# Extract configuration
config = config_gui.result
input_path = config['paths']['input_path']
cal_file = config['paths']['cal_file']
output_path = config['paths']['output_path']

use_ml_filtering = config['ml_settings']['use_ml_filtering']
classifier_path = config['ml_settings']['classifier_path'] if use_ml_filtering else None
fish_probability_threshold = config['ml_settings']['fish_probability_threshold']

parameters = config['parameters']

print("\n" + "="*60)
print("CONFIGURATION SUMMARY")
print("="*60)
print(f"Input folder: {input_path}")
print(f"Calibration file: {cal_file}")
print(f"Output folder: {output_path}")
print(f"ML filtering: {'Enabled' if use_ml_filtering else 'Disabled'}")
if use_ml_filtering:
    print(f"  Classifier: {classifier_path}")
    print(f"  Threshold: {fish_probability_threshold}")
print("\nAnalysis Parameters:")
for key, value in parameters.items():
    print(f"  {key}: {value}")
print("="*60 + "\n")

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load calibration factors
cor_factors_all = pd.read_csv(cal_file)

# Parameters for event creation
if parameters['create_events']:
    print("Setting event extraction parameters...")
    event_counter = 0           # Keep track of total events across files
    event_summaries = []

    print(f"  max_ipi_seconds: {parameters['max_ipi_seconds']}")
    print(f"  min_eods_premerge: {parameters['min_eods_premerge']}")
    print(f"  max_merge_gap_seconds: {parameters['max_merge_gap_seconds']}")
    print(f"  min_eods_postmerge: {parameters['min_eods_postmerge']}")
    print(f"  min_amplitude: {parameters['min_amplitude']}")

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
if use_ml_filtering and classifier_path:
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
cor_factors = np.array(cor_factors_all.iloc[np.where(cor_factors_all['file_id']==os.path.basename(file_set['filename'][0]).split('.')[0])[0][0]])
for i in range(n_channels):
    data[:, i] *= cor_factors[i]

# Plot raw data
offset = np.max(abs(data))#*1.5
plt.figure(figsize=(40,12))
for i in range(n_channels):
    plt.plot(data[0:int(60*rate-1), i] + i*offset, label=str(i+1))
    plt.hlines(y=parameters['thresh'] + i*offset, xmin=0, xmax=int(60*rate-1))

plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
plt.savefig('%s\\%s_one_minute_raw.png'%(output_path, file_set['filename'][0].split('\\')[-1][:-4]))
plt.show(block=False)

# Save parameters to CSV
parameters_df = pd.DataFrame({k: [v] for k, v in parameters.items()})
parameters_df.to_csv('%s\\analysis_parameters.csv' % output_path, index=False)
print(f"Analysis parameters saved to: {output_path}\\analysis_parameters.csv")
plt.close()
gc.collect()
del data

#%%
# Initialize cross-file continuation variables
retained_data = None
retained_data_start_time = None
retained_eod_table = None
retained_eod_waveforms = None
retained_audio_start_idx = None

# Process each file
for n, filepath in enumerate(file_set['filename']):
    fname = os.path.basename(filepath)
    print(f"Processing {n+1}/{len(file_set)}: {fname}")
    
    gc.collect()

    # Release large arrays left from any previous iteration that exited early
    for _cleanup_var in [
        'data_diff', 'detection_data', 'detection_signal',
        'unique_midpoints', 'unique_peaks', 'unique_troughs', 'unique_widths',
        'eod_snippets', 'eod_amps', 'eod_widths', 'eod_chan', 'is_differential',
        'snippet_p1_idc', 'snippet_p2_idc', 'raw_p1_idc', 'raw_p2_idc',
        'pulse_orientations', 'amp_ratios', 'fft_peak_freqs', 'pulse_locations', 'wf_lengths',
        'raw_midpoint_idc', 'snippet_midpoint_idc', 'eod_timestamps',
        'complete_eod_table', 'eod_table', 'filteredout_eod_table',
        'combined_eod_table', 'combined_eod_waveforms',
        'filtered_eod_waveforms', 'final_events', 'events_at_end',
        'filteredout_eod_chan', 'filteredout_final_p1_idc', 'filteredout_final_p2_idc',
    ]:
        if _cleanup_var in globals():
            del globals()[_cleanup_var]
    gc.collect()

    # Load file
    data, rate = aio.load_audio(filepath)
    n_channels = data.shape[1]
    file_duration = len(data) / rate
    print(f"    Loaded file: {file_duration:.1f}s, {n_channels} channels, {len(data)} samples")
    
    # Calibrate with correction factor
    cor_factors = np.array(cor_factors_all.iloc[np.where(cor_factors_all['file_id']==fname.split('.')[0])[0][0]])
    for i in range(n_channels):
        data[:, i] *= cor_factors[i]
    
    # EOD detection and waveform extraction run on the new file only.
    # Audio is still concatenated with the retained tail for event WAV export.
    detection_data = data
    retain_len = 0

    # Handle cross-file continuation: prepend retained audio from previous file
    if retained_data is not None:
        print(f"    Concatenating {len(retained_data)} samples from previous file (audio only)")
        original_data_length = len(data)
        retained_duration = len(retained_data) / rate
        retain_len = len(retained_data)
        data = np.concatenate([retained_data, data], axis=0)
        retained_data = None  # free immediately — no longer needed
        file_start_time = retained_data_start_time
        print(f"    Extended audio: {original_data_length/rate:.1f}s + {retained_duration:.2f}s = {len(data)/rate:.1f}s total")
        print(f"    Adjusted file_start_time to: {file_start_time}")
        print(f"    EOD detection on new file only ({original_data_length/rate:.1f}s)")
    else:
        file_start_time = file_set['timestamp'][n]
        print(f"    Using file timestamp: {file_start_time}")
        
    # Create differential data and determine detection channels
    if parameters['source'] == 'multich_linear':
        data_diff = np.diff(detection_data, axis=1)
        n_detect_channels = n_channels - 1  # Differential pairs
    elif parameters['source'] == '1ch_diff':
        data_diff = detection_data
        n_detect_channels = 1  # Only first channel for single-channel differential
    else:
        raise ValueError(f"Unknown source: {parameters['source']}")

    # Detect pulses across appropriate channels
    peaks = []
    troughs = []
    pulse_widths = []
    
    # Check for bandpass filter settings
    enable_bp = parameters.get('enable_bandpass_filter', False)
    bp_low = parameters.get('bandpass_low_cutoff', 300)
    bp_high = parameters.get('bandpass_high_cutoff', 2000)

    if enable_bp:
        print(f"    Applying bandpass filter ({bp_low}-{bp_high} Hz) for detection")

    for i in range(n_detect_channels):
        # Prepare signal for detection (optionally filtered)
        detection_signal = data_diff[:, i]
        if enable_bp:
            detection_signal = bandpass_filter(detection_signal, rate, bp_low, bp_high)

        ch_peaks, ch_troughs, _, ch_pulse_widths = \
            pulses.detect_pulses(detection_signal, rate, 
                                    thresh=parameters['thresh'], 
                                    min_rel_slope_diff=parameters['min_rel_slope_diff'],
                                    min_width=parameters['min_width_us'] / 1e6,
                                    max_width=parameters['max_width_us'] / 1e6,
                                    width_fac=parameters['width_fac_detection'],
                                    verbose=0,
                                    return_data=False)
        peaks.append(ch_peaks)
        troughs.append(ch_troughs)
        pulse_widths.append(ch_pulse_widths)

    # Unify pulses across channels
    unique_midpoints, unique_peaks, unique_troughs, unique_widths = unify_across_channels(
        peaks, troughs, pulse_widths, proximity_threshold=parameters['duplicate_samples'])

    print(f"  Found {len(unique_midpoints)} unique pulses")
    del peaks, troughs, pulse_widths
    gc.collect()

    # Extract and analyze waveforms
    if len(unique_midpoints) > 0:
        # Extract variable-width snippets and analyze them

        if parameters['waveform_extraction'] == 'Differential':
            (
                eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
                snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc, 
                pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations,
                wf_lengths, snippet_p3_idc, final_p3_idc
            ) = extract_pulse_snippets(
                detection_data, unique_peaks, unique_troughs, rate=rate,
                source='multich_linear', return_differential=parameters['return_diff'],
                interp_factor=parameters['interp_factor'],
                use_pca=False,
                window_mode = parameters['extraction_window'],
                window_factor = parameters['extraction_window_factor'],
                window_length = parameters['extraction_window_length_us'],
                search_window = parameters['search_window'],
                symmetry_threshold=parameters.get('symmetry_threshold', 0.3)
            )
        elif parameters['waveform_extraction'] == 'PCA':
            (
                eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
                snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc, 
                pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations,
                wf_lengths, snippet_p3_idc, final_p3_idc
            ) = extract_pulse_snippets(
                detection_data, unique_peaks, unique_troughs, rate=rate,
                source='multich_linear', return_differential=parameters['return_diff'],
                interp_factor=parameters['interp_factor'],
                use_pca=True, pca_interp_points=300,
                window_mode = parameters['extraction_window'],
                window_factor = parameters['extraction_window_factor'],
                window_length = parameters['extraction_window_length_us'],
                search_window = parameters['search_window']
            )

        if len(eod_snippets) == 0:
            print("    No valid EOD snippets extracted after waveform extraction")
            continue

        # Filter first — eliminates noise before the O(n·k) dedup walk
        if use_ml_filtering and loaded_classifier is not None and loaded_scaler is not None:
            keep_indices, filtered_features, filteredout_features = filter_waveforms_with_classifier(
                eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
                classifier=loaded_classifier,
                scaler=loaded_scaler,
                dur_min=parameters['min_width_us'],
                dur_max=parameters['max_width_us'],
                pp_r_min=parameters['amplitude_ratio_min'],
                pp_r_max=parameters['amplitude_ratio_max'],
                fft_freq_min=parameters['peak_fft_freq_min'],
                fft_freq_max=parameters['peak_fft_freq_max'],
                fish_probability_threshold=fish_probability_threshold,
                use_basic_filtering=True,
                return_features=True,
                return_filteredout_features=True
            )
        else:
            keep_indices, filtered_features, filteredout_features = filter_waveforms(
                eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
                dur_min=parameters['min_width_us'],
                dur_max=parameters['max_width_us'],
                pp_r_min=parameters['amplitude_ratio_min'],
                pp_r_max=parameters['amplitude_ratio_max'],
                fft_freq_min=parameters['peak_fft_freq_min'],
                fft_freq_max=parameters['peak_fft_freq_max'],
                return_features=True, return_filteredout_features=True
            )

        if len(keep_indices) == 0:
            print("    No pulses remaining after filtering")
            continue

        print(f"    Filtered {len(eod_snippets) - len(keep_indices)} out of {len(eod_snippets)} pulses")
        del filtered_features, filteredout_features

        # Build QC table from full pre-filter set and extract filtered-out rows
        filtered_out_indices = np.setdiff1d(np.arange(len(eod_snippets)), keep_indices)
        raw_midpoint_idc = (raw_p1_idc + raw_p2_idc) // 2
        snippet_midpoint_idc = (snippet_p1_idc + snippet_p2_idc) // 2
        pre_filter_table = pd.DataFrame({
            'timestamp': [file_set['timestamp'][n] + dt.timedelta(seconds=t) for t in raw_midpoint_idc / rate],
            'file_timestamp': [file_set['timestamp'][n]] * len(raw_midpoint_idc),
            'filename': [fname] * len(raw_midpoint_idc),
            'midpoint_idx': raw_midpoint_idc,
            'relative_time_s': raw_midpoint_idc / rate,
            'p1_idx': raw_p1_idc,
            'p2_idx': raw_p2_idc,
            'eod_channel': eod_chan,
            'pulse_location': pulse_locations,
            'snippet_p1_idx': snippet_p1_idc,
            'snippet_p2_idx': snippet_p2_idc,
            'snippet_midpoint_idx': snippet_midpoint_idc,
            'wf_length': wf_lengths,
            'eod_amplitude': eod_amps,
            'eod_width_us': eod_widths,
            'eod_amplitude_ratio': amp_ratios,
            'pulse_orientation': pulse_orientations,
            'fft_freq_max': fft_peak_freqs,
            'snippet_p3_idx': snippet_p3_idc,
            'p3_idx': final_p3_idc
        })
        filteredout_eod_table = pre_filter_table.iloc[filtered_out_indices].copy().reset_index(drop=True)
        del pre_filter_table

        # Subset all arrays to the kept pulses, then run dedup on the reduced set
        eod_snippets = [eod_snippets[i] for i in keep_indices]
        eod_amps = eod_amps[keep_indices]
        eod_widths = eod_widths[keep_indices]
        eod_chan = eod_chan[keep_indices]
        is_differential = is_differential[keep_indices]
        snippet_p1_idc = snippet_p1_idc[keep_indices]
        snippet_p2_idc = snippet_p2_idc[keep_indices]
        raw_p1_idc = raw_p1_idc[keep_indices]
        raw_p2_idc = raw_p2_idc[keep_indices]
        pulse_orientations = pulse_orientations[keep_indices]
        amp_ratios = amp_ratios[keep_indices]
        fft_peak_freqs = fft_peak_freqs[keep_indices]
        pulse_locations = pulse_locations[keep_indices]
        wf_lengths = wf_lengths[keep_indices]
        snippet_p3_idc = snippet_p3_idc[keep_indices]
        final_p3_idc = final_p3_idc[keep_indices]

        (
            eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
            snippet_p1_idc, snippet_p2_idc,
            raw_p1_idc, raw_p2_idc,
            pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations, wf_lengths,
            snippet_p3_idc, final_p3_idc
        ) = remove_duplicates(
            eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
            snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
            pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations, wf_lengths,
            snippet_p3_idc, final_p3_idc, parameters
        )

        if len(eod_snippets) == 0:
            print("    No valid EOD snippets remaining after duplicate removal")
            continue

        # Build eod_table from post-filter, post-dedup arrays
        raw_midpoint_idc = (raw_p1_idc + raw_p2_idc) // 2
        snippet_midpoint_idc = (snippet_p1_idc + snippet_p2_idc) // 2
        eod_table = pd.DataFrame({
            'timestamp': [file_set['timestamp'][n] + dt.timedelta(seconds=t) for t in raw_midpoint_idc / rate],
            'file_timestamp': [file_set['timestamp'][n]] * len(raw_midpoint_idc),
            'filename': [fname] * len(raw_midpoint_idc),
            'midpoint_idx': raw_midpoint_idc,
            'relative_time_s': raw_midpoint_idc / rate,
            'p1_idx': raw_p1_idc,
            'p2_idx': raw_p2_idc,
            'eod_channel': eod_chan,
            'pulse_location': pulse_locations,
            'snippet_p1_idx': snippet_p1_idc,
            'snippet_p2_idx': snippet_p2_idc,
            'snippet_midpoint_idx': snippet_midpoint_idc,
            'wf_length': wf_lengths,
            'eod_amplitude': eod_amps,
            'eod_width_us': eod_widths,
            'eod_amplitude_ratio': amp_ratios,
            'pulse_orientation': pulse_orientations,
            'fft_freq_max': fft_peak_freqs,
            'snippet_p3_idx': snippet_p3_idc,
            'p3_idx': final_p3_idc
        })
        filtered_eod_waveforms = eod_snippets

        # Combine with EODs retained from the previous file (event-creation mode only).
        # retained_eod_table sample indices are in the previous combined-audio frame;
        # subtract retained_audio_start_idx to bring them to the retained-audio frame
        # (= start of the current combined audio). New-file indices shift up by retain_len.
        if retained_eod_table is not None:
            idx_cols = ['midpoint_idx', 'p1_idx', 'p2_idx', 'p3_idx']
            ret_adj = retained_eod_table.drop(columns=['original_index'], errors='ignore').copy()
            for col in idx_cols:
                if col in ret_adj.columns:
                    ret_adj[col] -= retained_audio_start_idx
            ret_adj['relative_time_s'] -= retained_audio_start_idx / rate
            new_adj = eod_table.copy()
            for col in idx_cols:
                if col in new_adj.columns:
                    new_adj[col] += retain_len
            new_adj['relative_time_s'] += retain_len / rate
            combined_eod_table = pd.concat([ret_adj, new_adj]).reset_index(drop=True)
            combined_eod_waveforms = retained_eod_waveforms + filtered_eod_waveforms
            retained_eod_table = None
            retained_eod_waveforms = None
            retained_audio_start_idx = None
            print(f"    Combined EOD table: {len(combined_eod_table)} EODs ({len(ret_adj)} retained + {len(new_adj)} new)")
        else:
            combined_eod_table = eod_table
            combined_eod_waveforms = filtered_eod_waveforms


        # =============================================================================
        # Save individual file results and create plots
        # =============================================================================
        # Save individual file results if not creating events
        if not parameters['create_events']:
            output_file = os.path.join(output_path, f'{fname[:-4]}_eod_table.csv')
            eod_table.to_csv(output_file, index=False)

            # Save waveforms
            if len(keep_indices) > 0:
                try:
                    waveform_base = os.path.join(output_path, f'{fname[:-4]}_eod_waveforms')
                    save_waveforms(
                        filtered_eod_waveforms, waveform_base, length=parameters['extraction_window']
                    )
                    
                except Exception as e:
                    print(f"  Warning: Waveform saving failed: {e}")
            else:
                # Create empty waveform files for consistency
                empty_file = os.path.join(output_path, f'{fname[:-4]}_eod_waveforms.npz')
                np.savez_compressed(empty_file, waveforms=np.array([]))
                
                metadata_file = os.path.join(output_path, f'{fname[:-4]}_eod_waveforms_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump({'n_waveforms': 0, 'waveform_length': 0, 'total_samples': 0}, 
                            f, separators=(',', ':'))
            
            # Control plot
            if parameters['create_plots']:
                # Control plots
                if len(keep_indices) > 0:
                    # Plot: EOD detections with appropriate data based on extraction method
                    eod_idc = np.arange(len(filtered_eod_waveforms))
                    
                    # Determine what to plot based on extraction method
                    if parameters['waveform_extraction'] == 'PCA':
                        # Plot single-ended multi-channel data
                        plot_data = data
                        n_plot_channels = data.shape[1]
                        channel_label_prefix = 'Ch'
                        plot_title = f'{fname} - Single-Ended EOD Detections (PCA) - Red=P1, Blue=P2, Grey=Filtered Out (n={len(eod_idc)} kept, {len(filtered_out_indices)} filtered)'
                    else:  # Differential
                        # Plot differential data
                        n_plot_channels = data.shape[1] - 1
                        channel_label_prefix = 'Ch'
                        plot_title = f'{fname} - Differential EOD Detections - Red=P1, Blue=P2, Grey=Filtered Out (n={len(eod_idc)} kept, {len(filtered_out_indices)} filtered)'
                    
                    offset = np.max(eod_table['eod_amplitude']) * 1.5
                    if len(filtered_out_indices) > 0:
                        filteredout_eod_chan = filteredout_eod_table['eod_channel']
                        filteredout_final_p1_idc = filteredout_eod_table['p1_idx']
                        filteredout_final_p2_idc = filteredout_eod_table['p2_idx']
                    plt.figure(figsize=(20, 8))
                    for ch in range(n_plot_channels):
                        if parameters['waveform_extraction'] == 'PCA':
                            # Plot single-ended channel
                            plot_ch_data = data[:, ch]
                            ch_label = f'{channel_label_prefix}{ch}'
                        else:
                            # Plot differential channel
                            plot_ch_data = np.diff(data[:, ch:ch+2]).flatten()
                            ch_label = f'{channel_label_prefix}{ch}-{ch+1}'
                        
                        # Downsample for plotting if needed
                        step = max(1, len(plot_ch_data) // 15000000)
                        x_coords = np.arange(0, len(plot_ch_data), step)
                        plt.plot(x_coords, plot_ch_data[::step] + ((ch + 0.5) * offset), linewidth=0.5, label=ch_label)
                        
                        # Find pulses on this channel and plot them
                        ch_idc = np.where(eod_table['eod_channel'] == ch)[0]
                        actual_idc = eod_idc[ch_idc]
                        
                        if len(actual_idc) > 0:
                            plt.plot(eod_table['p1_idx'].iloc[actual_idc], 
                                    plot_ch_data[eod_table['p1_idx'].iloc[actual_idc]] + ((ch + 0.5) * offset), 
                                    'o', markersize=1, color='red')
                            plt.plot(eod_table['p2_idx'].iloc[actual_idc], 
                                    plot_ch_data[eod_table['p2_idx'].iloc[actual_idc]] + ((ch + 0.5) * offset), 
                                    'o', markersize=1, color='blue')
                            
                            # Plot pulse_location visualization for this channel
                            if 'pulse_location' in eod_table.columns:
                                for idx in actual_idc:
                                    peak_loc = pulse_locations[idx]
                                    p1_idx = eod_table['p1_idx'].iloc[idx]
                                    # Draw thin line from channel offset to pulse_location offset
                                    plt.plot([p1_idx, p1_idx], [(ch + 0.5) * offset, peak_loc * offset], 
                                            'k-', linewidth=0.5, alpha=0.6)
                                    # Mark pulse_location with small black marker
                                    plt.plot(p1_idx, peak_loc * offset, 'ko', markersize=2, alpha=0.8)
                        
                        # Plot filtered-out pulses in grey
                        if len(filtered_out_indices) > 0:
                            filteredout_ch_idc = np.where(filteredout_eod_chan == ch)[0]
                            if len(filteredout_ch_idc) > 0:
                                plt.plot(filteredout_final_p1_idc.iloc[filteredout_ch_idc], 
                                        plot_ch_data[filteredout_final_p1_idc.iloc[filteredout_ch_idc]] + ((ch + 0.5) * offset), 
                                        'o', markersize=1, color='grey', alpha=0.6)
                                plt.plot(filteredout_final_p2_idc.iloc[filteredout_ch_idc], 
                                        plot_ch_data[filteredout_final_p2_idc.iloc[filteredout_ch_idc]] + ((ch + 0.5) * offset), 
                                        'o', markersize=1, color='grey', alpha=0.6)
                    
                    plt.ylim(bottom=None, top=(n_plot_channels-0.5)*offset)
                    plt.title(plot_title)
                    plt.legend(loc='upper right')
                    plt.xlabel('Sample')
                    plt.ylabel('Voltage')
                    plt.savefig(f'{output_path}\\{fname[:-4]}_detection_plot.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    gc.collect()

       
        
            # Optional: Save filtered-out pulses for quality control
            if len(filtered_out_indices) > 0 and parameters['save_filtered_out']:
                try:
                    # Create filtered-out data structures  
                    filteredout_eod_waveforms = [eod_snippets[i] for i in filtered_out_indices]
                    # Save filtered-out data
                    filteredout_file = os.path.join(output_path, f'{fname[:-4]}_eod_table_filteredout.csv')
                    filteredout_eod_table.to_csv(filteredout_file, index=False)
    
                    # Save filtered-out waveforms
                    if len(filteredout_eod_waveforms) > 0:
                        filteredout_waveform_base = os.path.join(output_path, f'{fname[:-4]}_eod_waveforms_filteredout')
                        save_waveforms(filteredout_eod_waveforms, filteredout_waveform_base, length=parameters['extraction_window'])

                except Exception as e:
                    print(f"    Warning: QC file creation failed: {e}")


        # =============================================================================
        # EXTRACT EVENTS
        # =============================================================================
        else: # Create events
            print("  Creating events from extracted EODs...")
            # Create events from eod_table
            print(f"    Stage 1 - Channel-wise temporal clustering using max_ipi_seconds = {parameters['max_ipi_seconds']}")

            if len(combined_eod_table) == 0:
                print("No data to process!")
                continue

            # Create Channel Events
            channel_events = create_channel_events(
                combined_eod_table,
                parameters['max_ipi_seconds']
            )

            if len(channel_events) == 0:
                print("No channel events created!")
                continue

            if parameters['pre_merge_filtering']:
                filtered_channel_events = filter_events(channel_events, parameters['min_eods_premerge'], parameters['min_amplitude'])
            else:
                filtered_channel_events = channel_events.copy()

            if len(filtered_channel_events) == 0:
                print("    No channel events remaining after pre-merge filtering!")
                continue

            if parameters['merge_events']:
                # Merge channel events across channels
                print(f"    Stage 2 - Merging channel events using max_merge_gap_seconds = {parameters['max_merge_gap_seconds']}")
                merged_events = merge_channel_events(filtered_channel_events, parameters['max_merge_gap_seconds'])

                if len(merged_events) == 0:
                    print("No merged events created!")
                    continue
            else:
                merged_events = filtered_channel_events.copy()
                merged_events['merged_event_id'] = merged_events['channel_event_id']
                merged_events['event_start_time'] = merged_events['channel_start_time']
                merged_events['event_end_time'] = merged_events['channel_end_time']

            # Flag events that reach the end of the file
            merged_events['reaches_file_end'] = merged_events['event_end_time'] >= file_start_time + dt.timedelta(seconds=len(data)/rate - parameters['max_ipi_seconds'])

            # Retain events that reach the end of the file and prepare data for next file
            events_at_end = merged_events[merged_events['reaches_file_end']].copy()
            if len(events_at_end) > 0 and n < len(file_set) - 1:  # Don't retain for last file
                print(f"    Found {len(events_at_end['merged_event_id'].unique())} event(s) reaching file end - will retain for next file")
                
                # Calculate retention start index
                retain_start_time = events_at_end['event_start_time'].min() - dt.timedelta(seconds=parameters['margin'])
                retain_start_idx = max(0, int((retain_start_time - file_start_time).total_seconds() * rate))
                
                # Store audio for next iteration (unchanged — used for event WAV export)
                retained_data = data[retain_start_idx:, :].copy()
                retained_data_start_time = retain_start_time
                # Store EOD table rows to skip re-detection of the retained region
                ret_eod_mask = combined_eod_table['midpoint_idx'] >= retain_start_idx
                retained_eod_table = combined_eod_table[ret_eod_mask].drop(columns=['original_index'], errors='ignore').copy()
                retained_eod_waveforms = [combined_eod_waveforms[i] for i in retained_eod_table.index]
                retained_audio_start_idx = retain_start_idx

                print(f"    Retaining {len(retained_data)} samples ({len(retained_data)/rate:.2f}s) starting from {retain_start_time}")
                print(f"    Retained {len(retained_eod_table)} EODs for next iteration")
            else:
                # Clear retention variables
                retained_data = None
                retained_data_start_time = None
                retained_eod_table = None
                retained_eod_waveforms = None
                retained_audio_start_idx = None
                if len(events_at_end) > 0:
                    print(f"    Found {len(events_at_end['merged_event_id'].unique())} event(s) reaching file end (last file - not retaining)")

            # Process events that don't reach file end normally.
            # Also exclude events whose end time falls inside the retained region to
            # avoid double-processing them in the next file iteration.
            if retained_data is not None:
                safe_end_mask = (
                    ~merged_events['reaches_file_end'] &
                    (merged_events['event_end_time'] <= retain_start_time)
                )
            else:
                safe_end_mask = ~merged_events['reaches_file_end']
            events_to_filter = merged_events[safe_end_mask].copy()
            print(f"    Events to process normally: {len(events_to_filter['merged_event_id'].unique())}")

            # Post-merge filter: remove events based on size AND amplitude criteria
            if parameters['post_merge_filtering']:
                final_events = filter_events(events_to_filter, parameters['min_eods_postmerge'], parameters['min_amplitude'])
            else:
                final_events = events_to_filter.copy()
            
            if len(final_events) == 0:
                print("    No final events to process after filtering!")
                continue

            # Reassign sequential event IDs
            unique_ids = sorted(final_events['merged_event_id'].unique())
            for id in unique_ids:
                event_counter += 1  # Increment global event counter
                final_events.loc[final_events['merged_event_id'] == id, 'event_id'] = int(event_counter)

            # Clean up event data
            del merged_events, channel_events, filtered_channel_events, events_to_filter
            gc.collect()

            # Add event information to event_summaries
            # Create event summary directly from event_table BEFORE the processing loop
            for event_id in sorted(final_events['event_id'].unique()):
                event_id = int(event_id)
                event_eods = final_events[final_events['event_id'] == event_id]
                
                # Calculate summary statistics
                duration = (event_eods['timestamp_dt'].max() - 
                            event_eods['timestamp_dt'].min()).total_seconds()

                summary = {
                    'event_id': event_id,
                    'eod_start_time': event_eods['timestamp_dt'].min(),
                    'eod_end_time': event_eods['timestamp_dt'].max(),
                    'event_start_time': max(file_start_time, event_eods['event_start_time'].iloc[0] - dt.timedelta(seconds=parameters['margin'])),
                    'event_end_time': event_eods['event_end_time'].iloc[0] + dt.timedelta(seconds=parameters['margin']),
                    'duration_seconds': duration,
                    'n_eods': len(event_eods),
                    'n_channels': event_eods['eod_channel'].nunique(),
                    'channels_used': ','.join(map(str, sorted(event_eods['eod_channel'].unique()))),
                    'mean_ipi_seconds': event_eods['channel_ipi_seconds'].mean(),
                    'ipi_cv': event_eods['channel_ipi_seconds'].std() / event_eods['channel_ipi_seconds'].mean() if event_eods['channel_ipi_seconds'].mean() > 0 else 0,
                    'median_ipi_seconds': event_eods['channel_ipi_seconds'].median(),
                    'mean_amplitude': event_eods['eod_amplitude'].mean(),
                    'max_amplitude': event_eods['eod_amplitude'].max(),
                    'mean_width_ms': event_eods['eod_width_us'].mean() / 1000 if 'eod_width_us' in event_eods.columns else 0,
                    'n_files': event_eods['file_index'].nunique() if 'file_index' in event_eods.columns else 1,
                    'file_names': ','.join(event_eods['filename'].unique()) if 'filename' in event_eods.columns else 'unknown'
                }
                
                event_summaries.append(summary)
                print("      Created event summary")

                event_start_time = summary['event_start_time']
                event_end_time = summary['event_end_time']

                # Save waveforms for this event
                event_eod_indices = event_eods['original_index'].values
                event_eod_waveforms = [combined_eod_waveforms[i] for i in event_eod_indices]
                if len(event_eod_waveforms) > 0:
                    event_waveform_base = os.path.join(output_path, f'{fname[:-4]}_event_{event_id}_waveforms')
                    save_waveforms(event_eod_waveforms, event_waveform_base, length=parameters['extraction_window'])
                    print(f"      Saved event waveforms: {event_waveform_base}_waveforms")

                # Save audio segment for this event
                event_audio_start_idx = max(0, int((event_start_time - file_start_time).total_seconds() * rate))
                event_audio_end_idx = int((event_end_time - file_start_time).total_seconds() * rate)

                # Extract and save the audio segment
                event_data = data[event_audio_start_idx:event_audio_end_idx,:]
                event_audio_output_file = os.path.join(output_path, f'{fname[:-4]}_event_{event_id}.wav')
                aio.write_audio(event_audio_output_file, event_data, rate)
                print(f"      Saved event audio segment: {event_audio_output_file}")

                # Compute p1/p2/midpoint indices relative to event audio segment
                event_eods['original_p1_idx'] = event_eods['p1_idx']
                event_eods['original_p2_idx'] = event_eods['p2_idx']
                event_eods['original_midpoint_idx'] = event_eods['midpoint_idx']
                event_eods['p1_idx'] = event_eods['p1_idx'] - event_audio_start_idx
                event_eods['p2_idx'] = event_eods['p2_idx'] - event_audio_start_idx
                event_eods['midpoint_idx'] = event_eods['midpoint_idx'] - event_audio_start_idx

                # Save event EOD table
                event_output_file = os.path.join(output_path, f'{fname[:-4]}_event_{event_id}_eod_table.csv')
                event_eods.to_csv(event_output_file, index=False)
                print(f"      Saved event EOD table: {event_output_file}")

                # Create event plots
                if parameters['create_plots']:
                    create_event_plots(
                        event_id=event_id,
                        event_eods=event_eods,
                        event_data=event_data,
                        event_start_time=event_start_time,
                        sample_rate=rate,   
                        output_path=output_path,
                        extraction_method=parameters['waveform_extraction']
                    )
                del event_eods, event_eod_waveforms, event_data

            del final_events, events_at_end

    # Clear memory
    del data, data_diff
    gc.collect()
            
# Save event summaries across all files
if parameters['create_events'] and len(event_summaries) > 0:
    event_summary_df = pd.DataFrame(event_summaries)
    event_summary_df = event_summary_df.sort_values(by=['event_start_time']).reset_index(drop=True)
    event_summary_file = os.path.join(output_path, 'all_event_summaries.csv')
    event_summary_df.to_csv(event_summary_file, index=False)
    print(f"\nSaved all event summaries to: {event_summary_file}")

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