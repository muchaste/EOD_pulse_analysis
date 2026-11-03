# -*- coding: utf-8 -*-
"""
02_3_Pulse_extraction_field.py

This script extracts EOD pulses from multi-channel field recordings

Authors: Stefan Mucha with Claude Sonnet 4.6

"""

import thunderfish.pulses as pulses
import matplotlib.pyplot as plt
import audioio as aio
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
import glob
import datetime as dt
import os
import json
import pickle
from scipy.signal import correlate
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import EOD functions
from pulse_functions import (
    bandpass_filter,
    unify_across_channels,
    extract_pulse_snippets,
    remove_duplicates,
    filter_waveforms,
    filter_waveforms_with_classifier,
    save_waveforms,
    EventState,
    flush_pending_events,
    process_events,
    calc_fwhm_width,
    detect_monophasic_pulses,
    detect_pulses_envelope,
    extract_pulse_snippets_p1
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

_total_eods_kept = 0  # Across all files - used for the zero-pulse guard rail at the end

# Parameters for event creation
if parameters['create_events']:
    print("Setting event extraction parameters...")

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

    # If the logger lost the time, 
    if timecol[0].year == 2019:
        print("Logger seems to have lost the time. Select another file with correct timestamp for the first recording")
        anchor_file = filedialog.askopenfilename()
        # anchor_timestamp = pd.to_datetime(anchor_file.split('/')[-1].split('-')[-1].split('.')[0], format='%Y%m%dT%H%M%S')
        anchor_timestamp = pd.to_datetime(os.path.basename(anchor_file).split('-')[-1][0:-4], format='%Y%m%dT%H%M%S')
        anchor_data,rate = aio.load_audio(anchor_file)
        first_file, _ = aio.load_audio(filelist[0])
        
        # calculate offset
        # Compute cross-correlation
        correlation = correlate(anchor_data[:,0], first_file[:,0])
        
        # Plot the cross-correlation
        plt.figure(figsize=(10, 5))
        plt.plot(correlation)
        plt.title('Cross-Correlation between Recordings')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.savefig('%s\\cross_correlation.png' % (output_path))
        plt.show(block=False)
        # Find the index of the peak
        offset_index = np.argmax(correlation) - (len(first_file[:,0]) - 1)
        offset_seconds = offset_index/rate
        first_timestamp = anchor_timestamp + dt.timedelta(seconds = offset_seconds)
        first_timedelta = first_timestamp - timecol[0]
        timecol = timecol + first_timedelta
        
        # Create a DataFrame with the offset information
        df = pd.DataFrame({'Offset_samples': [offset_index], 'Offset_seconds': [offset_seconds],\
                        'File': [filelist[0]], 'Anchor': [anchor_file],\
                            'New_first_timestamp':[first_timestamp]})
        # Save the DataFrame to a CSV file
        df.to_csv('%s\\cross_correlation_offset.csv' %(output_path), index=False)
        
        del anchor_data, first_file, correlation


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
_preview_file_id = os.path.basename(file_set['filename'][0]).split('.')[0]
_ch_cols = [c for c in cor_factors_all.columns if c.startswith('ch_')]
_preview_row = cor_factors_all.loc[cor_factors_all['file_id'] == _preview_file_id].iloc[0]
_preview_cf = _preview_row[_ch_cols].values.astype(float)
_preview_dead_0idx = np.where(np.isnan(_preview_cf))[0]
_preview_live_0idx = np.where(~np.isnan(_preview_cf))[0]
for i in _preview_live_0idx:
    data[:, i] *= _preview_cf[i]
if len(_preview_dead_0idx) > 0:
    print("\n*** WARNING: dead channel(s) in preview file %s: %s — those channels uncalibrated ***" %
          (_preview_file_id, ", ".join("ch%d" % (c + 1) for c in _preview_dead_0idx)))

# Plot raw data
offset = np.max(abs(data))#*1.5
plt.figure(figsize=(40,12))
for i in range(n_channels):
    if i in _preview_dead_0idx:
        plt.plot(data[0:int(60*rate-1), i] + i*offset, color='red', linestyle='--',
                 linewidth=1.5, label='ch%d DEAD' % (i + 1))
    else:
        plt.plot(data[0:int(60*rate-1), i] + i*offset, label=str(i+1))
    plt.hlines(y=parameters['thresh'] + i*offset, xmin=0, xmax=int(60*rate-1))

plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
if len(_preview_dead_0idx) > 0:
    plt.title('Preview — DEAD CHANNEL(S): %s' % ", ".join("ch%d" % (c + 1) for c in _preview_dead_0idx),
              color='red', fontweight='bold')
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
_ch_cols = [c for c in cor_factors_all.columns if c.startswith('ch_')]
_dead_channel_log = []  # accumulates per-file dead channel entries
state = EventState()    # cross-file continuation state, event counter and event summaries




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
    
    # Calibrate with correction factor — live channels only, preserve physical channel space
    _file_id_str = fname.split('.')[0]
    _row_cf = cor_factors_all.loc[cor_factors_all['file_id'] == _file_id_str].iloc[0]
    _cf_values = _row_cf[_ch_cols].values.astype(float)  # NaN for dead channels
    _dead_ch_0idx = np.where(np.isnan(_cf_values))[0]    # 0-indexed dead channel indices
    _live_ch_0idx = np.where(~np.isnan(_cf_values))[0]
    # pairs where BOTH endpoints are live — preserves physical pair numbering
    _live_pair_indices = [i for i in range(n_channels - 1)
                          if i not in _dead_ch_0idx and (i + 1) not in _dead_ch_0idx]
    _dead_pair_indices = [i for i in range(n_channels - 1) if i not in _live_pair_indices]
    for i in _live_ch_0idx:
        data[:, i] *= _cf_values[i]
    if len(_dead_ch_0idx) > 0:
        _dead_str = ", ".join("ch%d" % (c + 1) for c in _dead_ch_0idx)
        _dead_pair_str = ", ".join("pair%d" % (p + 1) for p in _dead_pair_indices)
        print(f"    *** DEAD CHANNEL(S): {_dead_str} — skipping detection pairs: {_dead_pair_str} ***")
        _dead_channel_log.append({
            'file_id': _file_id_str,
            'dead_channels': ",".join(str(c + 1) for c in _dead_ch_0idx),
            'dead_pairs': ",".join(str(p + 1) for p in _dead_pair_indices),
            'n_live_pairs': len(_live_pair_indices)
        })
    _dead_channels_file_str = ",".join("ch%d" % (c + 1) for c in _dead_ch_0idx)
    
    # EOD detection and waveform extraction run on the new file only.
    # Audio is still concatenated with the retained tail for event WAV export.
    detection_data = data
    retain_len = 0

    # Handle cross-file continuation: prepend retained audio from previous file
    if state.retained_data is not None:
        print(f"    Concatenating {len(state.retained_data)} samples from previous file (audio only)")
        original_data_length = len(data)
        retained_duration = len(state.retained_data) / rate
        retain_len = len(state.retained_data)
        data = np.concatenate([state.retained_data, data], axis=0)
        state.retained_data = None  # free immediately — no longer needed
        # Anchor the combined audio to THIS file's name time, so that the new file's pulse
        # timestamps (file name + index / rate) and the audio frame agree even when the file
        # names are not exactly 600 s apart (N7). The retained pulses keep the timestamps of
        # the previous file's frame.
        file_start_time = file_set['timestamp'][n] - dt.timedelta(seconds=retain_len / rate)
        print(f"    Extended audio: {original_data_length/rate:.1f}s + {retained_duration:.2f}s = {len(data)/rate:.1f}s total")
        print(f"    Adjusted file_start_time to: {file_start_time} (retained buffer started {state.retained_data_start_time})")
        print(f"    EOD detection on new file only ({original_data_length/rate:.1f}s)")
    else:
        file_start_time = file_set['timestamp'][n]
        print(f"    Using file timestamp: {file_start_time}")
        
    # Create detection signal(s) and determine which channels/pairs to scan.
    # extraction_method selects WHICH signal the loop below detects on (differential pairs
    # vs raw single-ended channels, matching whichever signal channel-selection will use);
    # pulse_mode independently selects WHICH algorithm runs on it.
    if parameters['source'] == 'multich_linear':
        if parameters['extraction_method'] == 'differential':
            data_diff = np.diff(detection_data, axis=1)
            # Use only live adjacent pairs; _live_pair_indices preserves physical channel numbering
            detect_indices = _live_pair_indices
        else:
            data_diff = detection_data
            detect_indices = list(_live_ch_0idx)
    elif parameters['source'] == '1ch_diff':
        data_diff = detection_data
        detect_indices = [0]
    else:
        raise ValueError(f"Unknown source: {parameters['source']}")

    # The envelope route detects on all channels at once (below), not per channel/pair
    use_envelope_detection = parameters.get('detection_method', 'per_channel') == 'envelope'
    # The amplitude ratio is meaningless for monophasic pulses (and the envelope route): its limits are not applied
    use_amplitude_ratio = parameters.get('pulse_mode', 'biphasic') != 'monophasic' and not use_envelope_detection
    if use_envelope_detection and parameters['extraction_method'] not in ('single_channel', 'pca'):
        raise ValueError("detection_method='envelope' supports only extraction_method='single_channel' or 'pca'")
    if use_envelope_detection:
        detect_indices = []

    # Detect pulses across appropriate channels
    peaks = []
    troughs = []
    pulse_widths = []
    
    # Check for bandpass filter settings
    enable_bp = parameters.get('enable_bandpass_filter', False) and not use_envelope_detection
    bp_low = parameters.get('bandpass_low_cutoff', 300)
    bp_high = parameters.get('bandpass_high_cutoff', 2000)

    if enable_bp:
        print(f"    Applying bandpass filter ({bp_low}-{bp_high} Hz) for detection")

    for ch_idx in detect_indices:
        # Prepare signal for detection (optionally filtered)
        detection_signal = data_diff[:, ch_idx]
        if enable_bp:
            detection_signal = bandpass_filter(detection_signal, rate, bp_low, bp_high)

        if parameters.get('pulse_mode', 'biphasic') == 'monophasic':
            # Simple threshold peak-finding instead of thunderfish's biphasic
            # peak-trough slope-pairing model (see detect_monophasic_pulses docstring).
            ch_peaks, ch_troughs, _, ch_pulse_widths = detect_monophasic_pulses(
                detection_signal, rate,
                thresh=parameters['thresh'],
                min_width_us=parameters['min_width_us'],
                max_width_us=parameters['max_width_us'],
                min_distance_us=parameters['min_distance_us'])
        else:
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

    # Envelope route (monophasic pulses): one P1 per pulse from the channel-wide envelope.
    # Dead channels are zeroed so they cannot win the envelope or enter the location profile.
    if use_envelope_detection:
        envelope_data = detection_data
        if len(_dead_ch_0idx) > 0:
            envelope_data = detection_data.copy()
            envelope_data[:, _dead_ch_0idx] = 0
        unique_peaks, envelope_channels, _ = detect_pulses_envelope(
            envelope_data, rate,
            thresh=parameters['thresh'],
            peak_window_us=parameters['envelope_peak_window_us'],
            shadow_us=parameters['envelope_shadow_us'],
            shadow_fraction=parameters['envelope_shadow_fraction'],
            highpass_hz=parameters['envelope_highpass_hz'],
            biphasic_merge_us=parameters.get('envelope_biphasic_merge_us', 2000))
        unique_midpoints = unique_peaks
        unique_troughs = unique_peaks

    print(f"  Found {len(unique_midpoints)} unique pulses")
    del peaks, troughs, pulse_widths
    gc.collect()

    # Extract and analyze waveforms
    if len(unique_midpoints) > 0:
        # Extract variable-width snippets and analyze them
        if use_envelope_detection:
            (
                eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
                snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
                pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations,
                wf_lengths, snippet_p3_idc, final_p3_idc, p1_amplitudes
            ) = extract_pulse_snippets_p1(
                envelope_data, unique_peaks, envelope_channels, rate=rate,
                window_length=parameters['extraction_window_length_us'],
                interp_factor=parameters['interp_factor'],
                location_method=parameters.get('location_method', 'auto'),
                extraction_method=parameters['extraction_method'],
                pca_interp_points=300
            )
        else:
            p1_amplitudes = None
            (
                eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
                snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc, 
                pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations,
                wf_lengths, snippet_p3_idc, final_p3_idc
            ) = extract_pulse_snippets(
                detection_data, unique_peaks, unique_troughs, rate=rate,
                source='multich_linear', extraction_method=parameters['extraction_method'],
                location_method=parameters.get('location_method', 'auto'),
                interp_factor=parameters['interp_factor'],
                pca_interp_points=300,
                window_mode = parameters['extraction_window'],
                window_factor = parameters['extraction_window_factor'],
                window_length = parameters['extraction_window_length_us'],
                search_window = parameters['search_window'],
                symmetry_threshold=parameters.get('symmetry_threshold', 0.3),
                pulse_mode=parameters.get('pulse_mode', 'biphasic'),
                triphasic_search=parameters.get('triphasic_search', False)
            )

        if len(eod_snippets) == 0:
            print("    No valid EOD snippets extracted after waveform extraction")
            flush_pending_events(state, parameters, rate, output_path, data, file_start_time, fname, n == len(file_set) - 1)
            continue

        # Monophasic pulses: peak-to-trough distance misrepresents true pulse duration
        # (small second phase), so use FWHM of the dominant phase instead. The envelope
        # route already returns FWHM in eod_widths.
        if parameters.get('pulse_mode', 'biphasic') == 'monophasic' and not use_envelope_detection:
            eod_widths = calc_fwhm_width(
                eod_snippets, snippet_p1_idc, rate, interp_factor=parameters['interp_factor'])

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
                use_amplitude_ratio=use_amplitude_ratio,
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
                use_amplitude_ratio=use_amplitude_ratio,
                return_features=True, return_filteredout_features=True
            )

        if len(keep_indices) == 0:
            print("    No pulses remaining after filtering")
            flush_pending_events(state, parameters, rate, output_path, data, file_start_time, fname, n == len(file_set) - 1)
            continue

        print(f"    Filtered {len(eod_snippets) - len(keep_indices)} out of {len(eod_snippets)} pulses")
        del filtered_features, filteredout_features

        # Build QC table from full pre-filter set and extract filtered-out rows
        filtered_out_indices = np.setdiff1d(np.arange(len(eod_snippets)), keep_indices)
        # P2 is -1 for the envelope route (P1-only): the pulse position is then P1 itself
        raw_midpoint_idc = np.where(raw_p2_idc >= 0, (raw_p1_idc + raw_p2_idc) // 2, raw_p1_idc)
        snippet_midpoint_idc = np.where(snippet_p2_idc >= 0, (snippet_p1_idc + snippet_p2_idc) // 2, snippet_p1_idc)
        pre_filter_table = pd.DataFrame({
            'timestamp': [file_set['timestamp'][n] + dt.timedelta(seconds=t) for t in raw_midpoint_idc / rate],
            'file_timestamp': [file_set['timestamp'][n]] * len(raw_midpoint_idc),
            'filename': [fname] * len(raw_midpoint_idc),
            'midpoint_idx': raw_midpoint_idc,
            'relative_time_s': raw_midpoint_idc / rate,
            'p1_idx': raw_p1_idc,
            'p2_idx': raw_p2_idc,
            'eod_channel': eod_chan,
            'is_differential': is_differential,
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
            'p3_idx': final_p3_idc,
            'dead_channels_file': [_dead_channels_file_str] * len(raw_midpoint_idc)
        })
        if p1_amplitudes is not None:
            pre_filter_table['p1_amplitude'] = p1_amplitudes
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
        if p1_amplitudes is not None:
            p1_amplitudes = p1_amplitudes[keep_indices]

        # The envelope route yields one unique P1 per pulse, so it needs no duplicate removal
        if not use_envelope_detection:
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
            flush_pending_events(state, parameters, rate, output_path, data, file_start_time, fname, n == len(file_set) - 1)
            continue

        _total_eods_kept += len(eod_snippets)

        # Build eod_table from post-filter, post-dedup arrays
        raw_midpoint_idc = np.where(raw_p2_idc >= 0, (raw_p1_idc + raw_p2_idc) // 2, raw_p1_idc)
        snippet_midpoint_idc = np.where(snippet_p2_idc >= 0, (snippet_p1_idc + snippet_p2_idc) // 2, snippet_p1_idc)
        eod_table = pd.DataFrame({
            'timestamp': [file_set['timestamp'][n] + dt.timedelta(seconds=t) for t in raw_midpoint_idc / rate],
            'file_timestamp': [file_set['timestamp'][n]] * len(raw_midpoint_idc),
            'filename': [fname] * len(raw_midpoint_idc),
            'midpoint_idx': raw_midpoint_idc,
            'relative_time_s': raw_midpoint_idc / rate,
            'p1_idx': raw_p1_idc,
            'p2_idx': raw_p2_idc,
            'eod_channel': eod_chan,
            'is_differential': is_differential,
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
            'p3_idx': final_p3_idc,
            'dead_channels_file': [_dead_channels_file_str] * len(raw_midpoint_idc)
        })
        if p1_amplitudes is not None:
            eod_table['p1_amplitude'] = p1_amplitudes
        filtered_eod_waveforms = eod_snippets

        # Combine with EODs retained from the previous file (event-creation mode only).
        # retained_eod_table sample indices are in the previous combined-audio frame;
        # subtract retained_audio_start_idx to bring them to the retained-audio frame
        # (= start of the current combined audio). New-file indices shift up by retain_len.
        state.n_prev_retained = len(state.retained_eod_table) if state.retained_eod_table is not None else 0

        if state.retained_eod_table is not None:
            idx_cols = ['midpoint_idx', 'p1_idx', 'p2_idx', 'p3_idx']
            ret_adj = state.retained_eod_table.drop(columns=['original_index'], errors='ignore').copy()
            for col in idx_cols:
                if col in ret_adj.columns:
                    # negative values (-1 = P2/P3 not set) are not sample indices; leave them
                    ret_adj[col] = ret_adj[col].where(ret_adj[col] < 0, ret_adj[col] - state.retained_audio_start_idx)
            ret_adj['relative_time_s'] -= state.retained_audio_start_idx / rate
            new_adj = eod_table.copy()
            for col in idx_cols:
                if col in new_adj.columns:
                    new_adj[col] = new_adj[col].where(new_adj[col] < 0, new_adj[col] + retain_len)
            new_adj['relative_time_s'] += retain_len / rate
            combined_eod_table = pd.concat([ret_adj, new_adj]).reset_index(drop=True)
            combined_eod_waveforms = state.retained_eod_waveforms + filtered_eod_waveforms
            state.retained_eod_table = None
            state.retained_eod_waveforms = None
            state.retained_audio_start_idx = None
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

                    offset = np.max(eod_table['eod_amplitude']) * 1.5
                    if len(filtered_out_indices) > 0:
                        filteredout_eod_chan = filteredout_eod_table['eod_channel']
                        filteredout_final_p1_idc = filteredout_eod_table['p1_idx']
                        filteredout_final_p2_idc = filteredout_eod_table['p2_idx']

                    plt.figure(figsize=(20, 8))

                    def _plot_channel_lane(plot_ch_data, ch_label, lane_idx, kept_mask, filteredout_mask):
                        # Downsample for plotting if needed
                        step = max(1, len(plot_ch_data) // 15000000)
                        x_coords = np.arange(0, len(plot_ch_data), step)
                        plt.plot(x_coords, plot_ch_data[::step] + ((lane_idx + 0.5) * offset), linewidth=0.5, label=ch_label)

                        actual_idc = eod_idc[np.where(kept_mask)[0]]
                        if len(actual_idc) > 0:
                            plt.plot(eod_table['p1_idx'].iloc[actual_idc],
                                    plot_ch_data[eod_table['p1_idx'].iloc[actual_idc]] + ((lane_idx + 0.5) * offset),
                                    'o', markersize=1, color='red')
                            kept_p2 = eod_table['p2_idx'].iloc[actual_idc]
                            kept_p2 = kept_p2[kept_p2 >= 0]  # -1 = no P2 (envelope route)
                            plt.plot(kept_p2,
                                    plot_ch_data[kept_p2] + ((lane_idx + 0.5) * offset),
                                    'o', markersize=1, color='blue')

                            if 'pulse_location' in eod_table.columns:
                                for idx in actual_idc:
                                    peak_loc = pulse_locations[idx]
                                    p1_idx = eod_table['p1_idx'].iloc[idx]
                                    plt.plot([p1_idx, p1_idx], [(lane_idx + 0.5) * offset, peak_loc * offset],
                                            'k-', linewidth=0.5, alpha=0.6)
                                    plt.plot(p1_idx, peak_loc * offset, 'ko', markersize=2, alpha=0.8)

                        if len(filtered_out_indices) > 0:
                            filteredout_ch_idc = np.where(filteredout_mask)[0]
                            if len(filteredout_ch_idc) > 0:
                                plt.plot(filteredout_final_p1_idc.iloc[filteredout_ch_idc],
                                        plot_ch_data[filteredout_final_p1_idc.iloc[filteredout_ch_idc]] + ((lane_idx + 0.5) * offset),
                                        'o', markersize=1, color='grey', alpha=0.6)
                                filteredout_p2 = filteredout_final_p2_idc.iloc[filteredout_ch_idc]
                                filteredout_p2 = filteredout_p2[filteredout_p2 >= 0]
                                plt.plot(filteredout_p2,
                                        plot_ch_data[filteredout_p2] + ((lane_idx + 0.5) * offset),
                                        'o', markersize=1, color='grey', alpha=0.6)

                    if parameters['extraction_method'] == 'pca':
                        # Single-ended multi-channel data; eod_channel always a raw channel index.
                        n_plot_lanes = data.shape[1]
                        for ch in range(n_plot_lanes):
                            fo_mask = filteredout_eod_chan.values == ch if len(filtered_out_indices) > 0 else np.array([])
                            _plot_channel_lane(
                                data[:, ch], f'Ch{ch}', ch,
                                eod_table['eod_channel'].values == ch,
                                fo_mask
                            )
                        plot_title = f'{fname} - Single-Ended EOD Detections (PCA) - Red=P1, Blue=P2, Grey=Filtered Out (n={len(eod_idc)} kept, {len(filtered_out_indices)} filtered)'
                    elif parameters['extraction_method'] == 'differential':
                        # A differential run only ever produces is_differential in {1, -1}
                        # (noise is already excluded from eod_table), so a single set of
                        # differential-pair lanes covers every kept/filtered-out pulse.
                        n_plot_lanes = data.shape[1] - 1
                        for ch in range(n_plot_lanes):
                            fo_mask = filteredout_eod_chan.values == ch if len(filtered_out_indices) > 0 else np.array([])
                            _plot_channel_lane(
                                np.diff(data[:, ch:ch+2]).flatten(), f'Ch{ch}-{ch+1}', ch,
                                eod_table['eod_channel'].values == ch,
                                fo_mask
                            )
                        plot_title = f'{fname} - Differential EOD Detections - Red=P1, Blue=P2, Grey=Filtered Out (n={len(eod_idc)} kept, {len(filtered_out_indices)} filtered)'
                    else:
                        # A single-ended run (extraction_method='single_channel') always assigns
                        # a real raw channel (_select_best_singleended_channel never returns -1),
                        # so a single set of single-ended lanes covers every pulse.
                        n_plot_lanes = data.shape[1]
                        for ch in range(n_plot_lanes):
                            fo_mask = filteredout_eod_chan.values == ch if len(filtered_out_indices) > 0 else np.array([])
                            _plot_channel_lane(
                                data[:, ch], f'Ch{ch}', ch,
                                eod_table['eod_channel'].values == ch,
                                fo_mask
                            )
                        plot_title = f'{fname} - Single-Ended EOD Detections - Red=P1, Blue=P2, Grey=Filtered Out (n={len(eod_idc)} kept, {len(filtered_out_indices)} filtered)'

                    plt.ylim(bottom=None, top=(n_plot_lanes-0.5)*offset)
                    plt.title(plot_title)
                    plt.legend(loc='upper right', fontsize=7)
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
            process_events(state, parameters, rate, output_path, data, file_start_time, fname,
                           combined_eod_table, combined_eod_waveforms, n == len(file_set) - 1)

    # File without any new pulses while an event is pending: finalise it now
    if state.retained_eod_table is not None and state.retained_data is None:
        flush_pending_events(state, parameters, rate, output_path, data, file_start_time, fname, n == len(file_set) - 1)

    # Clear memory
    del data, data_diff
    gc.collect()
            
# Save event summaries across all files
if parameters['create_events'] and len(state.event_summaries) > 0:
    event_summary_df = pd.DataFrame(state.event_summaries)
    event_summary_df = event_summary_df.sort_values(by=['event_start_time']).reset_index(drop=True)
    event_summary_file = os.path.join(output_path, 'all_event_summaries.csv')
    event_summary_df.to_csv(event_summary_file, index=False)
    print(f"\nSaved all event summaries to: {event_summary_file}")

if _total_eods_kept == 0:
    print(f"\n*** No pulses survived filtering across the whole run (pulse_mode="
          f"{parameters.get('pulse_mode', 'biphasic')}, extraction_method={parameters['extraction_method']}). "
          f"This combination may not suit the data - e.g. monophasic signals rarely produce "
          f"clean differential flips, so try switching extraction_method or loosening width/"
          f"amplitude_ratio/fft bounds. ***")

# Save dead channel log
if len(_dead_channel_log) > 0:
    dead_log_df = pd.DataFrame(_dead_channel_log)
    dead_log_file = os.path.join(output_path, 'dead_channel_log.csv')
    dead_log_df.to_csv(dead_log_file, index=False)
    print(f"\n*** Dead channels were detected in {len(_dead_channel_log)} file(s). Log saved to: {dead_log_file} ***")
else:
    print("\nNo dead channels detected across all files.")

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
