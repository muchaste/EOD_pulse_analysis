# -*- coding: utf-8 -*-

"""
Shuttle-box Matlab pulse extraction

Extract EOD pulses from Shuttle-box Matlab recordings.

This script:
    - selects a root folder containing .bin recording files and matching .txt logfiles
    - tunes detection parameters once on a test segment from the first file
    - applies tuned parameters to all files in the session
    - runs segmentwise extraction (default 10-minute segments) per file
    - saves per-file pulse tables and waveforms
    - saves combined EOD table across all files
    - computes and saves EOD rate summaries at 1s, 1min, and 10min resolution
    
"""

import tkinter
from tkinter import messagebox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import detrend
import datetime as dt
import gc
import mat73
import math
import glob
import thunderfish.pulses as pulses
from scipy import signal
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulse_functions import (
    extract_pulse_snippets,
    save_waveforms,
    bandpass_filter,
    unify_across_channels,
    filter_waveforms,
    remove_duplicates,
    normalize_waveforms)

from parameter_gui import ShuttleboxConfigGUI

tuned = False 


# ============================================================================
# FILE SELECTION AND PARAMETER CONFIGURATION
# ============================================================================
root = tkinter.Tk()
config_gui = ShuttleboxConfigGUI(root)
root.mainloop()

if config_gui.result is None:
    print("Configuration cancelled by user")
    sys.exit()

root_folder = config_gui.result['root_folder']
output_path = config_gui.result['output_path']
os.makedirs(output_path, exist_ok=True)

bin_files = sorted(glob.glob(os.path.join(root_folder, '*.bin')))
file_pairs = []
for bf in bin_files:
    lf = os.path.dirname(bf) + '/log_' + os.path.basename(bf).replace('.bin', '.txt')
    if os.path.exists(lf):
        file_pairs.append((bf, lf))
    else:
        print(f"Warning: no logfile found for {os.path.basename(bf)}, skipping")

if not file_pairs:
    print("No valid .bin/.txt file pairs found. Exiting.")
    sys.exit()

print(f"Found {len(file_pairs)} file pair(s)")



# ============================================================================
# INITIAL DETECTION PARAMETERS (from GUI, tuned on first file's test segment)
# ============================================================================
parameters = config_gui.result['parameters']

# # Unpack for use in the tuning phase below
# test_seg_length = int(parameters['test_seg_length'])
# enable_bp = parameters['enable_bp']
# tuning_thresh = parameters['tuning_thresh']
# thresh = parameters['thresh']
# bp_low_cutoff = parameters['bp_low_cutoff']
# bp_high_cutoff = parameters['bp_high_cutoff']
# min_rel_slope_diff = parameters['min_rel_slope_diff']
# min_width_us = parameters['min_width_us']
# max_width_us = parameters['max_width_us']
# width_fac_detection = parameters['width_fac_detection']
# duplicate_samples = int(parameters['duplicate_samples'])
# interp_factor = int(parameters['interp_factor'])
# amplitude_ratio_min = parameters['amplitude_ratio_min']
# amplitude_ratio_max = parameters['amplitude_ratio_max']
# peak_fft_freq_min = parameters['peak_fft_freq_min']
# peak_fft_freq_max = parameters['peak_fft_freq_max']
# extraction_window_length_us = parameters['extraction_window_length_us']
# extraction_window_factor = parameters['extraction_window_factor']
# search_window = int(parameters['search_window'])


# ============================================================================
# TUNING PHASE: single test segment from the first file
# ============================================================================
tune_fname, tune_logfname = file_pairs[0]
tune_stem = os.path.splitext(os.path.basename(tune_fname))[0]

logtext = open(tune_logfname, "r").readlines()
filetime = pd.to_datetime(logtext[0], format='%Y\\%m\\%d ; %H:%M:%S.%f\n')
fish_id = logtext[1].split(':')[1][1:-1]
sex = logtext[2].split(':')[1][1:-1]
rate = int(logtext[8].split(':')[1][1:-1])
gain = float(logtext[12].split(':')[1][1:-1])

print(f"\nTuning on: {tune_stem}  (fish: {fish_id}, {sex}, rate: {rate} Hz)")

data_dict = mat73.loadmat(tune_fname)
data_raw = pd.DataFrame(data_dict['data']) / gain
del data_dict
gc.collect()

# Build test segment detection DataFrame
left_v = detrend(data_raw.iloc[:(int(parameters['test_seg_length']) * rate), 0])
left_h = detrend(data_raw.iloc[:(int(parameters['test_seg_length']) * rate), 1])
right_v = detrend(data_raw.iloc[:(int(parameters['test_seg_length']) * rate), 2])
right_h = detrend(data_raw.iloc[:(int(parameters['test_seg_length']) * rate), 3])

test_data_df = pd.DataFrame({
    'right_v_dt': right_v,
    'right_h_dt': right_h,
    'right_diff_dt': right_v - right_h,
    'right_sum_dt': right_v + right_h,
    'left_v_dt': left_v,
    'left_h_dt': left_h,
    'left_diff_dt': left_v - left_h,
    'left_sum_dt': left_v + left_h,
    'side_diff_sum': right_v - right_h + left_v - left_h,
    'side_diff_diff': right_v - right_h - left_v - left_h,
    'all_diff': right_v - right_h - left_v - left_h
})

# Detect pulses in test segment across all channels, then unify
peaks = []
troughs = []
pulse_widths = []

for i in range(test_data_df.shape[1]):
    detection_signal = np.array(test_data_df.iloc[:, i])
    if parameters['enable_bp']:
        detection_signal = bandpass_filter(detection_signal, rate, parameters['bp_low_cutoff'], parameters['bp_high_cutoff'])
    ch_peaks, ch_troughs, _, ch_pulse_widths = pulses.detect_pulses(
        detection_signal, rate,
        thresh=parameters['tuning_thresh'],
        min_rel_slope_diff=parameters['min_rel_slope_diff'],
        min_width=parameters['min_width_us'] / 1e6,
        max_width=parameters['max_width_us'] / 1e6,
        width_fac=parameters['width_fac_detection'],
        verbose=0,
        return_data=False)
    peaks.append(ch_peaks)
    troughs.append(ch_troughs)
    pulse_widths.append(ch_pulse_widths)

# Unify AFTER collecting all channels
unique_midpoints, unique_peaks, unique_troughs, unique_widths = unify_across_channels(
    peaks, troughs, pulse_widths, proximity_threshold=int(parameters['duplicate_samples']))

del peaks, troughs, pulse_widths
gc.collect()

print(f"  Found {len(unique_midpoints)} unique pulses in test segment")

# Multi-panel detection plot
ncol = 2
fig, axs = plt.subplots(int(np.ceil(test_data_df.shape[1] / ncol)), ncol, figsize=(12, 8), sharex=True)
for i, channel in enumerate(test_data_df.columns):
    row = i // ncol
    col = i % ncol
    axs[row, col].plot(test_data_df[channel], label=channel)
    axs[row, col].scatter(unique_peaks, test_data_df[channel].iloc[unique_peaks],
                          color='red', s=5, label='Peaks')
    axs[row, col].scatter(unique_troughs, test_data_df[channel].iloc[unique_troughs],
                          color='blue', s=5, label='Troughs')
    axs[row, col].set_title(channel)
    axs[row, col].legend(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(output_path, f'{tune_stem}_test_detection.png'), dpi=150)
plt.show()

# Extract snippets from test segment
# Concatenate left_v, left_h, right_v, right_h into an array for extraction
detection_data = np.array(test_data_df[['left_v_dt', 'left_h_dt', 'right_v_dt', 'right_h_dt']])

tuned = False

while not tuned:
    parameters = config_gui.result['parameters']


    (
        eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
        snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
        pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations,
        wf_lengths, snippet_p3_idc, final_p3_idc
    ) = extract_pulse_snippets(
        detection_data, unique_peaks, unique_troughs, rate=rate,
        source='shuttlebox_matlab', return_differential=True,
        interp_factor=parameters['interp_factor'],
        use_pca=False,
        window_mode='variable',
        window_factor=parameters['extraction_window_factor'],
        window_length=parameters['extraction_window_length_us'],
        search_window=parameters['search_window']
    )

    keep_indices, filtered_features, filteredout_features = filter_waveforms(
        eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
        dur_min=parameters['min_width_us'], dur_max=parameters['max_width_us'],
        pp_r_min=parameters['amplitude_ratio_min'], pp_r_max=parameters['amplitude_ratio_max'],
        fft_freq_min=parameters['peak_fft_freq_min'], fft_freq_max=parameters['peak_fft_freq_max'],
        return_features=True, return_filteredout_features=True
    )
    del filtered_features, filteredout_features
    print(f"    Filtered {len(eod_snippets) - len(keep_indices)} out of {len(eod_snippets)} pulses")

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
        snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
        pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations, wf_lengths,
        snippet_p3_idc, final_p3_idc
    ) = remove_duplicates(
        eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
        snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
        pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations, wf_lengths,
        snippet_p3_idc, final_p3_idc, parameters
    )
    print(f"    {len(eod_snippets)} valid EOD snippets after duplicate removal")

    normalized_waveforms = np.array(normalize_waveforms(
        eod_snippets, snippet_p1_idc, snippet_p2_idc,
        crop_and_interpolate=True, crop_factor=10, target_length=300
    ))

    # Tune parameters from test segment statistics
    min_width_us = eod_widths.min() * 0.9
    max_width_us = eod_widths.max() * 1.1
    amplitude_ratio_min = amp_ratios.min() * 0.9
    amplitude_ratio_max = amp_ratios.max() * 1.1
    peak_fft_freq_min = fft_peak_freqs.min() * 0.9
    peak_fft_freq_max = fft_peak_freqs.max() * 1.1
    bp_low_cutoff = peak_fft_freq_min // 2
    bp_high_cutoff = peak_fft_freq_max * 2

    # Analysis figure
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(14, 9))
    n_samples = normalized_waveforms.shape[1]
    # Waveforms are interpolated by interp_factor, so effective sample rate is rate*interp_factor
    time_axis = (np.arange(n_samples) - n_samples // 2) / (rate * int(parameters['interp_factor'])) * 1000

    for wf in normalized_waveforms:
        ax1.plot(time_axis, wf, 'b-', alpha=0.3, linewidth=0.5)
    mean_waveform = np.mean(normalized_waveforms, axis=0)
    ax1.plot(time_axis, mean_waveform, 'r-', linewidth=2, label=f'Mean (n={len(normalized_waveforms)})')
    ax1.set_title('Normalized Waveforms Overlay')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(amp_ratios, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(amp_ratios), color='r', linestyle='dashed', linewidth=1)
    ax2.axvline(amplitude_ratio_min, color='g', linestyle='dashed', linewidth=1)
    ax2.axvline(amplitude_ratio_max, color='g', linestyle='dashed', linewidth=1)
    ax2.set_title(f'Amplitude Ratios\nMean: {np.mean(amp_ratios):.3f} ± {np.std(amp_ratios):.3f}, Min: {amp_ratios.min():.3f}, Max: {amp_ratios.max():.3f}')
    ax2.set_xlabel('Peak/Trough Ratio')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)

    ax3.hist(eod_widths, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(eod_widths), color='r', linestyle='dashed', linewidth=1)
    ax3.axvline(min_width_us, color='g', linestyle='dashed', linewidth=1)
    ax3.axvline(max_width_us, color='g', linestyle='dashed', linewidth=1)
    ax3.set_title(f'Pulse Durations\nMean: {np.mean(eod_widths):.1f} ± {np.std(eod_widths):.1f} us, Min: {eod_widths.min():.1f}, Max: {eod_widths.max():.1f}')
    ax3.set_xlabel('Duration (us)')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)

    ax4.hist(fft_peak_freqs, bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(fft_peak_freqs), color='r', linestyle='dashed', linewidth=1)
    ax4.axvline(peak_fft_freq_min, color='g', linestyle='dashed', linewidth=1)
    ax4.axvline(peak_fft_freq_max, color='g', linestyle='dashed', linewidth=1)
    ax4.set_title(f'Peak FFT Frequencies\nMean: {np.mean(fft_peak_freqs):.0f} ± {np.std(fft_peak_freqs):.0f} Hz, Min: {fft_peak_freqs.min():.0f}, Max: {fft_peak_freqs.max():.0f}')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3)

    ax5.axis('off')

    psd_list = []
    for wf in normalized_waveforms:
        freqs_psd, psd = signal.welch(wf, fs=rate * int(parameters['interp_factor']), nperseg=min(len(wf), 1024))
        psd_list.append(psd)
    mean_psd = np.mean(psd_list, axis=0)
    freq_mask = freqs_psd <= 100000
    for psd in psd_list:
        ax6.loglog(freqs_psd[freq_mask], psd[freq_mask], 'b-', alpha=0.3, linewidth=0.5)
    ax6.loglog(freqs_psd[freq_mask], mean_psd[freq_mask], 'r-', linewidth=1)
    ax6.set_title('Power Spectral Density (0-100 kHz)')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Power Spectral Density')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{tune_stem}_EOD_analysis.png'), dpi=300)
    plt.show()

    # Update parameters dictionary with tuned values
    parameters.update({
        'min_width_us': min_width_us,
        'max_width_us': max_width_us,
        'amplitude_ratio_min': amplitude_ratio_min,
        'amplitude_ratio_max': amplitude_ratio_max,
        'peak_fft_freq_min': peak_fft_freq_min,
        'peak_fft_freq_max': peak_fft_freq_max,
        'bp_low_cutoff': bp_low_cutoff,
        'bp_high_cutoff': bp_high_cutoff
    })
    # Popup message box to inform user of tuned parameters and to ask if they want to proceed
    message = "Tuned parameters:\n"
    for k, v in parameters.items():
        message += f"{k}: {v}\n"
    message += "\nDo you want to proceed with these parameters?"

    if messagebox.askyesno("Tuned Parameters", message):
        tuned = True

    else:
        root = tkinter.Tk()
        config_gui = ShuttleboxConfigGUI(root)
        root.mainloop()
        parameters = config_gui.result['parameters']



print(f"\nTuned parameters:")
for k, v in parameters.items():
    print(f"  {k}: {v}")

params_file = os.path.join(output_path, 'tuned_parameters.txt')
with open(params_file, 'w') as f:
    f.write(f"Tuned from: {tune_stem}\n\n")
    for key, value in parameters.items():
        f.write(f"{key}: {value}\n")

del test_data_df, detection_data, eod_snippets, normalized_waveforms
gc.collect()


# ============================================================================
# MAIN EXTRACTION LOOP: all files in session
# ============================================================================
eod_waveforms_all = []
eod_table_all = pd.DataFrame()

for fname, logfname in file_pairs:
    file_stem = os.path.splitext(os.path.basename(fname))[0]
    print(f"\nProcessing: {file_stem}")

    # Skip loading the first file since it was already loaded during tuning
    if fname is not file_pairs[0][0]:
        logtext = open(logfname, "r").readlines()
        filetime = pd.to_datetime(logtext[0], format='%Y\\%m\\%d ; %H:%M:%S.%f\n')
        fish_id = logtext[1].split(':')[1][1:-1]
        sex = logtext[2].split(':')[1][1:-1]
        rate = int(logtext[8].split(':')[1][1:-1])
        gain = float(logtext[12].split(':')[1][1:-1])

        data_dict = mat73.loadmat(fname)
        data_raw = pd.DataFrame(data_dict['data']) / gain
        del data_dict
        gc.collect()

    seglength = int(parameters['seg_length_min']) * 60 * rate
    no_segments = math.ceil(len(data_raw) / seglength)
    left_v = detrend(data_raw.iloc[:, 0])
    left_h = detrend(data_raw.iloc[:, 1])
    right_v = detrend(data_raw.iloc[:, 2])
    right_h = detrend(data_raw.iloc[:, 3])

    data_df = pd.DataFrame({
        'right_v_dt': right_v,
        'right_h_dt': right_h,
        'right_diff_dt': right_v - right_h,
        'right_sum_dt': right_v + right_h,
        'left_v_dt': left_v,
        'left_h_dt': left_h,
        'left_diff_dt': left_v - left_h,
        'left_sum_dt': left_v + left_h,
        'side_diff_sum': right_v - right_h + left_v - left_h,
        'side_diff_diff': right_v - right_h - left_v - left_h,
        'all_diff': right_v - right_h - left_v - left_h
    })

    eod_waveforms_file = []
    eod_table_file = pd.DataFrame()

    for seg_i in range(no_segments):
        print(f"  Segment {seg_i + 1}/{no_segments}...")
        start_idx = seg_i * seglength
        end_idx = min((seg_i + 1) * seglength, len(data_df))

        # Subset segment and only left_v, left_h, right_v, right_h for detection
        segment_data = np.array(data_df.iloc[start_idx:end_idx, :][['left_v_dt', 'left_h_dt', 'right_v_dt', 'right_h_dt']])

        # Collect detections across all channels
        peaks = []
        troughs = []
        pulse_widths = []

        for j in range(segment_data.shape[1]):
            detection_signal = segment_data[:, j]
            if parameters['enable_bp']:
                detection_signal = bandpass_filter(
                    detection_signal, rate,
                    parameters['bp_low_cutoff'], parameters['bp_high_cutoff'])
            ch_peaks, ch_troughs, _, ch_pulse_widths = pulses.detect_pulses(
                detection_signal, rate,
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

        # Unify after all channels are collected
        unique_midpoints, unique_peaks, unique_troughs, unique_widths = unify_across_channels(
            peaks, troughs, pulse_widths, proximity_threshold=parameters['duplicate_samples'])
        del peaks, troughs, pulse_widths
        gc.collect()

        if len(unique_midpoints) == 0:
            print("    No pulses detected in segment")
            continue

        print(f"    Found {len(unique_midpoints)} unique pulses")

        (
            eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
            snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
            pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations,
            wf_lengths, snippet_p3_idc, final_p3_idc
        ) = extract_pulse_snippets(
            segment_data, unique_peaks, unique_troughs, rate=rate,
            source='shuttlebox_matlab', return_differential=True,
            interp_factor=parameters['interp_factor'],
            use_pca=False,
            window_mode='variable',
            window_factor=parameters['extraction_window_factor'],
            window_length=parameters['extraction_window_length_us'],
            search_window=parameters['search_window']
        )

        keep_indices, filtered_features, filteredout_features = filter_waveforms(
            eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
            dur_min=parameters['min_width_us'], dur_max=parameters['max_width_us'],
            pp_r_min=parameters['amplitude_ratio_min'],
            pp_r_max=parameters['amplitude_ratio_max'],
            fft_freq_min=parameters['peak_fft_freq_min'],
            fft_freq_max=parameters['peak_fft_freq_max'],
            return_features=True, return_filteredout_features=True
        )
        del filtered_features, filteredout_features

        if len(keep_indices) == 0:
            print("    No pulses remaining after filtering")
            continue

        print(f"    Filtered {len(eod_snippets) - len(keep_indices)} / {len(eod_snippets)} pulses")

        eod_snippets = [eod_snippets[k] for k in keep_indices]
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
            snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
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

        print(f"    {len(eod_snippets)} valid EOD snippets after duplicate removal")


        raw_midpoint_idc = start_idx + (raw_p1_idc + raw_p2_idc) // 2
        snippet_midpoint_idc = (snippet_p1_idc + snippet_p2_idc) // 2

        # Loop over raw_midpoint_idc and extract amplitudes and orientations for each channel
        # eod_chan_amps = np.zeros((len(raw_midpoint_idc), 4))  # 4 channels: left_v, left_h, right_v, right_h
        # eod_chan_orientations = np.zeros((len(raw_midpoint_idc), 4))

        eod_chan_amps = np.abs(segment_data[raw_p1_idc] - segment_data[raw_p2_idc])
        eod_chan_orientations = np.sign(segment_data[raw_p1_idc] - segment_data[raw_p2_idc]).astype(int)

        # for i, raw_p1_idx in enumerate(raw_p1_idc):
        #     if raw_p1_idx < 0 or raw_p1_idx >= len(segment_data):
        #         continue
        #     p1_dat = segment_data[raw_p1_idx, :]
        #     p2_dat = segment_data[raw_p2_idc[i], :]
            
        #     eod_chan_amps[i, :] = np.abs(p1_dat - p2_dat)
        #     eod_chan_orientations[i, :] = np.sign(p1_dat - p2_dat)

        eod_table_segment = pd.DataFrame({
            'timestamp': [filetime + dt.timedelta(seconds=t) for t in raw_midpoint_idc / rate],
            'file_timestamp': [filetime] * len(raw_midpoint_idc),
            'filename': [os.path.basename(fname)] * len(raw_midpoint_idc),
            'fish_id': [fish_id] * len(raw_midpoint_idc),
            'sex': [sex] * len(raw_midpoint_idc),
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
            'p3_idx': final_p3_idc,
            'amp_left_v':     eod_chan_amps[:, 0],
            'amp_left_h':     eod_chan_amps[:, 1],
            'amp_right_v':    eod_chan_amps[:, 2],
            'amp_right_h':    eod_chan_amps[:, 3],
            'orient_left_v':  eod_chan_orientations[:, 0],
            'orient_left_h':  eod_chan_orientations[:, 1],
            'orient_right_v': eod_chan_orientations[:, 2],
            'orient_right_h': eod_chan_orientations[:, 3]
        })
        eod_table_file = pd.concat([eod_table_file, eod_table_segment], ignore_index=True)
        eod_waveforms_file.extend(eod_snippets)

    if len(eod_table_file) > 0:
        per_file_csv = os.path.join(output_path, f'{file_stem}_pulse_extraction_results.csv')
        eod_table_file.to_csv(per_file_csv, index=False)
        waveform_base = os.path.join(output_path, f'{file_stem}_eod_waveforms')
        save_waveforms(eod_waveforms_file, waveform_base, format='npz', length='variable')
        print(f"  Saved {len(eod_table_file)} EODs -> {os.path.basename(per_file_csv)}")

    eod_table_all = pd.concat([eod_table_all, eod_table_file], ignore_index=True)
    eod_waveforms_all.extend(eod_waveforms_file)

    del data_raw, data_df, eod_table_file, eod_waveforms_file
    gc.collect()


# ============================================================================
# SAVE COMBINED RESULTS
# ============================================================================
if len(eod_table_all) > 0:
    combined_csv = os.path.join(output_path, 'all_files_eod_table.csv')
    eod_table_all.to_csv(combined_csv, index=False)
    print(f"\nSaved combined table: {len(eod_table_all)} EODs total -> {combined_csv}")


# ============================================================================
# EOD RATE SUMMARIES
# ============================================================================
if len(eod_table_all) > 0:
    eod_table_all = eod_table_all.sort_values('timestamp').reset_index(drop=True)

    # IPI: diff between consecutive timestamps; NaN at file boundaries
    eod_table_all['ipi_s'] = eod_table_all['timestamp'].diff().dt.total_seconds()
    file_first_rows = eod_table_all.groupby('filename').head(1).index
    eod_table_all.loc[file_first_rows, 'ipi_s'] = np.nan

    for bin_label, freq_str in [('1s', 's'), ('1min', 'min'), ('10min', '10min')]:
        bin_key = eod_table_all['timestamp'].dt.floor(freq_str)
        rate_df = eod_table_all.groupby(bin_key).agg(
            n_eods=('timestamp', 'count'),
            mean_ipi_s=('ipi_s', 'mean'),
            std_ipi_s=('ipi_s', 'std'),
            mean_amplitude=('eod_amplitude', 'mean'),
            std_amplitude=('eod_amplitude', 'std')
        ).reset_index()
        rate_df.rename(columns={'timestamp': 'bin_time'}, inplace=True)
        rate_df['cv_ipi'] = rate_df['std_ipi_s'] / rate_df['mean_ipi_s']
        rate_df['cv_amplitude'] = rate_df['std_amplitude'] / rate_df['mean_amplitude']
        out_csv = os.path.join(output_path, f'eod_rate_{bin_label}.csv')
        rate_df.to_csv(out_csv, index=False)
        print(f"Saved EOD rate summary ({bin_label}): {out_csv}")

print("\nDone.")