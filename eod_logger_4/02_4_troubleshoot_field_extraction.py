"""
03_4_troubleshoot_field_extraction.py
Re-extracts EODs from a single calibrated event WAV for diagnostic purposes.
Run stepwise in Spyder using #%% cell markers.
No event creation, no file saving.
"""

#%% -- IMPORTS ---------------------------------------------------------------

import configparser
import tkinter as tk
from tkinter import filedialog
import audioio as aio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import thunderfish.pulses as pulses
import gc

from pulse_functions import (
    bandpass_filter,
    unify_across_channels,
    extract_pulse_snippets,
    remove_duplicates,
    filter_waveforms,
    filter_waveforms_with_classifier,
)

root = tk.Tk()
root.withdraw()

#%% -- LOAD CONFIG -----------------------------------------------------------

config_file = filedialog.askopenfilename(
    title="Load Configuration (.cfg)",
    filetypes=[("Config files", "*.cfg"), ("All files", "*.*")]
)

config = configparser.ConfigParser()
config.read(config_file)

bool_keys = {
    "enable_bandpass_filter", "return_diff", "create_events", "merge_events",
    "pre_merge_filtering", "post_merge_filtering", "save_filtered_out", "create_plots"
}
int_keys = {
    "duplicate_samples", "interp_factor", "search_window",
    "min_eods_premerge", "min_eods_postmerge"
}
str_keys = {"source", "waveform_extraction", "extraction_window"}

parameters = {}
if "Parameters" in config:
    for key, val in config["Parameters"].items():
        if key in bool_keys:
            parameters[key] = val.lower() in ("true", "1", "yes")
        elif key in int_keys:
            parameters[key] = int(float(val))
        elif key in str_keys:
            parameters[key] = val
        else:
            parameters[key] = float(val)

use_ml_filtering = False
classifier_path = ""
fish_probability_threshold = 0.5
if "MachineLearning" in config:
    use_ml_filtering = config["MachineLearning"].getboolean("use_ml_filtering", fallback=False)
    classifier_path = config["MachineLearning"].get("classifier_path", "")
    fish_probability_threshold = config["MachineLearning"].getfloat("fish_probability_threshold", fallback=0.5)

print("Parameters loaded:")
for k, v in parameters.items():
    print(f"  {k}: {v}")
print(f"  use_ml_filtering: {use_ml_filtering}")

loaded_classifier = None
loaded_scaler = None
if use_ml_filtering and classifier_path:
    import pickle
    with open(classifier_path, "rb") as f:
        clf_data = pickle.load(f)
    loaded_classifier = clf_data["classifier"]
    loaded_scaler = clf_data["scaler"]
    print(f"Loaded classifier: {clf_data['classifier_name']} (accuracy {clf_data['accuracy']:.3f})")

#%% -- LOAD FILE -------------------------------------------------------------

event_file = filedialog.askopenfilename(
    title="Select Event Audio File",
    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
)

data, rate = aio.load_audio(event_file)
n_channels = data.shape[1]
file_duration = len(data) / rate
print(f"Loaded: {event_file}")
print(f"  {n_channels} channels, {rate} Hz, {file_duration:.3f}s, {len(data)} samples")

#%% -- FULL FILE PLOT --------------------------------------------------------

data_diff_full = np.diff(data, axis=1)
offset = np.max(np.abs(data_diff_full)) * 1.5

plt.figure(figsize=(20, 8))
for ch in range(n_channels - 1):
    step = max(1, len(data_diff_full) // 5000000)
    x = np.arange(0, len(data_diff_full), step) / rate
    plt.plot(x, data_diff_full[::step, ch] + ch * offset, linewidth=0.5, label=f"Ch{ch}-{ch+1}")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (offset)")
plt.title(f"Full event - differential - {event_file.split(chr(92))[-1]}")
plt.legend(loc="upper right", fontsize=7)
plt.tight_layout()
plt.show()

#%% -- TRIM WINDOW -----------------------------------------------------------
# Edit t_start_s and t_end_s, then re-run this cell and all cells below.
# Set both to 0.0 to use the full file.

t_start_s = 0.0
t_end_s = 0.0

s_idx = int(t_start_s * rate)
e_idx = int(t_end_s * rate) if t_end_s > 0 else len(data)
detection_data = data[s_idx:e_idx, :]
print(f"Detection window: {t_start_s:.4f}s - {e_idx/rate:.4f}s  ({len(detection_data)} samples)")

#%% -- TRIMMED PLOT ----------------------------------------------------------

data_diff = np.diff(detection_data, axis=1)
offset = np.max(np.abs(data_diff)) * 1.5

plt.figure(figsize=(20, 8))
for ch in range(n_channels - 1):
    step = max(1, len(data_diff) // 5000000)
    x = (np.arange(0, len(data_diff), step) + s_idx) / rate
    plt.plot(x, data_diff[::step, ch] + ch * offset, linewidth=0.5, label=f"Ch{ch}-{ch+1}")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (offset)")
plt.title(f"Trimmed window [{t_start_s:.4f}s - {e_idx/rate:.4f}s]")
plt.legend(loc="upper right", fontsize=7)
plt.tight_layout()
plt.show()

#%% -- DETECT ----------------------------------------------------------------

enable_bp = parameters.get("enable_bandpass_filter", False)
bp_low = parameters.get("bandpass_low_cutoff", 300)
bp_high = parameters.get("bandpass_high_cutoff", 2000)

peaks_list = []
troughs_list = []
widths_list = []

for pair_idx in range(n_channels - 1):
    sig = data_diff[:, pair_idx]
    if enable_bp:
        sig = bandpass_filter(sig, rate, bp_low, bp_high)
    ch_peaks, ch_troughs, _, ch_widths = pulses.detect_pulses(
        sig, rate,
        thresh=parameters["thresh"],
        min_rel_slope_diff=parameters["min_rel_slope_diff"],
        min_width=parameters["min_width_us"] / 1e6,
        max_width=parameters["max_width_us"] / 1e6,
        width_fac=parameters["width_fac_detection"],
        verbose=0,
        return_data=False
    )
    peaks_list.append(ch_peaks)
    troughs_list.append(ch_troughs)
    widths_list.append(ch_widths)
    print(f"  Pair {pair_idx} (Ch{pair_idx}-Ch{pair_idx+1}): {len(ch_peaks)} pulses detected")

unique_midpoints, unique_peaks, unique_troughs, unique_widths = unify_across_channels(
    peaks_list, troughs_list, widths_list,
    proximity_threshold=parameters["duplicate_samples"]
)
print(f"After unification: {len(unique_midpoints)} unique pulses")
del peaks_list, troughs_list, widths_list

#%% -- EXTRACT ---------------------------------------------------------------
# return_diff_override is read from config but can be overridden here.
# Change it manually and re-run from this cell to compare behaviour.

return_diff_override = parameters["return_diff"]   # override: True or False
print(f"return_differential = {return_diff_override}")

(
    eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
    snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
    pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations,
    wf_lengths, snippet_p3_idc, final_p3_idc
) = extract_pulse_snippets(
    detection_data, unique_peaks, unique_troughs, rate=rate,
    source="multich_linear",
    return_differential=return_diff_override,
    interp_factor=int(parameters["interp_factor"]),
    use_pca=(parameters.get("waveform_extraction", "Differential") == "PCA"),
    window_mode=parameters.get("extraction_window", "fixed"),
    window_factor=int(parameters.get("extraction_window_factor", 10)),
    window_length=parameters.get("extraction_window_length_us", 4000),
    search_window=int(parameters["search_window"]),
    symmetry_threshold=parameters.get("symmetry_threshold", 0.3)
)

print(f"Snippets extracted: {len(eod_snippets)}")
print(f"  is_differential: 0={np.sum(is_differential==0)}  1={np.sum(is_differential==1)}  2={np.sum(is_differential==2)}")

#%% -- FILTER + DEDUP --------------------------------------------------------

if use_ml_filtering and loaded_classifier is not None and loaded_scaler is not None:
    keep_indices, _, _ = filter_waveforms_with_classifier(
        eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
        classifier=loaded_classifier, scaler=loaded_scaler,
        dur_min=parameters["min_width_us"], dur_max=parameters["max_width_us"],
        pp_r_min=parameters["amplitude_ratio_min"], pp_r_max=parameters["amplitude_ratio_max"],
        fft_freq_min=parameters["peak_fft_freq_min"], fft_freq_max=parameters["peak_fft_freq_max"],
        fish_probability_threshold=fish_probability_threshold,
        use_basic_filtering=True, return_features=True, return_filteredout_features=True
    )
else:
    keep_indices, _, _ = filter_waveforms(
        eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
        dur_min=parameters["min_width_us"], dur_max=parameters["max_width_us"],
        pp_r_min=parameters["amplitude_ratio_min"], pp_r_max=parameters["amplitude_ratio_max"],
        fft_freq_min=parameters["peak_fft_freq_min"], fft_freq_max=parameters["peak_fft_freq_max"],
        return_features=True, return_filteredout_features=True
    )

filtered_out_indices = np.setdiff1d(np.arange(len(eod_snippets)), keep_indices)
print(f"After filter: {len(keep_indices)} kept, {len(filtered_out_indices)} filtered out")

fo_raw_p1  = raw_p1_idc[filtered_out_indices]
fo_raw_p2  = raw_p2_idc[filtered_out_indices]
fo_eod_chan = eod_chan[filtered_out_indices]

eod_snippets    = [eod_snippets[i] for i in keep_indices]
eod_amps        = eod_amps[keep_indices]
eod_widths      = eod_widths[keep_indices]
eod_chan        = eod_chan[keep_indices]
is_differential = is_differential[keep_indices]
snippet_p1_idc  = snippet_p1_idc[keep_indices]
snippet_p2_idc  = snippet_p2_idc[keep_indices]
raw_p1_idc      = raw_p1_idc[keep_indices]
raw_p2_idc      = raw_p2_idc[keep_indices]
pulse_orientations = pulse_orientations[keep_indices]
amp_ratios      = amp_ratios[keep_indices]
fft_peak_freqs  = fft_peak_freqs[keep_indices]
pulse_locations = pulse_locations[keep_indices]
wf_lengths      = wf_lengths[keep_indices]
snippet_p3_idc  = snippet_p3_idc[keep_indices]
final_p3_idc    = final_p3_idc[keep_indices]

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

raw_midpoint_idc = (raw_p1_idc + raw_p2_idc) // 2
print(f"After dedup: {len(eod_snippets)} pulses")

#%% -- DIAGNOSTIC TABLE ------------------------------------------------------

eod_table = pd.DataFrame({
    "relative_time_s":    raw_midpoint_idc / rate,
    "p1_idx":             raw_p1_idc,
    "p2_idx":             raw_p2_idc,
    "eod_channel":        eod_chan,
    "is_differential":    is_differential,
    "pulse_orientation":  pulse_orientations,
    "eod_amplitude":      eod_amps,
    "eod_width_us":       eod_widths,
    "eod_amplitude_ratio": amp_ratios,
    "fft_freq_max":       fft_peak_freqs,
    "pulse_location":     pulse_locations,
})

print(f"\nreturn_differential={return_diff_override}")
print(f"Total kept: {len(eod_table)}  |  is_diff=0: {(is_differential==0).sum()}  is_diff=1: {(is_differential==1).sum()}")
print(eod_table.to_string())

#%% -- DIAGNOSTIC PLOT -------------------------------------------------------

offset = np.max(np.abs(data_diff)) * 1.5
n_diff_ch = n_channels - 1

plt.figure(figsize=(20, 8))
for ch in range(n_diff_ch):
    step = max(1, len(data_diff) // 5000000)
    x = (np.arange(0, len(data_diff), step) + s_idx) / rate
    plt.plot(x, data_diff[::step, ch] + ch * offset, linewidth=0.5,
             label=f"Ch{ch}-{ch+1}", color="steelblue")

    ch_mask = eod_chan == ch
    if ch_mask.any():
        p1 = raw_p1_idc[ch_mask]
        p2 = raw_p2_idc[ch_mask]
        plt.plot((p1 + s_idx) / rate, data_diff[p1, ch] + ch * offset,
                 "o", markersize=5, color="red", zorder=3)
        plt.plot((p2 + s_idx) / rate, data_diff[p2, ch] + ch * offset,
                 "o", markersize=5, color="blue", zorder=3)

    fo_ch_mask = fo_eod_chan == ch
    if fo_ch_mask.any():
        p1_fo = fo_raw_p1[fo_ch_mask]
        p2_fo = fo_raw_p2[fo_ch_mask]
        plt.plot((p1_fo + s_idx) / rate, data_diff[p1_fo, ch] + ch * offset,
                 "o", markersize=4, color="grey", alpha=0.6, zorder=2)
        plt.plot((p2_fo + s_idx) / rate, data_diff[p2_fo, ch] + ch * offset,
                 "o", markersize=4, color="grey", alpha=0.6, zorder=2)

plt.ylim(-0.5 * offset, (n_diff_ch - 0.5) * offset)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (offset)")
plt.title(f"return_diff={return_diff_override} -- {len(eod_table)} kept (red=P1 blue=P2), {len(filtered_out_indices)} grey=filtered")
plt.legend(loc="upper right", fontsize=7)
plt.tight_layout()
plt.show()
