"""
02_4_troubleshoot_field_extraction.py
Re-extracts EODs from a single calibrated event WAV for diagnostic purposes.
Mirrors the current 02_3_Pulse_extraction_field.py pipeline: return_diff selects the
extraction mode (differential pair selection vs single-ended largest-peak selection,
see pulse_functions._select_differential_channel_pointwise /
_select_best_singleended_channel), pulse_mode independently selects the detection
algorithm (thunderfish vs detect_monophasic_pulses) and width/orientation logic.
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
    detect_monophasic_pulses,
    calc_fwhm_width,
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
str_keys = {"source", "waveform_extraction", "extraction_window", "pulse_mode"}

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

# Defaults for configs saved before pulse_mode/min_distance_us existed.
parameters.setdefault("pulse_mode", "biphasic")
parameters.setdefault("min_distance_us", 2000)

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

#%% -- FULL FILE PLOT (DIFFERENTIAL) ------------------------------------------

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

#%% -- FULL FILE PLOT (SINGLE-ENDED) ------------------------------------------
# Monophasic/single-electrode-dominant pulses can be large on one raw channel while
# barely visible in its neighbouring differential pairs - check both views regardless
# of which extraction mode (return_diff) you intend to run below.

offset_se = np.max(np.abs(data)) * 1.5

plt.figure(figsize=(20, 8))
for ch in range(n_channels):
    step = max(1, len(data) // 5000000)
    x = np.arange(0, len(data), step) / rate
    plt.plot(x, data[::step, ch] + ch * offset_se, linewidth=0.5, label=f"Ch{ch}")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (offset)")
plt.title(f"Full event - single-ended - {event_file.split(chr(92))[-1]}")
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
plt.title(f"Trimmed window [{t_start_s:.4f}s - {e_idx/rate:.4f}s] - differential")
plt.legend(loc="upper right", fontsize=7)
plt.tight_layout()
plt.show()

offset_se = np.max(np.abs(detection_data)) * 1.5

plt.figure(figsize=(20, 8))
for ch in range(n_channels):
    step = max(1, len(detection_data) // 5000000)
    x = (np.arange(0, len(detection_data), step) + s_idx) / rate
    plt.plot(x, detection_data[::step, ch] + ch * offset_se, linewidth=0.5, label=f"Ch{ch}")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (offset)")
plt.title(f"Trimmed window [{t_start_s:.4f}s - {e_idx/rate:.4f}s] - single-ended")
plt.legend(loc="upper right", fontsize=7)
plt.tight_layout()
plt.show()

#%% -- DETECT ----------------------------------------------------------------
# return_diff selects WHICH signal detection runs on (differential pairs vs raw
# single-ended channels) - same role as in 02_3. pulse_mode independently selects the
# detection algorithm (detect_monophasic_pulses vs thunderfish.pulses.detect_pulses)
# applied to whichever signal is chosen here. Both can be overridden here manually and
# re-run from this cell to compare behaviour.

return_diff_override = parameters["return_diff"]
pulse_mode_override = parameters["pulse_mode"]
print(f"return_diff = {return_diff_override}, pulse_mode = {pulse_mode_override}")

enable_bp = parameters.get("enable_bandpass_filter", False)
bp_low = parameters.get("bandpass_low_cutoff", 300)
bp_high = parameters.get("bandpass_high_cutoff", 2000)

if return_diff_override:
    detect_signal_source = data_diff
    detect_indices = list(range(n_channels - 1))
else:
    detect_signal_source = detection_data
    detect_indices = list(range(n_channels))

peaks_list = []
troughs_list = []
widths_list = []

for ch_idx in detect_indices:
    sig = detect_signal_source[:, ch_idx]
    if enable_bp:
        sig = bandpass_filter(sig, rate, bp_low, bp_high)

    if pulse_mode_override == "monophasic":
        ch_peaks, ch_troughs, _, ch_widths = detect_monophasic_pulses(
            sig, rate,
            thresh=parameters["thresh"],
            min_width_us=parameters["min_width_us"],
            max_width_us=parameters["max_width_us"],
            min_distance_us=parameters["min_distance_us"]
        )
    else:
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
    label = f"Ch{ch_idx}-{ch_idx+1}" if return_diff_override else f"Ch{ch_idx}"
    print(f"  {label}: {len(ch_peaks)} pulses detected")

unique_midpoints, unique_peaks, unique_troughs, unique_widths = unify_across_channels(
    peaks_list, troughs_list, widths_list,
    proximity_threshold=parameters["duplicate_samples"]
)
print(f"After unification: {len(unique_midpoints)} unique pulses")
del peaks_list, troughs_list, widths_list

#%% -- EXTRACT ---------------------------------------------------------------

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
    symmetry_threshold=parameters.get("symmetry_threshold", 0.3),
    pulse_mode=pulse_mode_override
)

print(f"Snippets extracted: {len(eod_snippets)}")
print(f"  is_differential: -1={np.sum(is_differential==-1)}  0={np.sum(is_differential==0)}  1={np.sum(is_differential==1)}  2={np.sum(is_differential==2)}")

# Monophasic pulses: peak-to-trough distance misrepresents true pulse duration (small
# second phase), so use FWHM of the dominant phase instead - same as 02_3.
if pulse_mode_override == "monophasic":
    eod_widths = calc_fwhm_width(eod_snippets, snippet_p1_idc, rate, interp_factor=int(parameters["interp_factor"]))

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

def _channel_description(chan, is_diff):
    if is_diff == 1:
        return f"diff {chan}-{chan+1}"
    elif is_diff in (0, 2):
        return f"single ch{chan}"
    else:
        return "none (-1)"

eod_table = pd.DataFrame({
    "relative_time_s":    raw_midpoint_idc / rate,
    "p1_idx":             raw_p1_idc,
    "p2_idx":             raw_p2_idc,
    "eod_channel":        eod_chan,
    "is_differential":    is_differential,
    "channel_description": [_channel_description(c, d) for c, d in zip(eod_chan, is_differential)],
    "pulse_orientation":  pulse_orientations,
    "eod_amplitude":      eod_amps,
    "eod_width_us":       eod_widths,
    "eod_amplitude_ratio": amp_ratios,
    "fft_freq_max":       fft_peak_freqs,
    "pulse_location":     pulse_locations,
})

print(f"\npulse_mode={pulse_mode_override}  return_diff={return_diff_override}")
print(f"Total kept: {len(eod_table)}  |  is_diff=-1: {(is_differential==-1).sum()}  is_diff=0/2: {((is_differential==0)|(is_differential==2)).sum()}  is_diff=1: {(is_differential==1).sum()}")
print(eod_table.to_string())

#%% -- DIAGNOSTIC PLOT --------------------------------------------------------
# A given run only ever produces one representation (return_diff is fixed for the whole
# run), so - unlike the old dual-lane design - a single set of lanes covers every pulse.

if return_diff_override:
    offset = np.max(np.abs(data_diff)) * 1.5
    n_plot_ch = n_channels - 1
    plot_data = data_diff
    ch_label_fmt = lambda ch: f"Ch{ch}-{ch+1}"
    plot_title_prefix = "Differential"
else:
    offset = np.max(np.abs(detection_data)) * 1.5
    n_plot_ch = n_channels
    plot_data = detection_data
    ch_label_fmt = lambda ch: f"Ch{ch}"
    plot_title_prefix = "Single-ended"

plt.figure(figsize=(20, 8))
for ch in range(n_plot_ch):
    step = max(1, len(plot_data) // 5000000)
    x = (np.arange(0, len(plot_data), step) + s_idx) / rate
    plt.plot(x, plot_data[::step, ch] + ch * offset, linewidth=0.5,
             label=ch_label_fmt(ch), color="steelblue")

    ch_mask = eod_chan == ch
    if ch_mask.any():
        p1 = raw_p1_idc[ch_mask]
        p2 = raw_p2_idc[ch_mask]
        plt.plot((p1 + s_idx) / rate, plot_data[p1, ch] + ch * offset,
                 "o", markersize=5, color="red", zorder=3)
        plt.plot((p2 + s_idx) / rate, plot_data[p2, ch] + ch * offset,
                 "o", markersize=5, color="blue", zorder=3)

    fo_ch_mask = fo_eod_chan == ch
    if fo_ch_mask.any():
        p1_fo = fo_raw_p1[fo_ch_mask]
        p2_fo = fo_raw_p2[fo_ch_mask]
        plt.plot((p1_fo + s_idx) / rate, plot_data[p1_fo, ch] + ch * offset,
                 "o", markersize=4, color="grey", alpha=0.6, zorder=2)
        plt.plot((p2_fo + s_idx) / rate, plot_data[p2_fo, ch] + ch * offset,
                 "o", markersize=4, color="grey", alpha=0.6, zorder=2)

plt.ylim(-0.5 * offset, (n_plot_ch - 0.5) * offset)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (offset)")
plt.title(f"{plot_title_prefix} - {len(eod_table)} kept (red=P1 blue=P2), {len(filtered_out_indices)} grey=filtered")
plt.legend(loc="upper right", fontsize=7)
plt.tight_layout()
plt.show()

