# -*- coding: utf-8 -*-
"""
02_5_Reextract_events.py

Re-extracts EOD pulses from the event WAV files of an existing 02_3 extraction folder
(calibrated, unfiltered 8-channel audio), without touching the raw 10-min recordings.
Use case: parameter changes that act after detection (search window, interpolation,
triphasic_search, location method, waveform filters) or a new library version.

Per event: event wav (or concatenated _part_N wavs) -> detection on bandpassed adjacent-pair
differentials of the live pairs -> unify -> extract_pulse_snippets -> filter_waveforms ->
remove_duplicates -> channel events / merge / event filters inside the window (as 02_3) ->
merged events whose pulses mostly match the source table (>= min_source_match_fraction within 1 ms)
form the re-extracted event; merged events that were separate events in the source run are dropped.
Pulses outside the original pulse span (eod_start_time .. eod_end_time of all_event_summaries.csv,
+- margin_tolerance_s) are dropped. Events are renumbered consecutively in the output (a source
event that yields no pulses leaves no gap); match events between extractions by their start times
(the log prints source id -> new id). The wav start time is taken from the source table
(timestamp - midpoint_idx / rate), which also covers source runs made before the N7 fix in 02_3.
original_* indices are relative to the source 10-min file of each pulse. Output: *_eod_table.csv,
*_waveforms_concatenated.npz/json, all_event_summaries.csv (+ n_eods_source), config.cfg (with the
overrides applied), reextraction_info.json (source folder, git hash), optional detection plots.
No wav files are written (the source folder keeps them, under the SOURCE event ids).
Validated 2026-11-03 on 3 Juma_L5 events (known_bugs_pulse_functions.md, "02_5").

Authors: Stefan Mucha with GitHub Copilot
"""

import configparser
import datetime as dt
import glob
import json
import os
import pickle
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog

import audioio as aio
import numpy as np
import pandas as pd
import thunderfish.pulses as pulses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse_functions import (
    bandpass_filter,
    unify_across_channels,
    extract_pulse_snippets,
    remove_duplicates,
    filter_waveforms,
    filter_waveforms_with_classifier,
    save_waveforms,
    create_channel_events,
    merge_channel_events,
    filter_events,
    event_amplitude_statistics,
    create_event_plots,
)

# Parameter overrides applied on top of the source folder's config.cfg (empty dict = same parameters)
parameter_overrides = {
}
margin_tolerance_s = 0.002   # pulses this far outside eod_start_time..eod_end_time are still kept
min_source_match_fraction = 0.5   # merged events in the window with fewer source matches (1 ms) were separate events in the source run: dropped
create_plots = True

root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Source extraction folder (event wavs + tables + config.cfg)")
output_path = filedialog.askdirectory(title="Output folder for the re-extraction")
root.destroy()
os.makedirs(output_path, exist_ok=True)

config = configparser.ConfigParser()
config.read(os.path.join(input_path, 'config.cfg'))
bool_keys = {
    "enable_bandpass_filter", "create_events", "merge_events",
    "pre_merge_filtering", "post_merge_filtering", "save_filtered_out", "create_plots", "triphasic_search",
    "ignore_channel", "return_diff"
}
int_keys = {
    "duplicate_samples", "interp_factor", "search_window",
    "min_eods_premerge", "min_eods_postmerge", "min_amplitude_run"
}
str_keys = {"source", "extraction_method", "detection_method", "location_method", "extraction_window", "pulse_mode", "waveform_extraction"}
parameters = {}
for key, val in config["Parameters"].items():
    if key in bool_keys:
        parameters[key] = val.lower() in ("true", "1", "yes")
    elif key in int_keys:
        parameters[key] = int(float(val))
    elif key in str_keys:
        parameters[key] = val
    else:
        parameters[key] = float(val)
parameters.setdefault("pulse_mode", "biphasic")
parameters.setdefault("extraction_method", "differential")
parameters.setdefault("location_method", "auto")
parameters.setdefault("detection_method", "per_channel")
parameters.setdefault("triphasic_search", True)
parameters.setdefault("ignore_channel", False)
parameters.setdefault("min_amplitude_run", 1)   # configs without the key use the old maximum rule
parameters.setdefault("split_duration", 0.0)
parameters.update(parameter_overrides)

if parameters['detection_method'] != 'per_channel' or parameters['source'] != 'multich_linear':
    raise ValueError("02_5 covers the Uganda route only (per_channel detection, multich_linear source)")

use_ml_filtering = config["MachineLearning"].getboolean("use_ml_filtering", fallback=False)
classifier_path = config["MachineLearning"].get("classifier_path", "")
fish_probability_threshold = config["MachineLearning"].getfloat("fish_probability_threshold", fallback=0.5)
loaded_classifier = None
loaded_scaler = None
if use_ml_filtering and classifier_path:
    with open(classifier_path, 'rb') as f:
        classifier_data = pickle.load(f)
    loaded_classifier = classifier_data['classifier']
    loaded_scaler = classifier_data['scaler']
    print(f"Loaded classifier: {classifier_data['classifier_name']}")

use_amplitude_ratio = parameters['pulse_mode'] != 'monophasic'

print("Parameters:")
for key, value in parameters.items():
    print(f"  {key}: {value}")

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
with open(os.path.join(output_path, 'reextraction_info.json'), 'w') as f:
    json.dump({'source_folder': input_path, 'git_hash': git_hash, 'date': dt.datetime.now().isoformat(),
               'parameter_overrides': parameter_overrides, 'margin_tolerance_s': margin_tolerance_s}, f, indent=2)
config.set('Paths', 'input_path', input_path)
config.set('Paths', 'output_path', output_path)
for key, value in parameter_overrides.items():
    config.set('Parameters', key, str(value))
with open(os.path.join(output_path, 'config.cfg'), 'w') as f:
    config.write(f)

source_summaries = pd.read_csv(os.path.join(input_path, 'all_event_summaries.csv'))
for col in ['event_start_time', 'event_end_time', 'eod_start_time', 'eod_end_time']:
    source_summaries[col] = pd.to_datetime(source_summaries[col])
source_summaries = source_summaries.sort_values('event_id').reset_index(drop=True)
print(f"Source folder: {len(source_summaries)} events")

event_summaries = []
event_counter = 0
for _, event_row in source_summaries.iterrows():
    source_event_id = int(event_row['event_id'])
    table_files = glob.glob(os.path.join(input_path, f'*_event_{source_event_id}_eod_table.csv'))
    if len(table_files) == 0:
        print(f"Source event {source_event_id}: no eod_table in the source folder, skipped")
        continue
    base_name = os.path.basename(table_files[0]).replace('_eod_table.csv', '')
    fname = base_name.split('_event_')[0] + '.wav'
    source_table = pd.read_csv(table_files[0], usecols=['timestamp', 'midpoint_idx'])
    source_times = np.sort(pd.to_datetime(source_table['timestamp']).values.astype('datetime64[ns]').astype(np.int64) / 1e9)
    print(f"\nSource event {source_event_id} ({base_name})")

    whole_wav = os.path.join(input_path, base_name + '.wav')
    if os.path.exists(whole_wav):
        data, rate = aio.load_audio(whole_wav)
    else:
        part_files = glob.glob(os.path.join(input_path, base_name + '_part_*.wav'))
        part_files = sorted(part_files, key=lambda p: int(p.rsplit('_part_', 1)[1][:-4]))
        if len(part_files) == 0:
            print("  no wav found, skipped")
            continue
        parts = []
        for part_file in part_files:
            part_data, rate = aio.load_audio(part_file)
            parts.append(part_data)
        data = np.concatenate(parts, axis=0)
        print(f"  concatenated {len(part_files)} parts")
    n_channels = data.shape[1]
    # The wav starts at the source run's event_start_time only when its audio frame and its timestamps
    # agree (not the case for runs before the N7 fix in 02_3), so the wav start is taken from the
    # source table itself (timestamp - midpoint_idx / rate)
    wav_starts = pd.to_datetime(source_table['timestamp']) - pd.to_timedelta(source_table['midpoint_idx'] / rate, unit='s')
    event_start_time = wav_starts.median().round('us')
    source_time_offset_s = (event_start_time - event_row['event_start_time']).total_seconds()
    if abs(source_time_offset_s) > 0.001:
        print(f"  wav start differs from the source event_start_time by {source_time_offset_s:.3f} s (source frame mismatch): using the table frame")
    print(f"  {len(data) / rate:.1f} s, {n_channels} channels, starts {event_start_time}")

    # Older source runs have no dead_channels_file column (summaries before 2026-10): no dead channels
    dead_str = event_row.get('dead_channels_file', '')
    if not isinstance(dead_str, str):
        dead_str = ''
    dead_ch_0idx = [int(c[2:]) - 1 for c in dead_str.split(',') if c.startswith('ch')]
    live_pair_indices = [i for i in range(n_channels - 1) if i not in dead_ch_0idx and (i + 1) not in dead_ch_0idx]
    if len(dead_ch_0idx) > 0:
        print(f"  dead channels {dead_str}: detection on pairs {[p + 1 for p in live_pair_indices]}")

    if parameters['extraction_method'] == 'differential':
        data_diff = np.diff(data, axis=1)
        detect_indices = live_pair_indices
    else:
        data_diff = data
        detect_indices = [i for i in range(n_channels) if i not in dead_ch_0idx]

    peaks = []
    troughs = []
    pulse_widths = []
    for ch_idx in detect_indices:
        detection_signal = data_diff[:, ch_idx]
        if parameters['enable_bandpass_filter']:
            detection_signal = bandpass_filter(detection_signal, rate, parameters['bandpass_low_cutoff'], parameters['bandpass_high_cutoff'])
        ch_peaks, ch_troughs, _, ch_pulse_widths = pulses.detect_pulses(
            detection_signal, rate,
            thresh=parameters['thresh'],
            min_rel_slope_diff=parameters['min_rel_slope_diff'],
            min_width=parameters['min_width_us'] / 1e6,
            max_width=parameters['max_width_us'] / 1e6,
            width_fac=parameters['width_fac_detection'],
            verbose=0, return_data=False)
        peaks.append(ch_peaks)
        troughs.append(ch_troughs)
        pulse_widths.append(ch_pulse_widths)
    del data_diff
    unique_midpoints, unique_peaks, unique_troughs, unique_widths = unify_across_channels(
        peaks, troughs, pulse_widths, proximity_threshold=parameters['duplicate_samples'])
    print(f"  {len(unique_peaks)} unique pulses detected")
    if len(unique_peaks) == 0:
        continue

    (
        eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
        snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc,
        pulse_orientations, amp_ratios, fft_peak_freqs, pulse_locations,
        wf_lengths, snippet_p3_idc, final_p3_idc
    ) = extract_pulse_snippets(
        data, unique_peaks, unique_troughs, rate=rate,
        source='multich_linear', extraction_method=parameters['extraction_method'],
        location_method=parameters['location_method'],
        interp_factor=parameters['interp_factor'],
        pca_interp_points=300,
        window_mode=parameters['extraction_window'],
        window_factor=parameters['extraction_window_factor'],
        window_length=parameters['extraction_window_length_us'],
        search_window=parameters['search_window'],
        symmetry_threshold=parameters['symmetry_threshold'],
        pulse_mode=parameters['pulse_mode'],
        triphasic_search=parameters['triphasic_search']
    )
    if len(eod_snippets) == 0:
        print("  no snippets extracted")
        continue

    if use_ml_filtering and loaded_classifier is not None:
        keep_indices = filter_waveforms_with_classifier(
            eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
            classifier=loaded_classifier, scaler=loaded_scaler,
            dur_min=parameters['min_width_us'], dur_max=parameters['max_width_us'],
            pp_r_min=parameters['amplitude_ratio_min'], pp_r_max=parameters['amplitude_ratio_max'],
            fft_freq_min=parameters['peak_fft_freq_min'], fft_freq_max=parameters['peak_fft_freq_max'],
            fish_probability_threshold=fish_probability_threshold,
            use_basic_filtering=True, use_amplitude_ratio=use_amplitude_ratio)[0]
    else:
        keep_indices = filter_waveforms(
            eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
            dur_min=parameters['min_width_us'], dur_max=parameters['max_width_us'],
            pp_r_min=parameters['amplitude_ratio_min'], pp_r_max=parameters['amplitude_ratio_max'],
            fft_freq_min=parameters['peak_fft_freq_min'], fft_freq_max=parameters['peak_fft_freq_max'],
            use_amplitude_ratio=use_amplitude_ratio)[0]
    print(f"  filter_waveforms: {len(eod_snippets)} -> {len(keep_indices)}")
    if len(keep_indices) == 0:
        continue

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
    print(f"  after remove_duplicates: {len(eod_snippets)}")
    if len(eod_snippets) == 0:
        continue

    # Source 10-min file of each pulse (for filename / file_timestamp / original_* columns as in 02_3)
    file_names = event_row['file_names'].split(',')
    file_starts = pd.to_datetime([n.split('-')[1][:-4] for n in file_names], format='%Y%m%dT%H%M%S')
    file_order = np.argsort(file_starts.values)
    file_names = [file_names[i] for i in file_order]
    file_starts = file_starts[file_order]

    raw_midpoint_idc = (raw_p1_idc + raw_p2_idc) // 2
    snippet_midpoint_idc = (snippet_p1_idc + snippet_p2_idc) // 2
    timestamps = pd.to_datetime([event_start_time + dt.timedelta(seconds=t) for t in raw_midpoint_idc / rate])
    file_idx = np.searchsorted(file_starts.values, timestamps.values, side='right') - 1
    file_idx = np.clip(file_idx, 0, len(file_names) - 1)
    pulse_file_starts = file_starts[file_idx]
    file_offset_samples = np.floor((event_start_time - pulse_file_starts).total_seconds().values * rate).astype(int)

    eod_table = pd.DataFrame({
        'timestamp': timestamps,
        'file_timestamp': pulse_file_starts,
        'filename': [file_names[i] for i in file_idx],
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
        'dead_channels_file': [dead_str] * len(raw_midpoint_idc),
    })

    # Keep only the original pulse span (the 1 s margin of the wav may hold pulses of other events)
    in_span = ((eod_table['timestamp'] >= event_row['eod_start_time'] - dt.timedelta(seconds=margin_tolerance_s)) &
               (eod_table['timestamp'] <= event_row['eod_end_time'] + dt.timedelta(seconds=margin_tolerance_s)))
    print(f"  outside the original pulse span: {int((~in_span).sum())} dropped")
    eod_table = eod_table[in_span.values].reset_index(drop=True)
    eod_snippets = [eod_snippets[i] for i in np.where(in_span.values)[0]]
    if len(eod_table) == 0:
        continue

    # Channel events, merge and event-size filters as in process_events
    channel_events = create_channel_events(eod_table, parameters['max_ipi_seconds'], ignore_channel=parameters['ignore_channel'])
    if parameters['pre_merge_filtering']:
        channel_events = filter_events(channel_events, parameters['min_eods_premerge'], parameters['min_amplitude'],
                                       min_run_pulses=parameters['min_amplitude_run'], max_ipi_seconds=parameters['max_ipi_seconds'])
    if len(channel_events) == 0:
        print("  no channel events left after pre-merge filtering")
        continue
    if parameters['merge_events']:
        merged_events = merge_channel_events(channel_events, parameters['max_merge_gap_seconds'])
    else:
        merged_events = channel_events.copy()
        merged_events['merged_event_id'] = merged_events['channel_event_id']
        merged_events['event_start_time'] = merged_events['channel_start_time']
        merged_events['event_end_time'] = merged_events['channel_end_time']
    if parameters['post_merge_filtering']:
        merged_events = filter_events(merged_events, parameters['min_eods_postmerge'], parameters['min_amplitude'],
                                      min_run_pulses=parameters['min_amplitude_run'], max_ipi_seconds=parameters['max_ipi_seconds'])
    if len(merged_events) == 0:
        print("  no events left after post-merge filtering")
        continue

    # The window may hold pulse trains that the source run numbered as separate events: keep only
    # merged events whose pulses mostly (>= min_source_match_fraction) have a source pulse within 1 ms
    new_times = merged_events['timestamp_dt'].values.astype('datetime64[ns]').astype(np.int64) / 1e9
    pos = np.clip(np.searchsorted(source_times, new_times), 1, len(source_times) - 1)
    nearest_dt = np.minimum(np.abs(source_times[pos] - new_times), np.abs(source_times[pos - 1] - new_times))
    merged_events['source_match'] = nearest_dt <= 0.001
    match_fraction = merged_events.groupby('merged_event_id')['source_match'].mean()
    keep_merged = match_fraction[match_fraction >= min_source_match_fraction].index
    for mid, frac in match_fraction.items():
        if mid not in keep_merged:
            print(f"  merged event {mid}: {int((merged_events['merged_event_id'] == mid).sum())} pulses, source match {frac:.2f} -> dropped (separate event in the source run)")
    merged_events = merged_events[merged_events['merged_event_id'].isin(keep_merged)].drop(columns=['source_match'])
    if len(merged_events) == 0:
        print("  no merged event matches the source event")
        continue
    event_counter += 1
    event_id = event_counter
    out_base_name = f'{fname[:-4]}_event_{event_id}'
    event_eods = merged_events.copy()
    event_eods['reaches_file_end'] = False
    event_eods['event_id'] = event_id
    print(f"  -> new event {event_id}: {len(event_eods)} pulses in {event_eods['merged_event_id'].nunique()} merged event(s) inside the window (source table: {int(event_row['n_eods'])})")

    # Index columns relative to the event audio (= the wav) and to the source 10-min file of each pulse
    event_eods['original_p1_idx'] = event_eods['p1_idx'] + file_offset_samples[event_eods['original_index'].values]
    event_eods['original_p2_idx'] = event_eods['p2_idx'] + file_offset_samples[event_eods['original_index'].values]
    event_eods['original_midpoint_idx'] = event_eods['midpoint_idx'] + file_offset_samples[event_eods['original_index'].values]
    event_eods['audio_part'] = 1
    event_eods['part_audio_start_sample'] = 0
    split_dur_samp = int(parameters['split_duration'] * rate) if parameters['split_duration'] > 0 else 0
    if split_dur_samp > 0 and len(data) > split_dur_samp:
        event_eods['audio_part'] = event_eods['midpoint_idx'] // split_dur_samp + 1
        event_eods['part_audio_start_sample'] = (event_eods['midpoint_idx'] // split_dur_samp) * split_dur_samp

    event_waveforms = [eod_snippets[i] for i in event_eods['original_index'].values]
    save_waveforms(event_waveforms, os.path.join(output_path, f'{out_base_name}_waveforms'), length=parameters['extraction_window'])
    event_eods.to_csv(os.path.join(output_path, f'{out_base_name}_eod_table.csv'), index=False)

    duration = (event_eods['timestamp_dt'].max() - event_eods['timestamp_dt'].min()).total_seconds()
    event_summaries.append({
        'event_id': event_id,
        'eod_start_time': event_eods['timestamp_dt'].min(),
        'eod_end_time': event_eods['timestamp_dt'].max(),
        'event_start_time': event_start_time,
        'event_end_time': event_row['event_end_time'] + dt.timedelta(seconds=source_time_offset_s),
        'duration_seconds': duration,
        'n_eods': len(event_eods),
        'n_channels': event_eods['eod_channel'].nunique(),
        'channels_used': ','.join(map(str, sorted(event_eods['eod_channel'].unique()))),
        'mean_ipi_seconds': event_eods['channel_ipi_seconds'].mean(),
        'ipi_cv': event_eods['channel_ipi_seconds'].std() / event_eods['channel_ipi_seconds'].mean() if event_eods['channel_ipi_seconds'].mean() > 0 else 0,
        'median_ipi_seconds': event_eods['channel_ipi_seconds'].median(),
        'mean_amplitude': event_eods['eod_amplitude'].mean(),
        'max_amplitude': event_eods['eod_amplitude'].max(),
        **event_amplitude_statistics(event_eods['timestamp_dt'].values.astype('datetime64[ns]').astype(np.int64) / 1e9,
                                     event_eods['eod_amplitude'].values, k=5, max_ipi=parameters['max_ipi_seconds']),
        'mean_width_ms': event_eods['eod_width_us'].mean() / 1000,
        'n_files': event_eods['filename'].nunique(),
        'file_names': ','.join(event_eods['filename'].unique()),
        'dead_channels_file': dead_str,
        'n_eods_source': int(event_row['n_eods']),
    })

    if create_plots:
        create_event_plots(event_id=event_id, event_eods=event_eods, event_data=data, event_start_time=event_start_time,
                           sample_rate=rate, output_path=output_path, extraction_method=parameters['extraction_method'])
    del data, eod_snippets, event_waveforms

summary_df = pd.DataFrame(event_summaries)
summary_df.to_csv(os.path.join(output_path, 'all_event_summaries.csv'), index=False)
print(f"\nDone: {len(summary_df)} events, {int(summary_df['n_eods'].sum()) if len(summary_df) else 0} pulses "
      f"(source: {len(source_summaries)} events, {int(source_summaries['n_eods'].sum())} pulses) -> {output_path}")
