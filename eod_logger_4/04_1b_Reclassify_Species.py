# -*- coding: utf-8 -*-
"""
04_1b_Reclassify_Species.py

Standalone species reclassification for an already-tracked session. Reuses the tracked
fish (tracked_fish_summary.csv) and their stored per-fish mean waveforms
(session_*_mean_waveforms.npz) from a 04_1_Track_Fish.py run, refits the species
classifier against a control reference library with new classification thresholds, and
writes a new/versioned set of session output files - without re-running tracking.

Authors: Stefan Mucha with Claude Sonnet 4.6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import glob
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulse_functions import load_waveforms, save_waveforms, fit_species_classifier, classify_fish_waveform
from parameter_gui import ClassificationParameterConfigGUI

print("="*70)
print("SPECIES RECLASSIFICATION")
print("="*70)

print("\nStarting Parameter Configuration GUI...")
root = tk.Tk()
config_gui = ClassificationParameterConfigGUI(root)
root.mainloop()

if config_gui.result is None:
    print("Configuration cancelled by user")
    exit()

config = config_gui.result
input_path   = config['paths']['input_path']
control_path = config['paths']['control_path']
output_path  = config['paths']['output_path']
params = config['parameters']

waveform_target_length = int(params['waveform_target_length'])
crop_factor             = int(params['crop_factor'])
lda_min_probability     = float(params['lda_min_probability'])
lda_max_dist_factor     = float(params['lda_max_dist_factor'])

os.makedirs(output_path, exist_ok=True)
print(f"✓ Input (tracked session) folder: {input_path}")
print(f"✓ Control folder: {control_path}")
print(f"✓ Output folder: {output_path}")

print("\n" + "="*70)
print("FITTING SPECIES CLASSIFIER")
print("="*70)
species_classifier = fit_species_classifier(
    control_path, waveform_target_length, crop_factor,
    lda_max_dist_factor=lda_max_dist_factor, output_folder=output_path
)
if species_classifier is None:
    raise ValueError("No usable control reference library found - cannot reclassify")
lda = species_classifier['lda']
all_species_codes = species_classifier['all_species_codes']

print("\n" + "="*70)
print("LOADING TRACKED FISH + MEAN WAVEFORMS")
print("="*70)

fish_summary_path = os.path.join(input_path, 'tracked_fish_summary.csv')
fish_summary_df = pd.read_csv(fish_summary_path)
fish_summary_df['entry_time'] = pd.to_datetime(fish_summary_df['entry_time'])
fish_summary_df['exit_time']  = pd.to_datetime(fish_summary_df['exit_time'])
print(f"✓ Loaded {len(fish_summary_df)} tracked fish from {os.path.basename(fish_summary_path)}")

# Keep only tracking-derived columns; species-derived columns are recomputed below.
base_columns = ['base_name', 'event_id', 'fish_key', 'fish_id', 'entry_time', 'exit_time',
                'n_pulses', 'duration_s', 'mean_rate_hz', 'std_rate_hz', 'cv_ipi',
                'mean_location', 'std_location', 'mean_width_us', 'std_width_us', 'cv_amplitude']
fish_summary_df = fish_summary_df[base_columns]

mean_wf_by_key = {}
wf_npz_files = glob.glob(os.path.join(input_path, "session_*_mean_waveforms.npz"))
for wf_file in wf_npz_files:
    wf_base = wf_file[:-4]  # strip '.npz'
    keys_file = wf_base + '_keys.csv'
    if not os.path.exists(keys_file):
        print(f"  ⚠ No keys file for {os.path.basename(wf_file)}, skipping")
        continue
    wf_list = load_waveforms(wf_base, format='npz', length='fixed')
    keys_df = pd.read_csv(keys_file)
    if len(wf_list) != len(keys_df):
        print(f"  ⚠ Waveform/key count mismatch for {os.path.basename(wf_file)}, skipping")
        continue
    for fish_key, wf in zip(keys_df['fish_key'], wf_list):
        mean_wf_by_key[fish_key] = wf
print(f"✓ Loaded {len(mean_wf_by_key)} per-fish mean waveforms from {len(wf_npz_files)} session file(s)")

missing_wf = ~fish_summary_df['fish_key'].isin(mean_wf_by_key)
if missing_wf.any():
    print(f"⚠ {missing_wf.sum()} fish have no stored mean waveform and will be dropped")
    fish_summary_df = fish_summary_df.loc[~missing_wf].reset_index(drop=True)

print("\n" + "="*70)
print("RECLASSIFYING SPECIES")
print("="*70)

cls_rows = []
for _, row in fish_summary_df.iterrows():
    mean_wf = mean_wf_by_key[row['fish_key']]
    cls_result = classify_fish_waveform(mean_wf, species_classifier, lda_min_probability)
    cls_row = {
        'fish_key':           row['fish_key'],
        'species_code':       cls_result['species_assigned'].rstrip('?'),
        'species_assigned':   cls_result['species_assigned'],
        'species_uncertain':  cls_result['species_uncertain'],
        'nearest_individual': cls_result['nearest_individual'],
        'dist_nearest':       cls_result['dist_nearest'],
        'dist_margin':        cls_result['dist_margin'],
    }
    if lda is not None:
        cls_row['lda_dist_centroid']  = cls_result['lda_dist_centroid']
        cls_row['lda_dist_threshold'] = cls_result['lda_dist_threshold']
        cls_row['lda_proba_assigned'] = cls_result['lda_proba_assigned']
        for sp in lda.classes_:
            cls_row[f'lda_proba_{sp}'] = cls_result[f'lda_proba_{sp}']
    else:
        for sp in all_species_codes:
            key = f'dist_{sp}'
            if key in cls_result:
                cls_row[key] = cls_result[key]
    cls_rows.append(cls_row)
    uncertain_flag = " (!)" if cls_result['species_uncertain'] else ""
    print(f"  {row['fish_key']}: {cls_result['species_assigned']}{uncertain_flag}")

cls_df = pd.DataFrame(cls_rows)
fish_summary_df = fish_summary_df.merge(cls_df, on='fish_key', how='left')

print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# --- 1: Per-fish summary ---
new_fish_summary_path = os.path.join(output_path, 'tracked_fish_summary.csv')
fish_summary_df.to_csv(new_fish_summary_path, index=False)
print(f"✓ Saved per-fish summary: {os.path.basename(new_fish_summary_path)} ({len(fish_summary_df)} rows)")

# --- 2: Per-species summary (one row per species per event, uncertain kept as own '?' category) ---
species_summary_df = (
    fish_summary_df.groupby(['base_name', 'event_id', 'species_assigned'])
    .size().reset_index(name='n_fish')
    .rename(columns={'species_assigned': 'species'})
)
species_summary_path = os.path.join(output_path, 'tracked_species_summary.csv')
species_summary_df.to_csv(species_summary_path, index=False)
print(f"✓ Saved per-species summary: {os.path.basename(species_summary_path)}")

# --- 3: Event summary - keep non-species columns from original, replace species columns ---
# Matches 04_1's convention: n_fish_{sp} counts confident assignments only (species_assigned
# with no '?' suffix); uncertain fish still count toward n_fish_total.
orig_event_summary_path = os.path.join(input_path, 'tracked_event_summary.csv')
orig_event_df = pd.read_csv(orig_event_summary_path)
old_species_cols = [c for c in orig_event_df.columns if c.startswith('n_fish_') and c != 'n_fish_total']
event_base_df = orig_event_df.drop(columns=old_species_cols)

event_sp_counts = (
    fish_summary_df.groupby(['event_id', 'species_assigned'])
    .size().unstack(fill_value=0)
)
event_sp_counts = event_sp_counts[[c for c in event_sp_counts.columns if c in all_species_codes]]
event_sp_counts.columns = [f'n_fish_{sp}' for sp in event_sp_counts.columns]
event_sp_counts = event_sp_counts.reindex(columns=[f'n_fish_{sp}' for sp in all_species_codes], fill_value=0)
event_sp_counts = event_sp_counts.reset_index()

event_out_df = event_base_df.merge(event_sp_counts, on='event_id', how='left')
for col in event_sp_counts.columns:
    if col != 'event_id':
        event_out_df[col] = event_out_df[col].fillna(0).astype(int)
event_summary_path = os.path.join(output_path, 'tracked_event_summary.csv')
event_out_df.to_csv(event_summary_path, index=False)
print(f"✓ Saved event summary: {os.path.basename(event_summary_path)} ({len(event_out_df)} rows)")

# --- 4: Session-level mean waveforms per species (re-bucketed by new species_code) ---
for sp, sp_group in fish_summary_df.groupby('species_code'):
    wf_out_base = os.path.join(output_path, f'session_{sp}_mean_waveforms')
    wf_list = [mean_wf_by_key[fk] for fk in sp_group['fish_key']]
    save_waveforms(wf_list, wf_out_base, format='npz', length='fixed')
    keys_df = sp_group[['fish_key', 'mean_location', 'mean_rate_hz', 'entry_time', 'exit_time']]
    keys_df.to_csv(wf_out_base + '_keys.csv', index=False)
    print(f"✓ Saved mean waveforms: {os.path.basename(wf_out_base)}.npz ({len(wf_list)} fish)")

# --- 5: Session summary figure (location histogram + 2 stacked time bar plots) ---
all_species_in_summ = sorted(fish_summary_df['species_code'].unique())
sp_pal_summ = plt.cm.Set1(np.linspace(0, 0.8, max(len(all_species_in_summ), 1)))
sp_color_summ = {sp: sp_pal_summ[i] for i, sp in enumerate(all_species_in_summ)}

all_int_channels = list(range(8))
ch_pal = plt.cm.tab10(np.linspace(0, 0.9, 8))
ch_color_summ = {ch: ch_pal[ch] for ch in all_int_channels}

time_bins_h = np.arange(0, 24.5, 0.5)
bin_centers = time_bins_h[:-1] + 0.25
n_bins = len(bin_centers)

counts_by_sp = {sp: np.zeros(n_bins, dtype=float) for sp in all_species_in_summ}
counts_by_ch = {ch: np.zeros(n_bins, dtype=float) for ch in all_int_channels}

for _, fd in fish_summary_df.iterrows():
    if pd.isnull(fd['entry_time']) or pd.isnull(fd['exit_time']):
        continue
    sp = fd['species_code']
    int_ch = max(0, min(7, int(round(fd['mean_location'])))) if pd.notnull(fd['mean_location']) else 0
    entry_h = (fd['entry_time'].hour + fd['entry_time'].minute / 60.0 + fd['entry_time'].second / 3600.0) % 24
    exit_h  = (fd['exit_time'].hour  + fd['exit_time'].minute  / 60.0 + fd['exit_time'].second  / 3600.0) % 24
    for b_idx, b_start in enumerate(time_bins_h[:-1]):
        b_end = b_start + 0.5
        if exit_h >= entry_h:
            active = entry_h < b_end and exit_h > b_start
        else:
            active = b_start < exit_h or b_end > entry_h
        if active:
            counts_by_sp[sp][b_idx] += 1
            counts_by_ch[int_ch][b_idx] += 1

fig_sess, (ax_loc_s, ax_time_ch, ax_time_sp) = plt.subplots(3, 1, figsize=(14, 11))
fig_sess.suptitle('Session summary (reclassified)', fontsize=11)

loc_bins = np.arange(-0.25, 8.25, 0.5)
for sp in all_species_in_summ:
    sp_locs = fish_summary_df.loc[fish_summary_df['species_code'] == sp, 'mean_location'].dropna()
    if len(sp_locs) > 0:
        ax_loc_s.hist(sp_locs, bins=loc_bins, alpha=0.6,
                      color=sp_color_summ[sp], label=sp, edgecolor='none')
ax_loc_s.set_xlabel('Mean location (electrode units)')
ax_loc_s.set_ylabel('Fish count')
ax_loc_s.set_title('Fish location distribution (0.5-unit bins)')
ax_loc_s.legend(fontsize=8)

bottom_ch = np.zeros(n_bins)
bar_width = 0.5
for ch in all_int_channels:
    ax_time_ch.bar(bin_centers, counts_by_ch[ch], width=bar_width,
                   bottom=bottom_ch, color=ch_color_summ[ch],
                   label=f'Ch{ch}', edgecolor='none', alpha=0.85)
    bottom_ch += counts_by_ch[ch]
ax_time_ch.set_xlabel('Hour of day')
ax_time_ch.set_ylabel('Active fish count')
ax_time_ch.set_title('Active fish over time — stacked by location (0.5-hour bins)')
ax_time_ch.set_xlim(0, 24)
ax_time_ch.set_xticks(np.arange(0, 25, 2))
ax_time_ch.legend(fontsize=7, ncol=min(len(all_int_channels), 8), loc='upper right')

bottom_sp = np.zeros(n_bins)
for sp in all_species_in_summ:
    ax_time_sp.bar(bin_centers, counts_by_sp[sp], width=bar_width,
                   bottom=bottom_sp, color=sp_color_summ[sp],
                   label=sp, edgecolor='none', alpha=0.85)
    bottom_sp += counts_by_sp[sp]
ax_time_sp.set_xlabel('Hour of day')
ax_time_sp.set_ylabel('Active fish count')
ax_time_sp.set_title('Active fish over time — stacked by species (0.5-hour bins)')
ax_time_sp.set_xlim(0, 24)
ax_time_sp.set_xticks(np.arange(0, 25, 2))
ax_time_sp.legend(fontsize=8, loc='upper right')

plt.tight_layout()
sess_fig_path = os.path.join(output_path, 'session_summary.png')
plt.savefig(sess_fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved session summary figure: {os.path.basename(sess_fig_path)}")

print("\n" + "="*70)
print("RECLASSIFICATION COMPLETE")
print("="*70)
