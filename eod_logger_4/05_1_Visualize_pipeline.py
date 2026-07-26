# -*- coding: utf-8 -*-
"""
poster_pipeline_visualization.py

Visualizes the EOD fish-tracking pipeline as individual vector-graphics panels
for a neuroethology conference poster. Reads outputs of 04_1_Track_Fish.py plus
the original raw WAV files. Saves one SVG per pipeline stage.

Black background, white axes, Okabe-Ito colorblind palette.

Authors: Stefan Mucha

"""

import audioio as aio
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pulse_functions import load_waveforms, normalize_waveforms

# =============================================================================
# CONFIGURATION
# =============================================================================

wav_folder      = r"C:\path\to\raw\wav\files"       # folder containing .wav recordings
input_folder    = r"C:\path\to\04_1\input"           # eod_table + waveforms_concatenated.npz
output_folder   = r"C:\path\to\04_1\output"          # tracked.csv, classifier_report.json
control_path    = r"C:\path\to\control\recordings"   # control lib for LDA; set '' to skip
svg_folder      = r"C:\path\to\svg\output"           # destination for saved SVGs

event_base_name = "event_001"   # matches *_tracked.csv, *_eod_table.csv, *_waveforms*

# Waveform normalization — must match 04_1 settings
waveform_target_length = 150
crop_factor            = 7

# Raw audio window (relative to event start time)
t_start_s  = 0.0    # seconds after event start to begin the window
t_window_s = 10.0   # window duration in seconds

# Number of waveforms to overlay in Panel 3
n_waveform_overlay = 200

# Figure size in inches (same for all panels)
FIG_W = 8.0
FIG_H = 5.0

# =============================================================================
# PALETTE & GLOBAL STYLE
# =============================================================================

# Okabe-Ito colorblind-safe palette (black entry omitted for black background use)
OKABE_ITO = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
]

BG    = '#000000'
FG    = '#FFFFFF'
GRAY  = '#666666'
BLUE  = '#56B4E9'  # default single-series highlight

plt.rcParams.update({
    'figure.facecolor':   BG,
    'axes.facecolor':     BG,
    'axes.edgecolor':     FG,
    'axes.labelcolor':    FG,
    'xtick.color':        FG,
    'ytick.color':        FG,
    'text.color':         FG,
    'legend.facecolor':   '#111111',
    'legend.edgecolor':   FG,
    'legend.framealpha':  0.8,
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'svg.fonttype':       'none',   # keep text editable in Illustrator
})

os.makedirs(svg_folder, exist_ok=True)

# =============================================================================
# LOAD TRACKED DATA
# =============================================================================

tracked_csv = os.path.join(output_folder, f"{event_base_name}_tracked.csv")
eod_data = pd.read_csv(tracked_csv)
eod_data['timestamp'] = pd.to_datetime(eod_data['timestamp'])
eod_data = eod_data.sort_values('timestamp').reset_index(drop=True)
t_sec = (eod_data['timestamp'] - eod_data['timestamp'].iloc[0]).dt.total_seconds().values

n_fish_tracked = int((eod_data['fish_id'] >= 0).any() and eod_data['fish_id'].max() + 1)
print(f"Tracked CSV: {len(eod_data)} pulses")

# =============================================================================
# LOAD & REORDER RAW WAVEFORMS
# Replicate the timestamp-sort reordering from 04_1 so waveform[i] matches
# eod_data row i. The original eod_table gives the pre-sort row order.
# =============================================================================

eod_table_path = os.path.join(input_folder, f"{event_base_name}_eod_table.csv")
eod_table_orig = pd.read_csv(eod_table_path)
eod_table_orig['timestamp'] = pd.to_datetime(eod_table_orig['timestamp'])
original_indices = (
    eod_table_orig
    .sort_values('timestamp')
    .index
    .tolist()
)

wf_base = os.path.join(input_folder, f"{event_base_name}_waveforms")
waveforms_all = load_waveforms(wf_base, format='npz', length='variable')
waveforms_raw = [waveforms_all[i] for i in original_indices]
del waveforms_all
print(f"Loaded {len(waveforms_raw)} raw waveforms")

# Slope subtraction: remove linear DC drift across each snippet window
waveforms_detrended = []
for wf in waveforms_raw:
    slope = np.linspace(float(wf[0]), float(wf[-1]), len(wf))
    waveforms_detrended.append(wf - slope)
del waveforms_raw

# P1-aligned normalization + L2 scaling
normalized_waveforms = normalize_waveforms(
    waveforms_detrended,
    snippet_p1_idc=eod_data['snippet_p1_idx'].values,
    snippet_p2_idc=eod_data['snippet_p2_idx'].values,
    method='p1_unity',
    crop_and_interpolate=True,
    crop_factor=crop_factor,
    target_length=waveform_target_length
)
normalized_waveforms = np.array(normalized_waveforms)
norms = np.linalg.norm(normalized_waveforms, axis=1, keepdims=True)
norms[norms == 0] = 1.0
waveforms_l2 = normalized_waveforms / norms
del normalized_waveforms, norms
print(f"Normalized waveforms: {waveforms_l2.shape}")

# =============================================================================
# LOAD RAW AUDIO (8-channel WAV for Panel 1)
# =============================================================================

event_start_time = eod_data['timestamp'].iloc[0]
window_start     = event_start_time + pd.Timedelta(seconds=t_start_s)
window_end       = window_start + pd.Timedelta(seconds=t_window_s)

wav_files = sorted(glob.glob(os.path.join(wav_folder, '*.wav')))

# Parse start timestamp from filename: *-YYYYMMDDTHHMMSS.wav
wav_timestamps = []
for f in wav_files:
    try:
        ts_str = os.path.basename(f).split('-')[1][:-4]
        wav_timestamps.append(pd.to_datetime(ts_str, format='%Y%m%dT%H%M%S'))
    except (IndexError, ValueError):
        wav_timestamps.append(pd.NaT)

# Last WAV file whose start timestamp is <= window_start
valid = [(i, ts) for i, ts in enumerate(wav_timestamps)
         if pd.notna(ts) and ts <= window_start]

audio_data = None
rate       = None

if valid:
    wav_idx, wav_start_ts = valid[-1]
    offset_s = (window_start - wav_start_ts).total_seconds()

    with aio.AudioLoader(wav_files[wav_idx]) as sf:
        rate         = sf.rate
        start_smp    = int(offset_s * rate)
        end_smp      = int((offset_s + t_window_s) * rate)
        audio_data   = sf[start_smp:end_smp, :].copy()

    n_channels = audio_data.shape[1]
    print(f"Loaded audio {audio_data.shape} from {os.path.basename(wav_files[wav_idx])}, "
          f"offset={offset_s:.1f}s, rate={rate} Hz")
else:
    print("WARNING: No matching WAV file found — Panel 1 will be skipped")

# =============================================================================
# SHARED: fish color map, widths, pulse assignments
# =============================================================================

fish_ids_assigned = sorted([fid for fid in eod_data['fish_id'].unique() if fid >= 0])
fish_color_map    = {fid: OKABE_ITO[i % len(OKABE_ITO)] for i, fid in enumerate(fish_ids_assigned)}

widths = eod_data['eod_width_us'].values

# =============================================================================
# PANEL 1: RAW MULTICHANNEL RECORDING WITH DETECTED PULSES
# =============================================================================

if audio_data is not None:
    fig1, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))

    # Vertical spacing between channel traces
    ch_range     = np.nanmax(np.abs(audio_data))
    offset_scale = ch_range * 2.2
    t_audio      = np.linspace(0, t_window_s, audio_data.shape[0])

    for ch in range(n_channels):
        ax1.plot(t_audio, audio_data[:, ch] + ch * offset_scale,
                 color=FG, linewidth=0.5, alpha=0.8, rasterized=False)
        ax1.text(-0.015 * t_window_s, ch * offset_scale, f'Ch {ch + 1}',
                 color=FG, ha='right', va='center', fontsize=8)

    # Mark detected pulses that fall within the window
    window_mask = (eod_data['timestamp'] >= window_start) & (eod_data['timestamp'] < window_end)
    window_pulses = eod_data.loc[window_mask].copy()
    window_pulses['t_rel'] = (window_pulses['timestamp'] - window_start).dt.total_seconds()

    if 'pulse_location' in window_pulses.columns:
        for _, row in window_pulses.iterrows():
            ch_idx = int(round(float(row['pulse_location'])))
            ch_idx = max(0, min(n_channels - 1, ch_idx))
            smp    = min(int(row['t_rel'] * rate), audio_data.shape[0] - 1)
            y_val  = float(audio_data[smp, ch_idx]) + ch_idx * offset_scale
            ax1.scatter(row['t_rel'], y_val,
                        color=OKABE_ITO[0], s=20, zorder=5, linewidths=0)

    ax1.set_xlim(0, t_window_s)
    ax1.set_yticks([])
    ax1.spines['left'].set_visible(False)
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Raw recording — detected pulses (orange)')

    plt.tight_layout()
    fig1.savefig(os.path.join(svg_folder, '01_raw_audio.svg'),
                 format='svg', bbox_inches='tight', facecolor=BG)
    plt.close(fig1)
    print("Saved: 01_raw_audio.svg")

# =============================================================================
# PANEL 2: FEATURE EXTRACTION — single representative pulse with landmarks
# =============================================================================

# Pick pulse closest to median width
median_width = float(np.median(widths))
rep_idx      = int(np.argmin(np.abs(widths - median_width)))

rep_wf  = waveforms_detrended[rep_idx]
p1_idx  = int(eod_data.loc[rep_idx, 'snippet_p1_idx'])
p2_idx  = int(eod_data.loc[rep_idx, 'snippet_p2_idx'])
t_smps  = np.arange(len(rep_wf))

fig2, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))

ax2.plot(t_smps, rep_wf, color=FG, linewidth=1.5)

# P1 peak
ax2.axvline(p1_idx, color=OKABE_ITO[0], linestyle='--', linewidth=1.0, alpha=0.8)
ax2.scatter(p1_idx, rep_wf[p1_idx], color=OKABE_ITO[0], s=70, zorder=5)
ax2.text(p1_idx + len(rep_wf) * 0.01, rep_wf[p1_idx],
         'P1', color=OKABE_ITO[0], fontsize=10, va='bottom')

# P2 trough
ax2.axvline(p2_idx, color=OKABE_ITO[1], linestyle='--', linewidth=1.0, alpha=0.8)
ax2.scatter(p2_idx, rep_wf[p2_idx], color=OKABE_ITO[1], s=70, zorder=5)
ax2.text(p2_idx + len(rep_wf) * 0.01, rep_wf[p2_idx],
         'P2', color=OKABE_ITO[1], fontsize=10, va='top')

# P1-P2 width bracket below the trace
y_min     = rep_wf.min()
y_bracket = y_min - abs(y_min) * 0.2
ax2.annotate('', xy=(p2_idx, y_bracket), xytext=(p1_idx, y_bracket),
             arrowprops=dict(arrowstyle='<->', color=OKABE_ITO[3], lw=1.5))
ax2.text((p1_idx + p2_idx) / 2, y_bracket,
         f'{widths[rep_idx]:.0f} µs',
         color=OKABE_ITO[3], ha='center', va='top', fontsize=9)

ax2.set_xlabel('Sample')
ax2.set_ylabel('Amplitude (V)')
ax2.set_title('EOD waveform — feature landmarks')

plt.tight_layout()
fig2.savefig(os.path.join(svg_folder, '02_feature_extraction.svg'),
             format='svg', bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print("Saved: 02_feature_extraction.svg")

# =============================================================================
# PANEL 3: PREPROCESSING — normalized waveform overlay
# =============================================================================

rng       = np.random.default_rng(seed=42)
n_overlay = min(n_waveform_overlay, len(waveforms_l2))
ol_idx    = rng.choice(len(waveforms_l2), size=n_overlay, replace=False)

# Normalized time axis: 0 is the P1-centred origin
x_norm = np.linspace(-1.0, 1.0, waveform_target_length)

fig3, ax3 = plt.subplots(figsize=(FIG_W, FIG_H))

for i in ol_idx:
    ax3.plot(x_norm, waveforms_l2[i], color=BLUE, alpha=0.05,
             linewidth=0.5, rasterized=False)

mean_wf = waveforms_l2.mean(axis=0)
ax3.plot(x_norm, mean_wf, color=OKABE_ITO[0], linewidth=2.0, label='Mean')

ax3.axvline(0, color=FG, linestyle=':', linewidth=0.7, alpha=0.4)
ax3.set_xlabel('Normalized time (P1 = 0)')
ax3.set_ylabel('Amplitude (L2-norm.)')
ax3.set_title(f'Normalized waveforms (n = {n_overlay} shown)')
ax3.legend(fontsize=9, framealpha=0.6)

plt.tight_layout()
fig3.savefig(os.path.join(svg_folder, '03_normalized_overlay.svg'),
             format='svg', bbox_inches='tight', facecolor=BG)
plt.close(fig3)
print("Saved: 03_normalized_overlay.svg")

# =============================================================================
# PANEL 4a: WIDTH DISTRIBUTION — histogram colored by width class
# =============================================================================

width_classes = sorted(eod_data['width_class'].unique())
wc_colors     = {wc: OKABE_ITO[i % len(OKABE_ITO)] for i, wc in enumerate(width_classes)}

fig4a, ax4a = plt.subplots(figsize=(FIG_W, FIG_H))

bin_edges = np.linspace(widths.min(), widths.max(), 60)
for wc in width_classes:
    wc_mask = eod_data['width_class'] == wc
    ax4a.hist(widths[wc_mask], bins=bin_edges, alpha=0.75,
              color=wc_colors[wc], label=f'Class {wc}', edgecolor='none')

# KDE overlay per class (bandwidth matches 04_1: 2× quantization step)
# interp_factor=1 assumed here; adjust if known
step_us = 1e6 / 96000.0
w_range = np.linspace(widths.min(), widths.max(), 500)
for wc in width_classes:
    wc_mask = eod_data['width_class'] == wc
    wc_w    = widths[wc_mask]
    if wc_mask.sum() > 5 and wc_w.min() < wc_w.max():
        wc_std   = float(np.std(wc_w))
        bw       = max(2.0 * step_us, 1.0) / wc_std if wc_std > 0 else 0.5
        kde_wc   = gaussian_kde(wc_w, bw_method=bw)
        scale    = wc_mask.sum() * (bin_edges[1] - bin_edges[0])
        ax4a.plot(w_range, kde_wc(w_range) * scale,
                  color=wc_colors[wc], linewidth=1.8)

ax4a.set_xlabel('EOD width (µs)')
ax4a.set_ylabel('Count')
ax4a.set_title('Pulse width distribution — width classes')
if len(width_classes) > 1:
    ax4a.legend(fontsize=9)

plt.tight_layout()
fig4a.savefig(os.path.join(svg_folder, '04a_width_histogram.svg'),
              format='svg', bbox_inches='tight', facecolor=BG)
plt.close(fig4a)
print("Saved: 04a_width_histogram.svg")

# =============================================================================
# PANEL 4b: SHAPE CLUSTERING — PCA scatter colored by shape class
# =============================================================================

pca_shape  = PCA(n_components=2)
pca_coords = pca_shape.fit_transform(waveforms_l2)

# Identify artifact-flagged pulses (shape_source == 'artifact')
artifact_mask = np.zeros(len(eod_data), dtype=bool)
if 'shape_source' in eod_data.columns:
    artifact_mask = (eod_data['shape_source'] == 'artifact').values

# Only color non-artifact, non-singleton shape classes; rest go to gray
shape_classes = sorted(eod_data['shape_class'].unique())
# Keep only classes large enough to be meaningful (>= 5 pulses), rest as gray
sc_sizes  = {sc: int((eod_data['shape_class'] == sc).sum()) for sc in shape_classes}
main_scs  = [sc for sc in shape_classes if sc_sizes[sc] >= 5]
sc_colors = {sc: OKABE_ITO[i % len(OKABE_ITO)] for i, sc in enumerate(main_scs)}

fig4b, ax4b = plt.subplots(figsize=(FIG_W, FIG_H))

# Gray: artifacts + small singletons
minor_mask = artifact_mask.copy()
for sc in shape_classes:
    if sc not in main_scs:
        minor_mask |= (eod_data['shape_class'] == sc).values

if minor_mask.any():
    ax4b.scatter(pca_coords[minor_mask, 0], pca_coords[minor_mask, 1],
                 c=GRAY, s=3, alpha=0.3, linewidths=0, rasterized=True)

for sc in main_scs:
    sc_mask = (eod_data['shape_class'] == sc).values & ~artifact_mask
    ax4b.scatter(pca_coords[sc_mask, 0], pca_coords[sc_mask, 1],
                 c=sc_colors[sc], s=5, alpha=0.55, linewidths=0,
                 rasterized=True, label=f'Shape {sc} (n={sc_sizes[sc]})')

ax4b.set_xlabel(f'PC1 ({pca_shape.explained_variance_ratio_[0] * 100:.1f}%)')
ax4b.set_ylabel(f'PC2 ({pca_shape.explained_variance_ratio_[1] * 100:.1f}%)')
ax4b.set_title('Waveform shape clustering (DBSCAN on PCA features)')
if len(main_scs) <= 10:
    ax4b.legend(fontsize=7, markerscale=2.5, loc='best')

plt.tight_layout()
fig4b.savefig(os.path.join(svg_folder, '04b_shape_clustering.svg'),
              format='svg', bbox_inches='tight', facecolor=BG)
plt.close(fig4b)
print("Saved: 04b_shape_clustering.svg")

# =============================================================================
# PANEL 5: TRACKING RESULT — pulse location vs time, colored by fish_id
# =============================================================================

fig5, ax5 = plt.subplots(figsize=(FIG_W, FIG_H))

unassigned_mask = (eod_data['fish_id'] < 0).values
if unassigned_mask.any():
    ax5.scatter(t_sec[unassigned_mask],
                eod_data.loc[unassigned_mask, 'pulse_location'],
                c=GRAY, s=3, alpha=0.35, linewidths=0,
                rasterized=True, label='unassigned')

for fid in fish_ids_assigned:
    fid_mask = (eod_data['fish_id'] == fid).values
    if 'species_assigned' in eod_data.columns:
        sp    = eod_data.loc[fid_mask, 'species_assigned'].iloc[0]
        label = f'Fish {fid} [{sp}]'
    else:
        label = f'Fish {fid}'
    ax5.scatter(t_sec[fid_mask], eod_data.loc[fid_mask, 'pulse_location'],
                c=fish_color_map[fid], s=4, alpha=0.75, linewidths=0,
                rasterized=True, label=label)

ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Electrode position')
ax5.set_title('Pulse tracking — individual fish')
n_leg_cols = max(1, len(fish_ids_assigned) // 8)
ax5.legend(markerscale=3, fontsize=7, ncol=n_leg_cols, loc='upper right')

plt.tight_layout()
fig5.savefig(os.path.join(svg_folder, '05_tracking_result.svg'),
             format='svg', bbox_inches='tight', facecolor=BG)
plt.close(fig5)
print("Saved: 05_tracking_result.svg")

# =============================================================================
# PANEL 6: SPECIES CLASSIFICATION — LDA scatterplot
# Requires control_path pointing to reference library built in 04_1.
# =============================================================================

if control_path:
    ctrl_concat_files = glob.glob(os.path.join(control_path, "*_eod_waveforms_concatenated.npz"))
    ctrl_fixed_files  = glob.glob(os.path.join(control_path, "*_eod_waveforms.npz"))
    concat_ids   = set(
        os.path.basename(f).replace("_eod_waveforms_concatenated.npz", "")
        for f in ctrl_concat_files
    )
    ctrl_entries = [(ind_id, 'variable') for ind_id in concat_ids]
    for f in ctrl_fixed_files:
        ind_id = os.path.basename(f).replace("_eod_waveforms.npz", "")
        if ind_id not in concat_ids:
            ctrl_entries.append((ind_id, 'fixed'))

    reference_library = {}
    for ind_id, wf_format in ctrl_entries:
        table_file = os.path.join(control_path, f"{ind_id}_eod_table.csv")
        if not os.path.exists(table_file):
            continue
        ctrl_table = pd.read_csv(table_file)
        if 'snippet_p1_idx' not in ctrl_table.columns:
            continue
        wf_base_ctrl = os.path.join(control_path, f"{ind_id}_eod_waveforms")
        if wf_format == 'variable':
            ctrl_wf_list = load_waveforms(wf_base_ctrl, format='npz', length='variable')
        else:
            ctrl_arr     = np.load(wf_base_ctrl + '.npz')['waveforms']
            ctrl_wf_list = [ctrl_arr[i] for i in range(ctrl_arr.shape[0])]
        if not ctrl_wf_list or len(ctrl_wf_list) != len(ctrl_table):
            continue
        ctrl_wf_norm = normalize_waveforms(
            ctrl_wf_list,
            ctrl_table['snippet_p1_idx'].values,
            ctrl_table['snippet_p2_idx'].values,
            method='p1_unity', crop_and_interpolate=True,
            crop_factor=crop_factor, target_length=waveform_target_length
        )
        ctrl_wf_norm = np.array(ctrl_wf_norm)
        ctrl_norms   = np.linalg.norm(ctrl_wf_norm, axis=1, keepdims=True)
        ctrl_norms[ctrl_norms == 0] = 1.0
        reference_library[ind_id] = {
            'mean_wf':      (ctrl_wf_norm / ctrl_norms).mean(axis=0),
            'species_code': ind_id[:2].upper()
        }
        print(f"  Ctrl: {ind_id} ({ind_id[:2].upper()})")

    all_species_codes = sorted(set(v['species_code'] for v in reference_library.values()))

    if len(all_species_codes) >= 2:
        ref_ids     = list(reference_library.keys())
        ref_matrix  = np.array([reference_library[rid]['mean_wf'] for rid in ref_ids])
        ref_species = np.array([reference_library[rid]['species_code'] for rid in ref_ids])

        n_pca_cls      = max(1, min(len(ref_matrix) - 1, waveform_target_length, 20))
        pca_cls        = PCA(n_components=n_pca_cls)
        ref_pca_scores = pca_cls.fit_transform(ref_matrix)
        lda            = LinearDiscriminantAnalysis()
        lda.fit(ref_pca_scores, ref_species)
        ctrl_lda       = lda.transform(ref_pca_scores)
        n_lda_axes     = len(all_species_codes) - 1

        # LOO accuracy from saved classifier report
        loo_str     = ''
        report_path = os.path.join(output_folder, 'classifier_report.json')
        if os.path.exists(report_path):
            with open(report_path) as _f:
                report = json.load(_f)
            if report.get('loo_accuracy') is not None:
                loo_str = f"  (LOO acc: {report['loo_accuracy']:.0%})"

        sp_colors_lda = {sp: OKABE_ITO[i % len(OKABE_ITO)]
                         for i, sp in enumerate(all_species_codes)}

        fig6, ax6 = plt.subplots(figsize=(FIG_W, FIG_H))

        # Control individual means — filled circles, semi-transparent
        for sp in all_species_codes:
            sp_mask = ref_species == sp
            cx = ctrl_lda[sp_mask, 0]
            cy = (ctrl_lda[sp_mask, 1] if n_lda_axes >= 2
                  else pca_cls.transform(ref_matrix[sp_mask])[:, 0])
            ax6.scatter(cx, cy,
                        color=sp_colors_lda[sp], s=80, alpha=0.45,
                        marker='o', label=f'{sp} ctrl', rasterized=False)

        # Field tracks — stars, fully opaque
        if 'species_assigned' in eod_data.columns:
            for fid in fish_ids_assigned:
                fid_mask      = (eod_data['fish_id'] == fid).values
                track_mean_wf = waveforms_l2[fid_mask].mean(axis=0)
                track_pca     = pca_cls.transform(track_mean_wf[None, :])
                track_lda_pt  = lda.transform(track_pca)
                tx = float(track_lda_pt[0, 0])
                ty = (float(track_lda_pt[0, 1]) if n_lda_axes >= 2
                      else float(track_pca[0, 0]))
                sp_assigned = eod_data.loc[fid_mask, 'species_assigned'].iloc[0].rstrip('?')
                ax6.scatter(tx, ty,
                            color=sp_colors_lda.get(sp_assigned, FG),
                            s=200, alpha=0.95, marker='*',
                            edgecolors=FG, linewidths=0.4,
                            zorder=5, rasterized=False)
                ax6.annotate(f'F{fid}', (tx, ty),
                             xytext=(0, 7), textcoords='offset points',
                             fontsize=7, color=FG, ha='center', va='bottom')

        x_label = 'LD1'
        y_label = 'LD2' if n_lda_axes >= 2 else 'PC1'
        ax6.set_xlabel(x_label)
        ax6.set_ylabel(y_label)
        ax6.set_title(f'Species classification — LDA space{loo_str}')
        ax6.legend(fontsize=8, markerscale=1.5, loc='best')

        plt.tight_layout()
        fig6.savefig(os.path.join(svg_folder, '06_lda_classification.svg'),
                     format='svg', bbox_inches='tight', facecolor=BG)
        plt.close(fig6)
        print("Saved: 06_lda_classification.svg")
    else:
        print("Skipping LDA panel: fewer than 2 species in reference library")
else:
    print("Skipping LDA panel: control_path not set")

print(f"\nAll panels saved to: {svg_folder}")
