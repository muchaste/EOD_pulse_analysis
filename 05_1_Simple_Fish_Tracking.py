import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import json
import glob
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# Import EOD functions
from pulse_functions import (
    load_waveforms,
    normalize_waveforms,
    create_tracking_plot
)

print("="*70)
print("SIMPLE FISH TRACKING")
print("="*70)

root = tk.Tk()
root.withdraw()

print("\nSelect INPUT folder containing EOD data...")
input_folder = filedialog.askdirectory(title="Select Input Folder")
if not input_folder:
    raise ValueError("No input folder selected")
print(f"✓ Input folder: {input_folder}")

print("\nSelect OUTPUT folder...")
output_folder = filedialog.askdirectory(title="Select Output Folder")
if not output_folder:
    raise ValueError("No output folder selected")
print(f"✓ Output folder: {output_folder}")

print("\n" + "="*70)
print("LOADING DATA FILES")
print("="*70)

eod_files = glob.glob(os.path.join(input_folder, "*_eod_table.csv"))
if not eod_files:
    raise ValueError(f"No EOD table files found in {input_folder}")
print(f"\n✓ Found {len(eod_files)} EOD table file(s)")

waveform_npz_files = glob.glob(os.path.join(input_folder, "*_waveforms_concatenated.npz"))
if not waveform_npz_files:
    raise ValueError(f"No waveform NPZ files found in {input_folder}")
print(f"✓ Found {len(waveform_npz_files)} waveform NPZ file(s)")
# prune "_waveforms_concatenated.npz" to get base names
# waveform_bases = set(os.path.basename(f).replace("_waveforms_concatenated.npz", "") for f in waveform_npz_files)

# waveform_metadata_files = glob.glob(os.path.join(input_folder, "*_waveforms_metadata.json"))
# if not waveform_metadata_files:
#     raise ValueError(f"No waveform metadata files found in {input_folder}")
# print(f"✓ Found {len(waveform_metadata_files)} waveform metadata file(s)")

audio_files = glob.glob(os.path.join(input_folder, "*.wav"))
if not audio_files:
    raise ValueError(f"No audio files found in {input_folder}")
print(f"✓ Found {len(audio_files)} audio file(s)")

event_summary_file = os.path.join(input_folder, "all_event_summaries.csv")
if os.path.exists(event_summary_file):
    event_summaries = pd.read_csv(event_summary_file)
    print(f"✓ Loaded event summaries: {len(event_summaries)} events")
else:
    event_summaries = None
    print("⚠ No all_event_summaries.csv found")

file_sets_list = []
for eod_file in eod_files:
    base_name = os.path.basename(eod_file).replace("_eod_table.csv", "")
    event_id = np.int64(base_name.split("event_")[1])
    
    waveform_base = os.path.join(input_folder, f"{base_name}_waveforms")
    # waveform_meta = os.path.join(input_folder, f"{base_name}_waveforms_metadata.json")
    audio_file = os.path.join(input_folder, f"{base_name}.wav")
    
    if os.path.exists(waveform_base+"_concatenated.npz") and os.path.exists(audio_file):
        file_sets_list.append({
            'base_name': base_name,
            'event_id': event_id,
            'eod_file': eod_file,
            'waveform_base': waveform_base,
            # 'waveform_meta': waveform_meta,
            'audio_file': audio_file
        })
    else:
        print(f"⚠ Incomplete file set for {base_name}")

if not file_sets_list:
    raise ValueError("No complete file sets found")

file_sets = pd.DataFrame(file_sets_list)

print(f"\n✓ Matched {len(file_sets)} complete file set(s)")
for idx, row in file_sets.iterrows():
    print(f"  - {row['base_name']}")

if event_summaries is not None:
    file_sets = file_sets.merge(event_summaries, on='event_id', how='left', suffixes=('', '_summary'))
    n_matched = file_sets['event_id'].notna().sum()
    print(f"✓ Merged with event summaries ({n_matched}/{len(file_sets)} matched)")

print("\n" + "="*70)
print("PROCESSING FILES")
print("="*70)

# =============================================================================
# TRACKING PARAMETERS
# =============================================================================

waveform_target_length = 150        # samples after normalization and resampling
min_ipi_s = 0.002                   # 2ms absolute refractory period
max_track_gap_s = 2                 # fragment times out after this gap in Pass 1
max_location_jump_per_s = 200.0       # max plausible velocity in electrode units per second (for gating candidates in Pass 1)

# Shape clustering parameters (applied within each width class)
shape_dbscan_eps = 0.3             # DBSCAN epsilon on L2-normalized waveforms (range [0, 2])
shape_dbscan_min_samples = 5       # minimum pulses to form a shape cluster
dbscan_max_direct = 3000           # above this pulse count, subsample before DBSCAN (caps O(n²) memory)
dbscan_sample_size = 2000          # number of pulses to subsample when width class exceeds dbscan_max_direct

# Pass 1 cost weights — waveform dropped (identity pre-established by shape cluster)
# location_weight + ipi_weight must sum to 1
location_weight = 0.2
ipi_weight = 0.8
location_tolerance = 20.0           # electrode units; more lenient given location noise
ipi_tolerance_fraction = 0.4       # min tolerance = 40% of median IPI
ipi_tolerance_min_s = 0.05         # absolute floor for IPI tolerance
n_recent_for_ipi = 8               # how many recent IPIs to use for median estimate
pass1_cost_threshold = 10.0         # normalized cost; raised slightly since waveform no longer contributes
debug_pass1 = True                 # print one line per new fragment with rejection reason
# Pass 2 (fragment stitching) parameters
pass2_max_gap_s = 2.0              # max time gap between fragment end and start to consider stitching
pass2_waveform_weight = 0.8
pass2_spatial_weight = 0.2
pass2_cost_threshold = 4.0          # normalized cost threshold for stitching fragments
pass2_max_iterations = 3           # max stitching passes
pass2_max_frags = 600              # skip Pass 2 if fragment count exceeds this (prevents O(n²) LAP blow-up)

# Pruning
min_track_pulses = 15               # discard tracks shorter than this
min_track_duration_s = 0.5         # discard tracks shorter than this

# Width-based pre-sorting
width_min_separation_us = 15       # KDE peaks must be at least this far apart to split into classes

# # Pre-sort events by estimated fish count (from event summaries if available)
# if event_summaries is not None and 'mean_ipi_seconds' in file_sets.columns:
#     single_fish_threshold = 0.03
#     single_fish_files = file_sets[file_sets['mean_ipi_seconds'] >= single_fish_threshold]
#     multiple_fish_files = file_sets[file_sets['mean_ipi_seconds'] < single_fish_threshold]
#     print(f"\n✓ Pre-sorting based on mean IPI:")
#     print(f"  - Single fish candidates: {len(single_fish_files)}")
#     print(f"  - Multiple fish candidates: {len(multiple_fish_files)}")
# else:
#     single_fish_files = file_sets
#     multiple_fish_files = pd.DataFrame()
#%%
# =============================================================================
# PER-FILE PROCESSING LOOP
# =============================================================================

all_tracked_data = []

# NOTE: RENAME FILE_SET AND FILE_SETS TO SOMETHING BETTER - EVENT_DATA OR SOMETHING

for file_idx, (row_idx, file_set) in enumerate(file_sets.iterrows()):
    print(f"\n{'='*70}")
    print(f"FILE {file_idx + 1}/{len(file_sets)}: {file_set['base_name']}")
    print(f"{'='*70}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading EOD table...")
    eod_data = pd.read_csv(file_set['eod_file'])
    eod_data['timestamp'] = pd.to_datetime(eod_data['timestamp'])
    eod_data = eod_data.sort_values('timestamp')
    original_indices = eod_data.index.tolist()
    eod_data.reset_index(drop=True, inplace=True)
    print(f"✓ Loaded {len(eod_data)} pulses")

    print("\nLoading waveforms...")
    waveforms_raw = load_waveforms(file_set['waveform_base'], format='npz', length='variable')
    print(f"✓ Loaded {len(waveforms_raw)} waveforms")
    
    # Reorder waveforms to match sorted eod_data
    waveforms_raw = [waveforms_raw[i] for i in original_indices]

    if len(waveforms_raw) != len(eod_data):
        print(f"⚠ WARNING: Waveform count ({len(waveforms_raw)}) != EOD table count ({len(eod_data)}), skipping")
        continue

    # -------------------------------------------------------------------------
    # Normalize waveforms: P1 to +1, baseline subtracted, resampled to fixed length
    # -------------------------------------------------------------------------
    print("\nNormalizing waveforms...")
    normalized_waveforms = normalize_waveforms(
        waveforms_raw,
        snippet_p1_idc=eod_data['snippet_p1_idx'].values,
        snippet_p2_idc=eod_data['snippet_p2_idx'].values,
        method='p1_unity',
        crop_and_interpolate=True,
        crop_factor=4,
        target_length=waveform_target_length
    )
    normalized_waveforms = np.array(normalized_waveforms)  # shape: (n_pulses, waveform_target_length)

    # L2-normalize each waveform so Euclidean distance is scale-independent (range [0, 2])
    norms = np.linalg.norm(normalized_waveforms, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    waveforms_l2 = normalized_waveforms / norms
    del normalized_waveforms, norms
    print(f"✓ Normalized, shape: {waveforms_l2.shape}")

    # -------------------------------------------------------------------------
    # Step 1: Diagnostic visualization
    # -------------------------------------------------------------------------
    print("\nGenerating diagnostic plot...")
    t_sec = (eod_data['timestamp'] - eod_data['timestamp'].iloc[0]).dt.total_seconds().values
    widths = eod_data['eod_width_us'].values

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=False)
    fig.suptitle(file_set['base_name'], fontsize=10)

    sc = axes[0].scatter(t_sec, eod_data['pulse_location'].values, c=widths,
                         cmap='viridis', s=3, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=axes[0], label='Width (µs)')
    axes[0].set_ylabel('Location (electrode units)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_title('Pulse location vs time (color = width)')

    axes[1].hist(widths, bins=80, color='steelblue', edgecolor='none')
    axes[1].set_xlabel('EOD width (µs)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Width distribution')

    axes[2].scatter(t_sec, eod_data['eod_amplitude'].values, s=2, alpha=0.4, c='steelblue', rasterized=True)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Amplitude vs time')

    # overlay of normalized waveforms (capped at 2000 random samples for memory/speed)
    axes[3] = fig.add_subplot(4, 1, 4)
    diag_plot_n = min(2000, len(waveforms_l2))
    diag_plot_indices = np.random.choice(len(waveforms_l2), size=diag_plot_n, replace=False)
    for i in diag_plot_indices:
        axes[3].plot(waveforms_l2[i], color='steelblue', alpha=0.1, linewidth=0.5, rasterized=True)
    axes[3].set_title('Normalized Waveforms')
    axes[3].set_xlabel('Time (resampled samples)')
    axes[3].set_ylabel('Amplitude (L2-normalized)')

    plt.tight_layout()
    diag_path = os.path.join(output_folder, f"{file_set['base_name']}_diagnostic.png")
    plt.savefig(diag_path, dpi=120)
    plt.close()
    print(f"✓ Saved diagnostic plot: {os.path.basename(diag_path)}")

    # -------------------------------------------------------------------------
    # Step 2a: Width-based pre-sorting into classes
    # -------------------------------------------------------------------------
    print("\nWidth-based pre-sorting...")
    width_range = np.linspace(widths.min(), widths.max(), 1000)
    kde = gaussian_kde(widths, bw_method=0.1)
    kde_vals = kde(width_range)

    min_peak_distance_bins = int(width_min_separation_us / (width_range[1] - width_range[0]))
    peaks_idx, _ = find_peaks(kde_vals, distance=max(1, min_peak_distance_bins),
                               prominence=0.05 * kde_vals.max())

    if len(peaks_idx) > 1:
        peak_positions = width_range[peaks_idx]
        pulse_width_class = np.argmin(
            np.abs(widths[:, None] - peak_positions[None, :]), axis=1
        )
        n_width_classes = len(peak_positions)
        print(f"✓ Found {n_width_classes} width classes at: {peak_positions.round(1)} µs")
    else:
        pulse_width_class = np.zeros(len(eod_data), dtype=int)
        n_width_classes = 1
        print(f"✓ Single width class (no clear modes separated by >{width_min_separation_us} µs)")

    eod_data['width_class'] = pulse_width_class

    # -------------------------------------------------------------------------
    # Step 2b: Waveform shape clustering within each width class
    # DBSCAN on L2-normalized waveforms; noise points (label=-1) get their own
    # unique shape class so they are still trackable (just not pre-grouped).
    # -------------------------------------------------------------------------
    print("\nShape clustering within width classes...")
    eod_data['shape_class'] = -1
    next_shape_id = 0

    for wc in range(n_width_classes):
        wc_mask = pulse_width_class == wc
        wc_indices = np.where(wc_mask)[0]
        wc_waveforms = waveforms_l2[wc_indices]
        n_wc = len(wc_indices)

        if n_wc > dbscan_max_direct:
            # Subsample to cap O(n²) DBSCAN memory; assign remaining pulses to nearest centroid
            rng = np.random.default_rng(seed=42)
            sample_pos = rng.choice(n_wc, size=min(dbscan_sample_size, n_wc), replace=False)
            sample_wf = wc_waveforms[sample_pos]
            db = DBSCAN(eps=shape_dbscan_eps, min_samples=shape_dbscan_min_samples, metric='euclidean')
            sample_labels = db.fit_predict(sample_wf)
            cluster_ids = np.unique(sample_labels[sample_labels >= 0])
            n_clusters = len(cluster_ids)
            if n_clusters > 0:
                centroids = np.array([sample_wf[sample_labels == cid].mean(axis=0) for cid in cluster_ids])
            else:
                centroids = np.empty((0, wc_waveforms.shape[1]))
            # Initialise all labels; propagate subsample labels first
            db_labels = np.full(n_wc, -1, dtype=int)
            for sp, sl in zip(sample_pos, sample_labels):
                db_labels[sp] = sl
            # Assign non-sampled pulses to nearest centroid by L2 distance
            unassigned_mask_wc = np.ones(n_wc, dtype=bool)
            unassigned_mask_wc[sample_pos] = False
            if n_clusters > 0 and unassigned_mask_wc.any():
                unassigned_wf = wc_waveforms[unassigned_mask_wc]
                dists = np.linalg.norm(unassigned_wf[:, None, :] - centroids[None, :, :], axis=2)
                db_labels[unassigned_mask_wc] = cluster_ids[np.argmin(dists, axis=1)]
            n_noise = (db_labels == -1).sum()
            print(f"  Width class {wc}: {n_clusters} shape cluster(s), {n_noise} noise pulses "
                  f"[subsampled {min(dbscan_sample_size, n_wc)}/{n_wc}]")
        else:
            db = DBSCAN(eps=shape_dbscan_eps, min_samples=shape_dbscan_min_samples, metric='euclidean')
            db_labels = db.fit_predict(wc_waveforms)
            n_clusters = (np.unique(db_labels[db_labels >= 0])).size
            n_noise = (db_labels == -1).sum()
            print(f"  Width class {wc}: {n_clusters} shape cluster(s), {n_noise} noise pulses")

        # Assign globally unique shape class ids; noise pulses each get a unique id
        for i, pulse_idx in enumerate(wc_indices):
            if db_labels[i] >= 0:
                eod_data.loc[pulse_idx, 'shape_class'] = next_shape_id + db_labels[i]
            else:
                eod_data.loc[pulse_idx, 'shape_class'] = next_shape_id + n_clusters + i

        next_shape_id += n_clusters + len(wc_indices)

    # Build list of (width_class, shape_class) groups for Pass 1
    shape_groups = (
        eod_data[['width_class', 'shape_class']]
        .drop_duplicates()
        .sort_values(['width_class', 'shape_class'])
        .values.tolist()
    )
    print(f"✓ Total shape groups for tracking: {len(shape_groups)}")

    # -------------------------------------------------------------------------
    # Step 3: Pass 1 - Sequential assignment within each width class
    # Result: each pulse gets a 'fragment_id'; fragments are continuous runs
    # with no gaps > max_track_gap_s
    # -------------------------------------------------------------------------
    print("\nPass 1: Sequential assignment...")

    eod_data['fragment_id'] = -1
    next_fragment_id = 0
    # fragments dict: fragment_id -> {
    #   'history': [pulse_idx, ...],
    #   'ipi_history': [ipi_s, ...],
    #   'last_timestamp': pd.Timestamp,
    #   'last_location': float,
    #   'width_class': int,
    #   'shape_class': int
    # }
    fragments = {}

    for wc, sc in shape_groups:
        group_mask = (eod_data['width_class'] == wc) & (eod_data['shape_class'] == sc)
        group_indices = np.where(group_mask)[0]

        for pulse_idx in group_indices:
            pulse_ts = eod_data.loc[pulse_idx, 'timestamp']
            pulse_loc = eod_data.loc[pulse_idx, 'pulse_location']

            # Candidates: same (width_class, shape_class), within time and velocity gates
            candidate_ids = []
            rejected_reasons = {}  # fid -> reason string, for debug
            for fid, f in fragments.items():
                if f['shape_class'] != sc or f['width_class'] != wc:
                    continue
                dt = (pulse_ts - f['last_timestamp']).total_seconds()
                if dt > max_track_gap_s:
                    rejected_reasons[fid] = f'gap={dt:.3f}s > max={max_track_gap_s}s'
                    continue
                if dt < min_ipi_s:
                    rejected_reasons[fid] = f'dt={dt*1000:.2f}ms < min_ipi={min_ipi_s*1000:.2f}ms'
                    continue
                loc_diff = abs(pulse_loc - f['last_location'])
                max_jump = max_location_jump_per_s * max(dt, 0.001)
                if loc_diff > max_jump:
                    rejected_reasons[fid] = f'loc_jump={loc_diff:.3f} > max={max_jump:.3f} (dt={dt:.3f}s)'
                    continue
                candidate_ids.append(fid)

            if not candidate_ids:
                if debug_pass1:
                    t_rel = (pulse_ts - eod_data.loc[0, 'timestamp']).total_seconds()
                    if rejected_reasons:
                        # There were same-class fragments but all were gated out
                        reason_summary = '; '.join(f'frag{fid}:[{r}]' for fid, r in list(rejected_reasons.items())[:3])
                        if len(rejected_reasons) > 3:
                            reason_summary += f' (+{len(rejected_reasons)-3} more)'
                        print(f'  [NEW frag {next_fragment_id}] t={t_rel:.3f}s loc={pulse_loc:.3f} '
                              f'wc={wc} sc={sc} | gated out: {reason_summary}')
                    else:
                        print(f'  [NEW frag {next_fragment_id}] t={t_rel:.3f}s loc={pulse_loc:.3f} '
                              f'wc={wc} sc={sc} | no same-class fragments active')
                fragments[next_fragment_id] = {
                    'history': [pulse_idx],
                    'ipi_history': [],
                    'last_timestamp': pulse_ts,
                    'last_location': pulse_loc,
                    'width_class': wc,
                    'shape_class': sc
                }
                eod_data.loc[pulse_idx, 'fragment_id'] = next_fragment_id
                next_fragment_id += 1
                continue

            if len(candidate_ids) == 1:
                # Only one candidate - assign directly without computing cost
                best_fid = candidate_ids[0]
                f = fragments[best_fid]
                dt = (pulse_ts - f['last_timestamp']).total_seconds()
                f['ipi_history'].append(dt)
                f['history'].append(pulse_idx)
                f['last_timestamp'] = pulse_ts
                f['last_location'] = pulse_loc
                eod_data.loc[pulse_idx, 'fragment_id'] = best_fid
                continue

            # Multiple candidates: score by location + IPI
            # (waveform identity is pre-established by shape cluster)
            best_fid = None
            best_cost = np.inf
            for fid in candidate_ids:
                f = fragments[fid]
                dt = (pulse_ts - f['last_timestamp']).total_seconds()

                # Location cost: linear velocity prediction from last two pulses
                if len(f['history']) >= 2:
                    prev_idx = f['history'][-2]
                    dt_prev = (f['last_timestamp'] - eod_data.loc[prev_idx, 'timestamp']).total_seconds()
                    velocity = (f['last_location'] - eod_data.loc[prev_idx, 'pulse_location']) / max(dt_prev, 1e-6)
                    predicted_loc = f['last_location'] + velocity * dt
                else:
                    predicted_loc = f['last_location']

                loc_cost = abs(pulse_loc - predicted_loc) / location_tolerance

                # IPI cost
                if len(f['ipi_history']) >= 1:
                    recent_ipis = f['ipi_history'][-n_recent_for_ipi:]
                    median_ipi = np.median(recent_ipis)
                    ipi_tol = max(median_ipi * ipi_tolerance_fraction, ipi_tolerance_min_s)
                    ipi_cost = abs(dt - median_ipi) / ipi_tol
                else:
                    ipi_cost = 0.0

                cost = location_weight * loc_cost + ipi_weight * ipi_cost

                if cost < best_cost:
                    best_cost = cost
                    best_fid = fid

            if best_cost < pass1_cost_threshold:
                f = fragments[best_fid]
                dt = (pulse_ts - f['last_timestamp']).total_seconds()
                f['ipi_history'].append(dt)
                f['history'].append(pulse_idx)
                f['last_timestamp'] = pulse_ts
                f['last_location'] = pulse_loc
                eod_data.loc[pulse_idx, 'fragment_id'] = best_fid
            else:
                if debug_pass1:
                    t_rel = (pulse_ts - eod_data.loc[0, 'timestamp']).total_seconds()
                    f_best = fragments[best_fid]
                    dt_best = (pulse_ts - f_best['last_timestamp']).total_seconds()
                    if len(f_best['history']) >= 2:
                        prev_idx_d = f_best['history'][-2]
                        dt_prev_d = (f_best['last_timestamp'] - eod_data.loc[prev_idx_d, 'timestamp']).total_seconds()
                        vel_d = (f_best['last_location'] - eod_data.loc[prev_idx_d, 'pulse_location']) / max(dt_prev_d, 1e-6)
                        pred_loc_d = f_best['last_location'] + vel_d * dt_best
                    else:
                        pred_loc_d = f_best['last_location']
                    loc_cost_d = abs(pulse_loc - pred_loc_d) / location_tolerance
                    if len(f_best['ipi_history']) >= 1:
                        med_ipi_d = np.median(f_best['ipi_history'][-n_recent_for_ipi:])
                        ipi_tol_d = max(med_ipi_d * ipi_tolerance_fraction, ipi_tolerance_min_s)
                        ipi_cost_d = abs(dt_best - med_ipi_d) / ipi_tol_d
                    else:
                        med_ipi_d = float('nan')
                        ipi_tol_d = float('nan')
                        ipi_cost_d = 0.0
                    print(f'  [NEW frag {next_fragment_id}] t={t_rel:.3f}s loc={pulse_loc:.3f} '
                          f'wc={wc} sc={sc} | cost_too_high: best_frag={best_fid} '
                          f'cost={best_cost:.3f} > thr={pass1_cost_threshold} | '
                          f'loc_cost={loc_cost_d:.3f} (pred={pred_loc_d:.3f}, actual={pulse_loc:.3f}, tol={location_tolerance}) | '
                          f'ipi_cost={ipi_cost_d:.3f} (dt={dt_best:.3f}s, med_ipi={med_ipi_d:.3f}s, tol={ipi_tol_d:.3f}s)')
                fragments[next_fragment_id] = {
                    'history': [pulse_idx],
                    'ipi_history': [],
                    'last_timestamp': pulse_ts,
                    'last_location': pulse_loc,
                    'width_class': wc,
                    'shape_class': sc
                }
                eod_data.loc[pulse_idx, 'fragment_id'] = next_fragment_id
                next_fragment_id += 1

    n_fragments_pass1 = len(fragments)
    n_assigned_pass1 = (eod_data['fragment_id'] >= 0).sum()
    print(f"✓ Pass 1 complete: {n_fragments_pass1} fragments, {n_assigned_pass1}/{len(eod_data)} pulses assigned")

    # -------------------------------------------------------------------------
    # Step 4: Pass 2 - Fragment stitching using LAP
    # Build a cost matrix of (fragment endings) x (fragment starts).
    # Merge fragment pairs whose assignment cost is below threshold.
    # Iterate until no more merges occur.
    # -------------------------------------------------------------------------
    print("\nPass 2: Fragment stitching...")

    for stitch_iter in range(pass2_max_iterations):
        frag_ids = list(fragments.keys())
        n_frags = len(frag_ids)
        if n_frags < 2:
            break
        if n_frags > pass2_max_frags:
            print(f"  Iteration {stitch_iter + 1}: {n_frags} fragments > pass2_max_frags={pass2_max_frags}, skipping Pass 2")
            break

        # Precompute per-fragment summary stats
        frag_start_ts = {fid: eod_data.loc[f['history'][0], 'timestamp'] for fid, f in fragments.items()}
        frag_end_ts = {fid: eod_data.loc[f['history'][-1], 'timestamp'] for fid, f in fragments.items()}
        frag_start_loc = {fid: eod_data.loc[f['history'][0], 'pulse_location'] for fid, f in fragments.items()}
        frag_end_loc = {fid: eod_data.loc[f['history'][-1], 'pulse_location'] for fid, f in fragments.items()}
        # Start/end waveform signatures (median of first/last 5 pulses)
        frag_start_wf = {fid: np.median(waveforms_l2[f['history'][:5]], axis=0) for fid, f in fragments.items()}
        frag_end_wf = {fid: np.median(waveforms_l2[f['history'][-5:]], axis=0) for fid, f in fragments.items()}

        INF = 1e6
        cost_matrix = np.full((n_frags, n_frags), INF)

        for i, fid_end in enumerate(frag_ids):
            for j, fid_start in enumerate(frag_ids):
                if fid_end == fid_start:
                    continue
                # start must come after end; gap must be within window
                gap = (frag_start_ts[fid_start] - frag_end_ts[fid_end]).total_seconds()
                if gap <= 0 or gap > pass2_max_gap_s:
                    continue
                # Must be same (width_class, shape_class)
                if fragments[fid_end]['width_class'] != fragments[fid_start]['width_class']:
                    continue
                if fragments[fid_end]['shape_class'] != fragments[fid_start]['shape_class']:
                    continue
                # Waveform distance
                wf_cost = np.linalg.norm(frag_end_wf[fid_end] - frag_start_wf[fid_start])
                # Spatial plausibility: velocity over the gap
                loc_diff = abs(frag_start_loc[fid_start] - frag_end_loc[fid_end])
                max_allowed_loc_diff = 3.0 * gap  # generous: 3 electrode units / second
                if loc_diff > max_allowed_loc_diff:
                    continue
                spatial_cost = loc_diff / max(location_tolerance, 0.01)
                cost = pass2_waveform_weight * wf_cost + pass2_spatial_weight * spatial_cost
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        merges = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < pass2_cost_threshold:
                merges.append((frag_ids[r], frag_ids[c]))

        if not merges:
            print(f"  Iteration {stitch_iter + 1}: no merges, stopping")
            break

        # Apply merges: append start fragment's history onto end fragment
        # Consume each fragment at most once (LAP guarantees one-to-one, but verify)
        consumed = set()
        for fid_end, fid_start in merges:
            if fid_end in consumed or fid_start in consumed:
                continue
            f_end = fragments[fid_end]
            f_start = fragments[fid_start]
            gap = (frag_start_ts[fid_start] - frag_end_ts[fid_end]).total_seconds()
            f_end['history'].extend(f_start['history'])
            f_end['ipi_history'].extend(f_start['ipi_history'])
            f_end['ipi_history'].append(gap)
            f_end['last_timestamp'] = f_start['last_timestamp']
            f_end['last_location'] = f_start['last_location']
            recent_wfs = waveforms_l2[f_end['history'][-10:]]
            f_end['waveform_signature'] = np.median(recent_wfs, axis=0)
            # Update fragment_id labels in eod_data
            for pidx in f_start['history']:
                eod_data.loc[pidx, 'fragment_id'] = fid_end
            del fragments[fid_start]
            consumed.add(fid_start)
            consumed.add(fid_end)

        print(f"  Iteration {stitch_iter + 1}: merged {len(merges)} fragment pair(s), {len(fragments)} fragments remain")

    print(f"✓ Pass 2 complete: {len(fragments)} fragments")

    # -------------------------------------------------------------------------
    # Step 5: Prune short/spurious fragments
    # -------------------------------------------------------------------------
    print("\nPruning short fragments...")
    fids_to_remove = []
    for fid, f in fragments.items():
        n_pulses = len(f['history'])
        duration = (eod_data.loc[f['history'][-1], 'timestamp'] -
                    eod_data.loc[f['history'][0], 'timestamp']).total_seconds()
        if n_pulses < min_track_pulses or duration < min_track_duration_s:
            fids_to_remove.append(fid)
            for pidx in f['history']:
                eod_data.loc[pidx, 'fragment_id'] = -1

    for fid in fids_to_remove:
        del fragments[fid]

    print(f"✓ Removed {len(fids_to_remove)} short fragments, {len(fragments)} surviving")

    # Remap fragment_ids to compact fish_ids (0, 1, 2, ...)
    fid_to_fish = {fid: i for i, fid in enumerate(sorted(fragments.keys()))}
    eod_data['fish_id'] = eod_data['fragment_id'].map(fid_to_fish).fillna(-1).astype(int)
    n_fish = len(fragments)

    # -------------------------------------------------------------------------
    # Step 6: Summary and validation plot
    # -------------------------------------------------------------------------
    assigned = (eod_data['fish_id'] >= 0).sum()
    print(f"\n--- Tracking summary ---")
    print(f"  Fish identified: {n_fish}")
    print(f"  Pulses assigned: {assigned}/{len(eod_data)} ({100*assigned/len(eod_data):.1f}%)")

    for fid in sorted(fid_to_fish):
        fish_id = fid_to_fish[fid]
        f = fragments[fid]
        n_p = len(f['history'])
        dur = (eod_data.loc[f['history'][-1], 'timestamp'] -
               eod_data.loc[f['history'][0], 'timestamp']).total_seconds()
        mean_loc = eod_data.loc[f['history'], 'pulse_location'].mean()
        mean_width = eod_data.loc[f['history'], 'eod_width_us'].mean()
        if len(f['ipi_history']) > 0:
            mean_rate = 1.0 / np.median(f['ipi_history'])
        else:
            mean_rate = float('nan')
        print(f"  Fish {fish_id:2d}: {n_p:5d} pulses, {dur:6.1f}s, {mean_rate:5.1f} Hz, "
              f"loc {mean_loc:.2f}, width {mean_width:.0f} µs")

    # Validation plot: fish_id-colored pulse location vs time
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"{file_set['base_name']} — tracking result", fontsize=10)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_fish, 1)))
    unassigned_mask = eod_data['fish_id'] < 0
    if unassigned_mask.any():
        axes[0].scatter(t_sec[unassigned_mask], eod_data.loc[unassigned_mask, 'pulse_location'],
                        c='lightgray', s=3, alpha=0.4, label='unassigned', rasterized=True)
        axes[1].scatter(t_sec[unassigned_mask], eod_data.loc[unassigned_mask, 'eod_width_us'],
                        c='lightgray', s=3, alpha=0.4, rasterized=True)

    for fish_id in range(n_fish):
        mask = eod_data['fish_id'] == fish_id
        c = colors[fish_id % len(colors)]
        axes[0].scatter(t_sec[mask], eod_data.loc[mask, 'pulse_location'],
                        color=c, s=4, alpha=0.7, label=f'Fish {fish_id}', rasterized=True)
        axes[1].scatter(t_sec[mask], eod_data.loc[mask, 'eod_width_us'],
                        color=c, s=4, alpha=0.7, rasterized=True)

    axes[0].set_ylabel('Location (electrode units)')
    axes[0].legend(markerscale=3, loc='upper right', fontsize=8)
    axes[1].set_ylabel('Width (µs)')
    axes[1].set_xlabel('Time (s)')
    plt.tight_layout()

    result_path = os.path.join(output_folder, f"{file_set['base_name']}_tracked.png")
    plt.savefig(result_path, dpi=120)
    plt.close()
    print(f"\n✓ Saved tracking plot: {os.path.basename(result_path)}")

    # Save tracked EOD table
    out_csv = os.path.join(output_folder, f"{file_set['base_name']}_tracked.csv")
    eod_data.to_csv(out_csv, index=False)
    print(f"✓ Saved tracked table: {os.path.basename(out_csv)}")

    all_tracked_data.append({
        'base_name': file_set['base_name'],
        'n_fish': n_fish,
        'n_pulses': len(eod_data),
        'assignment_rate': assigned / len(eod_data) if len(eod_data) > 0 else 0.0,
    })
    del eod_data, waveforms_raw, waveforms_l2

print("\n" + "="*70)
print("TRACKING COMPLETE")
print("="*70)
print(f"Processed {len(all_tracked_data)}/{len(file_sets)} file(s) successfully")
