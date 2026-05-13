import os
import glob
import random
import itertools
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from pulse_functions import load_waveforms, normalize_waveforms

# ============================================================
# CONFIGURATION — edit before running
# ============================================================

N_RANDOM_SAMPLES = 500    # set to 0 for exhaustive search
RANDOM_SEED      = 42
VERBOSE          = False  # per-event diagnostic prints

# Fixed tracking constants (not tuned)
DBSCAN_MIN_SAMPLES = 5
DBSCAN_MAX_DIRECT  = 30000
DBSCAN_SAMPLE_SIZE = 20000
N_RECENT_IPI       = 8
PASS2_MAX_FRAGS    = 1200
MIN_TRACK_PULSES   = 15
MIN_TRACK_DUR_S    = 0.5

# ============================================================
# PARAMETER GRID
# ============================================================

GRID = {
    'waveform_target_length':       [150, 300],
    'crop_factor':                  [4, 7],
    'min_ipi_s':                    [0.01, 0.005],
    'max_track_gap_s':              [2, 5, 10],
    'max_location_jump_per_s':      [100, 200, 400],
    'knn_percentile':               [70, 80, 90],
    'min_shape_eps':                [0.2, 0.4, 0.6],
    'fft_artifact_threshold':       [0.5, 0.75, 0.9],
    'location_tolerance':           [5, 10, 20],
    'ipi_tolerance_fraction':       [0.2, 0.4, 0.6],
    'ipi_tolerance_min_s':          [0.01, 0.05, 0.1],
    'pass1_new_frag_cost':          [1.0, 2.0, 4.0],
    'pass2_max_gap_s':              [1.0, 2.0, 4.0],
    'pass2_cost_threshold':         [1.0, 2.0, 4.0],
    'pass2_max_iterations':         [2, 4, 6],
    'pass2_overlap_wf_threshold':   [0.2, 0.4, 0.6],
    'pass2_overlap_min_s':          [0.05, 0.1, 0.2],
    'pass2_overlap_max_iterations': [2, 4, 6],
    'width_min_separation_us':      [15.0, 30.0],
}

_w = [0.2, 0.4, 0.6]
WEIGHT_TRIPLETS = [
    (lw, iw, ww) for lw in _w for iw in _w for ww in _w
    if abs(lw + iw + ww - 1.0) < 1e-9
]

WEIGHT_PAIRS = [(ww, 1.0 - ww) for ww in [0.2, 0.4, 0.5, 0.6, 0.8]]

# ============================================================
# FOLDER SELECTION
# ============================================================

tk_root = tk.Tk()
tk_root.withdraw()
folder_1fish  = filedialog.askdirectory(title="Select 1-fish event folder")
folder_2fish  = filedialog.askdirectory(title="Select 2-fish event folder")
output_folder = filedialog.askdirectory(title="Select output folder")
tk_root.destroy()

if not folder_1fish or not folder_2fish or not output_folder:
    raise ValueError("Folder selection cancelled")

os.makedirs(output_folder, exist_ok=True)

# ============================================================
# BUILD GROUND TRUTH LIST
# ============================================================

ground_truth = []

for eod_file in sorted(glob.glob(os.path.join(folder_1fish, "*_eod_table.csv"))):
    base_name = os.path.basename(eod_file).replace("_eod_table.csv", "")
    wf_base   = os.path.join(folder_1fish, f"{base_name}_waveforms")
    if os.path.exists(wf_base + "_concatenated.npz"):
        ground_truth.append({
            'eod_file':         eod_file,
            'wf_base':          wf_base,
            'folder':           folder_1fish,
            'event_id':         base_name,
            'n_fish_annotated': 1,
        })
    else:
        print(f"  ⚠ {base_name}: no waveform NPZ found, skipping")

for eod_file in sorted(glob.glob(os.path.join(folder_2fish, "*_eod_table.csv"))):
    base_name = os.path.basename(eod_file).replace("_eod_table.csv", "")
    wf_base   = os.path.join(folder_2fish, f"{base_name}_waveforms")
    if os.path.exists(wf_base + "_concatenated.npz"):
        ground_truth.append({
            'eod_file':         eod_file,
            'wf_base':          wf_base,
            'folder':           folder_2fish,
            'event_id':         base_name,
            'n_fish_annotated': 2,
        })
    else:
        print(f"  ⚠ {base_name}: no waveform NPZ found, skipping")

print(f"\nFound {len(ground_truth)} events total")
print(f"  1-fish: {sum(1 for g in ground_truth if g['n_fish_annotated'] == 1)}")
print(f"  2-fish: {sum(1 for g in ground_truth if g['n_fish_annotated'] == 2)}")

# ============================================================
# BUILD PARAMETER COMBO LIST
# ============================================================

grid_keys  = list(GRID.keys())
grid_vals  = [GRID[k] for k in grid_keys]
total_indep = 1
for v in grid_vals:
    total_indep *= len(v)
total_combos = total_indep * len(WEIGHT_TRIPLETS) * len(WEIGHT_PAIRS)
print(f"\nTotal possible combinations: {total_combos:,}  |  weight triplets: {len(WEIGHT_TRIPLETS)}  |  weight pairs: {len(WEIGHT_PAIRS)}")

rng_combo = random.Random(RANDOM_SEED)

if N_RANDOM_SAMPLES <= 0 or total_combos <= N_RANDOM_SAMPLES:
    param_combos = []
    for indep in itertools.product(*grid_vals):
        d = dict(zip(grid_keys, indep))
        for lw, iw, ww in WEIGHT_TRIPLETS:
            for p2ww, p2sw in WEIGHT_PAIRS:
                c = dict(d)
                c['location_weight']      = lw
                c['ipi_weight']           = iw
                c['waveform_weight']      = ww
                c['pass2_waveform_weight'] = p2ww
                c['pass2_spatial_weight']  = p2sw
                param_combos.append(c)
    print(f"Mode: exhaustive ({len(param_combos)} combinations)")
else:
    param_combos = []
    for _ in range(N_RANDOM_SAMPLES):
        d = {k: rng_combo.choice(v) for k, v in GRID.items()}
        lw, iw, ww           = rng_combo.choice(WEIGHT_TRIPLETS)
        p2ww, p2sw           = rng_combo.choice(WEIGHT_PAIRS)
        d['location_weight']       = lw
        d['ipi_weight']            = iw
        d['waveform_weight']       = ww
        d['pass2_waveform_weight'] = p2ww
        d['pass2_spatial_weight']  = p2sw
        param_combos.append(d)
    print(f"Mode: random search ({len(param_combos)} of {total_combos:,} combinations)")

# ============================================================
# PRE-LOAD ALL EVENTS (once, normalize for every (wt, cf) pair)
# ============================================================

wt_cf_combos = list(set(
    (wt, cf)
    for wt in GRID['waveform_target_length']
    for cf in GRID['crop_factor']
))

events_data = []
n_skipped   = 0
print(f"\nPre-loading {len(ground_truth)} events...")

for gt in ground_truth:
    eod_file         = gt['eod_file']
    wf_base          = gt['wf_base']
    folder           = gt['folder']
    event_id         = gt['event_id']
    n_fish_annotated = gt['n_fish_annotated']

    ap_file = os.path.join(folder, "analysis_parameters.csv")
    if os.path.exists(ap_file):
        ap_df         = pd.read_csv(ap_file)
        interp_factor = float(ap_df['interp_factor'].iloc[0]) if 'interp_factor' in ap_df.columns else 3.0
    else:
        interp_factor = 3.0
    step_us = 1e6 / (96000.0 * interp_factor)

    eod_df = pd.read_csv(eod_file)
    eod_df['timestamp'] = pd.to_datetime(eod_df['timestamp'])
    eod_df = eod_df.sort_values('timestamp')
    orig_idx = eod_df.index.tolist()
    eod_df.reset_index(drop=True, inplace=True)

    wf_raw = load_waveforms(wf_base, format='npz', length='variable')
    wf_raw = [wf_raw[i] for i in orig_idx]

    if len(wf_raw) != len(eod_df):
        print(f"  ⚠ {event_id}: waveform/table count mismatch, skipping")
        n_skipped += 1
        continue

    # Slope subtraction (remove linear DC drift)
    wf_detrended = []
    for wf in wf_raw:
        slope = np.linspace(float(wf[0]), float(wf[-1]), len(wf))
        wf_detrended.append(wf - slope)
    del wf_raw

    # Normalize for every (wt, cf) combination
    pre_normalized = {}
    for wt, cf in wt_cf_combos:
        wf_p1 = np.array(normalize_waveforms(
            wf_detrended,
            snippet_p1_idc=eod_df['snippet_p1_idx'].values,
            snippet_p2_idc=eod_df['snippet_p2_idx'].values,
            method='p1_unity', crop_and_interpolate=True,
            crop_factor=cf, target_length=wt
        ), dtype=np.float64)
        norms = np.linalg.norm(wf_p1, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        wf_p1 /= norms

        wf_p2 = np.array(normalize_waveforms(
            wf_detrended,
            snippet_p1_idc=eod_df['snippet_p2_idx'].values,
            snippet_p2_idc=eod_df['snippet_p1_idx'].values,
            method='p1_unity', crop_and_interpolate=True,
            crop_factor=cf, target_length=wt
        ), dtype=np.float64)
        norms2 = np.linalg.norm(wf_p2, axis=1, keepdims=True)
        norms2[norms2 == 0] = 1.0
        wf_p2 /= norms2

        pre_normalized[(wt, cf)] = (wf_p1, wf_p2)
    del wf_detrended

    events_data.append({
        'event_id':         event_id,
        'n_fish_annotated': n_fish_annotated,
        'timestamp':        eod_df['timestamp'],
        'pulse_locations':  eod_df['pulse_location'].values.astype(np.float64),
        'widths':           eod_df['eod_width_us'].values.astype(np.float64),
        'step_us':          step_us,
        'pre_normalized':   pre_normalized,
    })

print(f"✓ Pre-loaded {len(events_data)} events ({n_skipped} skipped)")

# ============================================================
# MAIN TUNING LOOP
# ============================================================

out_csv        = os.path.join(output_folder, 'parameter_tuning_results.csv')
header_written = False

print(f"\nStarting {len(param_combos)} combos × {len(events_data)} events...")
print(f"Output: {out_csv}\n")

for combo_idx, p in enumerate(param_combos):

    # Unpack all params as scalars — variable names match 05_1 exactly
    waveform_target_length       = int(p['waveform_target_length'])
    crop_factor                  = int(p['crop_factor'])
    min_ipi_s                    = float(p['min_ipi_s'])
    max_track_gap_s              = float(p['max_track_gap_s'])
    max_location_jump_per_s      = float(p['max_location_jump_per_s'])
    knn_percentile               = int(p['knn_percentile'])
    min_shape_eps                = float(p['min_shape_eps'])
    fft_artifact_threshold       = float(p['fft_artifact_threshold'])
    location_weight              = float(p['location_weight'])
    ipi_weight                   = float(p['ipi_weight'])
    waveform_weight              = float(p['waveform_weight'])
    location_tolerance           = float(p['location_tolerance'])
    ipi_tolerance_fraction       = float(p['ipi_tolerance_fraction'])
    ipi_tolerance_min_s          = float(p['ipi_tolerance_min_s'])
    pass1_new_frag_cost          = float(p['pass1_new_frag_cost'])
    pass2_max_gap_s              = float(p['pass2_max_gap_s'])
    pass2_waveform_weight        = float(p['pass2_waveform_weight'])
    pass2_spatial_weight         = float(p['pass2_spatial_weight'])
    pass2_cost_threshold         = float(p['pass2_cost_threshold'])
    pass2_max_iterations         = int(p['pass2_max_iterations'])
    pass2_overlap_wf_threshold   = float(p['pass2_overlap_wf_threshold'])
    pass2_overlap_min_s          = float(p['pass2_overlap_min_s'])
    pass2_overlap_max_iterations = int(p['pass2_overlap_max_iterations'])
    width_min_separation_us      = float(p['width_min_separation_us'])

    # Fixed constants aliased to match 05_1 variable names
    shape_dbscan_min_samples = DBSCAN_MIN_SAMPLES
    dbscan_max_direct        = DBSCAN_MAX_DIRECT
    dbscan_sample_size       = DBSCAN_SAMPLE_SIZE
    n_recent_for_ipi         = N_RECENT_IPI
    pass2_max_frags          = PASS2_MAX_FRAGS
    min_track_pulses         = MIN_TRACK_PULSES
    min_track_duration_s     = MIN_TRACK_DUR_S

    combo_results = []

    for ev in events_data:
        waveforms_l2, waveforms_l2_p2 = ev['pre_normalized'][(waveform_target_length, crop_factor)]

        # Reconstruct eod_data fresh for this event
        eod_data = pd.DataFrame({
            'timestamp':      ev['timestamp'],
            'pulse_location': ev['pulse_locations'],
            'eod_width_us':   ev['widths'],
        })
        widths  = ev['widths']
        step_us = ev['step_us']

        # ------------------------------------------------------------------
        # Step 2a: Width-based pre-sorting into classes
        # ------------------------------------------------------------------
        width_range = np.linspace(widths.min(), widths.max(), 1000)
        width_std   = np.std(widths)
        kde_bw      = max(2.0 * step_us, 1.0) / width_std if width_std > 0 else 0.5
        kde         = gaussian_kde(widths, bw_method=kde_bw)
        kde_vals    = kde(width_range)

        bin_width_us = width_range[1] - width_range[0]
        if bin_width_us > 0:
            min_peak_distance_bins = int(width_min_separation_us / bin_width_us)
        else:
            min_peak_distance_bins = 1
        peaks_idx, _ = find_peaks(kde_vals, distance=max(1, min_peak_distance_bins),
                                   prominence=0.01 * kde_vals.max())

        if len(peaks_idx) > 1:
            peak_positions    = width_range[peaks_idx]
            pulse_width_class = np.argmin(
                np.abs(widths[:, None] - peak_positions[None, :]), axis=1
            )
            n_width_classes = len(peak_positions)
        else:
            pulse_width_class = np.zeros(len(eod_data), dtype=int)
            n_width_classes   = 1

        eod_data['width_class'] = pulse_width_class

        # ------------------------------------------------------------------
        # Step 2b: Shape clustering within each width class
        # ------------------------------------------------------------------
        eod_data['shape_class']  = -1
        eod_data['shape_source'] = ''
        next_shape_id = 0
        n_fft_total   = waveform_target_length // 2 + 1
        n_fft_low     = max(1, n_fft_total // 5)

        for wc in range(n_width_classes):
            wc_mask         = pulse_width_class == wc
            wc_indices      = np.where(wc_mask)[0]
            wc_waveforms_p1 = waveforms_l2[wc_indices]
            wc_waveforms_p2 = waveforms_l2_p2[wc_indices]
            n_wc            = len(wc_indices)

            mean_wf       = wc_waveforms_p1.mean(axis=0)
            mean_wf       = mean_wf - np.mean(mean_wf)
            fft_power     = np.abs(np.fft.rfft(mean_wf))
            low_freq_ratio = fft_power[:n_fft_low].sum() / (fft_power.sum() + 1e-12)
            if low_freq_ratio < fft_artifact_threshold:
                for i, pulse_idx in enumerate(wc_indices):
                    eod_data.loc[pulse_idx, 'shape_class']  = next_shape_id + i
                    eod_data.loc[pulse_idx, 'shape_source'] = 'artifact'
                next_shape_id += len(wc_indices)
                continue

            if n_wc > dbscan_max_direct:
                rng        = np.random.default_rng(seed=42)
                sample_pos = rng.choice(n_wc, size=min(dbscan_sample_size, n_wc), replace=False)
            else:
                sample_pos = np.arange(n_wc)
            n_sample = len(sample_pos)

            if n_sample < 2:
                for pulse_idx in wc_indices:
                    eod_data.loc[pulse_idx, 'shape_class']  = next_shape_id
                    eod_data.loc[pulse_idx, 'shape_source'] = 'p1'
                next_shape_id += 1
                continue

            n_pca_shape = min(5, n_sample - 1, waveform_target_length)
            min_pts     = max(shape_dbscan_min_samples, int(n_sample * 0.01))

            pca_p1          = PCA(n_components=n_pca_shape)
            feat_p1_sample  = pca_p1.fit_transform(wc_waveforms_p1[sample_pos])
            knn_p1          = np.sort(pairwise_distances(feat_p1_sample), axis=1)
            knn_col         = min(min_pts, knn_p1.shape[1] - 1)
            eps_p1          = max(float(np.percentile(knn_p1[:, knn_col], knn_percentile)), min_shape_eps)

            pca_p2          = PCA(n_components=n_pca_shape)
            feat_p2_sample  = pca_p2.fit_transform(wc_waveforms_p2[sample_pos])
            knn_p2          = np.sort(pairwise_distances(feat_p2_sample), axis=1)
            eps_p2          = max(float(np.percentile(knn_p2[:, knn_col], knn_percentile)), min_shape_eps)

            sample_labels_p1 = DBSCAN(eps=eps_p1, min_samples=min_pts, metric='euclidean').fit_predict(feat_p1_sample)
            sample_labels_p2 = DBSCAN(eps=eps_p2, min_samples=min_pts, metric='euclidean').fit_predict(feat_p2_sample)

            cluster_ids_p1 = np.unique(sample_labels_p1[sample_labels_p1 >= 0])
            cluster_ids_p2 = np.unique(sample_labels_p2[sample_labels_p2 >= 0])

            if n_wc > dbscan_max_direct:
                labels_p1 = np.full(n_wc, -1, dtype=int)
                labels_p2 = np.full(n_wc, -1, dtype=int)
                for sp, sl in zip(sample_pos, sample_labels_p1):
                    labels_p1[sp] = sl
                for sp, sl in zip(sample_pos, sample_labels_p2):
                    labels_p2[sp] = sl
                unassigned = np.ones(n_wc, dtype=bool)
                unassigned[sample_pos] = False
                if len(cluster_ids_p1) > 0 and unassigned.any():
                    centroids_p1        = np.array([feat_p1_sample[sample_labels_p1 == c].mean(axis=0) for c in cluster_ids_p1])
                    feats_unassigned_p1 = pca_p1.transform(wc_waveforms_p1[unassigned])
                    d_p1                = np.linalg.norm(feats_unassigned_p1[:, None, :] - centroids_p1[None, :, :], axis=2)
                    labels_p1[unassigned] = cluster_ids_p1[np.argmin(d_p1, axis=1)]
                if len(cluster_ids_p2) > 0 and unassigned.any():
                    centroids_p2        = np.array([feat_p2_sample[sample_labels_p2 == c].mean(axis=0) for c in cluster_ids_p2])
                    feats_unassigned_p2 = pca_p2.transform(wc_waveforms_p2[unassigned])
                    d_p2                = np.linalg.norm(feats_unassigned_p2[:, None, :] - centroids_p2[None, :, :], axis=2)
                    labels_p2[unassigned] = cluster_ids_p2[np.argmin(d_p2, axis=1)]
            else:
                labels_p1 = sample_labels_p1
                labels_p2 = sample_labels_p2

            final_labels  = np.full(n_wc, -1, dtype=int)
            source_labels = np.full(n_wc, '', dtype=object)
            done_p1       = set()
            done_p2       = set()
            size_p1       = {c: int((labels_p1 == c).sum()) for c in cluster_ids_p1}
            size_p2       = {c: int((labels_p2 == c).sum()) for c in cluster_ids_p2}
            next_merged_id = 0

            while True:
                avail_p1 = {c: size_p1[c] for c in cluster_ids_p1 if c not in done_p1}
                avail_p2 = {c: size_p2[c] for c in cluster_ids_p2 if c not in done_p2}
                if not avail_p1 and not avail_p2:
                    break
                best_c1 = max(avail_p1, key=avail_p1.get) if avail_p1 else None
                best_c2 = max(avail_p2, key=avail_p2.get) if avail_p2 else None
                s1 = avail_p1[best_c1] if best_c1 is not None else 0
                s2 = avail_p2[best_c2] if best_c2 is not None else 0
                if s1 >= s2:
                    chosen = (labels_p1 == best_c1)
                    final_labels[chosen]  = next_merged_id
                    source_labels[chosen] = 'p1'
                    for c2 in np.unique(labels_p2[chosen]):
                        if c2 >= 0:
                            done_p2.add(c2)
                    done_p1.add(best_c1)
                else:
                    chosen = (labels_p2 == best_c2)
                    final_labels[chosen]  = next_merged_id
                    source_labels[chosen] = 'p2'
                    for c1 in np.unique(labels_p1[chosen]):
                        if c1 >= 0:
                            done_p1.add(c1)
                    done_p2.add(best_c2)
                next_merged_id += 1

            n_clusters = next_merged_id
            if n_clusters > 0:
                noise_mask = (final_labels == -1)
                if noise_mask.any():
                    merged_centroids = np.array(
                        [wc_waveforms_p1[final_labels == c].mean(axis=0) for c in range(n_clusters)]
                    )
                    d_noise = np.linalg.norm(
                        wc_waveforms_p1[noise_mask][:, None, :] - merged_centroids[None, :, :], axis=2
                    )
                    final_labels[noise_mask]  = np.argmin(d_noise, axis=1)
                    source_labels[noise_mask] = 'noise'
            else:
                final_labels[:]  = 0
                source_labels[:] = 'noise'
                n_clusters       = 1

            for i, pulse_idx in enumerate(wc_indices):
                eod_data.loc[pulse_idx, 'shape_class']  = next_shape_id + final_labels[i]
                eod_data.loc[pulse_idx, 'shape_source'] = source_labels[i]
            next_shape_id += n_clusters

        shape_groups = (
            eod_data[['width_class', 'shape_class']]
            .drop_duplicates()
            .sort_values(['width_class', 'shape_class'])
            .values.tolist()
        )
        if VERBOSE:
            print(f"\nEvent {ev['event_id']} ({ev['n_fish_annotated']} fish): "
                  f"{len(eod_data)} pulses | {n_width_classes} width cls | {len(shape_groups)} shape groups")

        # ------------------------------------------------------------------
        # Step 3: Pass 1 — sequential assignment within each shape group
        # ------------------------------------------------------------------
        eod_data['fragment_id'] = -1
        next_fragment_id        = 0
        fragments               = {}

        for wc, sc in shape_groups:
            group_mask    = (eod_data['width_class'] == wc) & (eod_data['shape_class'] == sc)
            group_indices = np.where(group_mask)[0]

            for pulse_idx in group_indices:
                pulse_ts  = eod_data.loc[pulse_idx, 'timestamp']
                pulse_loc = eod_data.loc[pulse_idx, 'pulse_location']

                candidate_ids = []
                for fid, f in fragments.items():
                    if f['shape_class'] != sc or f['width_class'] != wc:
                        # print("wrong shape or width class")
                        continue
                    dt = (pulse_ts - f['last_timestamp']).total_seconds()
                    if dt > max_track_gap_s:
                        # print("gap too large")
                        continue
                    if dt < min_ipi_s:
                        # print("min_ipi")
                        continue
                    loc_diff = abs(pulse_loc - f['last_location'])
                    max_jump = max_location_jump_per_s * max(dt, 0.001)
                    if loc_diff > max_jump:
                        # print("loc_jump too large")
                        continue
                    candidate_ids.append(fid)

                if not candidate_ids:
                    fragments[next_fragment_id] = {
                        'history':            [pulse_idx],
                        'ipi_history':        [],
                        'last_timestamp':     pulse_ts,
                        'last_location':      pulse_loc,
                        'width_class':        wc,
                        'shape_class':        sc,
                        'waveform_signature': waveforms_l2[pulse_idx].copy(),
                    }
                    eod_data.loc[pulse_idx, 'fragment_id'] = next_fragment_id
                    next_fragment_id += 1
                    continue

                best_fid  = None
                best_cost = np.inf
                for fid in candidate_ids:
                    f  = fragments[fid]
                    dt = (pulse_ts - f['last_timestamp']).total_seconds()

                    if len(f['history']) >= 2:
                        prev_idx      = f['history'][-2]
                        dt_prev       = (f['last_timestamp'] - eod_data.loc[prev_idx, 'timestamp']).total_seconds()
                        velocity      = (f['last_location'] - eod_data.loc[prev_idx, 'pulse_location']) / max(dt_prev, 1e-6)
                        predicted_loc = f['last_location'] + velocity * dt
                    else:
                        predicted_loc = f['last_location']

                    loc_cost = abs(pulse_loc - predicted_loc) / location_tolerance

                    if len(f['ipi_history']) >= 1:
                        recent_ipis = f['ipi_history'][-n_recent_for_ipi:]
                        median_ipi  = np.median(recent_ipis)
                        ipi_tol     = max(median_ipi * ipi_tolerance_fraction, ipi_tolerance_min_s)
                        ipi_cost    = abs(dt - median_ipi) / ipi_tol
                    else:
                        ipi_cost = 0.0

                    waveform_cost = np.linalg.norm(waveforms_l2[pulse_idx] - f['waveform_signature'])
                    cost = location_weight * loc_cost + ipi_weight * ipi_cost + waveform_weight * waveform_cost

                    if cost < best_cost:
                        best_cost = cost
                        best_fid  = fid

                if best_cost < pass1_new_frag_cost:
                    f  = fragments[best_fid]
                    dt = (pulse_ts - f['last_timestamp']).total_seconds()
                    f['ipi_history'].append(dt)
                    f['history'].append(pulse_idx)
                    f['last_timestamp']     = pulse_ts
                    f['last_location']      = pulse_loc
                    f['waveform_signature'] = np.median(waveforms_l2[f['history'][-10:]], axis=0)
                    eod_data.loc[pulse_idx, 'fragment_id'] = best_fid
                else:
                    fragments[next_fragment_id] = {
                        'history':            [pulse_idx],
                        'ipi_history':        [],
                        'last_timestamp':     pulse_ts,
                        'last_location':      pulse_loc,
                        'width_class':        wc,
                        'shape_class':        sc,
                        'waveform_signature': waveforms_l2[pulse_idx].copy(),
                    }
                    eod_data.loc[pulse_idx, 'fragment_id'] = next_fragment_id
                    next_fragment_id += 1

        if VERBOSE:
            _frag_sizes = sorted([len(f['history']) for f in fragments.values()], reverse=True)
            _assigned = int((eod_data['fragment_id'] >= 0).sum())
            print(f"  Pass1: {len(fragments)} frags | {_assigned}/{len(eod_data)} assigned | "
                  f"sizes top5={_frag_sizes[:5]} min={_frag_sizes[-1] if _frag_sizes else 0}")

        # ------------------------------------------------------------------
        # Step 4a: Pass 2a — overlap merge
        # ------------------------------------------------------------------
        for overlap_iter in range(pass2_overlap_max_iterations):
            frag_ids = list(fragments.keys())
            n_frags  = len(frag_ids)
            if n_frags < 2:
                break

            frag_start_ts  = {fid: eod_data.loc[f['history'][0],  'timestamp'] for fid, f in fragments.items()}
            frag_end_ts    = {fid: eod_data.loc[f['history'][-1], 'timestamp'] for fid, f in fragments.items()}
            frag_median_wf = {fid: np.median(waveforms_l2[f['history']], axis=0) for fid, f in fragments.items()}

            candidates = []
            for i in range(n_frags):
                for j in range(i + 1, n_frags):
                    fid_a = frag_ids[i]
                    fid_b = frag_ids[j]
                    overlap_start = max(frag_start_ts[fid_a], frag_start_ts[fid_b])
                    overlap_end   = min(frag_end_ts[fid_a],   frag_end_ts[fid_b])
                    overlap_s     = (overlap_end - overlap_start).total_seconds() if overlap_end > overlap_start else 0.0
                    if overlap_s < pass2_overlap_min_s:
                        continue
                    if fragments[fid_a]['width_class'] != fragments[fid_b]['width_class']:
                        continue
                    wf_dist = np.linalg.norm(frag_median_wf[fid_a] - frag_median_wf[fid_b])
                    if wf_dist >= pass2_overlap_wf_threshold:
                        continue
                    merged_indices   = fragments[fid_a]['history'] + fragments[fid_b]['history']
                    merged_ts_sorted = np.sort([eod_data.loc[pidx, 'timestamp'].timestamp() for pidx in merged_indices])
                    min_gap = np.min(np.diff(merged_ts_sorted)) if len(merged_ts_sorted) > 1 else 1.0
                    if min_gap < min_ipi_s:
                        continue
                    candidates.append((wf_dist, fid_a, fid_b))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            consumed = set()
            n_merged = 0
            for wf_dist, fid_a, fid_b in candidates:
                if fid_a in consumed or fid_b in consumed:
                    continue
                fid_keep = min(fid_a, fid_b)
                fid_drop = max(fid_a, fid_b)
                f_keep   = fragments[fid_keep]
                merged_ts_raw = [(eod_data.loc[pidx, 'timestamp'], pidx)
                                 for pidx in f_keep['history'] + fragments[fid_drop]['history']]
                merged_ts_raw.sort(key=lambda x: x[0])
                sorted_indices = [pidx for _, pidx in merged_ts_raw]
                sorted_ts      = [ts   for ts, _   in merged_ts_raw]
                f_keep['history']            = sorted_indices
                f_keep['ipi_history']        = [(sorted_ts[k] - sorted_ts[k - 1]).total_seconds()
                                                for k in range(1, len(sorted_ts))]
                f_keep['last_timestamp']     = sorted_ts[-1]
                f_keep['last_location']      = eod_data.loc[sorted_indices[-1], 'pulse_location']
                f_keep['waveform_signature'] = np.median(waveforms_l2[sorted_indices[-10:]], axis=0)
                for pidx in fragments[fid_drop]['history']:
                    eod_data.loc[pidx, 'fragment_id'] = fid_keep
                del fragments[fid_drop]
                consumed.add(fid_a)
                consumed.add(fid_b)
                n_merged += 1
            if n_merged == 0:
                break

        # ------------------------------------------------------------------
        # Step 4b: Pass 2b — LAP sequential stitching
        # ------------------------------------------------------------------
        for stitch_iter in range(pass2_max_iterations):
            frag_ids = list(fragments.keys())
            n_frags  = len(frag_ids)
            if n_frags < 2:
                break
            if n_frags > pass2_max_frags:
                break

            frag_start_ts  = {fid: eod_data.loc[f['history'][0],  'timestamp']       for fid, f in fragments.items()}
            frag_end_ts    = {fid: eod_data.loc[f['history'][-1], 'timestamp']       for fid, f in fragments.items()}
            frag_start_loc = {fid: eod_data.loc[f['history'][0],  'pulse_location']  for fid, f in fragments.items()}
            frag_end_loc   = {fid: eod_data.loc[f['history'][-1], 'pulse_location']  for fid, f in fragments.items()}
            frag_start_wf  = {fid: np.median(waveforms_l2[f['history'][:5]],  axis=0) for fid, f in fragments.items()}
            frag_end_wf    = {fid: np.median(waveforms_l2[f['history'][-5:]], axis=0) for fid, f in fragments.items()}

            INF         = 1e6
            cost_matrix = np.full((n_frags, n_frags), INF)

            for i, fid_end in enumerate(frag_ids):
                for j, fid_start in enumerate(frag_ids):
                    if fid_end == fid_start:
                        continue
                    gap = (frag_start_ts[fid_start] - frag_end_ts[fid_end]).total_seconds()
                    if gap <= 0 or gap > pass2_max_gap_s:
                        continue
                    if fragments[fid_end]['width_class'] != fragments[fid_start]['width_class']:
                        continue
                    loc_diff             = abs(frag_start_loc[fid_start] - frag_end_loc[fid_end])
                    max_allowed_loc_diff = max_location_jump_per_s * gap
                    if loc_diff > max_allowed_loc_diff:
                        continue
                    wf_cost      = np.linalg.norm(frag_end_wf[fid_end] - frag_start_wf[fid_start])
                    spatial_cost = loc_diff / max(location_tolerance, 0.01)
                    cost_matrix[i, j] = pass2_waveform_weight * wf_cost + pass2_spatial_weight * spatial_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            merges = [
                (frag_ids[r], frag_ids[c])
                for r, c in zip(row_ind, col_ind)
                if cost_matrix[r, c] < pass2_cost_threshold
            ]

            if not merges:
                break

            consumed = set()
            for fid_end, fid_start in merges:
                if fid_end in consumed or fid_start in consumed:
                    continue
                f_end   = fragments[fid_end]
                f_start = fragments[fid_start]
                gap     = (frag_start_ts[fid_start] - frag_end_ts[fid_end]).total_seconds()
                f_end['history'].extend(f_start['history'])
                f_end['ipi_history'].extend(f_start['ipi_history'])
                f_end['ipi_history'].append(gap)
                f_end['last_timestamp']     = f_start['last_timestamp']
                f_end['last_location']      = f_start['last_location']
                f_end['waveform_signature'] = np.median(waveforms_l2[f_end['history'][-10:]], axis=0)
                for pidx in f_start['history']:
                    eod_data.loc[pidx, 'fragment_id'] = fid_end
                del fragments[fid_start]
                consumed.add(fid_start)
                consumed.add(fid_end)

        # ------------------------------------------------------------------
        # Step 5: Prune short/spurious fragments
        # ------------------------------------------------------------------
        if VERBOSE:
            print(f"  Pre-prune: {len(fragments)} fragments (threshold: >={min_track_pulses} pulses, >={min_track_duration_s}s)")
        fids_to_remove = []
        for fid, f in fragments.items():
            n_pulses = len(f['history'])
            duration = (eod_data.loc[f['history'][-1], 'timestamp'] -
                        eod_data.loc[f['history'][0],  'timestamp']).total_seconds()
            keep = n_pulses >= min_track_pulses and duration >= min_track_duration_s
            if VERBOSE:
                print(f"    fid={fid}: {n_pulses} pulses, {duration:.3f}s -> {'KEEP' if keep else 'PRUNE'}")
            if not keep:
                fids_to_remove.append(fid)
        for fid in fids_to_remove:
            del fragments[fid]

        n_fish = len(fragments)
        if VERBOSE:
            print(f"  >>> n_fish_tracked={n_fish} (annotated={ev['n_fish_annotated']})")
        combo_results.append({
            'event_id':         ev['event_id'],
            'n_fish_annotated': ev['n_fish_annotated'],
            'n_fish_tracked':   n_fish,
        })

    # ----------------------------------------------------------------
    # Compute combo-level metrics
    # ----------------------------------------------------------------
    res_1 = [r for r in combo_results if r['n_fish_annotated'] == 1]
    res_2 = [r for r in combo_results if r['n_fish_annotated'] == 2]
    n_1   = len(res_1)
    n_2   = len(res_2)

    n_exact_1 = sum(1 for r in res_1 if r['n_fish_tracked'] == 1)
    n_over_1  = sum(1 for r in res_1 if r['n_fish_tracked'] >  1)
    n_under_1 = sum(1 for r in res_1 if r['n_fish_tracked'] <  1)
    n_exact_2 = sum(1 for r in res_2 if r['n_fish_tracked'] == 2)
    n_over_2  = sum(1 for r in res_2 if r['n_fish_tracked'] >  2)
    n_under_2 = sum(1 for r in res_2 if r['n_fish_tracked'] <  2)

    acc_1 = n_exact_1 / n_1 if n_1 > 0 else float('nan')
    acc_2 = n_exact_2 / n_2 if n_2 > 0 else float('nan')
    mae   = float(np.mean([abs(r['n_fish_tracked'] - r['n_fish_annotated']) for r in combo_results]))

    if n_1 > 0 and n_2 > 0:
        score = 0.6 * acc_1 + 0.4 * acc_2
    elif n_1 > 0:
        score = acc_1
    elif n_2 > 0:
        score = acc_2
    else:
        score = float('nan')

    row = {
        'combo_idx':  combo_idx,
        'score':      round(score, 4) if not np.isnan(score) else float('nan'),
        'acc_1fish':  round(acc_1,  4) if not np.isnan(acc_1) else float('nan'),
        'acc_2fish':  round(acc_2,  4) if not np.isnan(acc_2) else float('nan'),
        'mae':        round(mae,    4),
        'n_exact_1':  n_exact_1, 'n_over_1':   n_over_1,  'n_under_1':  n_under_1,
        'n_exact_2':  n_exact_2, 'n_over_2':   n_over_2,  'n_under_2':  n_under_2,
        'n_events_1': n_1,       'n_events_2': n_2,
    }
    row.update(p)

    row_df = pd.DataFrame([row])
    row_df.to_csv(out_csv, mode='a', header=not header_written, index=False)
    header_written = True

    if (combo_idx + 1) % 10 == 0 or combo_idx == 0:
        print(f"  combo {combo_idx + 1:4d}/{len(param_combos)}  "
              f"score={score:.4f}  acc_1={acc_1:.3f}  acc_2={acc_2:.3f}  mae={mae:.3f}")

print(f"\n✓ Done. Results saved to: {out_csv}")

final_df = pd.read_csv(out_csv).sort_values('score', ascending=False)
top_cols = [
    'combo_idx', 'score', 'acc_1fish', 'acc_2fish', 'mae',
    'pass1_new_frag_cost', 'max_track_gap_s', 'location_weight',
    'ipi_weight', 'waveform_weight', 'location_tolerance',
    'ipi_tolerance_fraction', 'pass2_cost_threshold', 'pass2_max_gap_s',
    'width_min_separation_us',
]
print("\nTop 10 combinations:\n")
print(final_df[top_cols].head(10).to_string(index=False))
