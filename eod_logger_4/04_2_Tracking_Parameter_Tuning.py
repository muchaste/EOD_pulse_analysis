# -*- coding: utf-8 -*-
"""
04_2_Tracking_Parameter_Tuning.py

This script performs tracking parameter tuning for fish tracking on EOD events extracted from field recordings. 
It evaluates different parameter combinations to optimize tracking performance (parameter grid search).

Authors: Stefan Mucha with Claude Sonnet 4.6

"""


import glob
import json
import random
import itertools
import multiprocessing
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
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulse_functions import load_waveforms, normalize_waveforms

# ============================================================
# CONFIGURATION — edit before running
# ============================================================

N_WORKERS        = 32     # parallel worker processes
N_RANDOM_SAMPLES = 10000   # max combos to evaluate; 0 = exhaustive
RANDOM_SEED      = 42
BATCH_SIZE       = 320    # combos per incremental save

# When set to a proposals_iter*.csv path from 05_5, skips grid/random generation
# and evaluates exactly those combos. Set to '' to use normal grid/random mode.
# PROPOSALS_CSV    = ''     # e.g. r'C:\...\proposals_iter01.csv'
PROPOSALS_CSV    = r'E:\track_tuning\output_combined_greedy\proposals_iter01.csv'     # e.g. r'C:\...\proposals_iter01.csv'

# Optional pass1 override for A/B testing: '', 'greedy', or 'lap'.
PASS1_MODE_OVERRIDE = ''

# Fixed parameters (not tuned)
_DBSCAN_MIN_SAMPLES = 5
_DBSCAN_MAX_DIRECT  = 30000
_DBSCAN_SAMPLE_SIZE = 20000
_N_RECENT_IPI       = 8
_MIN_TRACK_PULSES   = 15
_MIN_TRACK_DUR_S    = 0.5
_PASS2_MAX_FRAGS    = 1200

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

# Valid (location_weight, ipi_weight, waveform_weight) triplets summing to 1.0
# from candidate values [0.2, 0.4, 0.6]
_w = [0.2, 0.4, 0.6]
WEIGHT_TRIPLETS = [
    (lw, iw, ww) for lw in _w for iw in _w for ww in _w
    if abs(lw + iw + ww - 1.0) < 1e-9
]

WEIGHT_PAIRS = [(ww, 1.0 - ww) for ww in [0.2, 0.4, 0.5, 0.6, 0.8]]

# ============================================================
# WORKER GLOBALS — set once per worker process via initializer
# ============================================================

_worker_events_data = None


def _worker_init(events_data):
    global _worker_events_data
    _worker_events_data = events_data


def _pass1_greedy(eod_data, wf_l2, shape_groups, p):
    fragments = {}
    next_fid  = 0
    loc_tol   = p['location_tolerance']
    ipi_tf    = p['ipi_tolerance_fraction']
    ipi_tmin  = p['ipi_tolerance_min_s']
    lw        = p['location_weight']
    iw        = p['ipi_weight']
    ww        = p['waveform_weight']
    max_gap   = p['max_track_gap_s']
    min_ipi   = p['min_ipi_s']
    max_jps   = p['max_location_jump_per_s']
    new_cost  = p['pass1_new_frag_cost']

    for wc, sc in shape_groups:
        grp_mask    = (eod_data['width_class'] == wc) & (eod_data['shape_class'] == sc)
        grp_indices = np.where(grp_mask)[0]

        for pulse_idx in grp_indices:
            pulse_ts  = eod_data.at[pulse_idx, 'timestamp']
            pulse_loc = eod_data.at[pulse_idx, 'pulse_location']

            candidate_ids = []
            for fid, f in fragments.items():
                if f['shape_class'] != sc or f['width_class'] != wc:
                    continue
                dt = (pulse_ts - f['last_timestamp']).total_seconds()
                if dt > max_gap or dt < min_ipi:
                    continue
                if abs(pulse_loc - f['last_location']) > max_jps * max(dt, 0.001):
                    continue
                candidate_ids.append(fid)

            if not candidate_ids:
                fragments[next_fid] = {
                    'history': [pulse_idx], 'ipi_history': [],
                    'last_timestamp': pulse_ts, 'last_location': pulse_loc,
                    'width_class': wc, 'shape_class': sc,
                    'waveform_signature': wf_l2[pulse_idx].copy(),
                }
                eod_data.at[pulse_idx, 'fragment_id'] = next_fid
                next_fid += 1
                continue

            best_fid  = None
            best_cost = np.inf
            for fid in candidate_ids:
                f  = fragments[fid]
                dt = (pulse_ts - f['last_timestamp']).total_seconds()
                if len(f['history']) >= 2:
                    prev_idx = f['history'][-2]
                    dt_prev  = (f['last_timestamp'] -
                                eod_data.at[prev_idx, 'timestamp']).total_seconds()
                    vel      = ((f['last_location'] - eod_data.at[prev_idx, 'pulse_location'])
                                / max(dt_prev, 1e-6))
                    pred_loc = f['last_location'] + vel * dt
                else:
                    pred_loc = f['last_location']
                loc_cost = abs(pulse_loc - pred_loc) / loc_tol
                if f['ipi_history']:
                    med_ipi  = np.median(f['ipi_history'][-_N_RECENT_IPI:])
                    ipi_tol  = max(med_ipi * ipi_tf, ipi_tmin)
                    ipi_cost = abs(dt - med_ipi) / ipi_tol
                else:
                    ipi_cost = 0.0
                wf_cost = float(np.linalg.norm(wf_l2[pulse_idx] - f['waveform_signature']))
                cost    = lw * loc_cost + iw * ipi_cost + ww * wf_cost
                if cost < best_cost:
                    best_cost = cost
                    best_fid  = fid

            if best_cost < new_cost:
                f = fragments[best_fid]
                dt = (pulse_ts - f['last_timestamp']).total_seconds()
                f['ipi_history'].append(dt)
                f['history'].append(pulse_idx)
                f['last_timestamp'] = pulse_ts
                f['last_location']  = pulse_loc
                f['waveform_signature'] = np.median(wf_l2[f['history'][-10:]], axis=0)
                eod_data.at[pulse_idx, 'fragment_id'] = best_fid
            else:
                fragments[next_fid] = {
                    'history': [pulse_idx], 'ipi_history': [],
                    'last_timestamp': pulse_ts, 'last_location': pulse_loc,
                    'width_class': wc, 'shape_class': sc,
                    'waveform_signature': wf_l2[pulse_idx].copy(),
                }
                eod_data.at[pulse_idx, 'fragment_id'] = next_fid
                next_fid += 1

    return fragments


def _pass1_window_lap(eod_data, wf_l2, shape_groups, p):
    fragments = {}
    next_fid  = 0
    loc_tol   = p['location_tolerance']
    ipi_tf    = p['ipi_tolerance_fraction']
    ipi_tmin  = p['ipi_tolerance_min_s']
    lw        = p['location_weight']
    iw        = p['ipi_weight']
    ww        = p['waveform_weight']
    max_gap   = p['max_track_gap_s']
    min_ipi   = p['min_ipi_s']
    max_jps   = p['max_location_jump_per_s']
    new_cost  = p['pass1_new_frag_cost']
    inf_cost  = 1e6

    for wc, sc in shape_groups:
        grp_mask    = (eod_data['width_class'] == wc) & (eod_data['shape_class'] == sc)
        grp_indices = np.where(grp_mask)[0]
        if len(grp_indices) == 0:
            continue

        grp_indices = sorted(grp_indices, key=lambda idx: eod_data.at[idx, 'timestamp'])
        g_pos = 0
        while g_pos < len(grp_indices):
            win_start_ts = eod_data.at[grp_indices[g_pos], 'timestamp']
            win_end_ts   = win_start_ts + pd.to_timedelta(min_ipi, unit='s')

            win_indices = []
            while g_pos < len(grp_indices):
                pidx = grp_indices[g_pos]
                if eod_data.at[pidx, 'timestamp'] < win_end_ts:
                    win_indices.append(pidx)
                    g_pos += 1
                else:
                    break

            active_fids = [
                fid for fid, f in fragments.items()
                if f['shape_class'] == sc and f['width_class'] == wc
            ]

            n_pulses = len(win_indices)
            n_active = len(active_fids)
            if n_pulses == 0:
                continue

            cost = np.full((n_pulses, n_active + n_pulses), inf_cost, dtype=float)

            for r, pulse_idx in enumerate(win_indices):
                pulse_ts  = eod_data.at[pulse_idx, 'timestamp']
                pulse_loc = eod_data.at[pulse_idx, 'pulse_location']

                # Per-pulse new-fragment option with fixed penalty.
                cost[r, n_active + r] = new_cost

                for c, fid in enumerate(active_fids):
                    f  = fragments[fid]
                    dt = (pulse_ts - f['last_timestamp']).total_seconds()
                    if dt > max_gap or dt < min_ipi:
                        continue
                    if abs(pulse_loc - f['last_location']) > max_jps * max(dt, 0.001):
                        continue

                    if len(f['history']) >= 2:
                        prev_idx = f['history'][-2]
                        dt_prev  = (f['last_timestamp'] -
                                    eod_data.at[prev_idx, 'timestamp']).total_seconds()
                        vel      = ((f['last_location'] -
                                     eod_data.at[prev_idx, 'pulse_location'])
                                    / max(dt_prev, 1e-6))
                        pred_loc = f['last_location'] + vel * dt
                    else:
                        pred_loc = f['last_location']

                    loc_cost = abs(pulse_loc - pred_loc) / loc_tol
                    if f['ipi_history']:
                        med_ipi  = np.median(f['ipi_history'][-_N_RECENT_IPI:])
                        ipi_tol  = max(med_ipi * ipi_tf, ipi_tmin)
                        ipi_cost = abs(dt - med_ipi) / ipi_tol
                    else:
                        ipi_cost = 0.0
                    wf_cost = float(np.linalg.norm(wf_l2[pulse_idx] - f['waveform_signature']))
                    cost[r, c] = lw * loc_cost + iw * ipi_cost + ww * wf_cost

            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                pulse_idx = win_indices[r]
                pulse_ts  = eod_data.at[pulse_idx, 'timestamp']
                pulse_loc = eod_data.at[pulse_idx, 'pulse_location']

                if c < n_active and cost[r, c] < inf_cost:
                    fid = active_fids[c]
                    f   = fragments[fid]
                    dt  = (pulse_ts - f['last_timestamp']).total_seconds()
                    f['ipi_history'].append(dt)
                    f['history'].append(pulse_idx)
                    f['last_timestamp'] = pulse_ts
                    f['last_location']  = pulse_loc
                    f['waveform_signature'] = np.median(wf_l2[f['history'][-10:]], axis=0)
                    eod_data.at[pulse_idx, 'fragment_id'] = fid
                else:
                    fragments[next_fid] = {
                        'history': [pulse_idx], 'ipi_history': [],
                        'last_timestamp': pulse_ts, 'last_location': pulse_loc,
                        'width_class': wc, 'shape_class': sc,
                        'waveform_signature': wf_l2[pulse_idx].copy(),
                    }
                    eod_data.at[pulse_idx, 'fragment_id'] = next_fid
                    next_fid += 1

    return fragments


# ============================================================
# TRACKING FUNCTION — one event, one parameter set
# No prints, no plots, no saves. Returns n_fish after pruning.
# ============================================================

def _track_event(ev, p):
    wt = p['waveform_target_length']
    cf = p['crop_factor']
    wf_l2, wf_l2_p2 = ev['pre_normalized'][(wt, cf)]

    eod_data = pd.DataFrame({
        'timestamp':      pd.to_datetime(ev['timestamp']),
        'pulse_location': ev['pulse_locations'],
    })
    eod_data['fragment_id'] = -1
    eod_data['shape_class'] = -1
    eod_data['shape_source'] = ''
    eod_data['width_class']  = 0

    widths   = ev['widths']
    step_us  = ev['step_us']
    n_pulses = len(eod_data)

    # Step 2a: Width-based pre-sorting
    width_range  = np.linspace(widths.min(), widths.max(), 1000)
    width_std    = np.std(widths)
    kde_bw       = max(2.0 * step_us, 1.0) / width_std if width_std > 0 else 0.5
    kde_vals     = gaussian_kde(widths, bw_method=kde_bw)(width_range)
    bin_width_us = width_range[1] - width_range[0]
    min_pk_dist  = int(p['width_min_separation_us'] / bin_width_us) if bin_width_us > 0 else 1
    peaks_idx, _ = find_peaks(kde_vals, distance=max(1, min_pk_dist),
                              prominence=0.01 * kde_vals.max())
    if len(peaks_idx) > 1:
        peak_pos = width_range[peaks_idx]
        pwc      = np.argmin(np.abs(widths[:, None] - peak_pos[None, :]), axis=1)
        n_wc     = len(peak_pos)
    else:
        pwc  = np.zeros(n_pulses, dtype=int)
        n_wc = 1
    eod_data['width_class'] = pwc

    # Step 2b: Shape clustering within each width class
    n_fft_total   = wt // 2 + 1
    n_fft_low     = max(1, n_fft_total // 5)
    next_shape_id = 0

    for wc in range(n_wc):
        wc_mask      = pwc == wc
        wc_indices   = np.where(wc_mask)[0]
        n_wc_count   = len(wc_indices)
        wc_wf_p1     = wf_l2[wc_indices]
        wc_wf_p2     = wf_l2_p2[wc_indices]

        mean_wf  = wc_wf_p1.mean(axis=0)
        mean_wf -= np.mean(mean_wf)
        fft_power = np.abs(np.fft.rfft(mean_wf))
        lfr       = fft_power[:n_fft_low].sum() / (fft_power.sum() + 1e-12)
        if lfr < p['fft_artifact_threshold']:
            for k, pidx in enumerate(wc_indices):
                eod_data.at[pidx, 'shape_class']  = next_shape_id + k
                eod_data.at[pidx, 'shape_source'] = 'artifact'
            next_shape_id += n_wc_count
            continue

        if n_wc_count > _DBSCAN_MAX_DIRECT:
            rng        = np.random.default_rng(42)
            sample_pos = rng.choice(n_wc_count, size=min(_DBSCAN_SAMPLE_SIZE, n_wc_count),
                                    replace=False)
        else:
            sample_pos = np.arange(n_wc_count)
        n_sample = len(sample_pos)

        if n_sample < 2:
            for pidx in wc_indices:
                eod_data.at[pidx, 'shape_class']  = next_shape_id
                eod_data.at[pidx, 'shape_source'] = 'p1'
            next_shape_id += 1
            continue

        n_pca   = min(5, n_sample - 1, wt)
        min_pts = max(_DBSCAN_MIN_SAMPLES, int(n_sample * 0.01))
        knn_col = min(min_pts, n_sample - 1)

        pca_p1  = PCA(n_components=n_pca)
        fp1     = pca_p1.fit_transform(wc_wf_p1[sample_pos])
        knn_p1  = np.sort(pairwise_distances(fp1), axis=1)
        eps_p1  = max(float(np.percentile(knn_p1[:, knn_col], p['knn_percentile'])),
                      p['min_shape_eps'])

        pca_p2  = PCA(n_components=n_pca)
        fp2     = pca_p2.fit_transform(wc_wf_p2[sample_pos])
        knn_p2  = np.sort(pairwise_distances(fp2), axis=1)
        eps_p2  = max(float(np.percentile(knn_p2[:, knn_col], p['knn_percentile'])),
                      p['min_shape_eps'])

        sl_p1   = DBSCAN(eps=eps_p1, min_samples=min_pts).fit_predict(fp1)
        sl_p2   = DBSCAN(eps=eps_p2, min_samples=min_pts).fit_predict(fp2)
        cids_p1 = np.unique(sl_p1[sl_p1 >= 0])
        cids_p2 = np.unique(sl_p2[sl_p2 >= 0])

        if n_wc_count > _DBSCAN_MAX_DIRECT:
            lbl_p1 = np.full(n_wc_count, -1, dtype=int)
            lbl_p2 = np.full(n_wc_count, -1, dtype=int)
            for sp_i, sp_l in zip(sample_pos, sl_p1):
                lbl_p1[sp_i] = sp_l
            for sp_i, sp_l in zip(sample_pos, sl_p2):
                lbl_p2[sp_i] = sp_l
            unassigned = np.ones(n_wc_count, dtype=bool)
            unassigned[sample_pos] = False
            if len(cids_p1) > 0 and unassigned.any():
                cents_p1 = np.array([fp1[sl_p1 == c].mean(axis=0) for c in cids_p1])
                d_p1     = np.linalg.norm(
                    pca_p1.transform(wc_wf_p1[unassigned])[:, None, :] - cents_p1[None, :, :], axis=2)
                lbl_p1[unassigned] = cids_p1[np.argmin(d_p1, axis=1)]
            if len(cids_p2) > 0 and unassigned.any():
                cents_p2 = np.array([fp2[sl_p2 == c].mean(axis=0) for c in cids_p2])
                d_p2     = np.linalg.norm(
                    pca_p2.transform(wc_wf_p2[unassigned])[:, None, :] - cents_p2[None, :, :], axis=2)
                lbl_p2[unassigned] = cids_p2[np.argmin(d_p2, axis=1)]
        else:
            lbl_p1 = sl_p1
            lbl_p2 = sl_p2

        final_labels  = np.full(n_wc_count, -1, dtype=int)
        source_labels = np.full(n_wc_count, '', dtype=object)
        done_p1 = set()
        done_p2 = set()
        sz_p1   = {c: int((lbl_p1 == c).sum()) for c in cids_p1}
        sz_p2   = {c: int((lbl_p2 == c).sum()) for c in cids_p2}
        next_mid = 0

        while True:
            av1 = {c: sz_p1[c] for c in cids_p1 if c not in done_p1}
            av2 = {c: sz_p2[c] for c in cids_p2 if c not in done_p2}
            if not av1 and not av2:
                break
            bc1 = max(av1, key=av1.get) if av1 else None
            bc2 = max(av2, key=av2.get) if av2 else None
            s1  = av1[bc1] if bc1 is not None else 0
            s2  = av2[bc2] if bc2 is not None else 0
            if s1 >= s2:
                chosen = lbl_p1 == bc1
                final_labels[chosen]  = next_mid
                source_labels[chosen] = 'p1'
                for c2 in np.unique(lbl_p2[chosen]):
                    if c2 >= 0:
                        done_p2.add(c2)
                done_p1.add(bc1)
            else:
                chosen = lbl_p2 == bc2
                final_labels[chosen]  = next_mid
                source_labels[chosen] = 'p2'
                for c1 in np.unique(lbl_p1[chosen]):
                    if c1 >= 0:
                        done_p1.add(c1)
                done_p2.add(bc2)
            next_mid += 1

        n_clusters = next_mid
        if n_clusters > 0:
            noise_mask = final_labels == -1
            if noise_mask.any():
                m_cents = np.array([wc_wf_p1[final_labels == c].mean(axis=0)
                                    for c in range(n_clusters)])
                d_noise    = np.linalg.norm(
                    wc_wf_p1[noise_mask][:, None, :] - m_cents[None, :, :], axis=2)
                best_dist  = np.min(d_noise, axis=1)
                best_clust = np.argmin(d_noise, axis=1)
                noise_idxs = np.where(noise_mask)[0]
                n_singletons = 0
                for ni in range(len(noise_idxs)):
                    idx = noise_idxs[ni]
                    if best_dist[ni] < p['min_shape_eps']:
                        final_labels[idx]  = best_clust[ni]
                        source_labels[idx] = 'noise'
                    else:
                        final_labels[idx]  = n_clusters + n_singletons
                        source_labels[idx] = 'artifact'
                        n_singletons      += 1
                n_clusters += n_singletons
        else:
            final_labels[:] = 0
            source_labels[:] = 'noise'
            n_clusters = 1

        for k, pidx in enumerate(wc_indices):
            eod_data.at[pidx, 'shape_class']  = next_shape_id + final_labels[k]
            eod_data.at[pidx, 'shape_source'] = source_labels[k]
        next_shape_id += n_clusters

    shape_groups = (
        eod_data[['width_class', 'shape_class']]
        .drop_duplicates()
        .sort_values(['width_class', 'shape_class'])
        .values.tolist()
    )

    # Step 3: Pass 1 — assignment mode dispatch
    pass1_mode = str(p.get('pass1_mode', 'greedy')).lower()
    if pass1_mode == 'lap':
        fragments = _pass1_window_lap(eod_data, wf_l2, shape_groups, p)
    else:
        fragments = _pass1_greedy(eod_data, wf_l2, shape_groups, p)

    loc_tol = p['location_tolerance']
    min_ipi = p['min_ipi_s']
    max_jps = p['max_location_jump_per_s']

    # Step 4a: Pass 2a — overlap merge
    p2_ovlp_wft = p['pass2_overlap_wf_threshold']
    p2_ovlp_min = p['pass2_overlap_min_s']
    p2_ovlp_itr = p['pass2_overlap_max_iterations']

    for _ in range(p2_ovlp_itr):
        frag_ids   = list(fragments.keys())
        if len(frag_ids) < 2:
            break
        f_start_ts = {fid: eod_data.at[f['history'][0],  'timestamp']
                      for fid, f in fragments.items()}
        f_end_ts   = {fid: eod_data.at[f['history'][-1], 'timestamp']
                      for fid, f in fragments.items()}
        f_med_wf   = {fid: np.median(wf_l2[f['history']], axis=0)
                      for fid, f in fragments.items()}
        n_frags    = len(frag_ids)

        candidates = []
        for i in range(n_frags):
            for j in range(i + 1, n_frags):
                fid_a, fid_b = frag_ids[i], frag_ids[j]
                ovlp_end   = min(f_end_ts[fid_a],   f_end_ts[fid_b])
                ovlp_start = max(f_start_ts[fid_a], f_start_ts[fid_b])
                ovlp_s     = ((ovlp_end - ovlp_start).total_seconds()
                              if ovlp_end > ovlp_start else 0.0)
                if ovlp_s < p2_ovlp_min:
                    continue
                if fragments[fid_a]['width_class'] != fragments[fid_b]['width_class']:
                    continue
                wf_dist = float(np.linalg.norm(f_med_wf[fid_a] - f_med_wf[fid_b]))
                if wf_dist >= p2_ovlp_wft:
                    continue
                merged_ts = np.sort([eod_data.at[pidx, 'timestamp'].timestamp()
                                     for pidx in
                                     fragments[fid_a]['history'] + fragments[fid_b]['history']])
                if len(merged_ts) > 1 and np.min(np.diff(merged_ts)) < min_ipi:
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
            fid_keep   = min(fid_a, fid_b)
            fid_drop   = max(fid_a, fid_b)
            merged_raw = sorted(
                [(eod_data.at[pidx, 'timestamp'], pidx)
                 for pidx in fragments[fid_keep]['history'] + fragments[fid_drop]['history']],
                key=lambda x: x[0])
            s_idxs = [pidx for _, pidx in merged_raw]
            s_ts   = [ts   for ts, _   in merged_raw]
            fk     = fragments[fid_keep]
            fk['history']     = s_idxs
            fk['ipi_history'] = [(s_ts[k] - s_ts[k - 1]).total_seconds()
                                 for k in range(1, len(s_ts))]
            fk['last_timestamp']     = s_ts[-1]
            fk['last_location']      = eod_data.at[s_idxs[-1], 'pulse_location']
            fk['waveform_signature'] = np.median(wf_l2[s_idxs[-10:]], axis=0)
            for pidx in fragments[fid_drop]['history']:
                eod_data.at[pidx, 'fragment_id'] = fid_keep
            del fragments[fid_drop]
            consumed.add(fid_a)
            consumed.add(fid_b)
            n_merged += 1
        if n_merged == 0:
            break

    # Step 4b: Pass 2b — LAP stitching
    p2_max_gap = p['pass2_max_gap_s']
    p2_wfw     = p['pass2_waveform_weight']
    p2_sw      = p['pass2_spatial_weight']
    p2_thresh  = p['pass2_cost_threshold']
    p2_maxiter = p['pass2_max_iterations']
    INF        = 1e6

    for _ in range(p2_maxiter):
        frag_ids = list(fragments.keys())
        n_frags  = len(frag_ids)
        if n_frags < 2 or n_frags > _PASS2_MAX_FRAGS:
            break
        f_start_ts  = {fid: eod_data.at[f['history'][0],  'timestamp']
                       for fid, f in fragments.items()}
        f_end_ts    = {fid: eod_data.at[f['history'][-1], 'timestamp']
                       for fid, f in fragments.items()}
        f_start_loc = {fid: eod_data.at[f['history'][0],  'pulse_location']
                       for fid, f in fragments.items()}
        f_end_loc   = {fid: eod_data.at[f['history'][-1], 'pulse_location']
                       for fid, f in fragments.items()}
        f_start_wf  = {fid: np.median(wf_l2[f['history'][:5]],  axis=0)
                       for fid, f in fragments.items()}
        f_end_wf    = {fid: np.median(wf_l2[f['history'][-5:]], axis=0)
                       for fid, f in fragments.items()}

        cost_matrix = np.full((n_frags, n_frags), INF)
        for i, fid_end in enumerate(frag_ids):
            for j, fid_start in enumerate(frag_ids):
                if fid_end == fid_start:
                    continue
                gap = (f_start_ts[fid_start] - f_end_ts[fid_end]).total_seconds()
                if gap <= 0 or gap > p2_max_gap:
                    continue
                if fragments[fid_end]['width_class'] != fragments[fid_start]['width_class']:
                    continue
                loc_diff = abs(f_start_loc[fid_start] - f_end_loc[fid_end])
                if loc_diff > max_jps * gap:
                    continue
                wf_cost          = float(np.linalg.norm(f_end_wf[fid_end] - f_start_wf[fid_start]))
                spatial_cost     = loc_diff / max(loc_tol, 0.01)
                cost_matrix[i, j] = p2_wfw * wf_cost + p2_sw * spatial_cost

        # Jaqaman (2008, Nat Methods 5:695) augmented cost matrix.
        # Dummy diagonal blocks make "unlinked" a first-class assignment option,
        # so the LAP globally decides whether to link or leave unlinked instead
        # of solving the wrong complete-assignment problem then thresholding.
        aug = np.full((2 * n_frags, 2 * n_frags), INF)
        aug[:n_frags, :n_frags] = cost_matrix
        np.fill_diagonal(aug[:n_frags, n_frags:], p2_thresh)   # cost of unlinked end
        np.fill_diagonal(aug[n_frags:, :n_frags], p2_thresh)   # cost of unlinked start
        aug[n_frags:, n_frags:] = 0.0                          # dummy-to-dummy (free)
        row_ind, col_ind = linear_sum_assignment(aug)
        merges = [
            (frag_ids[r], frag_ids[c])
            for r, c in zip(row_ind, col_ind)
            if r < n_frags and c < n_frags and cost_matrix[r, c] < INF
        ]
        if not merges:
            break

        consumed = set()
        for fid_end, fid_start in merges:
            if fid_end in consumed or fid_start in consumed:
                continue
            f_end    = fragments[fid_end]
            f_start_f = fragments[fid_start]
            gap      = (f_start_ts[fid_start] - f_end_ts[fid_end]).total_seconds()
            f_end['history'].extend(f_start_f['history'])
            f_end['ipi_history'].extend(f_start_f['ipi_history'])
            f_end['ipi_history'].append(gap)
            f_end['last_timestamp']     = f_start_f['last_timestamp']
            f_end['last_location']      = f_start_f['last_location']
            f_end['waveform_signature'] = np.median(wf_l2[f_end['history'][-10:]], axis=0)
            for pidx in f_start_f['history']:
                eod_data.at[pidx, 'fragment_id'] = fid_end
            del fragments[fid_start]
            consumed.add(fid_start)
            consumed.add(fid_end)

    # Step 5: Prune short/low-count fragments
    to_del = [
        fid for fid, f in fragments.items()
        if (len(f['history']) < _MIN_TRACK_PULSES
            or (eod_data.at[f['history'][-1], 'timestamp'] -
                eod_data.at[f['history'][0],  'timestamp']).total_seconds() < _MIN_TRACK_DUR_S)
    ]
    for fid in to_del:
        del fragments[fid]

    return len(fragments)


def _evaluate_combo_worker(args):
    combo_idx, params = args
    rows = []
    for ev in _worker_events_data:
        try:
            n_fish = _track_event(ev, params)
        except Exception:
            n_fish = -1
        rows.append({
            'event_id':         ev['event_id'],
            'n_fish_annotated': ev['n_fish_annotated'],
            'n_fish_tracked':   n_fish,
        })
    return combo_idx, params, rows


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TRACKING PARAMETER TUNING")
    print("=" * 70)

    tk_root = tk.Tk()
    tk_root.withdraw()
    root_folder   = os.environ.get('TUNING_ROOT_FOLDER', '')
    output_folder = os.environ.get('TUNING_OUTPUT_FOLDER', '')
    proposals_env = os.environ.get('TUNING_PROPOSALS_CSV', '')
    pass1_mode_env = os.environ.get('TUNING_PASS1_MODE', '')
    if proposals_env:
        PROPOSALS_CSV = proposals_env
    if pass1_mode_env:
        PASS1_MODE_OVERRIDE = pass1_mode_env
    if not root_folder:
        root_folder   = filedialog.askdirectory(title="Select root folder (e.g. E:\\)")
    if not output_folder:
        output_folder = filedialog.askdirectory(title="Select output folder for tuning results")
    tk_root.destroy()

    if not root_folder or not output_folder:
        raise ValueError("Folder selection cancelled")

    PASS1_MODE_OVERRIDE = str(PASS1_MODE_OVERRIDE).strip().lower()
    if PASS1_MODE_OVERRIDE not in ('', 'greedy', 'lap'):
        raise ValueError("PASS1_MODE_OVERRIDE must be '', 'greedy', or 'lap'")

    os.makedirs(output_folder, exist_ok=True)

    # Load all annotation JSONs recursively
    annotation_files = sorted(set(
        glob.glob(os.path.join(root_folder, "**", "annotations*.json"), recursive=True) +
        glob.glob(os.path.join(root_folder, "**", "annotations*.JSON"), recursive=True)
    ))
    if not annotation_files:
        raise ValueError(f"No annotation files found under {root_folder}")
    print(f"\n✓ Found {len(annotation_files)} annotation file(s)")

    # Parse: keep clear_fish with fish_count 1 or 2
    ground_truth = []
    for af in annotation_files:
        session_dir = os.path.dirname(af)
        with open(af, 'r') as fh:
            ann_data = json.load(fh)
        annotations = ann_data.get('annotations', {})
        fish_counts = ann_data.get('fish_counts', {})
        for key, category in annotations.items():
            if category != 'clear_fish':
                continue
            try:
                event_id = int(key)
            except ValueError:
                try:
                    event_id = int(key.split("event_")[-1])
                except (ValueError, IndexError):
                    continue
            count_str = fish_counts.get(key, fish_counts.get(str(event_id), ''))
            if count_str not in ('1', '2'):
                continue
            ground_truth.append({
                'event_id':         event_id,
                'session_dir':      session_dir,
                'n_fish_annotated': int(count_str),
            })

    print(f"✓ Qualifying events (clear_fish, count 1 or 2): {len(ground_truth)}")
    if not ground_truth:
        raise ValueError("No qualifying events found in annotation files")

    # Pre-load all event data; normalize for each (wt, cf) combination once.
    # If running from proposals, read the proposals CSV early to pick up any
    # (wt, cf) values not in GRID (future-proofing against expanded DISCRETE_ONLY).
    wt_cf_set = {(wt, cf)
                 for wt in GRID['waveform_target_length']
                 for cf in GRID['crop_factor']}
    if PROPOSALS_CSV and os.path.isfile(PROPOSALS_CSV):
        _prop_preview = pd.read_csv(PROPOSALS_CSV,
                                    usecols=['waveform_target_length', 'crop_factor'])
        for _, _r in _prop_preview.iterrows():
            wt_cf_set.add((int(_r['waveform_target_length']), int(_r['crop_factor'])))
    wt_cf_combos = sorted(wt_cf_set)
    events_data = []
    n_skipped   = 0

    print("\nPre-loading event data...")
    for gt in ground_truth:
        session_dir      = gt['session_dir']
        event_id         = gt['event_id']
        n_fish_annotated = gt['n_fish_annotated']

        eod_matches = glob.glob(os.path.join(session_dir, f"*event_{event_id}_eod_table.csv"))
        if not eod_matches:
            n_skipped += 1
            continue
        eod_file  = eod_matches[0]
        base_name = os.path.basename(eod_file).replace("_eod_table.csv", "")
        wf_base   = os.path.join(session_dir, f"{base_name}_waveforms")
        if not os.path.exists(wf_base + "_concatenated.npz"):
            n_skipped += 1
            continue

        ap_file = os.path.join(session_dir, "analysis_parameters.csv")
        if os.path.exists(ap_file):
            ap_df         = pd.read_csv(ap_file)
            interp_factor = (float(ap_df['interp_factor'].iloc[0])
                             if 'interp_factor' in ap_df.columns else 1.0)
        else:
            interp_factor = 1.0
        step_us = 1e6 / (96000.0 * interp_factor)

        eod_df = pd.read_csv(eod_file)
        eod_df['timestamp'] = pd.to_datetime(eod_df['timestamp'])
        eod_df = eod_df.sort_values('timestamp')
        orig_idx = eod_df.index.tolist()
        eod_df.reset_index(drop=True, inplace=True)

        wf_raw = load_waveforms(wf_base, format='npz', length='variable')
        wf_raw = [wf_raw[i] for i in orig_idx]
        if len(wf_raw) != len(eod_df):
            n_skipped += 1
            continue

        wf_detrended = []
        bg_ratio_arr = np.zeros(len(wf_raw))
        for i, wf in enumerate(wf_raw):
            slope = np.linspace(float(wf[0]), float(wf[-1]), len(wf))
            wf_d  = wf - slope
            wf_detrended.append(wf_d)
            peak_amp        = np.max(np.abs(wf_d))
            bg_ratio_arr[i] = (abs(float(wf[0]) - float(wf[-1])) / peak_amp
                               if peak_amp > 0 else 0.0)
        del wf_raw

        pre_normalized = {}
        for wt, cf in wt_cf_combos:
            wf_p1 = np.array(normalize_waveforms(
                wf_detrended,
                snippet_p1_idc=eod_df['snippet_p1_idx'].values,
                snippet_p2_idc=eod_df['snippet_p2_idx'].values,
                method='p1_unity', crop_and_interpolate=True,
                crop_factor=cf, target_length=wt
            ), dtype=np.float32)
            norms = np.linalg.norm(wf_p1, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            wf_p1 /= norms

            wf_p2 = np.array(normalize_waveforms(
                wf_detrended,
                snippet_p1_idc=eod_df['snippet_p2_idx'].values,
                snippet_p2_idc=eod_df['snippet_p1_idx'].values,
                method='p1_unity', crop_and_interpolate=True,
                crop_factor=cf, target_length=wt
            ), dtype=np.float32)
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
            'bg_ratio_arr':     bg_ratio_arr,
            'widths':           eod_df['eod_width_us'].values.astype(np.float64),
            'step_us':          step_us,
            'pre_normalized':   pre_normalized,
        })
        if len(events_data) % 50 == 0:
            print(f"  Loaded {len(events_data)} / {len(ground_truth)} events...")

    print(f"✓ Pre-loaded {len(events_data)} events ({n_skipped} skipped: missing files)")
    n_1fish = sum(1 for ev in events_data if ev['n_fish_annotated'] == 1)
    n_2fish = sum(1 for ev in events_data if ev['n_fish_annotated'] == 2)
    print(f"  1-fish events: {n_1fish},  2-fish events: {n_2fish}")

    # Generate parameter combinations
    if PROPOSALS_CSV and os.path.isfile(PROPOSALS_CSV):
        prop_df      = pd.read_csv(PROPOSALS_CSV)
        param_combos = prop_df.to_dict(orient='records')
        # ensure int params are correct type
        int_params = ['pass2_max_iterations', 'pass2_overlap_max_iterations',
                      'crop_factor', 'waveform_target_length', 'knn_percentile']
        for combo in param_combos:
            for k in int_params:
                if k in combo:
                    combo[k] = int(round(combo[k]))
        print(f"Mode: proposals from {os.path.basename(PROPOSALS_CSV)} ({len(param_combos)} combos)")
    else:
        grid_keys    = list(GRID.keys())
        grid_vals    = [GRID[k] for k in grid_keys]
        total_indep  = 1
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
                        c['location_weight']       = lw
                        c['ipi_weight']            = iw
                        c['waveform_weight']       = ww
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

    # Parallel evaluation with incremental CSV saves
    if PASS1_MODE_OVERRIDE:
        for combo in param_combos:
            combo['pass1_mode'] = PASS1_MODE_OVERRIDE

    # In proposals mode, save to a separate file named after the proposals CSV
    # so 05_5 can glob all parameter_tuning_results*.csv and load every iteration.
    if PROPOSALS_CSV and os.path.isfile(PROPOSALS_CSV):
        prop_stem = os.path.splitext(os.path.basename(PROPOSALS_CSV))[0]  # e.g. proposals_iter01
        iter_tag  = prop_stem.replace('proposals_', '')                    # e.g. iter01
        out_csv   = os.path.join(output_folder, f'parameter_tuning_results_{iter_tag}.csv')
    else:
        out_csv   = os.path.join(output_folder, 'parameter_tuning_results.csv')
    header_written = os.path.exists(out_csv)
    n_done         = 0

    print(f"\nStarting pool ({N_WORKERS} workers) — events_data serialized once per worker")
    print(f"Output: {out_csv}\n")

    with multiprocessing.Pool(N_WORKERS,
                              initializer=_worker_init,
                              initargs=(events_data,)) as pool:
        for batch_start in range(0, len(param_combos), BATCH_SIZE):
            batch_args = [
                (batch_start + i, p)
                for i, p in enumerate(param_combos[batch_start: batch_start + BATCH_SIZE])
            ]
            batch_raw = pool.map(_evaluate_combo_worker, batch_args)

            rows = []
            for combo_idx, params, event_results in batch_raw:
                valid = [r for r in event_results if r['n_fish_tracked'] >= 0]
                res_1 = [r for r in valid if r['n_fish_annotated'] == 1]
                res_2 = [r for r in valid if r['n_fish_annotated'] == 2]
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
                mae   = (float(np.mean([abs(r['n_fish_tracked'] - r['n_fish_annotated'])
                                        for r in valid]))
                         if valid else float('nan'))

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
                    'acc_1fish':  round(acc_1, 4) if not np.isnan(acc_1) else float('nan'),
                    'acc_2fish':  round(acc_2, 4) if not np.isnan(acc_2) else float('nan'),
                    'mae':        round(mae,   4) if not np.isnan(mae)   else float('nan'),
                    'n_exact_1':  n_exact_1, 'n_over_1':   n_over_1,  'n_under_1':  n_under_1,
                    'n_exact_2':  n_exact_2, 'n_over_2':   n_over_2,  'n_under_2':  n_under_2,
                    'n_events_1': n_1,       'n_events_2': n_2,
                    'n_failed':   sum(1 for r in event_results if r['n_fish_tracked'] < 0),
                }
                row.update(params)
                rows.append(row)

            results_df = pd.DataFrame(rows)
            results_df.to_csv(out_csv, mode='a', header=not header_written, index=False)
            header_written = True
            n_done += len(batch_args)
            print(f"  {n_done}/{len(param_combos)} done  |  "
                  f"batch best score: {results_df['score'].max():.4f}")

    print(f"\n✓ Complete. Results saved to: {out_csv}")

    final_df = pd.read_csv(out_csv).sort_values('score', ascending=False)
    top_cols = [
        'combo_idx', 'score', 'acc_1fish', 'acc_2fish', 'mae',
        'pass1_new_frag_cost', 'max_track_gap_s', 'location_weight',
        'ipi_weight', 'waveform_weight', 'location_tolerance',
        'ipi_tolerance_fraction', 'pass2_cost_threshold',
        'pass2_max_gap_s', 'width_min_separation_us',
    ]
    print(f"\nTop 10 combinations:\n")
    print(final_df[top_cols].head(10).to_string(index=False))
