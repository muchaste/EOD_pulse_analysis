import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import json
import glob
import itertools
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from pulse_functions import load_waveforms, normalize_waveforms

print("=" * 70)
print("TRACKING PARAMETER TUNING")
print("=" * 70)

root = tk.Tk()
root.withdraw()

print("\nSelect ROOT folder to scan for annotated events...")
root_folder = filedialog.askdirectory(title="Select Root Folder (e.g. E:\\)")
if not root_folder:
    raise ValueError("No root folder selected")
print(f"✓ Root folder: {root_folder}")

print("\nSelect OUTPUT folder for tuning results...")
output_folder = filedialog.askdirectory(title="Select Output Folder")
if not output_folder:
    raise ValueError("No output folder selected")
print(f"✓ Output folder: {output_folder}")

# ---------------------------------------------------------------------------
# Scan recursively for annotation files; match event files in same directory
# ---------------------------------------------------------------------------
ann_files = glob.glob(os.path.join(root_folder, "**", "annotations*.json"), recursive=True)
ann_files += glob.glob(os.path.join(root_folder, "**", "annotations*.JSON"), recursive=True)
ann_files = sorted(set(ann_files))

if not ann_files:
    raise FileNotFoundError(f"No annotation JSON files found under {root_folder}")
print(f"\n✓ Found {len(ann_files)} annotation file(s) — scanning for matched events...")

matched = []
seen_keys = set()
n_skip_label = 0
n_skip_files = 0
n_skip_dup = 0

for ann_path in ann_files:
    event_dir = os.path.dirname(ann_path)
    with open(ann_path, "r") as fh:
        ann_data = json.load(fh)
    file_annotations = ann_data.get("annotations", {})
    file_fish_counts = ann_data.get("fish_counts", {})

    for key, label in file_annotations.items():
        if label != "clear_fish" or file_fish_counts.get(key) not in ("1", "2"):
            n_skip_label += 1
            continue
        gt_count = int(file_fish_counts[key])
        eod_file = os.path.join(event_dir, f"{key}_eod_table.csv")
        waveform_base = os.path.join(event_dir, f"{key}_waveforms")
        if not os.path.exists(eod_file) or not os.path.exists(waveform_base + "_concatenated.npz"):
            n_skip_files += 1
            continue
        unique_key = os.path.join(event_dir, key)
        if unique_key in seen_keys:
            n_skip_dup += 1
            continue
        seen_keys.add(unique_key)
        matched.append({
            "event_key": key,
            "event_dir": event_dir,
            "gt_fish_count": gt_count,
            "eod_file": eod_file,
            "waveform_base": waveform_base,
        })

if not matched:
    raise ValueError("No annotated events with matching data files found under root folder")

n_single = sum(m["gt_fish_count"] == 1 for m in matched)
n_two = sum(m["gt_fish_count"] == 2 for m in matched)
print(f"✓ Matched {len(matched)} annotated events "
      f"({n_single} single-fish, {n_two} two-fish)")
print(f"  Skipped: {n_skip_label} wrong label/count, "
      f"{n_skip_files} missing data files, {n_skip_dup} duplicates")

# ---------------------------------------------------------------------------
# Fixed tracking parameters (not tuned)
# ---------------------------------------------------------------------------
waveform_target_length = 150
min_ipi_s = 0.002
max_track_gap_s = 2.0
max_location_jump_per_s = 200.0
location_tolerance = 20.0
ipi_tolerance_min_s = 0.05
n_recent_for_ipi = 8
shape_dbscan_min_samples = 5
dbscan_max_direct = 3000
dbscan_sample_size = 2000
pass2_max_gap_s = 2.0
pass2_waveform_weight = 0.8
pass2_spatial_weight = 0.2
pass2_max_iterations = 3
pass2_max_frags = 600
min_track_pulses = 15
min_track_duration_s = 0.5
width_min_separation_us = 15
debug_pass1 = False
min_shape_eps = 0.1

# ---------------------------------------------------------------------------
# Parameter grid (tuned parameters)
# ---------------------------------------------------------------------------
param_grid = {
    "knn_percentile":         [70, 85],
    "ipi_tolerance_fraction": [0.2, 0.4, 0.6],
    "pass1_cost_threshold":   [5.0, 10.0, 20.0],
    "location_weight":        [0.1, 0.3],
    "waveform_weight":        [0.1, 0.3, 0.5],
    "pass2_cost_threshold":   [2.0, 4.0, 8.0],
}

param_keys = list(param_grid.keys())
param_combos = list(itertools.product(*[param_grid[k] for k in param_keys]))
print(f"\n✓ Parameter combinations: {len(param_combos)}")

# ---------------------------------------------------------------------------
# Pre-load and normalize all matched events once (expensive; do it once)
# ---------------------------------------------------------------------------
print("\nPre-loading event data...")
event_cache = {}
for m in matched:
    key = m["event_key"]
    eod_data = pd.read_csv(m["eod_file"])
    eod_data["timestamp"] = pd.to_datetime(eod_data["timestamp"])
    eod_data = eod_data.sort_values("timestamp")
    original_indices = eod_data.index.tolist()
    eod_data.reset_index(drop=True, inplace=True)

    waveforms_raw = load_waveforms(m["waveform_base"], format="npz", length="variable")
    waveforms_raw = [waveforms_raw[i] for i in original_indices]

    if len(waveforms_raw) != len(eod_data):
        print(f"  ⚠ Skipping {key}: waveform/table mismatch")
        continue

    normalized_waveforms = normalize_waveforms(
        waveforms_raw,
        snippet_p1_idc=eod_data["snippet_p1_idx"].values,
        snippet_p2_idc=eod_data["snippet_p2_idx"].values,
        method="p1_unity",
        crop_and_interpolate=True,
        crop_factor=4,
        target_length=waveform_target_length,
    )
    normalized_waveforms = np.array(normalized_waveforms)
    norms = np.linalg.norm(normalized_waveforms, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    waveforms_l2 = normalized_waveforms / norms
    del normalized_waveforms, norms

    event_cache[key] = {
        "eod_data": eod_data,
        "waveforms_l2": waveforms_l2,
        "gt_fish_count": m["gt_fish_count"],
    }
    print(f"  ✓ {key}: {len(eod_data)} pulses loaded")

print(f"\n✓ {len(event_cache)} events ready for tuning")

# Amplitude distribution: derive location-quality thresholds from data
all_amps = np.concatenate([
    cache["eod_data"]["eod_amplitude"].values for cache in event_cache.values()
])
amp_location_low = float(np.percentile(all_amps, 15))
amp_location_high = float(np.percentile(all_amps, 75))
print(f"✓ Amplitude thresholds: low={amp_location_low:.4f} V (p15), "
      f"high={amp_location_high:.4f} V (p75)")

fig_amp, ax_amp = plt.subplots(figsize=(8, 4))
ax_amp.hist(all_amps, bins=100, color="steelblue", alpha=0.7)
ax_amp.axvline(amp_location_low, color="orange", linestyle="--",
               label=f"p15 = {amp_location_low:.4f} V  (loc_quality=0)")
ax_amp.axvline(amp_location_high, color="green", linestyle="--",
               label=f"p75 = {amp_location_high:.4f} V  (loc_quality=1)")
ax_amp.set_xlabel("EOD amplitude (V)")
ax_amp.set_ylabel("Pulse count")
ax_amp.set_title(
    "Amplitude distribution — annotated events\n"
    "Edit percentiles (15, 75) in script above if thresholds look wrong"
)
ax_amp.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "amplitude_distribution.png"), dpi=120)
plt.close()
print("✓ Saved amplitude distribution diagnostic")

# ---------------------------------------------------------------------------
# Grid search loop
# ---------------------------------------------------------------------------
results = []
n_total = len(event_cache) * len(param_combos)
run_idx = 0

for combo in param_combos:
    params = dict(zip(param_keys, combo))
    knn_percentile = int(params["knn_percentile"])
    ipi_tolerance_fraction = params["ipi_tolerance_fraction"]
    pass1_cost_threshold = params["pass1_cost_threshold"]
    location_weight = params["location_weight"]
    waveform_weight = params["waveform_weight"]
    ipi_weight = max(0.0, 1.0 - location_weight - waveform_weight)
    pass2_cost_threshold = params["pass2_cost_threshold"]

    for event_key, cache in event_cache.items():
        run_idx += 1
        if run_idx % 500 == 0:
            print(f"  Progress: {run_idx}/{n_total}")

        eod_data = cache["eod_data"].copy()
        waveforms_l2 = cache["waveforms_l2"]
        gt_fish_count = cache["gt_fish_count"]
        widths = eod_data["eod_width_us"].values

        # --- Width-based pre-sorting ---
        width_range = np.linspace(widths.min(), widths.max(), 1000)
        kde = gaussian_kde(widths, bw_method=0.1)
        kde_vals = kde(width_range)
        min_peak_distance_bins = int(width_min_separation_us / (width_range[1] - width_range[0]))
        peaks_idx, _ = find_peaks(kde_vals, distance=max(1, min_peak_distance_bins),
                                   prominence=0.05 * kde_vals.max())
        if len(peaks_idx) > 1:
            peak_positions = width_range[peaks_idx]
            pulse_width_class = np.argmin(np.abs(widths[:, None] - peak_positions[None, :]), axis=1)
            n_width_classes = len(peak_positions)
        else:
            pulse_width_class = np.zeros(len(eod_data), dtype=int)
            n_width_classes = 1
        eod_data["width_class"] = pulse_width_class

        # --- Shape clustering: adaptive KNN eps, noise→centroid reassignment ---
        eod_data["shape_class"] = -1
        next_shape_id = 0
        for wc in range(n_width_classes):
            wc_mask = pulse_width_class == wc
            wc_indices = np.where(wc_mask)[0]
            wc_waveforms = waveforms_l2[wc_indices]
            n_wc = len(wc_indices)
            if n_wc > dbscan_max_direct:
                rng = np.random.default_rng(seed=42)
                sample_pos = rng.choice(n_wc, size=min(dbscan_sample_size, n_wc), replace=False)
            else:
                sample_pos = np.arange(n_wc)
            sample_wf = wc_waveforms[sample_pos]
            n_sample = len(sample_pos)
            min_pts = max(shape_dbscan_min_samples, int(n_sample * 0.01))
            n_pca = min(5, n_sample - 1, waveform_target_length)
            if n_sample < 2 or n_pca < 1:
                db_labels = np.zeros(n_wc, dtype=int)
                n_clusters = 1
            else:
                pca_s = PCA(n_components=n_pca)
                feat_sample = pca_s.fit_transform(sample_wf)
                knn_dists = np.sort(pairwise_distances(feat_sample), axis=1)
                knn_col = min(min_pts, knn_dists.shape[1] - 1)
                adaptive_eps = max(
                    float(np.percentile(knn_dists[:, knn_col], knn_percentile)),
                    min_shape_eps
                )
                db = DBSCAN(eps=adaptive_eps, min_samples=min_pts, metric="euclidean")
                sample_labels = db.fit_predict(feat_sample)
                cluster_ids = np.unique(sample_labels[sample_labels >= 0])
                n_clusters = len(cluster_ids)
                if n_clusters == 0:
                    cluster_ids = np.array([0], dtype=int)
                    n_clusters = 1
                    centroids = feat_sample.mean(axis=0, keepdims=True)
                    sample_labels = np.zeros(n_sample, dtype=int)
                else:
                    centroids = np.array(
                        [feat_sample[sample_labels == cid].mean(axis=0) for cid in cluster_ids]
                    )
                if n_wc > dbscan_max_direct:
                    db_labels = np.full(n_wc, -1, dtype=int)
                    for sp, sl in zip(sample_pos, sample_labels):
                        db_labels[sp] = sl
                    unassigned = np.where(db_labels == -1)[0]
                    if unassigned.size > 0:
                        feat_un = pca_s.transform(wc_waveforms[unassigned])
                        dists = np.linalg.norm(
                            feat_un[:, None, :] - centroids[None, :, :], axis=2
                        )
                        db_labels[unassigned] = cluster_ids[np.argmin(dists, axis=1)]
                else:
                    db_labels = sample_labels.copy()
                    noise_mask = db_labels == -1
                    if noise_mask.any():
                        feat_noise = pca_s.transform(wc_waveforms[noise_mask])
                        dists_noise = np.linalg.norm(
                            feat_noise[:, None, :] - centroids[None, :, :], axis=2
                        )
                        db_labels[noise_mask] = cluster_ids[np.argmin(dists_noise, axis=1)]
            for i, pulse_idx in enumerate(wc_indices):
                eod_data.loc[pulse_idx, "shape_class"] = next_shape_id + db_labels[i]
            next_shape_id += n_clusters

        shape_groups = (
            eod_data[["width_class", "shape_class"]]
            .drop_duplicates()
            .sort_values(["width_class", "shape_class"])
            .values.tolist()
        )

        # --- Pass 1: sequential assignment ---
        eod_data["fragment_id"] = -1
        next_fragment_id = 0
        fragments = {}

        for wc, sc in shape_groups:
            group_mask = (eod_data["width_class"] == wc) & (eod_data["shape_class"] == sc)
            group_indices = np.where(group_mask)[0]
            for pulse_idx in group_indices:
                pulse_ts = eod_data.loc[pulse_idx, "timestamp"]
                pulse_loc = eod_data.loc[pulse_idx, "pulse_location"]
                pulse_amp = eod_data.loc[pulse_idx, "eod_amplitude"]
                loc_quality = np.clip(
                    (pulse_amp - amp_location_low) / (amp_location_high - amp_location_low + 1e-12),
                    0.0, 1.0
                )
                candidate_ids = []
                for fid, f in fragments.items():
                    if f["shape_class"] != sc or f["width_class"] != wc:
                        continue
                    dt = (pulse_ts - f["last_timestamp"]).total_seconds()
                    if dt > max_track_gap_s or dt < min_ipi_s:
                        continue
                    if abs(pulse_loc - f["last_location"]) > max_location_jump_per_s * max(dt, 0.001):
                        continue
                    candidate_ids.append(fid)

                if not candidate_ids:
                    fragments[next_fragment_id] = {
                        "history": [pulse_idx], "ipi_history": [],
                        "last_timestamp": pulse_ts, "last_location": pulse_loc,
                        "width_class": wc, "shape_class": sc,
                    }
                    eod_data.loc[pulse_idx, "fragment_id"] = next_fragment_id
                    next_fragment_id += 1
                    continue

                if len(candidate_ids) == 1:
                    best_fid = candidate_ids[0]
                    f = fragments[best_fid]
                    dt = (pulse_ts - f["last_timestamp"]).total_seconds()
                    f["ipi_history"].append(dt)
                    f["history"].append(pulse_idx)
                    f["last_timestamp"] = pulse_ts
                    f["last_location"] = pulse_loc
                    eod_data.loc[pulse_idx, "fragment_id"] = best_fid
                    continue

                best_fid = None
                best_cost = np.inf
                for fid in candidate_ids:
                    f = fragments[fid]
                    dt = (pulse_ts - f["last_timestamp"]).total_seconds()
                    if len(f["history"]) >= 2:
                        prev_idx = f["history"][-2]
                        dt_prev = (f["last_timestamp"] - eod_data.loc[prev_idx, "timestamp"]).total_seconds()
                        velocity = (f["last_location"] - eod_data.loc[prev_idx, "pulse_location"]) / max(dt_prev, 1e-6)
                        predicted_loc = f["last_location"] + velocity * dt
                    else:
                        predicted_loc = f["last_location"]
                    loc_cost = abs(pulse_loc - predicted_loc) / location_tolerance
                    if len(f["ipi_history"]) >= 1:
                        recent_ipis = f["ipi_history"][-n_recent_for_ipi:]
                        median_ipi = np.median(recent_ipis)
                        ipi_tol = max(median_ipi * ipi_tolerance_fraction, ipi_tolerance_min_s)
                        ipi_cost = abs(dt - median_ipi) / ipi_tol
                    else:
                        ipi_cost = 0.0
                    recent_wf = waveforms_l2[f["history"][-10:]]
                    frag_median_wf = np.median(recent_wf, axis=0)
                    wf_cost = float(np.linalg.norm(waveforms_l2[pulse_idx] - frag_median_wf))
                    cost = (location_weight * loc_quality * loc_cost
                            + ipi_weight * ipi_cost
                            + waveform_weight * wf_cost)
                    if cost < best_cost:
                        best_cost = cost
                        best_fid = fid

                if best_cost < pass1_cost_threshold:
                    f = fragments[best_fid]
                    dt = (pulse_ts - f["last_timestamp"]).total_seconds()
                    f["ipi_history"].append(dt)
                    f["history"].append(pulse_idx)
                    f["last_timestamp"] = pulse_ts
                    f["last_location"] = pulse_loc
                    eod_data.loc[pulse_idx, "fragment_id"] = best_fid
                else:
                    fragments[next_fragment_id] = {
                        "history": [pulse_idx], "ipi_history": [],
                        "last_timestamp": pulse_ts, "last_location": pulse_loc,
                        "width_class": wc, "shape_class": sc,
                    }
                    eod_data.loc[pulse_idx, "fragment_id"] = next_fragment_id
                    next_fragment_id += 1

        # --- Pass 2: fragment stitching ---
        for stitch_iter in range(pass2_max_iterations):
            frag_ids = list(fragments.keys())
            n_frags = len(frag_ids)
            if n_frags < 2:
                break
            if n_frags > pass2_max_frags:
                break

            frag_start_ts = {fid: eod_data.loc[f["history"][0], "timestamp"] for fid, f in fragments.items()}
            frag_end_ts = {fid: eod_data.loc[f["history"][-1], "timestamp"] for fid, f in fragments.items()}
            frag_start_loc = {fid: eod_data.loc[f["history"][0], "pulse_location"] for fid, f in fragments.items()}
            frag_end_loc = {fid: eod_data.loc[f["history"][-1], "pulse_location"] for fid, f in fragments.items()}
            frag_start_wf = {fid: np.median(waveforms_l2[f["history"][:5]], axis=0) for fid, f in fragments.items()}
            frag_end_wf = {fid: np.median(waveforms_l2[f["history"][-5:]], axis=0) for fid, f in fragments.items()}

            INF = 1e6
            cost_matrix = np.full((n_frags, n_frags), INF)
            for i, fid_end in enumerate(frag_ids):
                for j, fid_start in enumerate(frag_ids):
                    if fid_end == fid_start:
                        continue
                    gap = (frag_start_ts[fid_start] - frag_end_ts[fid_end]).total_seconds()
                    if gap <= 0 or gap > pass2_max_gap_s:
                        continue
                    if fragments[fid_end]["width_class"] != fragments[fid_start]["width_class"]:
                        continue
                    loc_diff = abs(frag_start_loc[fid_start] - frag_end_loc[fid_end])
                    if loc_diff > 3.0 * gap:
                        continue
                    wf_cost = np.linalg.norm(frag_end_wf[fid_end] - frag_start_wf[fid_start])
                    spatial_cost = loc_diff / max(location_tolerance, 0.01)
                    cost_matrix[i, j] = pass2_waveform_weight * wf_cost + pass2_spatial_weight * spatial_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            merges = [(frag_ids[r], frag_ids[c]) for r, c in zip(row_ind, col_ind)
                      if cost_matrix[r, c] < pass2_cost_threshold]
            if not merges:
                break

            consumed = set()
            for fid_end, fid_start in merges:
                if fid_end in consumed or fid_start in consumed:
                    continue
                f_end = fragments[fid_end]
                f_start = fragments[fid_start]
                gap = (frag_start_ts[fid_start] - frag_end_ts[fid_end]).total_seconds()
                f_end["history"].extend(f_start["history"])
                f_end["ipi_history"].extend(f_start["ipi_history"])
                f_end["ipi_history"].append(gap)
                f_end["last_timestamp"] = f_start["last_timestamp"]
                f_end["last_location"] = f_start["last_location"]
                for pidx in f_start["history"]:
                    eod_data.loc[pidx, "fragment_id"] = fid_end
                del fragments[fid_start]
                consumed.add(fid_start)
                consumed.add(fid_end)

        # --- Pruning ---
        fids_to_remove = []
        for fid, f in fragments.items():
            n_pulses = len(f["history"])
            duration = (eod_data.loc[f["history"][-1], "timestamp"] -
                        eod_data.loc[f["history"][0], "timestamp"]).total_seconds()
            if n_pulses < min_track_pulses or duration < min_track_duration_s:
                fids_to_remove.append(fid)
        for fid in fids_to_remove:
            del fragments[fid]

        n_fish_tracked = len(fragments)
        assigned = sum(len(f["history"]) for f in fragments.values())
        assignment_rate = assigned / len(eod_data) if len(eod_data) > 0 else 0.0
        fragmentation_ratio = n_fish_tracked / gt_fish_count if gt_fish_count > 0 else np.nan
        correct = int(n_fish_tracked == gt_fish_count)
        over_fragmented = int(n_fish_tracked > gt_fish_count)
        merged = int(n_fish_tracked < gt_fish_count)
        composite_score = correct - 0.5 * abs(fragmentation_ratio - 1.0) if not np.isnan(fragmentation_ratio) else np.nan

        row = {
            "event_key": event_key,
            "gt_fish_count": gt_fish_count,
            "n_fish_tracked": n_fish_tracked,
            "correct": correct,
            "over_fragmented": over_fragmented,
            "merged": merged,
            "fragmentation_ratio": fragmentation_ratio,
            "composite_score": composite_score,
            "assignment_rate": assignment_rate,
        }
        row.update(params)
        results.append(row)

print(f"\n✓ Grid search complete: {len(results)} runs")

results_df = pd.DataFrame(results)
results_path = os.path.join(output_folder, "parameter_tuning_results.csv")
results_df.to_csv(results_path, index=False)
print(f"✓ Saved full results: {os.path.basename(results_path)}")

# ---------------------------------------------------------------------------
# Summary: aggregate by parameter combination
# ---------------------------------------------------------------------------
summary = results_df.groupby(param_keys)[["correct", "over_fragmented", "merged",
                                           "fragmentation_ratio", "composite_score",
                                           "assignment_rate"]].mean().reset_index()
for gt_count in [1, 2]:
    sub = results_df[results_df["gt_fish_count"] == gt_count].groupby(param_keys)["correct"].mean().reset_index()
    sub = sub.rename(columns={"correct": f"correct_gt{gt_count}"})
    summary = summary.merge(sub, on=param_keys, how="left")
    sub_cs = results_df[results_df["gt_fish_count"] == gt_count].groupby(param_keys)["composite_score"].mean().reset_index()
    sub_cs = sub_cs.rename(columns={"composite_score": f"composite_gt{gt_count}"})
    summary = summary.merge(sub_cs, on=param_keys, how="left")

summary = summary.sort_values("composite_score", ascending=False)
summary_path = os.path.join(output_folder, "parameter_tuning_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"✓ Saved summary: {os.path.basename(summary_path)}")

print("\n--- Top 10 parameter sets by composite score (all events) ---")
print(summary.head(10).to_string(index=False))

print("\n--- Top 10 by composite score on 2-fish events ---")
if "composite_gt2" in summary.columns:
    print(summary.sort_values("composite_gt2", ascending=False).head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# Heatmaps: 2D slices through parameter space (fixing remaining params at median)
# Metric shown: mean correct across all events
# ---------------------------------------------------------------------------
# 6 pairs from 5 parameters: use middle value of each non-plotted parameter
param_middle = {k: sorted(param_grid[k])[len(param_grid[k]) // 2] for k in param_keys}
pair_list = list(itertools.combinations(param_keys, 2))  # 10 pairs total, show first 9 in 3x3

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle("Parameter tuning: composite score (all events)\n"
             "(each panel: other params fixed at middle value)", fontsize=10)

for ax_idx, (pk1, pk2) in enumerate(pair_list[:9]):
    ax = axes[ax_idx // 3][ax_idx % 3]
    # Filter rows where all other params are at their middle value
    mask = pd.Series([True] * len(summary))
    for pk in param_keys:
        if pk in (pk1, pk2):
            continue
        mask = mask & (summary[pk] == param_middle[pk])
    sub = summary[mask.values]
    if sub.empty:
        ax.set_visible(False)
        continue
    pivot = sub.pivot_table(index=pk1, columns=pk2, values="composite_score", aggfunc="mean")
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2g}" for v in pivot.columns], fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2g}" for v in pivot.index], fontsize=7)
    ax.set_xlabel(pk2, fontsize=8)
    ax.set_ylabel(pk1, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6)

plt.tight_layout()
heatmap_path = os.path.join(output_folder, "parameter_tuning_heatmaps.png")
plt.savefig(heatmap_path, dpi=120)
plt.close()
print(f"\n✓ Saved heatmaps: {os.path.basename(heatmap_path)}")

# Separate heatmap for 2-fish events only (most diagnostic for multi-fish tracking)
if "composite_gt2" in summary.columns and summary["composite_gt2"].notna().any():
    fig2, axes2 = plt.subplots(3, 3, figsize=(14, 12))
    fig2.suptitle("Parameter tuning: composite score on 2-FISH events\n"
                  "(each panel: other params fixed at middle value)", fontsize=10)
    for ax_idx, (pk1, pk2) in enumerate(pair_list[:9]):
        ax = axes2[ax_idx // 3][ax_idx % 3]
        mask = pd.Series([True] * len(summary))
        for pk in param_keys:
            if pk in (pk1, pk2):
                continue
            mask = mask & (summary[pk] == param_middle[pk])
        sub = summary[mask.values]
        if sub.empty or sub["composite_gt2"].isna().all():
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(index=pk1, columns=pk2, values="composite_gt2", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2g}" for v in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2g}" for v in pivot.index], fontsize=7)
        ax.set_xlabel(pk2, fontsize=8)
        ax.set_ylabel(pk1, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6)
    plt.tight_layout()
    heatmap2_path = os.path.join(output_folder, "parameter_tuning_heatmaps_2fish.png")
    plt.savefig(heatmap2_path, dpi=120)
    plt.close()
    print(f"✓ Saved 2-fish heatmaps: {os.path.basename(heatmap2_path)}")

print("\n" + "=" * 70)
print("TUNING COMPLETE")
print("=" * 70)
