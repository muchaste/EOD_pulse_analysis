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
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut

# Import EOD functions
from pulse_functions import (
    load_waveforms,
    normalize_waveforms,
    create_tracking_plot
)
from parameter_gui import TrackingParameterConfigGUI

print("="*70)
print("SIMPLE FISH TRACKING")
print("="*70)

print("\nStarting Parameter Configuration GUI...")
root = tk.Tk()
config_gui = TrackingParameterConfigGUI(root)
root.mainloop()

if config_gui.result is None:
    print("Configuration cancelled by user")
    exit()

config = config_gui.result
input_folder = config['paths']['input_path']
output_folder = config['paths']['output_path']
control_path = config['paths'].get('control_path', '')
params = config['parameters']

print(f"✓ Input folder: {input_folder}")
print(f"✓ Output folder: {output_folder}")
if params['use_species_matching']:
    print(f"✓ Control folder: {control_path}")

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
# TRACKING PARAMETERS (from GUI)
# =============================================================================

waveform_target_length  = int(params['waveform_target_length'])
crop_factor             = int(params['crop_factor'])
min_ipi_s               = float(params['min_ipi_s'])
max_track_gap_s         = float(params['max_track_gap_s'])
max_location_jump_per_s = float(params['max_location_jump_per_s'])

shape_dbscan_min_samples = int(params['shape_dbscan_min_samples'])
knn_percentile           = int(params['knn_percentile'])
min_shape_eps            = float(params['min_shape_eps'])
fft_artifact_threshold   = float(params['fft_artifact_threshold'])
dbscan_max_direct        = int(params['dbscan_max_direct'])
dbscan_sample_size       = int(params['dbscan_sample_size'])

location_weight       = float(params['location_weight'])
ipi_weight            = float(params['ipi_weight'])
waveform_weight       = float(params['waveform_weight'])
location_tolerance    = float(params['location_tolerance'])
ipi_tolerance_fraction = float(params['ipi_tolerance_fraction'])
ipi_tolerance_min_s   = float(params['ipi_tolerance_min_s'])
n_recent_for_ipi      = int(params['n_recent_for_ipi'])
pass1_new_frag_cost   = float(params['pass1_new_frag_cost'])
debug_pass1           = True

pass2_max_gap_s          = float(params['pass2_max_gap_s'])
pass2_waveform_weight    = float(params['pass2_waveform_weight'])
pass2_spatial_weight     = float(params['pass2_spatial_weight'])
pass2_cost_threshold     = float(params['pass2_cost_threshold'])
pass2_max_iterations     = int(params['pass2_max_iterations'])
pass2_max_frags          = 600
pass2_overlap_wf_threshold  = 0.4
pass2_overlap_min_s         = 0.1
pass2_overlap_max_iterations = 3

min_track_pulses     = int(params['min_track_pulses'])
min_track_duration_s = float(params['min_track_duration_s'])

width_min_separation_us = float(params['width_min_separation_us'])
use_species_matching    = bool(params['use_species_matching'])
lda_min_probability     = float(params.get('lda_min_probability', 0.0))

# Load interp_factor from analysis_parameters.csv for physics-based KDE bandwidth
_ap_file = os.path.join(input_folder, "analysis_parameters.csv")
if os.path.exists(_ap_file):
    _ap_df = pd.read_csv(_ap_file)
    if 'interp_factor' in _ap_df.columns:
        interp_factor = float(_ap_df['interp_factor'].iloc[0])
    else:
        interp_factor = 1.0
        print("\u26a0 'interp_factor' not found in analysis_parameters.csv, defaulting to 1")
else:
    interp_factor = 1.0
    print("\u26a0 analysis_parameters.csv not found, defaulting interp_factor=1")
# Width quantization step at 96 kHz × interp_factor oversampling
step_us = 1e6 / (96000.0 * interp_factor)
print(f"\u2713 Width quantization step: {step_us:.2f} \u00b5s (96 kHz \u00d7 {interp_factor:.0f}\u00d7 interp)")

# =============================================================================
# SPECIES MATCHING: Load control reference library
# =============================================================================
reference_library = {}
lda = None
pca_cls = None
loo_accuracy = None
if use_species_matching:
    print("\n" + "="*70)
    print("LOADING CONTROL REFERENCE LIBRARY")
    print("="*70)
    ctrl_concat_files = glob.glob(os.path.join(control_path, "*_eod_waveforms_concatenated.npz"))
    ctrl_fixed_files  = glob.glob(os.path.join(control_path, "*_eod_waveforms.npz"))
    # Build set of individual ids already covered by concat files to avoid double-loading
    concat_ids = set(os.path.basename(f).replace("_eod_waveforms_concatenated.npz", "")
                     for f in ctrl_concat_files)
    # Merge: variable-length concat files + fixed-length files not already covered
    ctrl_entries = [(ind_id, 'variable') for ind_id in concat_ids]
    for f in ctrl_fixed_files:
        ind_id = os.path.basename(f).replace("_eod_waveforms.npz", "")
        if ind_id not in concat_ids:
            ctrl_entries.append((ind_id, 'fixed'))

    if not ctrl_entries:
        print("\u26a0 No control waveform files found \u2014 species matching disabled")
        use_species_matching = False
    else:
        for ind_id, wf_format in ctrl_entries:
            table_file = os.path.join(control_path, f"{ind_id}_eod_table.csv")
            if not os.path.exists(table_file):
                print(f"  \u26a0 No eod_table for {ind_id}, skipping")
                continue
            ctrl_table = pd.read_csv(table_file)
            if 'snippet_p1_idx' not in ctrl_table.columns or 'snippet_p2_idx' not in ctrl_table.columns:
                print(f"  \u26a0 Missing p1/p2 idx columns in {ind_id}_eod_table.csv, skipping")
                continue
            wf_base = os.path.join(control_path, f"{ind_id}_eod_waveforms")
            if wf_format == 'variable':
                ctrl_wf_list = load_waveforms(wf_base, format='npz', length='variable')
            else:
                ctrl_wf_arr = np.load(wf_base + '.npz')['waveforms']
                ctrl_wf_list = [ctrl_wf_arr[i] for i in range(ctrl_wf_arr.shape[0])]
            if len(ctrl_wf_list) == 0:
                print(f"  \u26a0 Empty waveforms for {ind_id}, skipping")
                continue
            ctrl_p1 = ctrl_table['snippet_p1_idx'].values
            ctrl_p2 = ctrl_table['snippet_p2_idx'].values
            if len(ctrl_wf_list) != len(ctrl_p1):
                print(f"  \u26a0 Waveform/table count mismatch for {ind_id} "
                      f"({len(ctrl_wf_list)} vs {len(ctrl_p1)}), skipping")
                continue
            ctrl_wf_norm = normalize_waveforms(
                ctrl_wf_list, ctrl_p1, ctrl_p2,
                method='p1_unity', crop_and_interpolate=True,
                crop_factor=crop_factor, target_length=waveform_target_length
            )
            ctrl_wf_norm = np.array(ctrl_wf_norm)
            ctrl_norms = np.linalg.norm(ctrl_wf_norm, axis=1, keepdims=True)
            ctrl_norms[ctrl_norms == 0] = 1.0
            ctrl_wf_l2 = ctrl_wf_norm / ctrl_norms
            species_code = ind_id[:2].upper()
            reference_library[ind_id] = {
                'mean_wf':      ctrl_wf_l2.mean(axis=0),
                'species_code': species_code
            }
            print(f"  \u2713 {ind_id} ({species_code}): {len(ctrl_wf_l2)} pulses")

        print(f"\n\u2713 Reference library: {len(reference_library)} individual(s)")
        species_counts = {}
        for info in reference_library.values():
            sp = info['species_code']
            species_counts[sp] = species_counts.get(sp, 0) + 1
        for sp, cnt in sorted(species_counts.items()):
            print(f"  {sp}: {cnt} individual(s)")
        ref_ids   = list(reference_library.keys())
        ref_matrix = np.array([reference_library[rid]['mean_wf'] for rid in ref_ids])  # (N_ref, waveform_target_length)
        ref_species = [reference_library[rid]['species_code'] for rid in ref_ids]
        all_species_codes = sorted(set(ref_species))

        # Fit PCA (dimensionality reduction) + LDA (species discriminant) on control reference
        # mean waveforms once. This stable classifier is reused for every field event.
        ref_species_arr = np.array(ref_species)
        n_pca_cls = max(1, min(len(ref_matrix) - 1, waveform_target_length, 20))
        if len(all_species_codes) >= 2:
            pca_cls = PCA(n_components=n_pca_cls)
            ref_pca_scores = pca_cls.fit_transform(ref_matrix)
            lda = LinearDiscriminantAnalysis()
            lda.fit(ref_pca_scores, ref_species_arr)
            print(f"\u2713 LDA fitted on {len(ref_matrix)} control mean waveforms "
                  f"({len(all_species_codes)} species, {n_pca_cls} PCA components)")
            # Leave-one-individual-out CV (each ref_matrix row = one individual mean waveform)
            sp_counts_loo = {sp: int((ref_species_arr == sp).sum()) for sp in all_species_codes}
            loo_feasible = all(c >= 2 for c in sp_counts_loo.values())
            if loo_feasible:
                loo = LeaveOneOut()
                loo_correct = 0
                loo_total = 0
                loo_sp_correct = {sp: 0 for sp in all_species_codes}
                loo_sp_total   = {sp: 0 for sp in all_species_codes}
                for train_idx, test_idx in loo.split(ref_pca_scores):
                    n_pca_cv = max(1, min(n_pca_cls, len(train_idx) - 1))
                    pca_cv = PCA(n_components=n_pca_cv)
                    scores_train = pca_cv.fit_transform(ref_matrix[train_idx])
                    scores_test = pca_cv.transform(ref_matrix[test_idx])
                    lda_cv = LinearDiscriminantAnalysis()
                    lda_cv.fit(scores_train, ref_species_arr[train_idx])
                    true_sp = ref_species_arr[test_idx][0]
                    pred_sp = lda_cv.predict(scores_test)[0]
                    is_correct = int(pred_sp == true_sp)
                    loo_correct += is_correct
                    loo_total += 1
                    loo_sp_correct[true_sp] += is_correct
                    loo_sp_total[true_sp] += 1
                loo_accuracy = loo_correct / loo_total
                print(f"✓ LOO CV accuracy: {loo_accuracy:.1%} ({loo_correct}/{loo_total})")
                for sp in all_species_codes:
                    sp_acc = loo_sp_correct[sp] / loo_sp_total[sp] if loo_sp_total[sp] > 0 else float('nan')
                    print(f"  {sp}: {sp_acc:.1%} ({loo_sp_correct[sp]}/{loo_sp_total[sp]})")
                loo_per_species = {
                    sp: {
                        'accuracy': loo_sp_correct[sp] / loo_sp_total[sp] if loo_sp_total[sp] > 0 else None,
                        'correct':  loo_sp_correct[sp],
                        'total':    loo_sp_total[sp],
                    }
                    for sp in all_species_codes
                }
            else:
                loo_accuracy = None
                loo_per_species = {}
                print("⚠ LOO CV skipped: at least one species has only 1 individual")

            # Save classifier report so each batch run has a record of the model used
            classifier_report = {
                'waveform_target_length': waveform_target_length,
                'crop_factor': crop_factor,
                'n_pca_components': n_pca_cls,
                'n_ref_individuals': len(ref_matrix),
                'species_n_individuals': {
                    sp: int((ref_species_arr == sp).sum()) for sp in all_species_codes
                },
                'loo_accuracy': loo_accuracy,
                'loo_per_species': loo_per_species,
                'ref_individuals': ref_ids,
            }
            classifier_report_path = os.path.join(output_folder, 'classifier_report.json')
            with open(classifier_report_path, 'w') as _f:
                json.dump(classifier_report, _f, indent=2)
            print(f"✓ Saved classifier report: {os.path.basename(classifier_report_path)}")
        else:
            lda = None
            pca_cls = None
            loo_accuracy = None
            print("⚠ Only 1 species in reference library — LDA disabled, using 1-NN fallback")
            print("\u26a0 Only 1 species in reference library \u2014 LDA disabled, using 1-NN fallback")

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

    # Slope subtraction: remove linear DC drift across each raw snippet window.
    # bg_ratio: boundary slope relative to peak amplitude — per-pulse noise quality metric.
    waveforms_detrended = []
    bg_ratio_arr = np.zeros(len(waveforms_raw))
    for i, wf in enumerate(waveforms_raw):
        slope = np.linspace(float(wf[0]), float(wf[-1]), len(wf))
        wf_d = wf - slope
        waveforms_detrended.append(wf_d)
        peak_amp = np.max(np.abs(wf_d))
        bg_ratio_arr[i] = abs(float(wf[0]) - float(wf[-1])) / peak_amp if peak_amp > 0 else 0.0
    del waveforms_raw
    eod_data['bg_ratio'] = bg_ratio_arr

    # -------------------------------------------------------------------------
    # Normalize waveforms: P1 to +1, baseline subtracted, resampled to fixed length
    # -------------------------------------------------------------------------
    print("\nNormalizing waveforms...")
    normalized_waveforms = normalize_waveforms(
        waveforms_detrended,
        snippet_p1_idc=eod_data['snippet_p1_idx'].values,
        snippet_p2_idc=eod_data['snippet_p2_idx'].values,
        method='p1_unity',
        crop_and_interpolate=True,
        crop_factor=crop_factor,
        target_length=waveform_target_length
    )
    normalized_waveforms = np.array(normalized_waveforms)  # shape: (n_pulses, waveform_target_length)

    # L2-normalize each waveform so Euclidean distance is scale-independent (range [0, 2])
    norms = np.linalg.norm(normalized_waveforms, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    waveforms_l2 = normalized_waveforms / norms
    del normalized_waveforms, norms
    print(f"✓ P1-aligned waveforms, shape: {waveforms_l2.shape}")

    # P2-aligned normalization: center on EOD trough for dual shape clustering.
    normalized_waveforms_p2 = normalize_waveforms(
        waveforms_detrended,
        snippet_p1_idc=eod_data['snippet_p2_idx'].values,
        snippet_p2_idc=eod_data['snippet_p1_idx'].values,
        method='p1_unity',
        crop_and_interpolate=True,
        crop_factor=crop_factor,
        target_length=waveform_target_length
    )
    normalized_waveforms_p2 = np.array(normalized_waveforms_p2)
    norms_p2 = np.linalg.norm(normalized_waveforms_p2, axis=1, keepdims=True)
    norms_p2[norms_p2 == 0] = 1.0
    waveforms_l2_p2 = normalized_waveforms_p2 / norms_p2
    del normalized_waveforms_p2, norms_p2, waveforms_detrended
    print(f"✓ P2-aligned waveforms, shape: {waveforms_l2_p2.shape}")

    # -------------------------------------------------------------------------
    # Step 1: Compute t_sec and widths (used throughout)
    # -------------------------------------------------------------------------
    t_sec = (eod_data['timestamp'] - eod_data['timestamp'].iloc[0]).dt.total_seconds().values
    widths = eod_data['eod_width_us'].values

    # -------------------------------------------------------------------------
    # Step 2a: Width-based pre-sorting into classes
    # -------------------------------------------------------------------------
    print("\nWidth-based pre-sorting...")
    width_range = np.linspace(widths.min(), widths.max(), 1000)
    width_std = np.std(widths)
    # bandwidth = desired_smoothing_µs / std(widths); smooth over ≥2 quantization steps
    kde_bw = max(2.0 * step_us, 1.0) / width_std if width_std > 0 else 0.5
    kde = gaussian_kde(widths, bw_method=kde_bw)
    kde_vals = kde(width_range)

    min_peak_distance_bins = int(width_min_separation_us / (width_range[1] - width_range[0]))
    peaks_idx, _ = find_peaks(kde_vals, distance=max(1, min_peak_distance_bins),
                               prominence=0.01 * kde_vals.max())

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
    # Dual P1+P2 alignment with adaptive DBSCAN epsilon.
    # Phase 3: FFT artifact rejection per width class.
    # Phase 2: Adaptive epsilon from KNN distances in PCA(5) space.
    # Phase 4: Independent DBSCAN on P1 and P2 features; greedy merge.
    # -------------------------------------------------------------------------
    print("\nShape clustering within width classes...")
    eod_data['shape_class'] = -1
    eod_data['shape_source'] = ''
    next_shape_id = 0
    n_fft_total = waveform_target_length // 2 + 1
    n_fft_low = max(1, n_fft_total // 5)  # lowest 20% of frequency bins

    for wc in range(n_width_classes):
        wc_mask = pulse_width_class == wc
        wc_indices = np.where(wc_mask)[0]
        wc_waveforms_p1 = waveforms_l2[wc_indices]
        wc_waveforms_p2 = waveforms_l2_p2[wc_indices]
        wc_bg = bg_ratio_arr[wc_indices]
        n_wc = len(wc_indices)

        # [Phase 3] FFT artifact rejection: skip width class if mean waveform is high-frequency dominated.
        # Real EOD pulses concentrate power in lower frequency bins (1-10 kHz band);
        # electrical artifacts are high-frequency dominated → low_freq_ratio < 0.75.
        mean_wf = wc_waveforms_p1.mean(axis=0)
        mean_wf = mean_wf - np.mean(mean_wf)  # remove DC before FFT (matches pulses.py remove_artefacts)
        fft_power = np.abs(np.fft.rfft(mean_wf))
        low_freq_ratio = fft_power[:n_fft_low].sum() / (fft_power.sum() + 1e-12)
        if low_freq_ratio < fft_artifact_threshold:
            print(f"  Width class {wc}: SKIPPED (FFT artifact, low-freq ratio={low_freq_ratio:.2f})")
            for i, pulse_idx in enumerate(wc_indices):
                eod_data.loc[pulse_idx, 'shape_class'] = next_shape_id + i
                eod_data.loc[pulse_idx, 'shape_source'] = 'artifact'
            next_shape_id += len(wc_indices)
            continue

        # Subsample if width class is too large for pairwise distance computation
        if n_wc > dbscan_max_direct:
            rng = np.random.default_rng(seed=42)
            sample_pos = rng.choice(n_wc, size=min(dbscan_sample_size, n_wc), replace=False)
        else:
            sample_pos = np.arange(n_wc)
        n_sample = len(sample_pos)

        # [Phase 2] Adaptive epsilon: PCA(5) features + KNN 80th percentile.
        # eps is taken directly from the 80th percentile of k-NN distances in PCA space,
        # giving a data-driven scale that adapts to actual waveform spread.
        n_pca_shape = min(5, n_sample - 1, waveform_target_length)
        min_pts = max(shape_dbscan_min_samples, int(n_sample * 0.01))

        pca_p1 = PCA(n_components=n_pca_shape)
        feat_p1_sample = pca_p1.fit_transform(wc_waveforms_p1[sample_pos])
        knn_p1 = np.sort(pairwise_distances(feat_p1_sample), axis=1)
        knn_col = min(min_pts, knn_p1.shape[1] - 1)
        eps_p1 = max(float(np.percentile(knn_p1[:, knn_col], knn_percentile)), min_shape_eps)

        pca_p2 = PCA(n_components=n_pca_shape)
        feat_p2_sample = pca_p2.fit_transform(wc_waveforms_p2[sample_pos])
        knn_p2 = np.sort(pairwise_distances(feat_p2_sample), axis=1)
        eps_p2 = max(float(np.percentile(knn_p2[:, knn_col], knn_percentile)), min_shape_eps)

        # [Phase 4] DBSCAN on P1 and P2 PCA features independently
        db_p1 = DBSCAN(eps=eps_p1, min_samples=min_pts, metric='euclidean')
        sample_labels_p1 = db_p1.fit_predict(feat_p1_sample)
        db_p2 = DBSCAN(eps=eps_p2, min_samples=min_pts, metric='euclidean')
        sample_labels_p2 = db_p2.fit_predict(feat_p2_sample)

        cluster_ids_p1 = np.unique(sample_labels_p1[sample_labels_p1 >= 0])
        cluster_ids_p2 = np.unique(sample_labels_p2[sample_labels_p2 >= 0])

        # Propagate sample labels to full width class via nearest centroid in PCA space
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
                centroids_p1 = np.array([feat_p1_sample[sample_labels_p1 == c].mean(axis=0) for c in cluster_ids_p1])
                feats_unassigned_p1 = pca_p1.transform(wc_waveforms_p1[unassigned])
                d_p1 = np.linalg.norm(feats_unassigned_p1[:, None, :] - centroids_p1[None, :, :], axis=2)
                labels_p1[unassigned] = cluster_ids_p1[np.argmin(d_p1, axis=1)]
            if len(cluster_ids_p2) > 0 and unassigned.any():
                centroids_p2 = np.array([feat_p2_sample[sample_labels_p2 == c].mean(axis=0) for c in cluster_ids_p2])
                feats_unassigned_p2 = pca_p2.transform(wc_waveforms_p2[unassigned])
                d_p2 = np.linalg.norm(feats_unassigned_p2[:, None, :] - centroids_p2[None, :, :], axis=2)
                labels_p2[unassigned] = cluster_ids_p2[np.argmin(d_p2, axis=1)]
        else:
            labels_p1 = sample_labels_p1
            labels_p2 = sample_labels_p2

        # [Phase 4] Greedy merge: largest cluster wins between P1 and P2 sets.
        # Picking a cluster from one set marks all overlapping clusters in the other as consumed.
        final_labels = np.full(n_wc, -1, dtype=int)
        source_labels = np.full(n_wc, '', dtype=object)
        done_p1 = set()
        done_p2 = set()
        size_p1 = {c: int((labels_p1 == c).sum()) for c in cluster_ids_p1}
        size_p2 = {c: int((labels_p2 == c).sum()) for c in cluster_ids_p2}
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
                final_labels[chosen] = next_merged_id
                source_labels[chosen] = 'p1'
                for c2 in np.unique(labels_p2[chosen]):
                    if c2 >= 0:
                        done_p2.add(c2)
                done_p1.add(best_c1)
            else:
                chosen = (labels_p2 == best_c2)
                final_labels[chosen] = next_merged_id
                source_labels[chosen] = 'p2'
                for c1 in np.unique(labels_p1[chosen]):
                    if c1 >= 0:
                        done_p1.add(c1)
                done_p2.add(best_c2)
            next_merged_id += 1

        n_clusters = next_merged_id

        # Assign DBSCAN noise points to nearest cluster centroid (waveform space).
        # Prevents each noise point becoming a singleton shape class that explodes fragment count.
        # shape_source='noise' is preserved as a diagnostic label.
        if n_clusters > 0:
            noise_mask = (final_labels == -1)
            n_noise_forced = int(noise_mask.sum())
            if n_noise_forced > 0:
                merged_centroids = np.array(
                    [wc_waveforms_p1[final_labels == c].mean(axis=0) for c in range(n_clusters)]
                )
                d_noise = np.linalg.norm(
                    wc_waveforms_p1[noise_mask][:, None, :] - merged_centroids[None, :, :], axis=2
                )
                final_labels[noise_mask] = np.argmin(d_noise, axis=1)
                source_labels[noise_mask] = 'noise'
        else:
            # All pulses were noise → single shared class
            final_labels[:] = 0
            source_labels[:] = 'noise'
            n_clusters = 1
            n_noise_forced = n_wc

        print(f"  Width class {wc}: {n_clusters} shape cluster(s) "
              f"(P1 eps={eps_p1:.3f}, P2 eps={eps_p2:.3f}), {n_noise_forced} noise\u2192assigned "
              f"[{'subsampled' if n_wc > dbscan_max_direct else 'direct'} {n_sample}/{n_wc}]")

        # Assign globally unique shape class IDs
        for i, pulse_idx in enumerate(wc_indices):
            eod_data.loc[pulse_idx, 'shape_class'] = next_shape_id + final_labels[i]
            eod_data.loc[pulse_idx, 'shape_source'] = source_labels[i]

        next_shape_id += n_clusters

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
                    'shape_class': sc,
                    'waveform_signature': waveforms_l2[pulse_idx].copy()
                }
                eod_data.loc[pulse_idx, 'fragment_id'] = next_fragment_id
                next_fragment_id += 1
                continue

            # Score all candidates; new-fragment competes at cost = pass1_new_frag_cost
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

                # Waveform cost: use rolling signature (median of last 10 assigned pulses)
                waveform_cost = np.linalg.norm(waveforms_l2[pulse_idx] - f['waveform_signature'])

                cost = location_weight * loc_cost + ipi_weight * ipi_cost + waveform_weight * waveform_cost

                if cost < best_cost:
                    best_cost = cost
                    best_fid = fid

            if best_cost < pass1_new_frag_cost:
                f = fragments[best_fid]
                dt = (pulse_ts - f['last_timestamp']).total_seconds()
                f['ipi_history'].append(dt)
                f['history'].append(pulse_idx)
                f['last_timestamp'] = pulse_ts
                f['last_location'] = pulse_loc
                f['waveform_signature'] = np.median(waveforms_l2[f['history'][-10:]], axis=0)
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
                    waveform_cost_d = np.linalg.norm(waveforms_l2[pulse_idx] - f_best['waveform_signature'])
                    print(f'  [NEW frag {next_fragment_id}] t={t_rel:.3f}s loc={pulse_loc:.3f} '
                          f'wc={wc} sc={sc} | new_frag_wins: best_frag={best_fid} '
                          f'cost={best_cost:.3f} >= new_frag_cost={pass1_new_frag_cost} | '
                          f'loc_cost={loc_cost_d:.3f} (pred={pred_loc_d:.3f}, actual={pulse_loc:.3f}, tol={location_tolerance}) | '
                          f'ipi_cost={ipi_cost_d:.3f} (dt={dt_best:.3f}s, med_ipi={med_ipi_d:.3f}s, tol={ipi_tol_d:.3f}s) | '
                          f'waveform_cost={waveform_cost_d:.3f}')
                fragments[next_fragment_id] = {
                    'history': [pulse_idx],
                    'ipi_history': [],
                    'last_timestamp': pulse_ts,
                    'last_location': pulse_loc,
                    'width_class': wc,
                    'shape_class': sc,
                    'waveform_signature': waveforms_l2[pulse_idx].copy()
                }
                eod_data.loc[pulse_idx, 'fragment_id'] = next_fragment_id
                next_fragment_id += 1

    n_fragments_pass1 = len(fragments)
    n_assigned_pass1 = (eod_data['fragment_id'] >= 0).sum()
    print(f"✓ Pass 1 complete: {n_fragments_pass1} fragments, {n_assigned_pass1}/{len(eod_data)} pulses assigned")

    # -------------------------------------------------------------------------
    # Step 4: Pass 2a - Overlap merge
    # Fragments whose time ranges genuinely overlap (interleaved pulse trains)
    # are merged if their median waveforms are similar and merging does not
    # create any consecutive pulse pair below min_ipi_s.
    # Greedy acceptance ranked by ascending waveform distance.
    # -------------------------------------------------------------------------
    print("\nPass 2a: Overlap merge...")

    for overlap_iter in range(pass2_overlap_max_iterations):
        frag_ids = list(fragments.keys())
        n_frags = len(frag_ids)
        if n_frags < 2:
            break

        # Median waveform and time-span per fragment
        frag_start_ts = {fid: eod_data.loc[f['history'][0], 'timestamp'] for fid, f in fragments.items()}
        frag_end_ts = {fid: eod_data.loc[f['history'][-1], 'timestamp'] for fid, f in fragments.items()}
        frag_median_wf = {fid: np.median(waveforms_l2[f['history']], axis=0) for fid, f in fragments.items()}

        candidates = []  # list of (wf_dist, fid_A, fid_B) where A < B (arbitrary order key)
        for i in range(n_frags):
            for j in range(i + 1, n_frags):
                fid_a = frag_ids[i]
                fid_b = frag_ids[j]
                # Check genuine time overlap: overlap window > pass2_overlap_min_s
                overlap_start = max(frag_start_ts[fid_a], frag_start_ts[fid_b])
                overlap_end = min(frag_end_ts[fid_a], frag_end_ts[fid_b])
                overlap_s = (overlap_end - overlap_start).total_seconds() if overlap_end > overlap_start else 0.0
                if overlap_s < pass2_overlap_min_s:
                    continue
                # Width class must match (different electrode distances = different species)
                if fragments[fid_a]['width_class'] != fragments[fid_b]['width_class']:
                    continue
                # Waveform similarity check
                wf_dist = np.linalg.norm(frag_median_wf[fid_a] - frag_median_wf[fid_b])
                if wf_dist >= pass2_overlap_wf_threshold:
                    continue
                # IPI validity: merge all pulse indices, sort by timestamp, check no gap < min_ipi_s
                merged_indices = fragments[fid_a]['history'] + fragments[fid_b]['history']
                merged_ts = np.array([eod_data.loc[pidx, 'timestamp'].timestamp() for pidx in merged_indices])
                merged_ts_sorted = np.sort(merged_ts)
                min_gap = np.min(np.diff(merged_ts_sorted)) if len(merged_ts_sorted) > 1 else 1.0
                if min_gap < min_ipi_s:
                    continue
                candidates.append((wf_dist, fid_a, fid_b))

        if not candidates:
            print(f"  Overlap iter {overlap_iter + 1}: no candidates, stopping")
            break

        # Greedy acceptance: sort by wf_dist ascending, consume each fragment at most once
        candidates.sort(key=lambda x: x[0])
        consumed = set()
        n_merged = 0
        for wf_dist, fid_a, fid_b in candidates:
            if fid_a in consumed or fid_b in consumed:
                continue
            # Merge fid_b into fid_a (keep lower fid as canonical)
            fid_keep = min(fid_a, fid_b)
            fid_drop = max(fid_a, fid_b)
            f_keep = fragments[fid_keep]
            f_drop = fragments[fid_drop]
            merged_indices = f_keep['history'] + f_drop['history']
            merged_ts_raw = [(eod_data.loc[pidx, 'timestamp'], pidx) for pidx in merged_indices]
            merged_ts_raw.sort(key=lambda x: x[0])
            sorted_indices = [pidx for _, pidx in merged_ts_raw]
            sorted_ts = [ts for ts, _ in merged_ts_raw]
            f_keep['history'] = sorted_indices
            f_keep['ipi_history'] = [(sorted_ts[k] - sorted_ts[k - 1]).total_seconds() for k in range(1, len(sorted_ts))]
            f_keep['last_timestamp'] = sorted_ts[-1]
            f_keep['last_location'] = eod_data.loc[sorted_indices[-1], 'pulse_location']
            f_keep['waveform_signature'] = np.median(waveforms_l2[sorted_indices[-10:]], axis=0)
            for pidx in f_drop['history']:
                eod_data.loc[pidx, 'fragment_id'] = fid_keep
            del fragments[fid_drop]
            consumed.add(fid_a)
            consumed.add(fid_b)
            n_merged += 1
            print(f"  [OVERLAP MERGE] fid {fid_drop} -> fid {fid_keep}  wf_dist={wf_dist:.3f}  overlap={overlap_s:.2f}s")

        print(f"  Overlap iter {overlap_iter + 1}: merged {n_merged} pair(s), {len(fragments)} fragments remain")
        if n_merged == 0:
            break

    print(f"✓ Pass 2a complete: {len(fragments)} fragments")

    # -------------------------------------------------------------------------
    # Step 4b: Pass 2b - Sequential fragment stitching using LAP
    # Build a cost matrix of (fragment endings) x (fragment starts).
    # Merge fragment pairs whose assignment cost is below threshold.
    # Iterate until no more merges occur.
    # shape_class gate removed — waveform cost already captures shape similarity.
    # -------------------------------------------------------------------------
    print("\nPass 2b: Sequential fragment stitching...")

    for stitch_iter in range(pass2_max_iterations):
        frag_ids = list(fragments.keys())
        n_frags = len(frag_ids)
        if n_frags < 2:
            break
        if n_frags > pass2_max_frags:
            print(f"  Iteration {stitch_iter + 1}: {n_frags} fragments > pass2_max_frags={pass2_max_frags}, skipping Pass 2b")
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
                # start must come strictly after end; gap must be within window
                gap = (frag_start_ts[fid_start] - frag_end_ts[fid_end]).total_seconds()
                if gap <= 0 or gap > pass2_max_gap_s:
                    continue
                # Width class must match (different electrode distances = different species)
                if fragments[fid_end]['width_class'] != fragments[fid_start]['width_class']:
                    continue
                # shape_class gate removed: waveform cost captures shape similarity
                # Waveform distance
                wf_cost = np.linalg.norm(frag_end_wf[fid_end] - frag_start_wf[fid_start])
                # Spatial plausibility: velocity over the gap
                loc_diff = abs(frag_start_loc[fid_start] - frag_end_loc[fid_end])
                max_allowed_loc_diff = max_location_jump_per_s * gap
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

    print(f"✓ Pass 2b complete: {len(fragments)} fragments")

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
    # Species assignment: LDA (PCA -> LDA on control means) or 1-NN fallback
    # -------------------------------------------------------------------------
    if use_species_matching and len(reference_library) > 0:
        print("\nAssigning species...")
        eod_data['species_assigned'] = ''
        eod_data['species_uncertain'] = False
        eod_data['nearest_individual'] = ''
        eod_data['dist_nearest'] = np.nan
        eod_data['dist_margin'] = np.nan
        if lda is not None:
            for sp in lda.classes_:
                eod_data[f'lda_proba_{sp}'] = np.nan
        else:
            for sp in all_species_codes:
                eod_data[f'dist_{sp}'] = np.nan

        fish_ids_for_matching = sorted([fid for fid in eod_data['fish_id'].unique() if fid >= 0])
        for fid in fish_ids_for_matching:
            fid_mask = eod_data['fish_id'] == fid
            track_mean_wf = waveforms_l2[fid_mask].mean(axis=0)  # (waveform_target_length,)
            # L2 distances to each control individual (kept for nearest_individual / dist diagnostics)
            dists = np.linalg.norm(ref_matrix - track_mean_wf[None, :], axis=1)  # (N_ref,)
            nn_idx = int(np.argmin(dists))
            nn_dist = float(dists[nn_idx])
            sorted_dists = np.sort(dists)
            margin = float(sorted_dists[1] - sorted_dists[0]) if len(sorted_dists) > 1 else np.nan
            eod_data.loc[fid_mask, 'nearest_individual'] = ref_ids[nn_idx]
            eod_data.loc[fid_mask, 'dist_nearest']       = nn_dist
            eod_data.loc[fid_mask, 'dist_margin']        = margin
            if lda is not None:
                track_pca_s = pca_cls.transform(track_mean_wf[None, :])
                sp_pred = lda.predict(track_pca_s)[0]
                sp_proba = lda.predict_proba(track_pca_s)[0]
                eod_data.loc[fid_mask, 'species_assigned'] = sp_pred
                for sp, p in zip(lda.classes_, sp_proba):
                    eod_data.loc[fid_mask, f'lda_proba_{sp}'] = float(p)
                assigned_proba = float(sp_proba[list(lda.classes_).index(sp_pred)])
                uncertain = assigned_proba < lda_min_probability
                eod_data.loc[fid_mask, 'species_uncertain'] = uncertain
                uncertain_flag = " (!)" if uncertain else ""
                print(f"  Fish {fid:2d}: {sp_pred}{uncertain_flag} (p={assigned_proba:.3f}, "
                      f"nearest: {ref_ids[nn_idx]}, dist={nn_dist:.4f}, margin={margin:.4f})")
            else:
                # 1-NN fallback when only 1 species in reference library
                eod_data.loc[fid_mask, 'species_assigned'] = ref_species[nn_idx]
                for sp in all_species_codes:
                    sp_mask_idx = [i for i, s in enumerate(ref_species) if s == sp]
                    if sp_mask_idx:
                        eod_data.loc[fid_mask, f'dist_{sp}'] = float(dists[sp_mask_idx].min())
                print(f"  Fish {fid:2d}: {ref_species[nn_idx]} (nearest: {ref_ids[nn_idx]}, "
                      f"dist={nn_dist:.4f}, margin={margin:.4f})")

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

    # -------------------------------------------------------------------------
    # Consolidated output figure (3 rows)
    # -------------------------------------------------------------------------
    print("\nGenerating consolidated tracking figure...")
    fish_ids_assigned = sorted([fid for fid in eod_data['fish_id'].unique() if fid >= 0])
    n_assigned = len(fish_ids_assigned)
    fish_colors = plt.cm.tab10(np.linspace(0, 1, max(n_assigned, 1)))
    fish_color_map = {fid: fish_colors[i % len(fish_colors)] for i, fid in enumerate(fish_ids_assigned)}

    # Row 3 determines number of columns (at least 1, at most 8 per row, wrap into multiple rows)
    n_cols_r3 = min(n_assigned, 8) if n_assigned > 0 else 1
    n_rows_r3 = max(1, int(np.ceil(n_assigned / n_cols_r3))) if n_assigned > 0 else 1
    total_rows = 3 + n_rows_r3 - 1  # row1 + row2 + n_rows_r3 waveform rows
    row_height_ratios = [2, 2] + [1.5] * n_rows_r3

    fig = plt.figure(figsize=(max(14, 3 * n_cols_r3), 4 * (total_rows)))
    fig.suptitle(f"{file_set['base_name']} — tracking result", fontsize=11, y=0.99)

    gs_outer = fig.add_gridspec(total_rows, 1, height_ratios=row_height_ratios, hspace=0.45)

    # --- Row 1: location vs time, colored by fish_id ---
    ax_loc = fig.add_subplot(gs_outer[0])
    unassigned_mask = eod_data['fish_id'] < 0
    if unassigned_mask.any():
        ax_loc.scatter(t_sec[unassigned_mask],
                       eod_data.loc[unassigned_mask, 'pulse_location'],
                       c='lightgray', s=3, alpha=0.4, label='unassigned', rasterized=True)
    for fid in fish_ids_assigned:
        mask = eod_data['fish_id'] == fid
        if use_species_matching and 'species_assigned' in eod_data.columns:
            _sp = eod_data.loc[mask, 'species_assigned'].iloc[0]
            _loc_label = f'Fish {fid} [{_sp}]'
        else:
            _loc_label = f'Fish {fid}'
        ax_loc.scatter(t_sec[mask], eod_data.loc[mask, 'pulse_location'],
                       color=fish_color_map[fid], s=4, alpha=0.7,
                       label=_loc_label, rasterized=True)
    ax_loc.set_ylabel('Location (electrode units)')
    ax_loc.set_xlabel('Time (s)')
    ax_loc.set_title('Pulse location vs time')
    ax_loc.legend(markerscale=3, loc='upper right', fontsize=7, ncol=max(1, n_assigned // 8))

    # --- Row 2: three columns: fish-ID PCA | species/control PCA | width histogram ---
    gs_r2 = gs_outer[1].subgridspec(1, 3, wspace=0.35)
    ax_pca  = fig.add_subplot(gs_r2[0])
    ax_pca2 = fig.add_subplot(gs_r2[1])
    ax_hist = fig.add_subplot(gs_r2[2])

    # PCA on all L2-normalized waveforms
    pca_model = PCA(n_components=2)
    pca_coords = pca_model.fit_transform(waveforms_l2)
    # plot unassigned
    if unassigned_mask.any():
        ax_pca.scatter(pca_coords[unassigned_mask, 0], pca_coords[unassigned_mask, 1],
                       c='lightgray', s=3, alpha=0.3, rasterized=True)
    for fid in fish_ids_assigned:
        mask = eod_data['fish_id'] == fid
        ax_pca.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                       color=fish_color_map[fid], s=4, alpha=0.5,
                       label=f'Fish {fid}', rasterized=True)
    ax_pca.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)')
    ax_pca.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)')
    ax_pca.set_title('PCA of waveforms (colored by fish ID)')

    # LDA discriminant space: control individual means + per-track means
    if use_species_matching and lda is not None and len(reference_library) > 0 and n_assigned > 0:
        sp_pal = plt.cm.Set3(np.linspace(0, 0.9, max(len(all_species_codes), 1)))
        sp_color_map_pca = {sp: sp_pal[i] for i, sp in enumerate(all_species_codes)}
        n_lda_axes = len(lda.classes_) - 1
        ctrl_lda = lda.transform(ref_pca_scores)  # (N_ref, n_lda_axes)
        if n_lda_axes >= 2:
            ctrl_x, ctrl_y = ctrl_lda[:, 0], ctrl_lda[:, 1]
            x_label, y_label = 'LD1', 'LD2'
        else:
            ctrl_x = ctrl_lda[:, 0]
            ctrl_y = pca_cls.transform(ref_matrix)[:, 0]
            x_label, y_label = 'LD1', 'PC1'
        for sp in all_species_codes:
            sp_ctrl_mask = [i for i, s in enumerate(ref_species) if s == sp]
            if sp_ctrl_mask:
                ax_pca2.scatter(ctrl_x[sp_ctrl_mask], ctrl_y[sp_ctrl_mask],
                                color=sp_color_map_pca[sp], s=40, alpha=0.35,
                                marker='o', label=f'{sp} (ctrl)', rasterized=True)
        for fid in fish_ids_assigned:
            fid_mask_pca = eod_data['fish_id'] == fid
            track_mean_wf = waveforms_l2[fid_mask_pca].mean(axis=0)
            track_pca_s = pca_cls.transform(track_mean_wf[None, :])
            track_lda = lda.transform(track_pca_s)
            tx = float(track_lda[0, 0])
            ty = float(track_lda[0, 1]) if n_lda_axes >= 2 else float(track_pca_s[0, 0])
            sp_assigned = eod_data.loc[fid_mask_pca, 'species_assigned'].iloc[0]
            proba_col = f'lda_proba_{sp_assigned}'
            assigned_p = (
                float(eod_data.loc[fid_mask_pca, proba_col].iloc[0])
                if proba_col in eod_data.columns else float('nan')
            )
            ax_pca2.scatter(tx, ty,
                            color=sp_color_map_pca.get(sp_assigned, 'black'),
                            s=150, alpha=0.9, marker='*', edgecolors='k',
                            linewidths=0.5, zorder=5)
            ax_pca2.annotate(f'{fid}\n(p={assigned_p:.2f})', (tx, ty),
                             fontsize=6, ha='center', va='bottom')
        ax_pca2.set_xlabel(x_label)
        ax_pca2.set_ylabel(y_label)
        loo_str = f'{loo_accuracy:.0%}' if loo_accuracy is not None else 'N/A'
        ax_pca2.set_title(f'LDA space: ctrl + tracks (LOO {loo_str})')
        ax_pca2.legend(fontsize=6, markerscale=1.5, loc='best')
    else:
        ax_pca2.axis('off')
        if not use_species_matching:
            _ax2_msg = 'Species matching\ndisabled'
        elif lda is None:
            _ax2_msg = 'Only 1 species\nin reference library'
        else:
            _ax2_msg = 'No fish assigned'
        ax_pca2.text(0.5, 0.5, _ax2_msg,
                     ha='center', va='center', transform=ax_pca2.transAxes, fontsize=9)

    # Width histogram + KDE, colored by width_class
    width_classes = sorted(eod_data['width_class'].unique())
    wc_cmap = plt.cm.Set1(np.linspace(0, 0.8, len(width_classes)))
    wc_color_map = {wc: wc_cmap[i] for i, wc in enumerate(width_classes)}
    bin_edges = np.linspace(widths.min(), widths.max(), 60)
    for wc in width_classes:
        wc_mask = eod_data['width_class'] == wc
        ax_hist.hist(widths[wc_mask], bins=bin_edges, alpha=0.6,
                     color=wc_color_map[wc], label=f'Class {wc}', edgecolor='none')
    # KDE overlay per class
    w_range = np.linspace(widths.min(), widths.max(), 400)
    for wc in width_classes:
        wc_mask = eod_data['width_class'] == wc
        if wc_mask.sum() > 5:
            wc_std = np.std(widths[wc_mask])
            wc_kde_bw = max(2.0 * step_us, 1.0) / wc_std if wc_std > 0 else kde_bw
            kde_wc = gaussian_kde(widths[wc_mask], bw_method=wc_kde_bw)
            kde_scale = wc_mask.sum() * (bin_edges[1] - bin_edges[0])
            ax_hist.plot(w_range, kde_wc(w_range) * kde_scale,
                         color=wc_color_map[wc], linewidth=1.5)
    ax_hist.set_xlabel('EOD width (µs)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Pulse width distribution by class')
    if len(width_classes) > 1:
        ax_hist.legend(fontsize=7)

    # --- Rows 3+: per-fish waveform overlays ---
    if n_assigned > 0:
        shape_classes = sorted(eod_data['shape_class'].unique())
        sc_cmap = plt.cm.Set2(np.linspace(0, 0.9, max(len(shape_classes), 1)))
        sc_color_map = {sc_id: sc_cmap[i] for i, sc_id in enumerate(shape_classes)}

        wf_panel_idx = 0
        for row_offset in range(n_rows_r3):
            gs_r3 = gs_outer[2 + row_offset].subgridspec(1, n_cols_r3, wspace=0.3)
            for col_idx in range(n_cols_r3):
                if wf_panel_idx >= n_assigned:
                    ax_wf = fig.add_subplot(gs_r3[col_idx])
                    ax_wf.axis('off')
                    wf_panel_idx += 1
                    continue
                fid = fish_ids_assigned[wf_panel_idx]
                ax_wf = fig.add_subplot(gs_r3[col_idx])
                fid_mask = eod_data['fish_id'] == fid
                fid_wf = waveforms_l2[fid_mask]
                fid_sc = eod_data.loc[fid_mask, 'shape_class'].values
                # subsample for overlay
                n_overlay = min(300, len(fid_wf))
                overlay_idx = np.random.choice(len(fid_wf), size=n_overlay, replace=False)
                for oi in overlay_idx:
                    sc_id = fid_sc[oi]
                    ax_wf.plot(fid_wf[oi], color=sc_color_map.get(sc_id, 'steelblue'),
                               alpha=0.08, linewidth=0.5, rasterized=True)
                mean_wf = fid_wf.mean(axis=0)
                ax_wf.plot(mean_wf, color=fish_color_map[fid], linewidth=2)
                if use_species_matching and 'species_assigned' in eod_data.columns:
                    sp_label = eod_data.loc[fid_mask, 'species_assigned'].iloc[0]
                    ax_wf.set_title(f'Fish {fid} [{sp_label}] (n={fid_mask.sum()})', fontsize=8)
                else:
                    ax_wf.set_title(f'Fish {fid} (n={fid_mask.sum()})', fontsize=8)
                ax_wf.set_xticks([])
                ax_wf.tick_params(labelsize=6)
                wf_panel_idx += 1

    result_path = os.path.join(output_folder, f"{file_set['base_name']}_tracked.png")
    plt.savefig(result_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved tracking plot: {os.path.basename(result_path)}")

    # Save tracked EOD table
    out_csv = os.path.join(output_folder, f"{file_set['base_name']}_tracked.csv")
    eod_data.to_csv(out_csv, index=False)
    print(f"✓ Saved tracked table: {os.path.basename(out_csv)}")

    # Collect per-fish details for summary outputs
    fish_details = []
    for fid in sorted(fid_to_fish):
        fish_id = fid_to_fish[fid]
        f = fragments[fid]
        n_p = len(f['history'])
        dur = (eod_data.loc[f['history'][-1], 'timestamp'] -
               eod_data.loc[f['history'][0], 'timestamp']).total_seconds()
        mean_loc = eod_data.loc[f['history'], 'pulse_location'].mean()
        mean_width_val = eod_data.loc[f['history'], 'eod_width_us'].mean()
        mean_rate_val = 1.0 / np.median(f['ipi_history']) if f['ipi_history'] else float('nan')
        fish_rec = {
            'fish_id':       fish_id,
            'n_pulses':      n_p,
            'duration_s':    dur,
            'mean_rate_hz':  mean_rate_val,
            'mean_location': mean_loc,
            'mean_width_us': mean_width_val,
        }
        if use_species_matching and 'species_assigned' in eod_data.columns:
            fid_mask_sum = eod_data['fish_id'] == fish_id
            fish_rec['species_assigned']   = eod_data.loc[fid_mask_sum, 'species_assigned'].iloc[0]
            fish_rec['species_uncertain']  = bool(eod_data.loc[fid_mask_sum, 'species_uncertain'].iloc[0])
            fish_rec['nearest_individual'] = eod_data.loc[fid_mask_sum, 'nearest_individual'].iloc[0]
            fish_rec['dist_nearest']       = float(eod_data.loc[fid_mask_sum, 'dist_nearest'].iloc[0])
            fish_rec['dist_margin']        = float(eod_data.loc[fid_mask_sum, 'dist_margin'].iloc[0])
            if lda is not None:
                for sp in lda.classes_:
                    fish_rec[f'lda_proba_{sp}'] = float(
                        eod_data.loc[fid_mask_sum, f'lda_proba_{sp}'].iloc[0]
                    )
                fish_rec['lda_proba_assigned'] = fish_rec[
                    f'lda_proba_{fish_rec["species_assigned"]}'
                ]
            else:
                for sp in all_species_codes:
                    fish_rec[f'dist_{sp}'] = float(
                        eod_data.loc[fid_mask_sum, f'dist_{sp}'].iloc[0]
                    )
        fish_details.append(fish_rec)

    all_tracked_data.append({
        'base_name':       file_set['base_name'],
        'event_id':        file_set['event_id'],
        'n_fish':          n_fish,
        'n_pulses':        len(eod_data),
        'assignment_rate': assigned / len(eod_data) if len(eod_data) > 0 else 0.0,
        'fish_details':    fish_details,
    })
    del eod_data, waveforms_l2, waveforms_l2_p2

print("\n" + "="*70)
print("TRACKING COMPLETE")
print("="*70)
print(f"Processed {len(all_tracked_data)}/{len(file_sets)} file(s) successfully")

# =============================================================================
# SUMMARY OUTPUTS
# =============================================================================

# --- 1: Per-fish summary (one row per fish per file) ---
fish_summary_rows = []
for entry in all_tracked_data:
    for fd in entry['fish_details']:
        row = {'base_name': entry['base_name'], 'event_id': entry['event_id']}
        row.update(fd)
        fish_summary_rows.append(row)

if fish_summary_rows:
    fish_summary_df = pd.DataFrame(fish_summary_rows)
    fish_summary_path = os.path.join(output_folder, 'tracked_fish_summary.csv')
    fish_summary_df.to_csv(fish_summary_path, index=False)
    print(f"\u2713 Saved per-fish summary: {os.path.basename(fish_summary_path)} "
          f"({len(fish_summary_df)} rows)")

# --- 2: Per-species summary (one row per species per file, when matching enabled) ---
if use_species_matching and fish_summary_rows:
    species_summary_rows = []
    for entry in all_tracked_data:
        sp_counts = {}
        for fd in entry['fish_details']:
            sp = fd.get('species_assigned', '')
            sp_counts[sp] = sp_counts.get(sp, 0) + 1
        for sp, cnt in sp_counts.items():
            species_summary_rows.append({
                'base_name': entry['base_name'],
                'event_id':  entry['event_id'],
                'species':   sp,
                'n_fish':    cnt,
            })
    if species_summary_rows:
        species_summary_df = pd.DataFrame(species_summary_rows)
        species_summary_path = os.path.join(output_folder, 'tracked_species_summary.csv')
        species_summary_df.to_csv(species_summary_path, index=False)
        print(f"\u2713 Saved per-species summary: {os.path.basename(species_summary_path)}")

# --- 3: Event summary — tracking results appended to input all_event_summaries ---
event_rows = []
for entry in all_tracked_data:
    row = {
        'event_id':         entry['event_id'],
        'n_fish_total':     entry['n_fish'],
        'n_pulses_tracked': entry['n_pulses'],
        'assignment_rate':  entry['assignment_rate'],
    }
    if (use_species_matching and entry['fish_details']
            and 'species_assigned' in entry['fish_details'][0]):
        sp_counts = {}
        for fd in entry['fish_details']:
            sp = fd.get('species_assigned', '')
            sp_counts[sp] = sp_counts.get(sp, 0) + 1
        for sp in all_species_codes:
            row[f'n_fish_{sp}'] = sp_counts.get(sp, 0)
    event_rows.append(row)

event_tracking_df = pd.DataFrame(event_rows)
if event_summaries is not None:
    event_out_df = event_summaries.merge(event_tracking_df, on='event_id', how='left')
else:
    event_out_df = event_tracking_df
event_summary_out_path = os.path.join(output_folder, 'tracked_event_summary.csv')
event_out_df.to_csv(event_summary_out_path, index=False)
print(f"\u2713 Saved event summary: {os.path.basename(event_summary_out_path)} "
      f"({len(event_out_df)} rows)")
