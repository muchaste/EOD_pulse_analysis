# -*- coding: utf-8 -*-
"""
04_1c_Batch_Reclassify_Species.py

Batch species reclassification (no GUI) across every already-tracked session folder
under a tracking-results root. Final thresholds (lda_max_dist_factor=3.0,
lda_min_probability=0.8) chosen from the baseline (factor=1.0/min_prob=0.0) calibration
sweep in 3a_Species_Classification_Calibration.R. Uses fit_species_classifier()/
reclassify_session() from pulse_functions.py - no re-tracking.

Authors: Stefan Mucha with Claude Sonnet 4.6
"""

import json
import glob
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulse_functions import fit_species_classifier, reclassify_session

tracking_root = r"C:\Users\muchaste\Seafile\Uni\01 Research Projects\04 Uganda 2023 EOD loggers\01 Data\EOD Loggers\Tracking Results"
control_path  = r"H:\Models and Classifiers\Species Assignment\Extraction_0.005"
output_root   = r"C:\Users\muchaste\Seafile\Uni\01 Research Projects\04 Uganda 2023 EOD loggers\01 Data\EOD Loggers\Tracking Results Reclassified\final_factor3.0_prob0.8"

lda_min_probability = 0.8
lda_max_dist_factor = 3.0

print("="*70)
print("BATCH SPECIES RECLASSIFICATION")
print("="*70)
print(f"✓ Tracking root: {tracking_root}")
print(f"✓ Control folder: {control_path}")
print(f"✓ Output root: {output_root}")
print(f"✓ lda_min_probability={lda_min_probability}, lda_max_dist_factor={lda_max_dist_factor}")

session_folders = sorted(glob.glob(os.path.join(tracking_root, "*", "*")))
session_folders = [f for f in session_folders if os.path.isdir(f)
                   and os.path.exists(os.path.join(f, "tracked_fish_summary.csv"))]
print(f"\n✓ Found {len(session_folders)} tracked session folder(s)")

n_ok, n_skipped = 0, 0
for session_folder in session_folders:
    rel_path = os.path.relpath(session_folder, tracking_root)
    output_folder = os.path.join(output_root, rel_path)
    os.makedirs(output_folder, exist_ok=True)

    report_path = os.path.join(session_folder, "classifier_report.json")
    if not os.path.exists(report_path):
        print(f"\n⚠ Skipping {rel_path}: no classifier_report.json found")
        n_skipped += 1
        continue
    with open(report_path, "r") as f:
        report = json.load(f)
    waveform_target_length = int(report["waveform_target_length"])
    crop_factor = int(report["crop_factor"])

    print(f"\n{'='*70}\n{rel_path}\n{'='*70}")
    species_classifier = fit_species_classifier(
        control_path, waveform_target_length, crop_factor,
        lda_max_dist_factor=lda_max_dist_factor, output_folder=output_folder
    )
    if species_classifier is None:
        print(f"⚠ Skipping {rel_path}: no usable control reference library")
        n_skipped += 1
        continue

    fish_summary_df = reclassify_session(session_folder, output_folder, species_classifier, lda_min_probability)
    if fish_summary_df is None:
        print(f"⚠ Skipping {rel_path}: no usable tracked fish / stored mean waveforms")
        n_skipped += 1
        continue
    n_ok += 1

print(f"\n{'='*70}")
print(f"BATCH RECLASSIFICATION COMPLETE: {n_ok} session(s) processed, {n_skipped} skipped")
print("="*70)
