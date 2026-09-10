# -*- coding: utf-8 -*-
"""
04_3b_Seed_Best_Combo.py

This script seeds the best parameter combination from the Phase-A analysis for iterative optimization.
It creates a single-row proposal CSV file with the best marginal values.

Authors: Stefan Mucha with Claude Sonnet 4.6
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# ============================================================
# CONFIGURATION
# ============================================================
SUMMARY_CSV   = r'c:\Users\Admin\Projects\EOD_pulse_analysis\05_4_analysis_summary.csv'
OUTPUT_FOLDER = ''   # leave empty to choose via dialog
# ============================================================

if not OUTPUT_FOLDER:
    root = tk.Tk()
    root.withdraw()
    OUTPUT_FOLDER = filedialog.askdirectory(title='Select the NEW (empty) output folder')
    root.destroy()
if not OUTPUT_FOLDER:
    raise RuntimeError('No output folder selected.')

summary = pd.read_csv(SUMMARY_CSV, index_col='parameter')
best    = summary['best_value'].to_dict()

# Single-row proposal built from the best marginal values in the Phase-A analysis.
# Weight triplet must sum to 1.0.  Phase A showed weights are near-irrelevant
# (eta2 < 0.005), so we use the confirmed best triplet: loc=0.1, ipi=0.1, wf=0.8.
row = {
    'pass2_max_iterations':         int(round(best['pass2_max_iterations'])),
    'pass2_max_gap_s':              float(best['pass2_max_gap_s']),
    'ipi_tolerance_min_s':          float(best['ipi_tolerance_min_s']),
    'pass1_new_frag_cost':          float(best['pass1_new_frag_cost']),
    'ipi_tolerance_fraction':       float(best['ipi_tolerance_fraction']),
    'fft_artifact_threshold':       float(best['fft_artifact_threshold']),
    'pass2_overlap_max_iterations': int(round(best['pass2_overlap_max_iterations'])),
    'min_ipi_s':                    float(best['min_ipi_s']),
    'max_track_gap_s':              float(best['max_track_gap_s']),
    'max_location_jump_per_s':      float(best['max_location_jump_per_s']),
    'knn_percentile':               int(round(best['knn_percentile'])),
    'min_shape_eps':                float(best['min_shape_eps']),
    'location_tolerance':           float(best['location_tolerance']),
    'pass2_cost_threshold':         float(best['pass2_cost_threshold']),
    'pass2_overlap_wf_threshold':   float(best['pass2_overlap_wf_threshold']),
    'pass2_overlap_min_s':          float(best['pass2_overlap_min_s']),
    'width_min_separation_us':      float(best['width_min_separation_us']),
    'waveform_target_length':       int(round(best['waveform_target_length'])),
    'crop_factor':                  int(round(best['crop_factor'])),
    'location_weight':              0.7,
    'ipi_weight':                   0.1,
    'waveform_weight':              0.2,
    'pass2_waveform_weight':        0.5,
    'pass2_spatial_weight':         0.5,
}

out_path = os.path.join(OUTPUT_FOLDER, 'proposals_iter01.csv')
pd.DataFrame([row]).to_csv(out_path, index=False)
print(f'Saved: {out_path}')
print('Run 05_6 — it will skip 05_5 for iter 1 and evaluate this combo directly in 05_2.')
