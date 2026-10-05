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

import tkinter as tk
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulse_functions import fit_species_classifier, reclassify_session
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

print("\n" + "="*70)
print("RECLASSIFYING SPECIES")
print("="*70)
fish_summary_df = reclassify_session(input_path, output_path, species_classifier, lda_min_probability)
if fish_summary_df is None:
    raise ValueError("No usable tracked fish / stored mean waveforms found - cannot reclassify")

print("\n" + "="*70)
print("RECLASSIFICATION COMPLETE")
print("="*70)

