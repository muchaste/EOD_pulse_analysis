"""
# 04_1_REF_Fish_Tracking.py
# Sequential Fish Tracking for Multi-Channel Field Recordings

This script tracks individual fish through sequential pulse-by-pulse analysis.
Combines waveform similarity, spatiotemporal consistency, and physiological
constraints to assign fish identities and infer swimming behaviors.

PROCEDURE:
==========
1. Load EOD table and waveforms from pulse extraction outputs (03_1_REF)
2. Initialize tracks using waveform clustering on first N pulses
3. Sequentially process remaining pulses, matching to active tracks
4. Apply spatiotemporal consistency checks and physiological constraints
5. Detect and annotate swimming behaviors (turning, hovering, movement, crossing)
6. Optionally classify species using pre-trained models (02_2_REF)
7. Output tracked EOD table and behavioral event summaries

INPUTS:
=======
- EOD table CSV files (*_eod_table.csv) from pulse extraction
- Waveform NPZ files (*_waveforms.npz) with normalized fixed-length waveforms
- [Optional] Pre-trained species classifier (*.pkl or *.joblib)

OUTPUTS:
========
- *_tracked.csv: EOD table with added fish_id, species, behavior columns
- *_behaviors.csv: Behavioral event summaries per fish
- *_trajectories.png: Spatiotemporal trajectory plots
- *_tracking_metrics.json: Quality metrics and statistics
- *_fish_waveforms.png: Per-fish waveform overlays for validation

"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import json
import glob
import datetime as dt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import warnings

# Import EOD functions
from pulse_functions import (
    load_fixed_length_waveforms,
    normalize_waveforms
)

warnings.filterwarnings('ignore')

print("="*70)
print("SEQUENTIAL FISH TRACKING ANALYSIS")
print("="*70)

# =============================================================================
# TRACKING PARAMETERS
# =============================================================================

print("\nSetting tracking parameters...")

# Waveform preprocessing parameters
waveform_crop_factor = 4  # Crop window size as multiple of peak-peak duration
waveform_target_length = 200  # Target number of samples after interpolation

# Waveform clustering parameters (for initial track seeding)
n_seed_pulses = 10  # Number of initial pulses to use for track initialization
dbscan_eps_percentile = 40  # Percentile for adaptive DBSCAN eps threshold - lower = stricter
dbscan_min_samples = 2  # Minimum samples per cluster

# Temporal constraints
max_discharge_rate_hz = 150  # Maximum EOD rate (minimum IPI = 6.67ms)
min_ipi_ms = 1000.0 / max_discharge_rate_hz
track_timeout_s = 5.0  # Mark track inactive after this gap
reappearance_window_s = 30.0  # Allow track reactivation within this window

# Spatial constraints
electrode_spacing_cm = 50.0  # Distance between electrodes
max_fish_velocity_m_s = 2.0  # Maximum swimming speed
max_spatial_jump = max_fish_velocity_m_s * 100 / electrode_spacing_cm  # In electrode units per second

# Waveform matching thresholds
waveform_distance_threshold_multiplier = 0.5  # Multiply initial DBSCAN eps by this
amplitude_ratio_tolerance = 0.3  # Relative change tolerance
width_tolerance_us = 50  # Microseconds
freq_tolerance_hz = 10  # Hz

# Behavioral detection parameters
turn_location_threshold = 0.3  # Electrode units - polarity flip without movement
hover_location_std = 0.1  # Stdev threshold for hovering detection
hover_min_duration_s = 1.0  # Minimum duration to classify as hovering
movement_velocity_threshold = 0.5  # m/s - sustained velocity for movement detection
movement_min_pulses = 3  # Minimum consecutive pulses for movement

# Track history
track_history_length = 10  # Number of recent pulses to store per track

# Output options
save_trajectory_plots = True
save_waveform_plots = True
save_tracking_metrics = True
plot_dpi = 150

print("✓ Parameters configured")

# =============================================================================
# FILE SELECTION
# =============================================================================

print("\n" + "="*70)
print("FILE SELECTION")
print("="*70)

# Set up GUI for folder selection
root = tk.Tk()
root.withdraw()

# Select input folder containing EOD tables and waveforms
print("\nSelect INPUT folder containing EOD tables and waveform files...")
input_folder = filedialog.askdirectory(title="Select Input Folder with EOD Data")

if not input_folder:
    raise ValueError("No input folder selected. Please restart and select a folder.")

print(f"✓ Input folder: {input_folder}")

# Select output folder
print("\nSelect OUTPUT folder for tracked results...")
output_folder = filedialog.askdirectory(title="Select Output Folder for Tracked Data")

if not output_folder:
    raise ValueError("No output folder selected. Please restart and select a folder.")

print(f"✓ Output folder: {output_folder}")

# Optional: Select species classifier
use_species_classification = messagebox.askyesno(
    "Species Classification",
    "Do you want to classify species using a pre-trained model?"
)

classifier_path = None
if use_species_classification:
    print("\nSelect pre-trained species classifier file (.pkl or .joblib)...")
    classifier_path = filedialog.askopenfilename(
        title="Select Species Classifier",
        filetypes=[("Model files", "*.pkl *.joblib"), ("All files", "*.*")]
    )
    if classifier_path:
        print(f"✓ Classifier: {os.path.basename(classifier_path)}")
    else:
        print("⚠ No classifier selected - proceeding without species classification")
        use_species_classification = False

print("\n" + "="*70)
print("CONFIGURATION SUMMARY")
print("="*70)
print(f"Input folder: {input_folder}")
print(f"Output folder: {output_folder}")
print(f"Species classification: {'Enabled' if use_species_classification else 'Disabled'}")
if use_species_classification and classifier_path:
    print(f"  Classifier: {os.path.basename(classifier_path)}")
print(f"\nTracking Parameters:")
print(f"  Seed pulses for initialization: {n_seed_pulses}")
print(f"  Waveform preprocessing: {waveform_crop_factor}× P-P duration → {waveform_target_length} samples")
print(f"  Max discharge rate: {max_discharge_rate_hz} Hz (min IPI: {min_ipi_ms:.2f} ms)")
print(f"  Track timeout: {track_timeout_s} s")
print(f"  Max fish velocity: {max_fish_velocity_m_s} m/s")
print(f"  Electrode spacing: {electrode_spacing_cm} cm")
print("="*70 + "\n")

# =============================================================================
# FIND AND LOAD DATA FILES
# =============================================================================

print("Searching for EOD data files...")

# Find all EOD table CSV files
eod_files = glob.glob(os.path.join(input_folder, "*_eod_table.csv"))

if not eod_files:
    raise ValueError(f"No EOD table files (*_eod_table.csv) found in {input_folder}")

print(f"✓ Found {len(eod_files)} EOD table file(s)")

# Find corresponding waveform NPZ files
waveform_files = glob.glob(os.path.join(input_folder, "*_waveforms.csv"))

if not waveform_files:
    raise ValueError(f"No waveform files (*_waveforms.npz) found in {input_folder}")

print(f"✓ Found {len(waveform_files)} waveform file(s)")

# Match EOD tables with waveform files
file_pairs = []
for eod_file in eod_files:
    base_name = os.path.basename(eod_file).replace("_eod_table.csv", "")
    waveform_file = os.path.join(input_folder, f"{base_name}_waveforms.csv")
    
    if os.path.exists(waveform_file):
        file_pairs.append({
            'base_name': base_name,
            'eod_file': eod_file,
            'waveform_file': waveform_file
        })
    else:
        print(f"⚠ Warning: No matching waveform file for {base_name}")

if not file_pairs:
    raise ValueError("No matching EOD table and waveform file pairs found")

print(f"\n✓ Matched {len(file_pairs)} EOD table + waveform file pair(s)")
print("\nFiles to process:")
for pair in file_pairs:
    print(f"  - {pair['base_name']}")

print("\n" + "="*70)
print("STARTING TRACKING ANALYSIS")
print("="*70)

# =============================================================================
# PROCESS EACH RECORDING
# =============================================================================

for file_idx, file_pair in enumerate(file_pairs):
    
    print(f"\n{'='*70}")
    print(f"PROCESSING FILE {file_idx + 1}/{len(file_pairs)}: {file_pair['base_name']}")
    print(f"{'='*70}")
    
    # Load EOD table
    print("\nLoading EOD table...")
    eod_data = pd.read_csv(file_pair['eod_file'])
    print(f"✓ Loaded {len(eod_data)} pulses")
    print(f"  Columns: {', '.join(eod_data.columns.tolist())}")
    
    # Sort by timestamp to ensure chronological processing
    eod_data = eod_data.sort_values('timestamp').reset_index(drop=True)
    print(f"✓ Sorted by timestamp")
    
    # Load waveforms
    print("\nLoading waveforms...")
    waveforms = load_fixed_length_waveforms(
        # strip .csv at end of filename
        file_pair['waveform_file'][:-4]
    )
    print(f"✓ Loaded {len(waveforms)} waveforms")
    
    # Verify indices match EOD table
    if len(waveforms) != len(eod_data):
        print(f"⚠ WARNING: Waveform count ({len(waveforms)}) doesn't match EOD table ({len(eod_data)})")
        print("  Skipping this file...")
        continue
    
    # Normalize waveforms for comparison (with adaptive crop and interpolation)
    print(f"\nNormalizing waveforms (crop {waveform_crop_factor}× P-P → interpolate to {waveform_target_length} samples)...")
    normalized_waveforms = normalize_waveforms(
        waveforms,
        snippet_p1_idc=eod_data['snippet_p1_idx'].values,
        snippet_p2_idc=eod_data['snippet_p2_idx'].values,
        method='p1_unity',  # Normalize P1 to unity for consistent comparison
        crop_and_interpolate=True,
        crop_factor=waveform_crop_factor,
        target_length=waveform_target_length
    )
    normalized_waveforms = np.array(normalized_waveforms)
    print(f"✓ Normalized and preprocessed {len(normalized_waveforms)} waveforms")
    print(f"  Waveform shape: {normalized_waveforms.shape}")
    
    # Initialize tracking data structures
    print("\nInitializing tracking data structures...")
    
    # Add fish_id column (initialize to -1 = unassigned)
    eod_data['fish_id'] = -1
    
    # Track management
    active_tracks = {}  # fish_id -> track_info dict
    inactive_tracks = {}  # fish_id -> track_info dict (for potential reactivation)
    next_fish_id = 0
    
    # Behavioral events
    behavioral_events = []
    
    print("✓ Ready to start tracking")
    
    # =============================================================================
    # STEP 2: INITIALIZE TRACKS WITH FIRST N PULSES
    # =============================================================================
    
    print(f"\n{'='*70}")
    print(f"STEP 2: INITIALIZING TRACKS (First {n_seed_pulses} pulses)")
    print(f"{'='*70}")
    
    # Determine actual number of seed pulses (might be less if recording is short)
    n_actual_seed = min(n_seed_pulses, len(eod_data))
    print(f"\nUsing {n_actual_seed} pulses for initialization")
    
    if n_actual_seed < 2:
        print("⚠ WARNING: Too few pulses for initialization - need at least 2 pulses")
        print("  Skipping this file...")
        continue
    
    # Extract seed waveforms
    seed_waveforms = normalized_waveforms[:n_actual_seed]
    
    # Compute pairwise distances
    print("\nComputing pairwise waveform distances...")
    distance_matrix = pairwise_distances(seed_waveforms, metric='euclidean')
    
    # Get upper triangle distances for eps calculation
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    
    if len(distances) == 0:
        print("⚠ WARNING: Cannot compute distances - only 1 seed pulse")
        # Assign single fish ID
        eod_data.loc[0, 'fish_id'] = 0
        active_tracks[0] = {
            'fish_id': 0,
            'last_pulse_idx': 0,
            'last_timestamp': eod_data.loc[0, 'timestamp'],
            'history': [0],
            'waveform_signature': seed_waveforms[0],
            'created_at': eod_data.loc[0, 'timestamp']
        }
        next_fish_id = 1
        print("✓ Created single track: Fish 0")
    else:
        # Adaptive eps based on distance distribution
        eps = np.percentile(distances, dbscan_eps_percentile)
        print(f"✓ Computed {len(distances)} pairwise distances")
        print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
        print(f"  Adaptive eps ({dbscan_eps_percentile}th percentile): {eps:.3f}")
        
        # DBSCAN clustering
        print(f"\nRunning DBSCAN clustering (eps={eps:.3f}, min_samples={dbscan_min_samples})...")
        dbscan = DBSCAN(eps=eps, min_samples=dbscan_min_samples, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"✓ Clustering complete")
        print(f"  Number of fish detected: {n_clusters}")
        print(f"  Noise/outlier pulses: {n_noise}")
        
        # Assign initial fish IDs
        print("\nAssigning initial fish IDs...")
        for seed_idx, cluster_label in enumerate(cluster_labels):
            if cluster_label == -1:
                # Noise - assign unique fish ID (each noise pulse is its own fish for now)
                fish_id = next_fish_id
                next_fish_id += 1
            else:
                # Cluster - use cluster label as fish ID
                fish_id = cluster_label
                if cluster_label >= next_fish_id:
                    next_fish_id = cluster_label + 1
            
            eod_data.loc[seed_idx, 'fish_id'] = fish_id
            
            # Create or update track
            if fish_id not in active_tracks:
                active_tracks[fish_id] = {
                    'fish_id': fish_id,
                    'last_pulse_idx': seed_idx,
                    'last_timestamp': eod_data.loc[seed_idx, 'timestamp'],
                    'history': [seed_idx],
                    'waveform_signatures': [seed_waveforms[seed_idx]],
                    'created_at': eod_data.loc[seed_idx, 'timestamp']
                }
            else:
                # Add to existing track
                active_tracks[fish_id]['last_pulse_idx'] = seed_idx
                active_tracks[fish_id]['last_timestamp'] = eod_data.loc[seed_idx, 'timestamp']
                active_tracks[fish_id]['history'].append(seed_idx)
                active_tracks[fish_id]['waveform_signatures'].append(seed_waveforms[seed_idx])
        
        print(f"✓ Initialized {len(active_tracks)} track(s)")
        
        # Compute representative waveform for each track (median)
        for fish_id, track in active_tracks.items():
            track_waveforms = np.array(track['waveform_signatures'])
            track['waveform_signature'] = np.median(track_waveforms, axis=0)
            n_pulses = len(track['history'])
            print(f"  Fish {fish_id}: {n_pulses} pulse(s)")
    
    # Store waveform distance threshold for matching
    waveform_distance_threshold = eps * waveform_distance_threshold_multiplier if 'eps' in locals() else 0.5
    print(f"\nWaveform distance threshold for matching: {waveform_distance_threshold:.3f}")
    
    print(f"\n✓ Track initialization complete")
    print(f"  Active tracks: {len(active_tracks)}")
    print(f"  Next fish ID: {next_fish_id}")
    
    # TODO: Continue with step 3 - sequential pulse processing
    
    print(f"\n⚠ Implementation in progress - stopping after initialization step")
    print(f"  Processed {n_actual_seed}/{len(eod_data)} pulses")
    break  # Stop after first file for now

print("\n" + "="*70)
print("TRACKING ANALYSIS PAUSED (PARTIAL IMPLEMENTATION)")
print("="*70)
print("\nNext steps:")
print("  - Implement sequential pulse-by-pulse matching (Step 3)")
print("  - Add spatiotemporal consistency checks (Step 4)")
print("  - Add behavioral detection (Step 5)")
print("  - Add species classification integration (Step 6)")
print("  - Add diagnostic outputs (Step 7)")
