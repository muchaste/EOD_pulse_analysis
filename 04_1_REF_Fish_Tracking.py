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
import audioio as aio

# Import EOD functions
from pulse_functions import (
    load_fixed_length_waveforms,
    normalize_waveforms,
    create_tracking_plot
)

warnings.filterwarnings('ignore')

print("="*70)
print("SEQUENTIAL FISH TRACKING ANALYSIS")
print("="*70)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_track_state(amplitude_history, polarity_history, window_size=5):
    """
    Detect the current state of a fish track based on amplitude trajectory and polarity.
    
    States:
    - 'fade_in': Amplitude increasing (fish entering detection zone)
    - 'stable': Amplitude relatively constant (fish in optimal zone)
    - 'turning': Amplitude decrease + polarity flip + recovery (fish rotating)
    - 'fade_out': Amplitude decreasing (fish leaving detection zone)
    - 'unknown': Insufficient data
    
    Parameters:
    -----------
    amplitude_history : list or array
        Recent amplitude values (most recent last)
    polarity_history : list or array
        Recent polarity values (most recent last)
    window_size : int
        Number of recent pulses to analyze
        
    Returns:
    --------
    state : str
        Detected track state
    confidence : float
        Confidence in state detection (0-1)
    """
    
    if len(amplitude_history) < 2:
        return 'unknown', 0.0
    
    # Get recent window
    recent_amps = amplitude_history[-window_size:] if len(amplitude_history) >= window_size else amplitude_history
    recent_pols = polarity_history[-window_size:] if len(polarity_history) >= window_size else polarity_history
    
    n_samples = len(recent_amps)
    
    if n_samples < 2:
        return 'unknown', 0.0
    
    # Calculate amplitude trend (linear fit slope)
    x = np.arange(n_samples)
    if n_samples >= 3:
        amp_slope = np.polyfit(x, recent_amps, 1)[0]
    else:
        amp_slope = (recent_amps[-1] - recent_amps[0]) / (n_samples - 1)
    
    # Calculate amplitude variability
    amp_mean = np.mean(recent_amps)
    amp_std = np.std(recent_amps)
    amp_cv = amp_std / amp_mean if amp_mean > 0 else np.inf
    
    # Detect polarity changes
    if n_samples >= 2:
        polarity_changes = np.sum([recent_pols[i] != recent_pols[i-1] for i in range(1, len(recent_pols))])
    else:
        polarity_changes = 0
    
    # STATE DETECTION LOGIC
    
    # Check for turning pattern: polarity flip + amplitude dip + recovery
    if polarity_changes > 0 and n_samples >= 3:
        # Find amplitude minimum position
        min_idx = np.argmin(recent_amps)
        min_amp = recent_amps[min_idx]
        
        # Check if amplitude dropped then recovered
        if min_idx > 0 and min_idx < n_samples - 1:
            # Amplitude before and after minimum
            before_amp = np.mean(recent_amps[:min_idx])
            after_amp = np.mean(recent_amps[min_idx+1:])
            
            # Turning if: dip to <60% and partial recovery
            dip_ratio = min_amp / before_amp if before_amp > 0 else 1.0
            recovery_ratio = after_amp / before_amp if before_amp > 0 else 1.0
            
            if dip_ratio < turn_amplitude_dip_factor and recovery_ratio > 0.6:
                confidence = min(1.0, 2.0 * (1.0 - dip_ratio))  # Higher confidence for deeper dips
                return 'turning', confidence
    
    # Normalize slope by mean amplitude (relative change)
    relative_slope = amp_slope / amp_mean if amp_mean > 0 else 0
    
    # Check for fade-in: increasing amplitude
    if relative_slope > fade_amplitude_slope_threshold:
        confidence = min(1.0, abs(relative_slope) / (2 * fade_amplitude_slope_threshold))
        return 'fade_in', confidence
    
    # Check for fade-out: decreasing amplitude
    if relative_slope < -fade_amplitude_slope_threshold:
        confidence = min(1.0, abs(relative_slope) / (2 * fade_amplitude_slope_threshold))
        return 'fade_out', confidence
    
    # Check for stable: low variability
    if amp_cv < stable_amplitude_cv:
        confidence = min(1.0, (stable_amplitude_cv - amp_cv) / stable_amplitude_cv)
        return 'stable', confidence
    
    # Default: stable with low confidence
    return 'stable', 0.3

# =============================================================================
# TRACKING PARAMETERS
# =============================================================================

print("\nSetting tracking parameters...")

# Waveform preprocessing parameters
waveform_crop_factor = 4  # Crop window size as multiple of peak-peak duration
waveform_target_length = 200  # Target number of samples after interpolation

# Track initialization parameters (spatiotemporal-first approach)
n_seed_pulses = 20  # Number of initial pulses to use for track initialization
spatial_grouping_threshold = 0.5  # Electrode units - pulses within this distance are spatially compatible
channel_adjacency = 1  # Consider channels within this range as compatible (0=same channel only, 1=adjacent allowed)
ipi_regularity_threshold = 0.6  # Coefficient of variation for IPI - lower = more regular discharge pattern
min_pulses_per_track = 2  # Minimum pulses to form a track during initialization

# Temporal constraints
max_discharge_rate_hz = 150  # Maximum EOD rate (minimum IPI = 6.67ms)
min_ipi_ms = 1000.0 / max_discharge_rate_hz
track_timeout_s = 5.0  # Mark track inactive after this gap

# Spatial constraints
electrode_spacing_cm = 50.0  # Distance between electrodes
max_fish_velocity_m_s = 1.0  # Maximum swimming speed
max_spatial_jump = max_fish_velocity_m_s * 100 / electrode_spacing_cm  # In electrode units per second

# Predictive tracking parameters
prediction_history_n = 10  # Number of recent pulses to use for state prediction
min_prediction_history = 3  # Minimum pulses needed for linear predictions
location_std_min = 0.05  # Minimum location std for normalization (electrode units)
amplitude_tolerance_relative = 0.3  # Relative tolerance for amplitude residuals (30%)
ipi_flexibility_factor = 0.5  # Minimum IPI variability tolerance (50% of median)
ipi_std_multiplier = 2.0  # Standard deviation multiplier for IPI tolerance

# Feature normalization parameters
width_tolerance_us = 20  # Width tolerance for normalization (microseconds)

# Amplitude trajectory and state detection parameters
fade_detection_window = 5  # Number of pulses to analyze for amplitude trends
fade_amplitude_slope_threshold = 0.15  # Minimum amplitude slope (units/pulse) for fade detection
stable_amplitude_cv = 0.3  # CV threshold - below this = stable amplitude
turn_amplitude_dip_factor = 0.6  # During turns, amplitude can drop to 60% before recovery
turn_recovery_threshold = 0.8  # After turn, amplitude should recover to 80% of pre-turn level
polarity_flip_ipi_multiplier = 10  # IPI can be 10x longer during polarity flips

# New track creation criteria (prevent spurious tracks from strong pulses)
new_track_amplitude_percentile = 40  # Only pulses weaker than this percentile can start tracks
new_track_fade_in_bonus = 0.5  # Score bonus (added to threshold) for potential fade-in tracks

# Scoring parameters (lower score = better match)
score_threshold = 2.5  # Maximum normalized score for accepting pulse-track match
# State-dependent threshold multipliers:
fade_in_threshold_multiplier = 1.8  # Relaxed matching during fade-in (fish entering)
stable_threshold_multiplier = 1.5  # Standard matching for stable fish
turning_threshold_multiplier = 2.5  # Very relaxed during turns (amplitude dip + polarity flip)
fade_out_threshold_multiplier = 1.5  # Relaxed during fade-out (fish leaving)
# Scoring weights (must sum to 1.0):
# - location_weight = 0.50 (predicted location residual - physics-based)
# - amplitude_weight = 0.20 (predicted amplitude residual - gradual changes)
# - waveform_weight = 0.15 (SNR-weighted waveform distance - invariant)
# - ipi_weight = 0.10 (IPI residual - behavioral flexibility)
# - width_weight = 0.05 (width difference - validation only)

# Behavioral detection parameters
turn_location_threshold = 0.3  # Electrode units - polarity flip without movement
hover_location_std = 0.1  # Stdev threshold for hovering detection
hover_min_duration_s = 1.0  # Minimum duration to classify as hovering
movement_velocity_threshold = 0.5  # m/s - sustained velocity for movement detection
movement_min_pulses = 3  # Minimum consecutive pulses for movement

# Track history management
track_history_max = 100  # Maximum number of pulses to keep in track history (for memory management)
track_recent_n = 10  # Number of recent pulses for rolling median waveform and spatial statistics

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
print(f"  Initialization: {n_seed_pulses} seed pulses (spatiotemporal-first)")
print(f"  Waveform preprocessing: {waveform_crop_factor}× P-P duration → {waveform_target_length} samples")
print(f"\n  Temporal constraints:")
print(f"    Max discharge rate: {max_discharge_rate_hz} Hz (min IPI: {min_ipi_ms:.2f} ms)")
print(f"    Track timeout: {track_timeout_s} s")
print(f"\n  Spatial constraints:")
print(f"    Electrode spacing: {electrode_spacing_cm} cm")
print(f"    Max fish velocity: {max_fish_velocity_m_s} m/s")
print(f"\n  Predictive tracking:")
print(f"    History for predictions: {prediction_history_n} pulses (min {min_prediction_history})")
print(f"    IPI flexibility: ±{int(ipi_flexibility_factor*100)}% or {ipi_std_multiplier}×std")
print(f"    Amplitude tolerance: ±{int(amplitude_tolerance_relative*100)}%")
print(f"    Score threshold: {score_threshold} (normalized)")
print(f"\n  State-based tracking:")
print(f"    Fade detection window: {fade_detection_window} pulses")
print(f"    Turn amplitude dip: {int(turn_amplitude_dip_factor*100)}% minimum")
print(f"    Threshold multipliers: fade-in={fade_in_threshold_multiplier}×, turning={turning_threshold_multiplier}×, fade-out={fade_out_threshold_multiplier}×")
print(f"\n  Track history: rolling {track_recent_n} pulses, max {track_history_max} stored")
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

# Find corresponding waveform files
waveform_files = glob.glob(os.path.join(input_folder, "*_waveforms.csv"))

if not waveform_files:
    raise ValueError(f"No waveform files (*_waveforms.csv) found in {input_folder}")
print(f"✓ Found {len(waveform_files)} waveform file(s)")

# Find corresponding .wav files for diagnostic plots
audio_files = glob.glob(os.path.join(input_folder, "*.wav"))
if not audio_files:
    raise ValueError(f"No .wav audio files found in {input_folder} for diagnostics")
print(f"✓ Found {len(audio_files)} .wav audio file(s) for diagnostics")

# Match EOD tables with waveform and audio files
file_sets = []
for eod_file in eod_files:
    base_name = os.path.basename(eod_file).replace("_eod_table.csv", "")
    event_name = '_'.join(base_name.split('_')[1:])
    waveform_file = os.path.join(input_folder, f"{base_name}_waveforms.csv")
    audio_file = os.path.join(input_folder, f"{base_name}.wav")
    
    if os.path.exists(waveform_file):
        file_sets.append({
            'base_name': base_name,
            'event_name': event_name,
            'eod_file': eod_file,
            'waveform_file': waveform_file,
            'audio_file': audio_file
        })
    else:
        print(f"⚠ Warning: No matching waveform file for {base_name}")

if not file_sets:
    raise ValueError("No matching EOD table, waveform and audio file sets found")

print(f"\n✓ Matched {len(file_sets)} EOD table + waveform + audio file set(s)")
print("\nFiles to process:")
for pair in file_sets:
    print(f"  - {pair['base_name']}")

print("\n" + "="*70)
print("STARTING TRACKING ANALYSIS")
print("="*70)
#%%
# =============================================================================
# PROCESS EACH RECORDING
# =============================================================================

for file_idx, file_set in enumerate(file_sets):
    
    print(f"\n{'='*70}")
    print(f"PROCESSING FILE {file_idx + 1}/{len(file_sets)}: {file_set['base_name']}")
    print(f"{'='*70}")
    
    # Load EOD table
    print("\nLoading EOD table...")
    eod_data = pd.read_csv(file_set['eod_file'])
    print(f"✓ Loaded {len(eod_data)} pulses")
    print(f"  Columns: {', '.join(eod_data.columns.tolist())}")

    # Convert timestamp column to datetime
    eod_data['timestamp'] = pd.to_datetime(eod_data['timestamp'])
    eod_data['file_timestamp'] = pd.to_datetime(eod_data['file_timestamp'])
    
    # Sort by timestamp to ensure chronological processing
    eod_data = eod_data.sort_values('timestamp').reset_index(drop=True)
    print(f"✓ Sorted by timestamp")
    
    # Load waveforms
    print("\nLoading waveforms...")
    waveforms = load_fixed_length_waveforms(
        # strip .csv at end of filename
        file_set['waveform_file'][:-4]
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

    # Load audio file path
    audio_file = file_set['audio_file']
    print(f"\nAudio file for diagnostics: {audio_file}")
    audio_data, audio_fs = aio.load_audio(audio_file)
    if audio_data is None:
        print(f"⚠ WARNING: Unable to load audio file {audio_file}")
        print("  Skipping this file...")
        continue
    
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
    # STEP 2: INITIALIZE TRACKS WITH FIRST N PULSES (SPATIOTEMPORAL-FIRST)
    # =============================================================================
    
    print(f"\n{'='*70}")
    print(f"STEP 2: INITIALIZING TRACKS - SPATIOTEMPORAL GROUPING (First {n_seed_pulses} pulses)")
    print(f"{'='*70}")
    
    # Determine actual number of seed pulses (might be less if recording is short)
    n_actual_seed = min(n_seed_pulses, len(eod_data))
    print(f"\nUsing {n_actual_seed} pulses for initialization")
    
    if n_actual_seed < 1:
        print("⚠ WARNING: No pulses available for initialization")
        print("  Skipping this file...")
        continue
    
    # Extract seed data
    seed_data = eod_data.iloc[:n_actual_seed].copy()
    seed_waveforms = normalized_waveforms[:n_actual_seed]
    
    # STAGE 1: SPATIOTEMPORAL GROUPING
    print("\n--- Stage 1: Spatiotemporal Grouping ---")
    print(f"Parameters: spatial threshold = {spatial_grouping_threshold} electrode units")
    print(f"            channel adjacency = ±{channel_adjacency} channels")
    print(f"            IPI regularity CV threshold = {ipi_regularity_threshold}")
    
    # Initialize groups
    pulse_groups = []  # Each group is a list of pulse indices
    pulse_assigned = np.zeros(n_actual_seed, dtype=bool)
    
    # Sort by channel and location for efficient grouping
    seed_data_sorted = seed_data.sort_values(['eod_channel', 'pulse_location']).copy()
    sorted_indices = seed_data_sorted.index.tolist()
    
    print(f"\nGrouping pulses by spatial proximity...")
    
    for i, idx in enumerate(sorted_indices):
        if pulse_assigned[idx]:
            continue  # Already assigned to a group
        
        # Start a new group with this pulse
        current_group = [idx]
        pulse_assigned[idx] = True
        
        ref_channel = seed_data.loc[idx, 'eod_channel']
        ref_location = seed_data.loc[idx, 'pulse_location']
        
        # Find spatially compatible pulses
        for j, other_idx in enumerate(sorted_indices[i+1:], start=i+1):
            if pulse_assigned[other_idx]:
                continue
            
            other_channel = seed_data.loc[other_idx, 'eod_channel']
            other_location = seed_data.loc[other_idx, 'pulse_location']
            
            # Check channel compatibility
            channel_diff = abs(other_channel - ref_channel)
            if channel_diff > channel_adjacency:
                # Pulses are sorted by channel, so if we exceed adjacency, stop searching
                if other_channel > ref_channel + channel_adjacency:
                    break
                continue
            
            # Check spatial proximity
            location_diff = abs(other_location - ref_location)
            if location_diff <= spatial_grouping_threshold:
                current_group.append(other_idx)
                pulse_assigned[other_idx] = True
        
        if len(current_group) >= min_pulses_per_track:
            pulse_groups.append(current_group)
        else:
            # Too few pulses - treat as individual tracks for now
            for pulse_idx in current_group:
                pulse_groups.append([pulse_idx])
    
    print(f"✓ Created {len(pulse_groups)} spatial group(s)")
    
    # STAGE 2: TEMPORAL CONSISTENCY CHECK
    print("\n--- Stage 2: Temporal Consistency Verification ---")
    
    final_tracks = []
    
    for group_idx, group in enumerate(pulse_groups):
        if len(group) < 2:
            # Single pulse - accept as is
            final_tracks.append(group)
            continue
        
        # Sort group by timestamp
        group_sorted = sorted(group, key=lambda idx: seed_data.loc[idx, 'timestamp'])
        group_timestamps = [seed_data.loc[idx, 'timestamp'] for idx in group_sorted]
        
        # Calculate inter-pulse intervals (IPIs) - timestamps are datetime strings -> convert to datetime
        ipis = np.diff(group_timestamps).astype('timedelta64[ms]').astype(float) / 1000.0  # in seconds
        
        # Check temporal regularity
        if len(ipis) > 0:
            mean_ipi = np.mean(ipis)
            std_ipi = np.std(ipis)
            cv_ipi = std_ipi / mean_ipi if mean_ipi > 0 else np.inf
            
            # Check if IPIs are reasonable (not too fast)
            min_ipi_s = min_ipi_ms / 1000.0
            valid_ipis = ipis >= min_ipi_s
            
            if cv_ipi <= ipi_regularity_threshold and np.all(valid_ipis):
                # Regular discharge pattern - keep as single track
                final_tracks.append(group_sorted)
                mean_rate = 1.0 / mean_ipi if mean_ipi > 0 else 0
                print(f"  Group {group_idx}: {len(group)} pulses, mean rate = {mean_rate:.1f} Hz, CV = {cv_ipi:.3f} ✓")
            else:
                # Irregular pattern - might be multiple fish or noisy data
                # Split by large gaps (>2x median IPI)
                median_ipi = np.median(ipis)
                gap_threshold = 2.0 * median_ipi
                
                current_subgroup = [group_sorted[0]]
                for i, ipi in enumerate(ipis):
                    if ipi > gap_threshold or ipi < min_ipi_s:
                        # Large gap or too-fast discharge - start new subgroup
                        if len(current_subgroup) >= min_pulses_per_track:
                            final_tracks.append(current_subgroup)
                        else:
                            # Add singles as individual tracks
                            for pulse_idx in current_subgroup:
                                final_tracks.append([pulse_idx])
                        current_subgroup = [group_sorted[i+1]]
                    else:
                        current_subgroup.append(group_sorted[i+1])
                
                # Add final subgroup
                if len(current_subgroup) >= min_pulses_per_track:
                    final_tracks.append(current_subgroup)
                else:
                    for pulse_idx in current_subgroup:
                        final_tracks.append([pulse_idx])
                
                print(f"  Group {group_idx}: irregular (CV = {cv_ipi:.3f}) → split into {len(final_tracks) - group_idx} subgroup(s)")
        else:
            # No IPIs to check (single pulse after filtering)
            final_tracks.append(group_sorted)
    
    print(f"\n✓ Final track count after temporal verification: {len(final_tracks)}")
    
    # STAGE 3: CREATE TRACK OBJECTS
    print("\n--- Stage 3: Creating Track Objects ---")
    
    for track_idx, track_indices in enumerate(final_tracks):
        fish_id = next_fish_id
        next_fish_id += 1
        
        # Assign fish_id to all pulses in track
        for pulse_idx in track_indices:
            eod_data.loc[pulse_idx, 'fish_id'] = fish_id
        
        # Compute representative waveform (median)
        track_waveforms = seed_waveforms[track_indices]
        median_waveform = np.median(track_waveforms, axis=0)
        
        # Get track statistics
        track_channels = seed_data.loc[track_indices, 'eod_channel'].values
        track_locations = seed_data.loc[track_indices, 'pulse_location'].values
        track_timestamps = seed_data.loc[track_indices, 'timestamp'].values
        track_amplitudes = seed_data.loc[track_indices, 'eod_amplitude'].values
        track_polarities = seed_data.loc[track_indices, 'pulse_orientation'].values
        
        last_idx = track_indices[-1]
        
        # Create track object
        active_tracks[fish_id] = {
            'fish_id': fish_id,
            'last_pulse_idx': last_idx,
            'last_timestamp': seed_data.loc[last_idx, 'timestamp'],
            'last_location': seed_data.loc[last_idx, 'pulse_location'],
            'last_channel': seed_data.loc[last_idx, 'eod_channel'],
            'last_polarity': seed_data.loc[last_idx, 'pulse_orientation'],
            'history': list(track_indices),
            'waveform_signatures': list(track_waveforms),
            'waveform_signature': median_waveform,
            'created_at': seed_data.loc[track_indices[0], 'timestamp'],
            'mean_location': np.mean(track_locations),
            'location_std': np.std(track_locations),
            'amplitude_history': list(track_amplitudes),
            'polarity_history': list(track_polarities),
            'max_amplitude_seen': np.max(track_amplitudes),
            'track_state': 'unknown',
            'state_confidence': 0.0,
            'state_duration': 0
        }
        
        # Print track summary
        mean_channel = np.mean(track_channels)
        mean_location = np.mean(track_locations)
        if len(track_timestamps) > 1:
            ipis = np.diff(track_timestamps).astype('timedelta64[ms]').astype(float) / 1000.0  # in seconds
            mean_rate = 1.0 / np.mean(ipis) if len(ipis) > 0 else 0
            print(f"  Fish {fish_id}: {len(track_indices)} pulses, ch {mean_channel:.1f}, loc {mean_location:.2f}, rate {mean_rate:.1f} Hz")
        else:
            print(f"  Fish {fish_id}: {len(track_indices)} pulse(s), ch {mean_channel:.1f}, loc {mean_location:.2f}")
    
    # Compute representative waveform distance threshold for sequential matching
    # Use median of within-track waveform distances
    print("\nComputing waveform distance threshold for sequential matching...")
    within_track_distances = []
    for fish_id, track in active_tracks.items():
        if len(track['waveform_signatures']) > 1:
            track_waveforms_array = np.array(track['waveform_signatures'])
            track_distances = pairwise_distances(track_waveforms_array, metric='euclidean')
            track_distances_upper = track_distances[np.triu_indices_from(track_distances, k=1)]
            within_track_distances.extend(track_distances_upper)
    
    if len(within_track_distances) > 0:
        waveform_distance_threshold = np.percentile(within_track_distances, 75)  # 75th percentile of within-track distances
        print(f"  Within-track distance range: [{np.min(within_track_distances):.3f}, {np.max(within_track_distances):.3f}]")
        print(f"  Threshold (75th percentile): {waveform_distance_threshold:.3f}")
    else:
        # Fallback: use default threshold
        waveform_distance_threshold = 1.0
        print(f"  No multi-pulse tracks - using default threshold: {waveform_distance_threshold:.3f}")
    
    print(f"\n✓ Track initialization complete")
    print(f"  Active tracks: {len(active_tracks)}")
    print(f"  Next fish ID: {next_fish_id}")
    
    # =============================================================================
    # STEP 3: SEQUENTIAL PULSE-BY-PULSE MATCHING
    # =============================================================================
    
    print(f"\n{'='*70}")
    print(f"STEP 3: SEQUENTIAL PULSE MATCHING (Remaining {len(eod_data) - n_actual_seed} pulses)")
    print(f"{'='*70}")
    
    # Process remaining pulses sequentially
    n_assigned = 0
    n_new_tracks = 0
    n_unassigned = 0
    
    for pulse_idx in range(n_actual_seed, len(eod_data)):
        # Progress indicator every 100 pulses
        if (pulse_idx - n_actual_seed) % 100 == 0 and pulse_idx > n_actual_seed:
            print(f"  Processed {pulse_idx - n_actual_seed}/{len(eod_data) - n_actual_seed} pulses...", end='\r')
        
        # Get pulse features
        pulse_timestamp = eod_data.loc[pulse_idx, 'timestamp']
        pulse_location = eod_data.loc[pulse_idx, 'pulse_location']
        pulse_channel = eod_data.loc[pulse_idx, 'eod_channel']
        pulse_waveform = normalized_waveforms[pulse_idx]
        pulse_width = eod_data.loc[pulse_idx, 'eod_width_us']
        pulse_freq = eod_data.loc[pulse_idx, 'fft_freq_max']
        pulse_amplitude_ratio = eod_data.loc[pulse_idx, 'eod_amplitude_ratio']
        pulse_polarity = eod_data.loc[pulse_idx, 'pulse_orientation']
        
        # PHASE 1: Check for inactive tracks (timeout management)
        tracks_to_deactivate = []
        for fish_id, track in active_tracks.items():
            time_since_last = (pulse_timestamp - track['last_timestamp']).total_seconds()
            if time_since_last > track_timeout_s:
                tracks_to_deactivate.append(fish_id)
        
        # Move timed-out tracks to inactive
        for fish_id in tracks_to_deactivate:
            inactive_tracks[fish_id] = active_tracks[fish_id]
            del active_tracks[fish_id]
        
        if len(tracks_to_deactivate) > 0:
            print(f"\n  t={pulse_timestamp:.2f}s: Deactivated {len(tracks_to_deactivate)} track(s) due to timeout")
        
        # PHASE 2: Find compatible tracks (temporal + spatial gating)
        candidate_tracks = []
        
        for fish_id, track in active_tracks.items():
            # Temporal constraint: check IPI
            time_since_last = (pulse_timestamp - track['last_timestamp']).total_seconds()
            min_ipi_s = min_ipi_ms / 1000.0
            
            if time_since_last < min_ipi_s:
                # Too fast - physiologically impossible
                continue
            
            # Spatial constraint: check velocity limit
            location_diff = abs(pulse_location - track['last_location'])
            max_displacement = max_spatial_jump * time_since_last  # max_spatial_jump is in electrode units per second
            
            if location_diff > max_displacement:
                # Moved too far too fast
                continue
            
            # Channel constraint: check if spatially compatible
            channel_diff = abs(pulse_channel - track['last_channel'])
            if channel_diff > channel_adjacency:
                # Too far away in channel space
                continue
            
            # Track passes gating criteria - add as candidate
            candidate_tracks.append(fish_id)
        
        # PHASE 3: Score candidates and find best match
        best_match = None
        best_score = np.inf
        match_info = {}
        
        for fish_id in candidate_tracks:
            track = active_tracks[fish_id]
            
            # Get track history for predictions
            recent_indices = track['history'][-prediction_history_n:]
            n_recent = len(recent_indices)
            
            # === TRACK STATE DETECTION ===
            # Detect current track state based on amplitude trajectory and polarity
            track_state, state_confidence = detect_track_state(
                track['amplitude_history'],
                track['polarity_history'],
                window_size=fade_detection_window
            )
            
            # Update track state tracking
            if track_state != track.get('track_state', 'unknown'):
                # State changed - reset duration
                track['track_state'] = track_state
                track['state_confidence'] = state_confidence
                track['state_duration'] = 1
            else:
                # Same state - increment duration
                track['state_duration'] = track.get('state_duration', 0) + 1
            
            # === PREDICTIVE TRACKING: Extrapolate expected state from history ===
            
            # 1. LOCATION PREDICTION (Linear velocity model)
            recent_locations = eod_data.loc[recent_indices, 'pulse_location'].values
            recent_timestamps = [eod_data.loc[idx, 'timestamp'] for idx in recent_indices]
            recent_times = [(t - recent_timestamps[0]).total_seconds() for t in recent_timestamps]
            
            if n_recent >= min_prediction_history:
                # Fit linear trend to estimate velocity
                velocity = np.polyfit(recent_times, recent_locations, 1)[0]  # electrode units per second
                time_since_last = (pulse_timestamp - track['last_timestamp']).total_seconds()
                predicted_location = track['last_location'] + velocity * time_since_last
            else:
                # Insufficient history - use last known location
                predicted_location = track['last_location']
                velocity = 0.0
            
            location_residual = abs(pulse_location - predicted_location)
            location_std = track.get('location_std', 0.1)
            location_residual_normalized = location_residual / max(location_std, location_std_min)
            
            # 2. IPI PREDICTION (Flexible - fish can vary discharge rate)
            # Use median IPI with tolerance for variability (fish behavior is context-dependent)
            if n_recent >= 2:
                recent_ipis = np.diff(recent_timestamps).astype('timedelta64[ms]').astype(float) / 1000.0
                median_ipi = np.median(recent_ipis)
                ipi_std = np.std(recent_ipis)
                
                # Allow wide tolerance since IPI can vary with behavior
                ipi_tolerance = max(ipi_std * ipi_std_multiplier, median_ipi * ipi_flexibility_factor)
                ipi_residual = abs(time_since_last - median_ipi)
                ipi_residual_normalized = ipi_residual / ipi_tolerance if ipi_tolerance > 0 else 0
            else:
                ipi_residual_normalized = 0
                median_ipi = time_since_last
            
            # 3. AMPLITUDE PREDICTION (Gradual changes with distance/orientation)
            recent_amplitudes = eod_data.loc[recent_indices, 'eod_amplitude'].values
            
            if n_recent >= min_prediction_history:
                # Fit linear trend to detect gradual amplitude changes
                amp_trend = np.polyfit(recent_times, recent_amplitudes, 1)[0]
                predicted_amplitude = recent_amplitudes[-1] + amp_trend * time_since_last
            else:
                predicted_amplitude = np.mean(recent_amplitudes)
            
            pulse_amplitude = eod_data.loc[pulse_idx, 'eod_amplitude']
            amplitude_residual = abs(pulse_amplitude - predicted_amplitude)
            amplitude_residual_normalized = amplitude_residual / max(predicted_amplitude * amplitude_tolerance_relative, 0.001)
            
            # 4. WAVEFORM DISTANCE (SNR-weighted - trust high-amplitude pulses more)
            waveform_distance = np.linalg.norm(pulse_waveform - track['waveform_signature'])
            
            # SNR weighting: high amplitude = more reliable waveform
            # Low amplitude pulses have more relative noise -> downweight waveform importance
            median_track_amplitude = np.median(recent_amplitudes)
            snr_weight = min(pulse_amplitude / max(median_track_amplitude, 0.001), 2.0)
            snr_weight = (snr_weight - 0.5) / 1.5  # Normalize to [0, 1] range (0.5->0, 2.0->1.0)
            snr_weight = max(0.0, min(1.0, snr_weight))  # Clamp to [0, 1]
            
            # State-based waveform weighting
            waveform_weight_factor = (0.5 + 0.5 * snr_weight)
            
            # During turns: heavily reduce waveform weight (waveform inverted/distorted)
            if track_state == 'turning':
                waveform_weight_factor *= 0.3  # Reduce to 30%
            # During fade-in/out: moderately reduce waveform weight (amplitude changing)
            elif track_state in ['fade_in', 'fade_out']:
                waveform_weight_factor *= 0.7  # Reduce to 70%
            
            waveform_score = waveform_distance / waveform_distance_threshold * waveform_weight_factor
            
            # 5. INVARIANT FEATURES (Should be constant - use for validation)
            track_widths = eod_data.loc[recent_indices, 'eod_width_us'].values
            track_freqs = eod_data.loc[recent_indices, 'fft_freq_max'].values
            
            mean_width = np.mean(track_widths)
            mean_freq = np.mean(track_freqs)
            
            width_diff = abs(pulse_width - mean_width)
            freq_diff = abs(pulse_freq - mean_freq)
            
            # === PREDICTIVE SCORING (lower is better) ===
            # Prioritize physics-based predictions over invariant features
            score = (0.50 * location_residual_normalized +    # HIGHEST - predicted location (physics)
                     0.20 * amplitude_residual_normalized +   # SECOND - predicted amplitude (gradual change)
                     0.15 * waveform_score +                  # SNR-weighted waveform (invariant but noisy)
                     0.10 * ipi_residual_normalized +         # IPI flexible (behavioral)
                     0.05 * (width_diff / width_tolerance_us))  # Validation only
            
            match_info[fish_id] = {
                'score': score,
                'location_residual': location_residual,
                'predicted_location': predicted_location,
                'velocity': velocity,
                'amplitude_residual': amplitude_residual,
                'predicted_amplitude': predicted_amplitude,
                'waveform_distance': waveform_distance,
                'snr_weight': snr_weight,
                'ipi_residual': time_since_last - median_ipi if n_recent >= 2 else 0,
                'median_ipi': median_ipi if n_recent >= 2 else 0,
                'track_state': track_state,
                'state_confidence': state_confidence
            }
            
            if score < best_score:
                best_score = score
                best_match = fish_id
        
        # PHASE 4: Assign pulse to track or create new track
        # Apply state-dependent threshold multiplier
        if best_match is not None:
            best_track_state = match_info[best_match]['track_state']
            
            if best_track_state == 'fade_in':
                effective_threshold = score_threshold * fade_in_threshold_multiplier
            elif best_track_state == 'turning':
                effective_threshold = score_threshold * turning_threshold_multiplier
            elif best_track_state == 'fade_out':
                effective_threshold = score_threshold * fade_out_threshold_multiplier
            else:  # stable or unknown
                effective_threshold = score_threshold * stable_threshold_multiplier
        else:
            effective_threshold = score_threshold
        
        if best_match is not None and best_score < effective_threshold:  # Score threshold for acceptance
            # Assign to best matching track
            fish_id = best_match
            track = active_tracks[fish_id]
            
            eod_data.loc[pulse_idx, 'fish_id'] = fish_id
            
            # Update track state
            track['last_pulse_idx'] = pulse_idx
            track['last_timestamp'] = pulse_timestamp
            track['last_location'] = pulse_location
            track['last_channel'] = pulse_channel
            track['last_polarity'] = pulse_polarity
            track['history'].append(pulse_idx)
            track['waveform_signatures'].append(pulse_waveform)
            
            # Update amplitude and polarity history for state tracking
            track['amplitude_history'].append(pulse_amplitude)
            track['polarity_history'].append(pulse_polarity)
            track['max_amplitude_seen'] = max(track['max_amplitude_seen'], pulse_amplitude)
            
            # Limit amplitude/polarity history to prevent memory issues
            if len(track['amplitude_history']) > track_history_max:
                track['amplitude_history'] = track['amplitude_history'][-track_history_max:]
                track['polarity_history'] = track['polarity_history'][-track_history_max:]
            
            # Log state transitions
            if match_info[best_match]['track_state'] != 'stable':
                state_name = match_info[best_match]['track_state']
                state_conf = match_info[best_match]['state_confidence']
                print(f"\n  Fish {fish_id} [{state_name.upper()}, conf={state_conf:.2f}] - score {best_score:.2f} < threshold {effective_threshold:.2f}")
            
            # Update representative waveform (rolling median of recent waveforms)
            recent_waveforms = track['waveform_signatures'][-track_recent_n:]
            track['waveform_signature'] = np.median(recent_waveforms, axis=0)
            
            # Update spatial statistics
            recent_locations = [eod_data.loc[idx, 'pulse_location'] for idx in track['history'][-track_recent_n:]]
            track['mean_location'] = np.mean(recent_locations)
            track['location_std'] = np.std(recent_locations)
            
            # Limit history size to prevent memory issues
            if len(track['history']) > track_history_max:
                track['history'] = track['history'][-track_history_max:]
                track['waveform_signatures'] = track['waveform_signatures'][-track_history_max:]
            
            n_assigned += 1
            
        else:
            # No good match - validate before creating new track
            # Check if pulse amplitude is consistent with a fade-in (entering fish)
            pulse_amplitude = eod_data.loc[pulse_idx, 'eod_amplitude']
            
            # Get amplitude distribution from existing tracks
            existing_max_amplitudes = [track['max_amplitude_seen'] for track in active_tracks.values()]
            
            create_new_track = True
            reassign_to_existing = False
            
            if len(existing_max_amplitudes) > 0:
                amplitude_threshold = np.percentile(existing_max_amplitudes, new_track_amplitude_percentile)
                
                if pulse_amplitude > amplitude_threshold:
                    # Strong pulse - suspicious for new track (fish don't teleport in)
                    # Check if we should force assignment to best candidate with relaxed threshold
                    if best_match is not None:
                        relaxed_threshold = effective_threshold + new_track_fade_in_bonus
                        
                        if best_score < relaxed_threshold:
                            # Accept with penalty
                            reassign_to_existing = True
                            fish_id = best_match
                            track = active_tracks[fish_id]
                            
                            print(f"\n  t={pulse_timestamp}: Strong pulse (amp={pulse_amplitude:.3f}) assigned to Fish {fish_id} with relaxed threshold (score {best_score:.2f} < {relaxed_threshold:.2f})")
                        else:
                            # Create anyway but log warning
                            print(f"\n  t={pulse_timestamp}: ⚠ Creating track with strong pulse (amp={pulse_amplitude:.3f} > threshold {amplitude_threshold:.3f})")
                else:
                    # Weak pulse - consistent with fade-in
                    if best_match is not None:
                        print(f"\n  t={pulse_timestamp}: Weak pulse (amp={pulse_amplitude:.3f}) - likely new fish entering")
            
            if reassign_to_existing:
                # Assign to existing track with relaxed criteria
                eod_data.loc[pulse_idx, 'fish_id'] = fish_id
                
                # Update track state (same as above)
                track['last_pulse_idx'] = pulse_idx
                track['last_timestamp'] = pulse_timestamp
                track['last_location'] = pulse_location
                track['last_channel'] = pulse_channel
                track['last_polarity'] = pulse_polarity
                track['history'].append(pulse_idx)
                track['waveform_signatures'].append(pulse_waveform)
                track['amplitude_history'].append(pulse_amplitude)
                track['polarity_history'].append(pulse_polarity)
                track['max_amplitude_seen'] = max(track['max_amplitude_seen'], pulse_amplitude)
                
                if len(track['amplitude_history']) > track_history_max:
                    track['amplitude_history'] = track['amplitude_history'][-track_history_max:]
                    track['polarity_history'] = track['polarity_history'][-track_history_max:]
                
                # Update representative waveform
                recent_waveforms = track['waveform_signatures'][-track_recent_n:]
                track['waveform_signature'] = np.median(recent_waveforms, axis=0)
                
                # Update spatial statistics
                recent_locations = [eod_data.loc[idx, 'pulse_location'] for idx in track['history'][-track_recent_n:]]
                track['mean_location'] = np.mean(recent_locations)
                track['location_std'] = np.std(recent_locations)
                
                # Limit history size
                if len(track['history']) > track_history_max:
                    track['history'] = track['history'][-track_history_max:]
                    track['waveform_signatures'] = track['waveform_signatures'][-track_history_max:]
                
                n_assigned += 1
                
            elif create_new_track:
                # Create new track
                fish_id = next_fish_id
                next_fish_id += 1
                
                eod_data.loc[pulse_idx, 'fish_id'] = fish_id
                
                active_tracks[fish_id] = {
                    'fish_id': fish_id,
                    'last_pulse_idx': pulse_idx,
                    'last_timestamp': pulse_timestamp,
                    'last_location': pulse_location,
                    'last_channel': pulse_channel,
                    'last_polarity': pulse_polarity,
                    'history': [pulse_idx],
                    'waveform_signatures': [pulse_waveform],
                    'waveform_signature': pulse_waveform,
                    'created_at': pulse_timestamp,
                    'mean_location': pulse_location,
                    'location_std': 0.0,
                    'amplitude_history': [pulse_amplitude],
                    'polarity_history': [pulse_polarity],
                    'max_amplitude_seen': pulse_amplitude,
                    'track_state': 'unknown',
                    'state_confidence': 0.0,
                    'state_duration': 0
                }
                
                n_new_tracks += 1
                
                if best_match is not None:
                    # Had candidates but score too high
                    info = match_info[best_match]
                    state_info = f" [{info['track_state']}]" if info.get('track_state') != 'stable' else ""
                    print(f"\n  t={pulse_timestamp}: New track (Fish {fish_id}) - best match score {best_score:.2f} > threshold {effective_threshold:.2f}{state_info}")
                    print(f"    (loc residual={info['location_residual']:.3f}, predicted loc={info['predicted_location']:.2f}, velocity={info['velocity']:.3f} units/s)")
                    print(f"    (waveform dist={info['waveform_distance']:.2f}, SNR weight={info['snr_weight']:.2f}, amp residual={info['amplitude_residual']:.3f})")
    
    # Clear progress line
    print(f"  Processed {len(eod_data) - n_actual_seed}/{len(eod_data) - n_actual_seed} pulses... Done!")
    
    print(f"\n✓ Sequential matching complete")
    print(f"  Pulses assigned to existing tracks: {n_assigned}")
    print(f"  New tracks created: {n_new_tracks}")
    print(f"  Total active tracks: {len(active_tracks)}")
    print(f"  Total inactive tracks: {len(inactive_tracks)}")
    print(f"  All tracks created: {next_fish_id}")
    
    # Print track summaries
    print(f"\n--- Track Summary ---")
    all_tracks = {**active_tracks, **inactive_tracks}
    for fish_id in sorted(all_tracks.keys()):
        track = all_tracks[fish_id]
        n_pulses = len(track['history'])
        duration = (track['last_timestamp'] - track['created_at']).total_seconds()
        status = "ACTIVE" if fish_id in active_tracks else "INACTIVE"
        
        if n_pulses >= 2:
            timestamps = [eod_data.loc[idx, 'timestamp'] for idx in track['history']]
            ipis = np.diff(timestamps).astype('timedelta64[ms]').astype(float) / 1000.0  # in seconds
            mean_rate = 1.0 / np.mean(ipis) if len(ipis) > 0 else 0
            print(f"  Fish {fish_id:2d} [{status:8s}]: {n_pulses:4d} pulses, {duration:6.2f}s, rate {mean_rate:5.1f} Hz, loc {track['mean_location']:.2f}")
        else:
            print(f"  Fish {fish_id:2d} [{status:8s}]: {n_pulses:4d} pulse(s), {duration:6.2f}s, loc {track['mean_location']:.2f}")
    
    # Create tracking plot
    create_tracking_plot(
        file_set['event_name'],
        eod_data,
        audio_data,
        event_start_time=pd.to_datetime(eod_data['event_start_time'].iloc[0]), 
        sample_rate=audio_fs,
        output_path=output_folder
    )

    # TODO: Continue with step 4 - behavioral detection
    
    # print(f"\n⚠ Implementation in progress - stopping after sequential matching")
    # break  # Stop after first file for now

print("\n" + "="*70)
print("TRACKING ANALYSIS PAUSED (PARTIAL IMPLEMENTATION)")
print("="*70)
print("\nNext steps:")
print("  - Add behavioral detection (Step 4)")
print("  - Add species classification integration (Step 5)")
print("  - Add diagnostic outputs (Step 6)")
