"""
FLATTENED Event Extraction from EOD Data

This is a linear, non-nested version of the event extraction script for easier debugging.
All function calls have been flattened into sequential code blocks.

This script implements the two-stage spatially-aware event extraction approach:
Stage 1: Extract channel-wise events using temporal criteria only
Stage 2: Merge neighboring channel events  
Stage 3: Apply size filtering to merged events (preserves complete multi-channel events)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import signal
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from pathlib import Path
import glob
import tkinter as tk
from tkinter import filedialog
import warnings
import os
import gc
import json
import audioio as aio
# from scipy.signal import find_peaks
from eod_functions import load_variable_length_waveforms, save_variable_length_waveforms
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
# from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# Memory monitoring function for debugging
def print_memory_usage(stage_name=""):
    """Print current memory usage statistics"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"  [MEMORY] {stage_name}: {memory_mb:.1f} MB RSS")
    except ImportError:
        # psutil not available, use basic gc info
        print(f"  [MEMORY] {stage_name}: {len(gc.get_objects())} objects in memory")

print("EOD Session-Level Event Extraction (FLATTENED VERSION)")
print("=" * 60)

# =============================================================================
# STEP 1: GET INPUT FOLDERS
# =============================================================================
print("\nStep 1: Selecting input and output folders...")

root = tk.Tk()
root.withdraw()

session_folder = filedialog.askdirectory(
    title="Select folder with EOD table/waveform files from a recording session"
)
print(f"Session folder: {session_folder}")

output_folder = filedialog.askdirectory(
    title="Select folder to save event extraction results"
)
print(f"Output folder: {output_folder}")

raw_data_folder = filedialog.askdirectory(
    title="Select folder with raw recording files (optional, can be empty)"
)
print(f"Raw data folder: {raw_data_folder}")

# Pick file for amplitude calibration
cal_file = filedialog.askopenfilename(
    title="Select File with Calibration Data"
)
# Extract calibration factors
cor_factors = np.array(pd.read_csv(cal_file))

print(f"Calibration file: {cal_file}")
#%%
# =============================================================================
# STEP 2: SET PARAMETERS
# =============================================================================
print("\nStep 2: Setting extraction parameters...")

# Parameters (can be adjusted based on your data)
max_ipi_seconds = 5.0       # Maximum inter-pulse interval for temporal clustering
min_eods_per_event = 30     # Minimum EODs required per merged event
max_merge_gap_seconds = 0.5  # Maximum gap for merging neighboring channel events
sample_rate = 96000         # Audio sample rate
file_duration = 600.0       # Audio file duration in seconds
margin = 1.0                # Safety margin around events in seconds
min_channel_event_size = 10  # Minimum size of channel events before merging (high-pass filter)
min_amplitude = 0.01          # Minimum amplitude threshold for events (at least one eod should have this size)
create_plots = True

# Clustering parameters
clustering_enabled = False   # Enable waveform clustering
dbscan_eps = 0.5           # DBSCAN distance threshold
dbscan_min_samples = 5     # DBSCAN minimum samples per cluster


print(f"  max_ipi_seconds: {max_ipi_seconds}")
print(f"  min_eods_per_event: {min_eods_per_event}")
print(f"  max_merge_gap_seconds: {max_merge_gap_seconds}")
print(f"  min_channel_event_size: {min_channel_event_size}")
print(f"  min_amplitude: {min_amplitude}")
print(f"  clustering_enabled: {clustering_enabled}")
if clustering_enabled:
    print(f"  dbscan_eps: {dbscan_eps}")
    print(f"  dbscan_min_samples: {dbscan_min_samples}")

# =============================================================================
# STEP 3: LOAD SESSION DATA
# =============================================================================
print("\nStep 3: Loading session data...")

try:
    # Find all CSV files in session folder
    session_path = Path(session_folder)
    eod_files = list(session_path.glob("*eod_table*.csv"))
    
    print(f"  Found {len(eod_files)} EOD table files")
    
    if not eod_files:
        raise ValueError("No EOD table files found!")
    
    # Load and combine EOD tables with file tracking for efficient waveform loading later
    combined_tables = []
    file_metadata = []  # Track file info for later waveform loading
    
    for i, file_path in enumerate(eod_files):
        print(f"  Loading {file_path.name}...")
        df = pd.read_csv(file_path)
        df['file_index'] = i
        df['filename'] = file_path.stem
        df['original_row_in_file'] = df.index  # Track original position in file for waveform mapping
        
        combined_tables.append(df)
        
        # Extract base name for waveform files (remove '_eod_table' suffix)
        base_name = file_path.stem.replace('_eod_table', '')
        waveform_base_path = session_path / f"{base_name}_eod_waveforms"
        
        # Check if corresponding waveform files exist
        concatenated_file = Path(str(waveform_base_path) + '_concatenated.npz')
        metadata_file = Path(str(waveform_base_path) + '_metadata.json')
        
        file_info = {
            'file_index': i,
            'eod_file': file_path,
            'base_name': base_name,
            'waveform_base_path': str(waveform_base_path),
            'has_waveforms': concatenated_file.exists() and metadata_file.exists(),
            'n_eods_in_file': len(df)
        }
        file_metadata.append(file_info)
        
        if file_info['has_waveforms']:
            print(f"    Found matching waveform files for {base_name}")
        else:
            print(f"    Warning: No waveform files found for {base_name}")
    
    combined_table = pd.concat(combined_tables, ignore_index=True)
    print(f"  Combined table shape: {combined_table.shape}")
    
    # Clean up the list of individual tables to free memory
    del combined_tables
    gc.collect()
    print("  Memory cleanup completed after table loading")
    print_memory_usage("After table loading")
    
    # Store file metadata for later use
    files_with_waveforms = [f for f in file_metadata if f['has_waveforms']]
    print(f"  Files with waveforms: {len(files_with_waveforms)}/{len(file_metadata)}")
    
except Exception as e:
    print(f"Error loading session data: {e}")
    exit(1)


# =============================================================================
# STEP 4: EXTRACT EVENTS - STAGE 1 (CHANNEL-WISE TEMPORAL CLUSTERING)
# =============================================================================
print(f"\nStep 4: Stage 1 - Channel-wise temporal clustering...")
print(f"  Using max_ipi_seconds = {max_ipi_seconds}")

if len(combined_table) == 0:
    print("No data to process!")
    exit(1)

# Sort by timestamp
sorted_table = combined_table.sort_values('timestamp').copy()
sorted_table['timestamp_dt'] = pd.to_datetime(sorted_table['timestamp'])

# STAGE 1: Extract channel-wise events using temporal criteria only
print("  Extracting events per channel (no size filtering yet)...")

channel_events_list = []
channel_event_counter = 0

# Get unique channels
channels = sorted(sorted_table['eod_channel'].unique())
print(f"  Processing {len(channels)} channels: {channels}")

for channel in channels:
    print(f"    Processing channel {channel}...")
    
    # Get EODs for this channel
    channel_eods = sorted_table[sorted_table['eod_channel'] == channel].copy()
    
    if len(channel_eods) == 0:
        print(f"      No EODs found for channel {channel}")
        continue
        
    print(f"      Found {len(channel_eods)} EODs")
    
    # Calculate inter-pulse intervals for this channel
    time_diffs = channel_eods['timestamp_dt'].diff().dt.total_seconds()
    
    # Identify event boundaries (gaps > max_ipi_seconds)
    event_boundaries = np.where(time_diffs > max_ipi_seconds)[0]
    
    # Create event segments for this channel
    event_starts = np.concatenate([[0], event_boundaries])
    event_ends = np.concatenate([event_boundaries, [len(channel_eods)]])
    
    print(f"      Found {len(event_starts)} temporal clusters")
    
    # Create channel events
    for start_idx, end_idx in zip(event_starts, event_ends):
        event_eods = channel_eods.iloc[start_idx:end_idx].copy()
        
        # Add channel event metadata
        event_eods['channel_event_id'] = channel_event_counter
        event_eods['channel'] = channel
        event_eods['channel_start_time'] = event_eods['timestamp_dt'].min()
        event_eods['channel_end_time'] = event_eods['timestamp_dt'].max()
        event_eods['channel_n_eods'] = len(event_eods)
        
        channel_events_list.append(event_eods)
        channel_event_counter += 1
        
        print(f"        Channel event {channel_event_counter-1}: {len(event_eods)} EODs")

if channel_events_list:
    channel_events = pd.concat(channel_events_list, ignore_index=True)
    print(f"  Stage 1 complete: {len(channel_events)} total EODs in {channel_event_counter} channel events")

    # Pre-merge filter: remove channel events based on size AND amplitude criteria
    print(f"  Pre-merge filtering: removing channel events with <{min_channel_event_size} pulses OR no pulse ≥{min_amplitude} amplitude...")
    
    # Criterion 1: Size filter (at least min_channel_event_size pulses)
    event_sizes = channel_events.groupby('channel_event_id').size()
    valid_size_ids = event_sizes[event_sizes >= min_channel_event_size].index
    
    # Criterion 2: Amplitude filter (at least one pulse with amplitude ≥ min_amplitude)
    event_max_amplitudes = channel_events.groupby('channel_event_id')['eod_amplitude'].max()
    valid_amplitude_ids = event_max_amplitudes[event_max_amplitudes >= min_amplitude].index
    
    # Combined filter: channel events must pass BOTH criteria
    valid_channel_event_ids = valid_size_ids.intersection(valid_amplitude_ids)
    
    # Calculate removal statistics
    n_total = channel_events['channel_event_id'].nunique()
    n_removed_size = n_total - len(valid_size_ids)
    n_removed_amplitude = n_total - len(valid_amplitude_ids)
    n_removed_total = n_total - len(valid_channel_event_ids)
    
    # Apply combined filter
    filtered_channel_events = channel_events[channel_events['channel_event_id'].isin(valid_channel_event_ids)].copy()
    
    print(f"    Original channel events: {n_total}")
    print(f"    Failed size criterion (<{min_channel_event_size} pulses): {n_removed_size}")
    print(f"    Failed amplitude criterion (max amplitude <{min_amplitude}): {n_removed_amplitude}")
    print(f"    Remaining after combined filter: {len(valid_channel_event_ids)}")
    print(f"    Total removed: {n_removed_total}")
    
    if len(filtered_channel_events) == 0:
        print(f"  All channel events removed by pre-merge filtering! Consider lowering min_channel_event_size or min_amplitude.")
        exit(1)
    
    # NOW EFFICIENTLY LOAD ONLY WAVEFORMS FOR FILTERED EODS
    print(f"\nStep 4.2: Loading waveforms...")
    combined_waveforms = None
    
    if len(files_with_waveforms) > 0:
        print(f"  Loading waveforms for {len(filtered_channel_events)} filtered EODs...")

        # Create mapping of which EODs we need from each file
        filtered_eods_by_file = {}
        for _, eod_row in filtered_channel_events.iterrows():
            file_idx = eod_row['file_index']
            original_row = eod_row['original_row_in_file']
            
            if file_idx not in filtered_eods_by_file:
                filtered_eods_by_file[file_idx] = []
            filtered_eods_by_file[file_idx].append(original_row)
        
        print(f"  Need to load waveforms from {len(filtered_eods_by_file)} files")
        
        all_filtered_waveforms = []
        all_filtered_metadata = []
        
        for file_idx, needed_rows in filtered_eods_by_file.items():
            file_info = file_metadata[file_idx]
            
            if not file_info['has_waveforms']:
                print(f"    Skipping {file_info['base_name']} - no waveform files")
                continue
                
            print(f"    Loading {len(needed_rows)} waveforms from {file_info['base_name']}")
            
            # Load ALL waveforms from this file (we need the full array structure)
            waveforms = load_variable_length_waveforms(file_info['waveform_base_path'])
            
            # Load metadata
            file_metadata_list = None
            try:
                with open(file_info['waveform_base_path'] + '_metadata.json', 'r') as f:
                    metadata = json.load(f)
                    
                if isinstance(metadata, dict) and 'individual_metadata' in metadata:
                    file_metadata_list = metadata['individual_metadata']
                elif isinstance(metadata, list):
                    file_metadata_list = metadata
                else:
                    # Create dummy metadata
                    file_metadata_list = [{'index': i} for i in range(len(waveforms))]
                    
                # Clean up the metadata dict immediately after extraction
                del metadata
                    
            except Exception as e:
                print(f"      Warning: Could not load metadata: {e}")
                file_metadata_list = [{'index': i} for i in range(len(waveforms))]
            
            # Extract only the waveforms we need
            for row_idx in needed_rows:
                if row_idx < len(waveforms):
                    all_filtered_waveforms.append(waveforms[row_idx])
                    if row_idx < len(file_metadata_list):
                        all_filtered_metadata.append(file_metadata_list[row_idx])
                    else:
                        all_filtered_metadata.append({'index': row_idx, 'file': file_info['base_name']})
                else:
                    print(f"      Warning: Row {row_idx} not found in waveforms (file has {len(waveforms)} waveforms)")
            
            # Clean up large temporary variables immediately after use
            del waveforms
            del file_metadata_list
            
            # Force garbage collection after processing each file to free memory
            gc.collect()
            print(f"      Memory freed after processing {file_info['base_name']}")
        
        if all_filtered_waveforms:
            combined_waveforms = {'waveforms': all_filtered_waveforms, 'metadata': all_filtered_metadata}
            print(f"  Loaded {len(all_filtered_waveforms)} waveforms for filtered EODs")
        else:
            print("  No waveforms loaded")
            
        # Clean up temporary waveform loading variables
        del all_filtered_waveforms
        del all_filtered_metadata
        del filtered_eods_by_file
        
        # Force garbage collection after all waveform loading is complete
        gc.collect()
        print("  Memory cleanup completed after waveform loading")
        print_memory_usage("After waveform loading")
        
    else:
        print("  No files with waveforms found")
    
    # Update channel_events to the filtered version
    channel_events = filtered_channel_events
    
    # Clean up intermediate filtering variables
    del filtered_channel_events, valid_size_ids, valid_amplitude_ids
    del valid_channel_event_ids, event_sizes, event_max_amplitudes
    gc.collect()
    print("  Memory cleanup completed after filtering")

# =============================================================================
# STEP 5: EOD FEATURE CLUSTERING AND NOISE FILTERING
# =============================================================================
print(f"\nStep 5: EOD feature clustering and noise filtering...")

if clustering_enabled and len(channel_events) > 0:
    print(f"  Starting EOD clustering on individual pulses...")
    print(f"  Processing {len(channel_events)} individual EOD pulses...")
    
    # Use pre-computed features from individual EOD pulses
    clustering_features = ['eod_amplitude_ratio', 'waveform_length', 'eod_width_us', 'fft_freq_max']
    
    # Check which features are available
    available_features = [f for f in clustering_features if f in channel_events.columns]
    missing_features = [f for f in clustering_features if f not in channel_events.columns]
    
    print(f"  Available features: {available_features}")
    if missing_features:
        print(f"  Missing features: {missing_features}")
    
    if len(available_features) == 0:
        print("  No clustering features available - skipping clustering")
        noise_filtered_events = channel_events.copy()
    else:
        # Extract feature matrix from individual EOD pulses
        feature_data = channel_events[available_features].copy()
        
        # Handle missing values
        print(f"  Feature matrix shape: {feature_data.shape}")
        print(f"  Checking for missing values...")
        
        # Check for NaN or infinite values
        nan_counts = feature_data.isnull().sum()
        inf_counts = np.isinf(feature_data.select_dtypes(include=[np.number])).sum()
        
        if nan_counts.sum() > 0:
            print(f"  Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        if inf_counts.sum() > 0:
            print(f"  Found infinite values: {inf_counts[inf_counts > 0].to_dict()}")
        
        # Fill missing values with feature medians
        for col in available_features:
            if feature_data[col].isnull().sum() > 0:
                median_val = feature_data[col].median()
                feature_data[col].fillna(median_val, inplace=True)
                print(f"    Filled {feature_data[col].isnull().sum()} NaN values in {col} with median: {median_val:.3f}")
        
        # Handle infinite values
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        for col in available_features:
            if feature_data[col].isnull().sum() > 0:
                median_val = feature_data[col].median()
                feature_data[col].fillna(median_val, inplace=True)
                print(f"    Replaced infinite values in {col} with median: {median_val:.3f}")
        
        print(f"  Feature statistics for individual pulses:")
        for col in available_features:
            vals = feature_data[col]
            print(f"    {col}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}, std={vals.std():.3f}")
        
        # More memory-efficient approach: Use length clustering with reduced dataset
        print("  Using memory-efficient length + feature clustering approach...")
        
        # Stage 1: Pre-filter by obvious outliers using waveform length
        print("    Stage 1: Pre-filtering by waveform length...")
        lengths = channel_events['waveform_length'].values
        
        # Use simple statistical outlier detection for length first
        length_q1 = np.percentile(lengths, 25)
        length_q3 = np.percentile(lengths, 75)
        length_iqr = length_q3 - length_q1
        length_lower = length_q1 - 1.5 * length_iqr
        length_upper = length_q3 + 1.5 * length_iqr
        
        # Keep pulses within reasonable length bounds
        length_mask = (lengths >= length_lower) & (lengths <= length_upper)
        length_filtered_events = channel_events[length_mask].copy()
        
        print(f"      Length filtering: kept {len(length_filtered_events)}/{len(channel_events)} pulses")
        print(f"      Length bounds: {length_lower:.0f} - {length_upper:.0f} samples")
        
        # Stage 2: Use enhanced feature clustering on length-filtered data
        print("    Stage 2: Feature-based clustering on length-filtered data...")
        
        # Enhanced feature set including length groups
        enhanced_features = ['eod_amplitude_ratio', 'waveform_length', 'eod_width_us', 'fft_freq_max']
        feature_data = length_filtered_events[enhanced_features].copy()
        
        # Add length category as additional feature
        length_bins = np.linspace(length_lower, length_upper, 10)
        length_categories = np.digitize(length_filtered_events['waveform_length'], length_bins)
        feature_data['length_category'] = length_categories
        
        # Normalize features for clustering        
        X = feature_data.values
        X = np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"      Enhanced feature matrix shape: {X_scaled.shape}")
        
        # Perform DBSCAN clustering with enhanced features
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')
        cluster_labels_filtered = clusterer.fit_predict(X_scaled)
        
        # Map back to full dataset
        final_cluster_labels = np.full(len(channel_events), -1, dtype=int)
        final_cluster_labels[length_mask] = cluster_labels_filtered
        
        # Add cluster labels to individual EOD pulses
        channel_events['pulse_cluster_id'] = final_cluster_labels
        
        # Analyze clustering results for individual pulses
        cluster_labels = final_cluster_labels
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise_pulses = np.sum(cluster_labels == -1)
        
        print(f"  Two-stage clustering results:")
        print(f"    Number of final clusters: {n_clusters}")
        print(f"    Number of noise pulses: {n_noise_pulses}")
        print(f"    Percentage noise pulses: {n_noise_pulses/len(cluster_labels)*100:.1f}%")
        
        # Create waveform visualization for each cluster
        print("  Creating waveform cluster visualization...")
        
        # Analyze cluster characteristics for individual pulses
        print("  Individual pulse cluster characteristics:")
        for cluster_id in sorted(unique_clusters):
            if cluster_id == -1:
                cluster_name = "DBSCAN Outliers"
            else:
                cluster_name = f"Cluster {cluster_id}"
                
            mask = cluster_labels == cluster_id
            cluster_pulses = channel_events[mask]
            
            print(f"    {cluster_name} ({np.sum(mask)} pulses):")
            for col in available_features:
                if col in cluster_pulses.columns:
                    vals = cluster_pulses[col]
                    print(f"      {col}: {vals.min():.3f}-{vals.max():.3f} (mean: {vals.mean():.3f})")
            
                    # Show which channels the pulses come from
                    if len(cluster_pulses) > 0:
                        channels_in_cluster = cluster_pulses['eod_channel'].unique()
                        print(f"      Channels: {sorted(channels_in_cluster)}")
                
                # Clean up per-cluster variables
                del cluster_pulses, mask
        
        # Clean up clustering analysis variables
        del available_features, unique_clusters        # Create multi-panel waveform plot
        if combined_waveforms is not None:
            print("  Creating waveform overlay plots for each cluster...")
            
            # Calculate subplot layout
            n_clusters_including_outliers = len(unique_clusters)
            n_cols = min(3, n_clusters_including_outliers)
            n_rows = int(np.ceil(n_clusters_including_outliers / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_clusters_including_outliers == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            colors = cm.get_cmap('tab10', n_clusters_including_outliers)
            
            for i, cluster_id in enumerate(sorted(unique_clusters)):
                ax = axes[i]
                
                # Get pulses in this cluster
                mask = cluster_labels == cluster_id
                cluster_pulses = channel_events[mask]
                
                if cluster_id == -1:
                    cluster_name = f"DBSCAN Outliers"
                    color = 'red'
                else:
                    cluster_name = f"Cluster {cluster_id}"
                    color = colors(i)
                
                # Sample up to 100 waveforms for visualization
                max_waveforms = 100
                if len(cluster_pulses) > max_waveforms:
                    cluster_sample = cluster_pulses.sample(n=max_waveforms, random_state=42)
                    sample_info = f" (showing {max_waveforms}/{len(cluster_pulses)})"
                else:
                    cluster_sample = cluster_pulses
                    sample_info = f" (all {len(cluster_pulses)})"
                
                waveform_count = 0
                for pulse_idx, pulse in cluster_sample.iterrows():
                    channel = pulse['eod_channel']
                    
                    # Get waveform data for this pulse using the original table index
                    if 'waveforms' in combined_waveforms and 'metadata' in combined_waveforms:
                        all_waveforms = combined_waveforms['waveforms']
                        all_metadata = combined_waveforms['metadata']
                        
                        # Find the corresponding waveform using the index in filtered_eods
                        # The pulse_idx corresponds to the row in filtered_eods
                        if pulse_idx < len(all_waveforms):
                            waveform_data = all_waveforms[pulse_idx]
                            if waveform_data is not None and len(waveform_data) > 0:
                                # Normalize waveform for overlay visualization
                                waveform_norm = (waveform_data - waveform_data.mean()) / (waveform_data.std() + 1e-8)
                                time_axis = np.arange(len(waveform_norm)) / 96000 * 1000  # Convert to ms
                                ax.plot(time_axis, waveform_norm, color=color, alpha=0.3, linewidth=0.5)
                                waveform_count += 1
                
                ax.set_title(f'{cluster_name}{sample_info}\n{waveform_count} waveforms plotted')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Normalized Amplitude')
                ax.grid(True, alpha=0.3)
                
                # Add cluster statistics as text
                if len(cluster_pulses) > 0:
                    stats_text = []
                    for col in available_features[:2]:  # Show first 2 features
                        if col in cluster_pulses.columns:
                            vals = cluster_pulses[col]
                            stats_text.append(f'{col}: {vals.mean():.2f}±{vals.std():.2f}')
                    ax.text(0.02, 0.98, '\n'.join(stats_text), 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=8)
            
            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            # Save the plot
            waveform_plot_path = Path(output_folder) / 'pulse_cluster_waveforms.png'
            plt.savefig(waveform_plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved waveform cluster visualization: {waveform_plot_path}")
            
            # Clean up plotting variables before interactive display
            del colors, axes, fig
            
            # Show interactive popup plot for inspection
            waveform_fig = plt.gcf()  # Get current figure for later cleanup
            print("  Displaying interactive cluster visualization...")
            plt.show(block=False)  # Non-blocking show
            
            # Give the plot a moment to appear
            plt.pause(1)
        else:
            print("  Skipping waveform visualization - no waveform data available")
        
        # Interactive user input for noise cluster selection
        print(f"\n{'='*50}")
        print("INTERACTIVE CLUSTER FILTERING")
        print(f"{'='*50}")
        print(f"Available clusters: {sorted(unique_clusters)}")
        print("Please inspect the waveform cluster plot and decide which clusters represent noise.")
        print("Note: DBSCAN automatically labels outliers as cluster -1")
        print("")
        
        # Get user input for noise clusters
        while True:
            try:
                user_input = input("Enter cluster IDs to REMOVE as noise (comma-separated integers, or 'none' to keep all): ").strip()
                
                if user_input.lower() == 'none':
                    noise_cluster_ids = []
                    break
                elif user_input == '':
                    print("Please enter cluster IDs or 'none'")
                    continue
                else:
                    # Parse comma-separated integers
                    noise_cluster_ids = [int(x.strip()) for x in user_input.split(',')]
                    
                    # Validate cluster IDs
                    invalid_ids = [cid for cid in noise_cluster_ids if cid not in unique_clusters]
                    if invalid_ids:
                        print(f"Invalid cluster IDs: {invalid_ids}. Available: {sorted(unique_clusters)}")
                        continue
                    break
                    
            except ValueError:
                print("Please enter valid integers separated by commas, or 'none'")
                continue
        
        # Close the interactive plot after user input
        plt.close('all')
        print("  Interactive plot closed")
        
        # Clean up waveform plotting variables
        if 'waveform_fig' in locals():
            del waveform_fig
        if 'combined_waveforms' in locals() and 'waveforms' in combined_waveforms:
            # Don't delete combined_waveforms completely as it's needed for saving later
            pass  
        gc.collect()
        
        # Filter based on user selection
        if len(noise_cluster_ids) > 0:
            print(f"  Removing pulses from clusters: {noise_cluster_ids}")
            
            # Create mask to filter out selected noise clusters
            noise_mask = np.isin(cluster_labels, noise_cluster_ids)
            non_noise_mask = ~noise_mask
            noise_filtered_events = channel_events[non_noise_mask].copy()
            
            n_removed_pulses = np.sum(noise_mask)
            
            print(f"  After user-defined cluster filtering:")
            print(f"    Original pulses: {len(channel_events)}")
            print(f"    Remaining pulses: {len(noise_filtered_events)}")
            print(f"    Removed pulses: {n_removed_pulses}")
            print(f"    Filtering efficiency: {n_removed_pulses/len(channel_events)*100:.1f}% pulses removed")
            
            # Check how filtering affects channel events
            original_channel_events = channel_events['channel_event_id'].nunique()
            remaining_channel_events = noise_filtered_events['channel_event_id'].nunique()
            print(f"    Original channel events: {original_channel_events}")
            print(f"    Remaining channel events: {remaining_channel_events}")
            print(f"    Channel events lost: {original_channel_events - remaining_channel_events}")
            
        else:
            print("  No clusters selected for removal - keeping all pulses")
            noise_filtered_events = channel_events.copy()
        
        # Clean up cluster filtering variables
        if 'noise_cluster_ids' in locals():
            del noise_cluster_ids
        if 'noise_mask' in locals():
            del noise_mask, non_noise_mask
        if 'cluster_labels' in locals():
            del cluster_labels
        gc.collect()
        
        # Clean up clustering variables to free memory
        del X_scaled, X, feature_data
        del scaler, clusterer
        del lengths, length_mask, final_cluster_labels
        if 'length_filtered_events' in locals():
            del length_filtered_events
        if 'cluster_labels_filtered' in locals():
            del cluster_labels_filtered
        if 'enhanced_features' in locals():
            del enhanced_features
        gc.collect()
        print("  Memory cleanup completed after pulse clustering")
        print_memory_usage("After pulse clustering")

else:
    print("  EOD pulse clustering disabled - skipping clustering step")
    noise_filtered_events = channel_events.copy()

# Continue with the noise-filtered individual pulses
channel_events = noise_filtered_events

if len(channel_events) == 0:
    print("  No channel events found after filtering!")
    exit(1)

# Post-cluster size filtering: Some channel events may now be too small after noise filtering
print(f"  Post-cluster size filtering: removing channel events with <{min_channel_event_size} pulses...")
pre_filter_channel_events = channel_events['channel_event_id'].nunique()
pre_filter_total_eods = len(channel_events)

# Calculate EODs per channel event after cluster filtering
channel_event_sizes_post_cluster = channel_events.groupby('channel_event_id').size()
valid_channel_events_post_cluster = channel_event_sizes_post_cluster[channel_event_sizes_post_cluster >= min_channel_event_size].index

# Filter channel events by post-cluster size
channel_events = channel_events[channel_events['channel_event_id'].isin(valid_channel_events_post_cluster)].copy()

post_filter_channel_events = len(valid_channel_events_post_cluster)
post_filter_total_eods = len(channel_events)
removed_channel_events = pre_filter_channel_events - post_filter_channel_events
removed_eods = pre_filter_total_eods - post_filter_total_eods

print(f"    Original channel events (post-cluster): {pre_filter_channel_events}")
print(f"    Remaining channel events: {post_filter_channel_events}")
print(f"    Removed channel events: {removed_channel_events}")
print(f"    Original EODs (post-cluster): {pre_filter_total_eods}")
print(f"    Remaining EODs: {post_filter_total_eods}")
print(f"    Removed EODs: {removed_eods}")

if len(channel_events) == 0:
    print("  No channel events found after post-cluster size filtering!")
    exit(1)

# =============================================================================
# STEP 6: EXTRACT EVENTS - STAGE 2 (MERGE NEIGHBORING CHANNELS)
# =============================================================================
print(f"\nStep 6: Stage 2 - Merging neighboring channel events...")
print(f"  Using max_merge_gap_seconds = {max_merge_gap_seconds}")

# Get unique channel event summaries
event_summaries = (channel_events
                  .groupby('channel_event_id')
                  .agg({
                      'channel': 'first',
                      'channel_start_time': 'first', 
                      'channel_end_time': 'first',
                      'channel_n_eods': 'first'
                  })
                  .reset_index())

print(f"  Processing {len(event_summaries)} channel events for merging...")

# Initialize merging data structures
merged_events_list = []
merged_event_counter = 0
processed_channel_events = set()

# Sort by start time for processing
event_summaries = event_summaries.sort_values('channel_start_time')

for idx, current_event in event_summaries.iterrows():
    current_id = current_event['channel_event_id']
    
    if current_id in processed_channel_events:
        continue
        
    print(f"    Processing channel event {current_id} (channel {current_event['channel']})...")
    
    # Start a new merged event with iterative growing
    merged_group = [current_id]
    processed_channel_events.add(current_id)
    
    # Initialize merged event bounds
    current_start = current_event['channel_start_time']
    current_end = current_event['channel_end_time']
    current_channels = {current_event['channel']}
    
    # ITERATIVE MERGING: Keep expanding until no more events can be merged
    found_new_merge = True
    iteration = 0
    merge_iteration = {}
    
    while found_new_merge:
        found_new_merge = False
        iteration += 1
        print(f"      Iteration {iteration}: bounds=({current_start.strftime('%H:%M:%S')}-{current_end.strftime('%H:%M:%S')}), channels={sorted(current_channels)}")

        # --- Temporal-edge-based merging ---
        # For each channel in the current merged event, track its temporal edges (start, end)
        # For each edge (channel, edge_time), check neighbor channels for unmerged events that are temporally adjacent/overlapping
        edge_points = []  # List of (channel, edge_time, 'start'/'end')
        for ch in current_channels:
            # Find all events in merged_group for this channel
            ch_event_ids = [eid for eid in merged_group if event_summaries.loc[event_summaries['channel_event_id'] == eid, 'channel'].values[0] == ch]
            # For each, get start and end
            for eid in ch_event_ids:
                ev = event_summaries.loc[event_summaries['channel_event_id'] == eid].iloc[0]
                edge_points.append((ch, ev['channel_start_time'], 'start'))
                edge_points.append((ch, ev['channel_end_time'], 'end'))

        # For each edge, check neighbor channels for unmerged events that are temporally adjacent/overlapping
        events_to_merge_this_iteration = []
        new_channels_this_iteration = set()
        for ch, edge_time, edge_type in edge_points:
            for neighbor_ch in [ch-1, ch+1]:
                if neighbor_ch < 0 or neighbor_ch > 6:
                    continue
                # Find unprocessed events on neighbor channel
                neighbor_events = event_summaries[
                    (event_summaries['channel'] == neighbor_ch) &
                    (~event_summaries['channel_event_id'].isin(processed_channel_events))
                ]
                for _, neighbor_event in neighbor_events.iterrows():
                    neighbor_id = neighbor_event['channel_event_id']
                    if neighbor_id in processed_channel_events:
                        continue
                    neighbor_start = neighbor_event['channel_start_time']
                    neighbor_end = neighbor_event['channel_end_time']
                    # True interval overlap with current merged event
                    interval_overlap = (neighbor_end >= current_start) and (neighbor_start <= current_end)
                    # Edge-based gap checks (for temporal adjacency)
                    edge_overlap = (neighbor_start <= edge_time) and (neighbor_end >= edge_time)
                    if interval_overlap:
                        gap = 0
                        gap_type = 'interval_overlap'
                        should_merge = True
                    elif edge_overlap:
                        gap = 0
                        gap_type = 'edge_overlap'
                        should_merge = True
                    else:
                        # Check if neighbor is just after or before this edge
                        if edge_type == 'end' and 0 <= (neighbor_start - edge_time).total_seconds() <= max_merge_gap_seconds:
                            gap = (neighbor_start - edge_time).total_seconds()
                            gap_type = 'sequential_after'
                            should_merge = True
                        elif edge_type == 'start' and 0 <= (edge_time - neighbor_end).total_seconds() <= max_merge_gap_seconds:
                            gap = (edge_time - neighbor_end).total_seconds()
                            gap_type = 'sequential_before'
                            should_merge = True
                        else:
                            gap = float('inf')
                            gap_type = 'too_distant'
                            should_merge = False
                    if should_merge:
                        print(f"          CANDIDATE for merging: event {neighbor_id} ch{neighbor_ch} ({gap_type}, gap: {gap:.2f}s) at edge ({ch}, {edge_type}, {edge_time.strftime('%H:%M:%S')})")
                        events_to_merge_this_iteration.append({
                            'event_id': neighbor_id,
                            'channel': neighbor_ch,
                            'start_time': neighbor_start,
                            'end_time': neighbor_end,
                            'gap': gap,
                            'gap_type': gap_type
                        })
                        processed_channel_events.add(neighbor_id)
                    # else:
                        # print(f"          REJECTED: event {neighbor_id} ch{neighbor_ch} too distant ({gap_type}, gap: {gap:.2f}s) at edge ({ch}, {edge_type}, {edge_time.strftime('%H:%M:%S')})")

        # Merge all qualifying events simultaneously
        if events_to_merge_this_iteration:
            print(f"        MERGING {len(events_to_merge_this_iteration)} events simultaneously:")
            events_to_merge_this_iteration.sort(key=lambda x: x['gap'])
            for event_info in events_to_merge_this_iteration:
                neighbor_id = event_info['event_id']
                neighbor_channel = event_info['channel']
                neighbor_start = event_info['start_time']
                neighbor_end = event_info['end_time']
                print(f"          - Event {neighbor_id} ch{neighbor_channel} ({event_info['gap_type']}, gap: {event_info['gap']:.2f}s)")
                if neighbor_id not in merged_group:
                    merged_group.append(neighbor_id)
                    merge_iteration[neighbor_id] = iteration
                if neighbor_id not in processed_channel_events:
                    processed_channel_events.add(neighbor_id)
                if neighbor_channel not in new_channels_this_iteration:
                    new_channels_this_iteration.add(neighbor_channel)
                if neighbor_channel not in current_channels:
                    current_channels.add(neighbor_channel)
                current_start = min(current_start, neighbor_start)
                current_end = max(current_end, neighbor_end)
            found_new_merge = True
            print(f"          UPDATED bounds after parallel merge: ({current_start.strftime('%H:%M:%S')}-{current_end.strftime('%H:%M:%S')})")
        else:
            print(f"        No qualifying events found for any temporal edge of channels {sorted(current_channels)}")
        if not found_new_merge:
            print(f"        No more mergeable neighbors found for any temporal edge of channels {sorted(current_channels)}")
        elif len(new_channels_this_iteration) == 0:
            print(f"        Temporal bounds changed but no new channels added - stopping to prevent infinite loop")
            found_new_merge = False
    
    # Assign merged event ID to all EODs in the final merged group
    print(f"      Final merged group after {iteration} iterations: {merged_group} -> merged_event_id {merged_event_counter}")
    print(f"      Final channels: {sorted(current_channels)}, final bounds: ({current_start.strftime('%H:%M:%S')}-{current_end.strftime('%H:%M:%S')})")
    
    for merge_order, channel_event_id in enumerate(merged_group):
        event_eods = channel_events[channel_events['channel_event_id'] == channel_event_id].copy()
        event_eods['merged_event_id'] = merged_event_counter
        event_eods['merge_order'] = merge_order  # Store the order in which this channel event was merged
        event_eods['merge_iteration'] = merge_iteration.get(channel_event_id, 1)
        merged_events_list.append(event_eods)
    
    merged_event_counter += 1

# Clean up merging loop variables
del processed_channel_events, merged_group, merge_iteration

if merged_events_list:
    merged_events = pd.concat(merged_events_list, ignore_index=True)
    print(f"  Stage 2 complete: {len(merged_events)} total EODs in {merged_event_counter} merged events")
    
    # Clean up merging variables to free memory
    del merged_events_list
    del event_summaries
    gc.collect()
    print("  Memory cleanup completed after merging")
    
else:
    print("  No merged events found!")
    exit(1)

# =============================================================================
# STEP 7: EXTRACT EVENTS - STAGE 3 (SIZE FILTERING)
# =============================================================================
print(f"\nStep 7: Stage 3 - Size filtering merged events...")
print(f"  Using min_eods_per_event = {min_eods_per_event}")

# Count EODs per merged event
event_sizes = merged_events.groupby('merged_event_id').size()
print(f"  Event sizes: min={event_sizes.min()}, max={event_sizes.max()}, mean={event_sizes.mean():.1f}")

valid_event_ids = event_sizes[event_sizes >= min_eods_per_event].index
print(f"  Events passing size filter: {len(valid_event_ids)} / {len(event_sizes)}")

if len(valid_event_ids) == 0:
    print("  No events passed size filtering!")
    print("  Consider reducing min_eods_per_event parameter")
    exit(1)

# Filter to keep only valid events
event_table = merged_events[merged_events['merged_event_id'].isin(valid_event_ids)].copy()

# Reassign sequential event IDs
unique_ids = sorted(event_table['merged_event_id'].unique())
id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
event_table['event_id'] = event_table['merged_event_id'].map(id_mapping)

print(f"  Stage 3 complete: {len(event_table)} EODs in {len(unique_ids)} final events")

# Clean up size filtering variables to free memory
del merged_events
del event_sizes
del valid_event_ids
del unique_ids
del id_mapping
gc.collect()
print("  Memory cleanup completed after size filtering")

# =============================================================================
# STEP 8: CREATE EVENT SUMMARY (BEFORE INDIVIDUAL PROCESSING)
# =============================================================================
print(f"\nStep 8: Creating event summary...")

# Create event summary directly from event_table BEFORE the processing loop
event_summaries = []
for event_id in sorted(event_table['event_id'].unique()):
    event_eods = event_table[event_table['event_id'] == event_id]
    
    # Calculate summary statistics
    duration = (event_eods['timestamp_dt'].max() - 
                event_eods['timestamp_dt'].min()).total_seconds()
    
    # Calculate IPIs for this event
    event_sorted = event_eods.sort_values('timestamp_dt')
    time_diffs = event_sorted['timestamp_dt'].diff().dt.total_seconds().dropna()
    
    summary = {
        'event_id': event_id,
        'start_time': event_eods['timestamp_dt'].min(),
        'end_time': event_eods['timestamp_dt'].max(),
        'duration_seconds': duration,
        'n_eods': len(event_eods),
        'n_channels': event_eods['eod_channel'].nunique(),
        'channels_used': ','.join(map(str, sorted(event_eods['eod_channel'].unique()))),
        'mean_ipi_seconds': time_diffs.mean() if len(time_diffs) > 0 else 0,
        'median_ipi_seconds': time_diffs.median() if len(time_diffs) > 0 else 0,
        'mean_amplitude': event_eods['eod_amplitude'].mean(),
        'max_amplitude': event_eods['eod_amplitude'].max(),
        'mean_width_ms': event_eods['eod_width_us'].mean() * 1000 if 'eod_width_us' in event_eods.columns else 0,
        'n_files': event_eods['file_index'].nunique() if 'file_index' in event_eods.columns else 1,
        'file_names': ','.join(event_eods['filename'].unique()) if 'filename' in event_eods.columns else 'unknown'
    }
    
    event_summaries.append(summary)

event_summary = pd.DataFrame(event_summaries)
print(f"  Created event summary with {len(event_summary)} events")

# Save event summary immediately (before processing individual events)
output_path_obj = Path(output_folder)
event_summary_path = output_path_obj / "session_events_summary.csv"
event_summary.to_csv(event_summary_path, index=False)
print(f"  Saved event summary: {event_summary_path}")

# Clean up
del event_summaries
gc.collect()
print("  Memory cleanup completed after event summary creation")
print_memory_usage("After event summary creation")

#%%
# =============================================================================
# STEP 9: PROCESS INDIVIDUAL EVENTS (AUDIO EXTRACTION + INDIVIDUAL SAVES)
# =============================================================================
print(f"\nStep 9: Processing individual events (audio extraction and individual saves)...")

# Find all wav files in raw data folder
wav_files = glob.glob(str(Path(raw_data_folder) / "*.wav"))
print(f"  Found {len(wav_files)} wav files")

# Add event snippet-relative index columns to event_table
event_table['peak_idx_event_snippet'] = -1
event_table['trough_idx_event_snippet'] = -1
event_table['midpoint_idx_event_snippet'] = -1

if len(wav_files) > 0:
    # Extract timestamps from filenames and create file timing DataFrame
    timecol = [pd.to_datetime(os.path.basename(fname).split('-')[1][0:-4], 
                             format='%Y%m%dT%H%M%S') for fname in wav_files]
    file_times_df = pd.DataFrame({'filename': wav_files, 'file_timestamp': timecol})
    file_times_df = file_times_df.sort_values('file_timestamp').reset_index(drop=True)
    # Add file_index column for proper EOD index mapping
    file_times_df['file_index'] = file_times_df.index
    
    # Pre-calculate sample indices
    samples_per_file = int(sample_rate * file_duration)
    
    # Add columns to event_summary for sample information
    event_summary['start_sample_global'] = 0
    event_summary['end_sample_global'] = 0
    event_summary['raw_event_data_filename'] = ''
    event_summary['margin_complete'] = True
    
    print(f"  Processing {len(event_summary)} events individually...")
    print(f"  Parameters: sample_rate={sample_rate}Hz, file_duration={file_duration}s, margin={margin}s")
    
    for idx, event in event_summary.iterrows():
        event_id = event['event_id']
        
        print(f"    Processing event {event_id}...")
        
        try:
            # Get EOD data for this event (fetch once and reuse)
            event_eods = event_table[event_table['event_id'] == event_id].copy()
            if len(event_eods) == 0:
                print(f"        Warning: No EOD data found for event {event_id}")
                continue
                        
            # Find first and last pulse midpoints
            first_pulse_time = event_eods['timestamp_dt'].min()
            last_pulse_time = event_eods['timestamp_dt'].max()
            
            # Calculate event boundaries with margin
            event_start_margin = first_pulse_time - pd.Timedelta(seconds=margin)
            event_end_margin = last_pulse_time + pd.Timedelta(seconds=margin)

            # Extract and save audio data
            n_files = event['n_files']
            samples_read = 0
            all_data = []
            for i in range(n_files):
                fname = event['file_names'].split(',')[i]
                file_basename = fname.split('_')[0]
                mask = file_times_df['filename'].str.contains(file_basename, regex=False)
                matching_files = file_times_df[mask]
                wav_file = matching_files['filename'].values[0]
                file_time = matching_files['file_timestamp'].values[0]
                
                # Calculate sample indices
                if i == 0: # first file: use the event start or the file start if the margin precedes file start
                    start_sample_file = int((event_start_margin - file_time).total_seconds() * sample_rate)
                    start_sample_file = max(0, start_sample_file)
                    event_start_time = file_time + pd.Timedelta(seconds=start_sample_file / sample_rate)
                else: # subsequent files: start from the beginning of the file
                    start_sample_file = 0

                if i == n_files - 1: # last file: use the event end or the file end if the margin exceeds file duration
                    end_sample_file = int((event_end_margin - file_time).total_seconds() * sample_rate)
                    end_sample_file = min(samples_per_file, end_sample_file) # only if the margin exceeds the file duration
                elif i < n_files - 1: # intermediate file - read completely
                    end_sample_file = samples_per_file
               
               # Load data
                with aio.AudioLoader(wav_file) as sf:
                    file_data = sf[start_sample_file:end_sample_file, :]
                    
                # Apply calibration factors
                for ch in range(file_data.shape[1]):
                    file_data[:, ch] *= cor_factors[ch, 1]

                # Append to all_data
                all_data.append(file_data)

                # Update the event snippet-relative indices
                file_mask = event_eods['filename'] == fname
                event_eods.loc[file_mask, 'peak_idx_event_snippet'] = event_eods.loc[file_mask, 'peak_idx'] - start_sample_file + samples_read
                event_eods.loc[file_mask, 'trough_idx_event_snippet'] = event_eods.loc[file_mask, 'trough_idx'] - start_sample_file + samples_read
                event_eods.loc[file_mask, 'midpoint_idx_event_snippet'] = event_eods.loc[file_mask, 'midpoint_idx'] - start_sample_file + samples_read

                samples_read += file_data.shape[0]
                
            # Save audio snippet
            output_filename = f"event_{event_id:03d}_{event_start_time.strftime('%Y%m%dT%H%M%S')}.wav"
            audio_output_path = Path(output_folder) / output_filename
            event_data = np.concatenate(all_data, axis=0)
            aio.write_audio(str(audio_output_path), event_data, sample_rate)
            
            # Update event_table with the event snippet indices
            event_table.loc[event_table['event_id'] == event_id, ['peak_idx_event_snippet', 'trough_idx_event_snippet', 'midpoint_idx_event_snippet']] = event_eods[['peak_idx_event_snippet', 'trough_idx_event_snippet', 'midpoint_idx_event_snippet']].values
                    
            # Update event_summary with audio info
            event_summary.loc[idx, 'raw_event_data_filename'] = output_filename
            
            actual_duration = len(event_data) / sample_rate
            print(f"        Saved audio: {output_filename} ({actual_duration:.1f}s)")

            # SAVE INDIVIDUAL EVENT DATA (EOD table and waveforms)
            print(f"        Saving individual event data...")
            
            # Save individual event EOD table
            event_eod_table_clean = event_eods.drop(columns=['peak_idx', 'trough_idx', 'midpoint_idx'], errors='ignore')
            event_eod_table_path = Path(output_folder) / f"event_{event_id:03d}_eod_table.csv"
            event_eod_table_clean.to_csv(event_eod_table_path, index=False)
            print(f"        Saved EOD table: {event_eod_table_path.name}")
            
            # Save individual event waveforms (if available)
            if combined_waveforms is not None and 'waveforms' in combined_waveforms:
                # Extract waveforms for this event
                event_waveforms = []
                event_waveform_metadata = []
                
                # Get the indices of EODs in this event within the original combined_waveforms
                for _, eod_row in event_eods.iterrows():
                    original_idx = eod_row.get('original_row_in_file', None)
                    file_idx = eod_row.get('file_index', None)
                    
                    # Try to find the corresponding waveform
                    # This is complex due to the filtering and reorganization that happened
                    # For now, we'll use a simple approach based on the row position
                    if hasattr(event_eods, 'index'):
                        waveform_idx = eod_row.name  # Use pandas index
                        if waveform_idx < len(combined_waveforms['waveforms']):
                            waveform = combined_waveforms['waveforms'][waveform_idx]
                            metadata = combined_waveforms['metadata'][waveform_idx] if 'metadata' in combined_waveforms and waveform_idx < len(combined_waveforms['metadata']) else {}
                            
                            event_waveforms.append(waveform)
                            event_waveform_metadata.append(metadata)
                
                if len(event_waveforms) > 0:
                    event_waveform_base_path = Path(output_folder) / f"event_{event_id:03d}_waveforms"
                    save_variable_length_waveforms(event_waveforms, str(event_waveform_base_path))
                    print(f"        Saved waveforms: {event_waveform_base_path.name}_concatenated.npz")
                else:
                    print(f"        Warning: No waveforms found for event {event_id}")
            
            # Create individual event plot with memory management
            if create_plots:
                print(f"        Creating event plot...")

                # Check event size and determine plotting strategy
                event_duration = len(event_data) / sample_rate
                event_size_mb = event_data.nbytes / (1024 * 1024)
                max_plot_duration = 300  # 5 minutes
                
                print(f"        Event duration: {event_duration:.1f}s, size: {event_size_mb:.1f}MB")
                
                try:
                    # Get all original channel_event_ids that contributed to this merged event
                    merged_channel_event_ids = event_eods['channel_event_id'].unique()

                    # Handle merge_iteration data with fallback
                    if 'merge_iteration' in event_eods.columns:
                        # Create list of (channel_event_id, merge_iteration) pairs and sort by merge_iteration
                        merge_event_data = event_eods[['channel_event_id', 'merge_iteration']].drop_duplicates().sort_values('merge_iteration').reset_index(drop=True)

                        if len(merge_event_data) > 0:
                            # Assign a color to each merge_iteration
                            max_iteration = max(merge_event_data['merge_iteration'])
                            color_map = cm.get_cmap('tab20', max_iteration)
                            channel_event_colors = {nit: color_map(i) for i, nit in enumerate(np.unique(merge_event_data['merge_iteration']))}
                        else:
                            # Fallback if no merge data
                            merge_event_data = pd.DataFrame({'channel_event_id': merged_channel_event_ids, 'merge_iteration': [1] * len(merged_channel_event_ids)})
                            color_map = cm.get_cmap('tab20', len(merged_channel_event_ids))
                            channel_event_colors = {1: color_map(0)}
                    else:
                        print(f"        No merged event ID data available for event {event_id}, using default coloring")
                        # Create fallback merge_event_data
                        merge_event_data = pd.DataFrame({'channel_event_id': merged_channel_event_ids, 'merge_iteration': [1] * len(merged_channel_event_ids)})
                        color_map = cm.get_cmap('tab20', len(merged_channel_event_ids))
                        channel_event_colors = {1: color_map(0)}

                    if len(event_eods) > 0:
                        # Calculate offset for stacking differential channels
                        offset_diff = np.max(event_eods['eod_amplitude']) * 1.5

                        plt.figure(figsize=(20, 8))
                        
                        # Simple downsampling strategy: 3x downsampling for events > 5 minutes
                        if event_duration > max_plot_duration:
                            plot_step = 3  # Fixed 3x downsampling for long events
                            print(f"        Long event detected - applying 3x downsampling")
                        else:
                            plot_step = 1  # Original resolution for shorter events
                            print(f"        Normal event - using original resolution")
                        
                        for i in range(event_data.shape[1]-1):
                            # Calculate differential signal with memory-efficient approach
                            data_diff = np.diff(event_data[::plot_step, i:i+2], axis=1).flatten()
                            x_coords = np.arange(0, len(event_data), plot_step)[:len(data_diff)]
                            plt.plot(x_coords, data_diff + i * offset_diff, linewidth=0.5, label=f'Ch{i}-{i+1}')

                            # Plot detected pulses for this channel
                            ch_eods = event_eods[event_eods['eod_channel'] == i]
                            if len(ch_eods) > 0:
                                # Plot peaks (red)
                                if 'peak_idx_event_snippet' in ch_eods.columns:
                                    peak_indices = ch_eods['peak_idx_event_snippet'].values.astype(np.int64)
                                    valid_peaks = (peak_indices >= 0) & (peak_indices < len(event_data)) & (~np.isnan(peak_indices))
                                    if np.any(valid_peaks):
                                        valid_peak_samples = peak_indices[valid_peaks]
                                        if len(valid_peak_samples) > 0:
                                            peak_diffs = np.diff(event_data[valid_peak_samples, i:i+2], axis=1).flatten()
                                            plt.plot(valid_peak_samples, peak_diffs + i * offset_diff, 
                                                    'o', markersize=3, color='red', alpha=0.8, label='Peaks' if i == 0 else "")
                                # Plot troughs (blue)
                                if 'trough_idx_event_snippet' in ch_eods.columns:
                                    trough_indices = ch_eods['trough_idx_event_snippet'].values.astype(np.int64)
                                    valid_troughs = (trough_indices >= 0) & (trough_indices < len(event_data)) & (~np.isnan(trough_indices))
                                    if np.any(valid_troughs):
                                        valid_trough_samples = trough_indices[valid_troughs]
                                        if len(valid_trough_samples) > 0:
                                            trough_diffs = np.diff(event_data[valid_trough_samples, i:i+2], axis=1).flatten()
                                            plt.plot(valid_trough_samples, trough_diffs + i * offset_diff, 
                                                    'o', markersize=3, color='blue', alpha=0.8, label='Troughs' if i == 0 else "")
                            
                            # Clean up large arrays immediately
                            del data_diff, x_coords
                            gc.collect()

                        # Overlay colored boxes for each original channel event
                        for _, merge_event_row in merge_event_data.iterrows():
                            cid = merge_event_row['channel_event_id']
                            iteration = merge_event_row['merge_iteration']
                            ce_eods = event_eods[event_eods['channel_event_id'] == cid]
                            if len(ce_eods) == 0:
                                continue
                            ch = ce_eods['eod_channel'].iloc[0]
                            # Get min/max sample for this channel event
                            if 'midpoint_idx_event_snippet' in ce_eods.columns:
                                min_sample = int(np.nanmin(ce_eods['midpoint_idx_event_snippet']))
                                max_sample = int(np.nanmax(ce_eods['midpoint_idx_event_snippet']))
                            else:
                                min_sample = 0
                                max_sample = len(event_data)-1
                            # Draw a colored rectangle for this channel event
                            rect = mpatches.Rectangle((min_sample, ch * offset_diff - 0.5 * offset_diff),
                                                    max_sample - min_sample,
                                                    offset_diff,
                                                    linewidth=2, edgecolor=channel_event_colors[iteration], facecolor='none', alpha=0.7, zorder=10)
                            plt.gca().add_patch(rect)
                            
                            # Add event number annotation in the top-left corner of the rectangle
                            text_x = min_sample + (max_sample - min_sample) * 0.05  # 5% from left edge
                            text_y = ch * offset_diff + 0.3 * offset_diff  # Upper part of the channel
                            plt.text(text_x, text_y, str(iteration), 
                                    fontsize=12, fontweight='bold', 
                                    color=channel_event_colors[iteration], 
                                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor=channel_event_colors[iteration], alpha=0.8),
                                    ha='center', va='center', zorder=15)
                            
                            # Draw a line at the mean midpoint
                            if 'midpoint_idx_event_snippet' in ce_eods.columns:
                                mean_sample = int(np.nanmean(ce_eods['midpoint_idx_event_snippet']))
                                plt.plot([mean_sample], [ch * offset_diff], marker='s', color=channel_event_colors[iteration], markersize=10, zorder=11)

                        # Draw lines connecting the mean midpoints of the merged channel events (showing merge path)
                        midpoints = []
                        for cid in merged_channel_event_ids:
                            ce_eods = event_eods[event_eods['channel_event_id'] == cid]
                            ch = ce_eods['eod_channel'].iloc[0]
                            if 'midpoint_idx_event_snippet' in ce_eods.columns:
                                mean_sample = int(np.nanmean(ce_eods['midpoint_idx_event_snippet']))
                            else:
                                mean_sample = 0
                            midpoints.append((mean_sample, ch * offset_diff))
                        midpoints = sorted(midpoints, key=lambda x: x[0])
                        if len(midpoints) > 1:
                            plt.plot([m[0] for m in midpoints], [m[1] for m in midpoints], color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=12)

                        plt.ylim(bottom=None, top=(event_data.shape[1]-1.5)*offset_diff)
                        if plot_step > 1:
                            plt.title(f'Event {event_id} - Differential EOD Detections (downsampled {plot_step}x)\nDuration: {event_duration:.1f}s - Colored boxes: merged channel events')
                        else:
                            plt.title(f'Event {event_id} - Differential EOD Detections\nDuration: {event_duration:.1f}s - Colored boxes: merged channel events')
                        plt.legend(loc='upper right')
                        plt.xlabel('Sample (relative to event snippet)')
                        plt.ylabel('Voltage (stacked by channel)')

                        # Save plot
                        plot_filename = f'event_{event_id:03d}_differential_detection.png'
                        plot_path = Path(output_folder) / plot_filename
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()

                        print(f"        Created plot: {plot_filename}")
                
                except Exception as e:
                    print(f"        WARNING: Plot creation failed for event {event_id}: {e}")
                    print(f"        Event will be processed without plots.")
                    # Create a simple text file with event info instead
                    try:
                        info_filename = f'event_{event_id:03d}_info.txt'
                        info_path = Path(output_folder) / info_filename
                        with open(info_path, 'w') as f:
                            f.write(f"Event {event_id} - Plot creation failed\n")
                            f.write(f"Duration: {event_duration:.1f} seconds\n")
                            f.write(f"Size: {event_size_mb:.1f} MB\n")
                            f.write(f"Number of EODs: {len(event_eods)}\n")
                            f.write(f"Error: {str(e)}\n")
                        print(f"        Created info file instead: {info_filename}")
                    except:
                        pass
            
            print(f"        Event {event_id} completed successfully")
            
        except Exception as e:
            print(f"        ERROR processing event {event_id}: {e}")
            # Continue with next event rather than crashing
            continue
            
        finally:
            # Clean up large variables after each event to free memory
            if 'all_data' in locals():
                del all_data
            if 'event_data' in locals():
                del event_data
            if 'data_diff' in locals():
                del data_diff
            if 'merge_event_data' in locals():
                del merge_event_data
            if 'channel_event_colors' in locals():
                del channel_event_colors
            if 'event_waveforms' in locals():
                del event_waveforms
            if 'event_waveform_metadata' in locals():
                del event_waveform_metadata
            gc.collect()
            
            if (idx + 1) % 5 == 0:  # Print memory status every 5 events
                print_memory_usage(f"After processing {idx + 1} events")

else:
    print("  No wav files found - skipping raw data extraction")
    event_summary['raw_event_data_filename'] = 'NO_WAV_FILES'

# Clean up audio processing variables
if 'wav_files' in locals():
    del wav_files
if 'file_times_df' in locals():
    del file_times_df
if 'cor_factors' in locals():
    del cor_factors
gc.collect()
print("  Memory cleanup completed after individual event processing")
print_memory_usage("After individual event processing")

#%%
# =============================================================================
# STEP 10: SAVE FINAL COMBINED RESULTS
# =============================================================================
print(f"\nStep 10: Saving final combined results...")

output_path = Path(output_folder)

# # Save combined event table (with snippet-relative indices)
# event_table_clean = event_table.drop(columns=['peak_idx', 'trough_idx', 'midpoint_idx'], errors='ignore')
# event_table_path = output_path / "session_events_eod_table.csv"
# event_table_clean.to_csv(event_table_path, index=False)
# print(f"  Saved combined event table: {event_table_path}")

# Save updated event summary (with audio filenames)
event_summary_path = output_path / "session_events_summary.csv"
event_summary.to_csv(event_summary_path, index=False)
print(f"  Saved updated event summary: {event_summary_path}")

# # Save combined waveforms if available (for reference)
# if combined_waveforms is not None:
#     waveform_base_path = output_path / "session_events_waveforms"
#     save_variable_length_waveforms(combined_waveforms['waveforms'], str(waveform_base_path))
#     print(f"  Saved combined waveforms: {waveform_base_path}")

# Save parameters
params_df = pd.DataFrame({
    'parameter': ['max_ipi_seconds', 'min_eods_per_event', 'max_merge_gap_seconds', 'sample_rate', 'file_duration', 'margin', 'min_channel_event_size', 'min_amplitude', 'clustering_enabled', 'dbscan_eps', 'dbscan_min_samples'],
    'value': [max_ipi_seconds, min_eods_per_event, max_merge_gap_seconds, sample_rate, file_duration, margin, min_channel_event_size, min_amplitude, clustering_enabled, dbscan_eps, dbscan_min_samples]
})
params_path = output_path / "session_events_extraction_params.csv"
params_df.to_csv(params_path, index=False)
print(f"  Saved parameters: {params_path}")

print(f"\nNote: Individual event data was saved during processing in Step 9:")
print(f"  - event_XXX_eod_table.csv: EOD data for each individual event")
print(f"  - event_XXX_waveforms_concatenated.npz: Waveforms for each event (if available)")
print(f"  - event_XXX_YYYYMMDDTHHMMSS.wav: Raw audio data for each event")

# =============================================================================
# STEP 11: CREATE TIMELINE PLOT
# =============================================================================
print(f"\nStep 11: Creating event timeline...")

if len(event_summary) > 0:
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    plt.figure(figsize=(15, 6))

    # Convert to datetime for plotting
    start_times = pd.to_datetime(event_summary['start_time'])
    durations = event_summary['duration_seconds']
    n_eods = event_summary['n_eods']

    # For coloring merged events
    color_map = cm.get_cmap('tab20', len(event_summary))

    for i, (start, duration, n_eod, event_id) in enumerate(zip(start_times, durations, n_eods, event_summary['event_id'])):
        # Draw main merged event bar
        plt.barh(i, duration/60, left=(start - start_times.min()).total_seconds()/60,
                height=0.8, alpha=0.7, color=color_map(i), label=f'Merged event {event_id}' if i < 5 else "")
        # Overlay boxes for constituent channel events
        event_eods = event_table[event_table['event_id'] == event_id]
        channel_event_ids = event_eods['channel_event_id'].unique()
        channel_event_colors = {cid: color_map(j % 20) for j, cid in enumerate(channel_event_ids)}
        for j, cid in enumerate(channel_event_ids):
            ce_eods = event_eods[event_eods['channel_event_id'] == cid]
            ce_start = ce_eods['timestamp_dt'].min()
            ce_end = ce_eods['timestamp_dt'].max()
            ce_left = (ce_start - start_times.min()).total_seconds()/60
            ce_width = (ce_end - ce_start).total_seconds()/60
            plt.barh(i, ce_width, left=ce_left, height=0.8, alpha=0.3, color=channel_event_colors[cid], edgecolor=channel_event_colors[cid], linewidth=2, zorder=10)
            # Draw a vertical line at the midpoint
            ce_mid = ce_left + ce_width/2
            plt.plot([ce_mid], [i], marker='s', color=channel_event_colors[cid], markersize=8, zorder=11)

    plt.xlabel('Time (minutes)')
    plt.ylabel('Event ID')
    plt.title(f'Event Timeline ({len(event_summary)} events)\nColored boxes: constituent channel events')
    plt.grid(True, alpha=0.3)
    if len(event_summary) <= 5:
        plt.legend()

    timeline_path = output_path / 'event_timeline.png'
    plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created event timeline: {timeline_path}")
    
    # Clean up timeline plotting variables
    del start_times, durations, n_eods, color_map
    if 'channel_event_ids' in locals():
        del channel_event_ids, channel_event_colors
    gc.collect()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n" + "="*60)
print("EVENT EXTRACTION COMPLETED SUCCESSFULLY!")
print("="*60)

print(f"\nExtraction Summary:")
print(f"  Total input EODs: {len(combined_table)}")
print(f"  Channel events (Stage 1): {channel_event_counter}")
print(f"  Merged events (Stage 2): {merged_event_counter}")
print(f"  Final events (Stage 3): {len(event_summary)}")
print(f"  Total EODs in events: {len(event_table)}")
print(f"  Extraction efficiency: {len(event_table)/len(combined_table)*100:.1f}%")

if len(event_summary) > 0:
    print(f"  Event duration range: {event_summary['duration_seconds'].min():.1f} - {event_summary['duration_seconds'].max():.1f} seconds")
    print(f"  EODs per event range: {event_summary['n_eods'].min()} - {event_summary['n_eods'].max()}")
    print(f"  Channels per event range: {event_summary['n_channels'].min()} - {event_summary['n_channels'].max()}")

print(f"\nOutput files saved to: {output_folder}")
print("  Combined files:")
print("    - session_events_eod_table.csv: Combined EOD data for all events")
print("    - session_events_summary.csv: Summary statistics for each event") 
print("    - session_events_waveforms.csv: Combined waveform data (if available)")
print("    - session_events_extraction_params.csv: Extraction parameters")
print("    - event_timeline.png: Timeline visualization of all events")
print("  Individual event files (crash-resistant):")
print("    - event_XXX_eod_table.csv: EOD data for individual events")
print("    - event_XXX_waveforms_concatenated.npz: Waveform data for individual events")
print("    - event_XXX_YYYYMMDDTHHMMSS.wav: Raw audio data for each event (with margins)")
print("  Event plots (memory-optimized):")
print("    - event_XXX_differential_detection.png: Plots for all events")
print("      * Events <5min: Original resolution")
print("      * Events ≥5min: 3x downsampled")
print("    - event_XXX_info.txt: Info files when plotting fails due to memory constraints")
print("="*60)

# Final memory cleanup
del combined_table, sorted_table, event_table, event_summary
if 'combined_waveforms' in locals():
    del combined_waveforms
if 'file_metadata' in locals():
    del file_metadata
gc.collect()
print_memory_usage("Final memory state")
print("Memory cleanup completed.")
