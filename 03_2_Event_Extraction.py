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
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.cm as cm
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
pre_merge_filtering = False  # Enable pre-merge filtering based on size and amplitude criteria

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

#%%
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
    
    if pre_merge_filtering:
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
        
    else:
        filtered_channel_events = channel_events.copy()

    # Note: Waveform loading is now deferred until after merging and filtering (Step 8.5)
    # This saves memory by only loading waveforms for final events that pass all filters
    
    # Update channel_events to the filtered version
    channel_events = filtered_channel_events
    
    # Clean up intermediate filtering variables
    del filtered_channel_events
    gc.collect()
    print("  Memory cleanup completed after filtering")

#%%
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
        # Convert datetime objects to pandas Timestamps for safe formatting
        current_start_pd = pd.Timestamp(current_start)
        current_end_pd = pd.Timestamp(current_end)
        print(f"      Iteration {iteration}: bounds=({current_start_pd.strftime('%H:%M:%S')}-{current_end_pd.strftime('%H:%M:%S')}), channels={sorted(current_channels)}")

        # --- CORRECTED: Spatially AND temporally connected merging ---
        # For each channel currently in the merged event, check its specific temporal bounds
        # Only merge neighbor events that are connected in BOTH space AND time
        events_to_merge_this_iteration = []
        new_channels_this_iteration = set()
        
        # For each channel in current merged event, get its specific temporal bounds
        channel_time_bounds = {}
        for ch in current_channels:
            # Find all events in merged_group for this channel and get their combined time bounds
            ch_event_ids = [eid for eid in merged_group if event_summaries.loc[event_summaries['channel_event_id'] == eid, 'channel'].values[0] == ch]
            if ch_event_ids:
                ch_starts = [event_summaries.loc[event_summaries['channel_event_id'] == eid, 'channel_start_time'].values[0] for eid in ch_event_ids]
                ch_ends = [event_summaries.loc[event_summaries['channel_event_id'] == eid, 'channel_end_time'].values[0] for eid in ch_event_ids]
                channel_time_bounds[ch] = (min(ch_starts), max(ch_ends))
        
        # For each channel with events in the merged group, check its spatial neighbors
        for ch, (ch_start, ch_end) in channel_time_bounds.items():
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
                    
                    # Direct temporal overlap between this channel and neighbor
                    direct_overlap = (neighbor_end >= ch_start) and (neighbor_start <= ch_end)
                    
                    # Sequential connection within gap tolerance
                    gap_after = (neighbor_start - ch_end).total_seconds() if neighbor_start > ch_end else float('inf')
                    gap_before = (ch_start - neighbor_end).total_seconds() if ch_start > neighbor_end else float('inf')
                    sequential_connection = min(gap_after, gap_before) <= max_merge_gap_seconds
                    
                    if direct_overlap:
                        gap = 0
                        gap_type = 'direct_overlap'
                        should_merge = True
                    elif sequential_connection:
                        gap = min(gap_after, gap_before)
                        gap_type = 'sequential_connection'
                        should_merge = True
                    else:
                        gap = min(gap_after, gap_before)
                        gap_type = 'too_distant'
                        should_merge = False
                    
                    if should_merge:
                        # Avoid duplicates - check if this neighbor event is already in the merge list
                        already_added = any(event_info['event_id'] == neighbor_id for event_info in events_to_merge_this_iteration)
                        if not already_added:
                            # Convert to pandas Timestamp for safe formatting
                            ch_start_pd = pd.Timestamp(ch_start)
                            ch_end_pd = pd.Timestamp(ch_end)
                            print(f"          CANDIDATE for merging: event {neighbor_id} ch{neighbor_ch} ({gap_type}, gap: {gap:.2f}s) connected to ch{ch} ({ch_start_pd.strftime('%H:%M:%S')}-{ch_end_pd.strftime('%H:%M:%S')})")
                            events_to_merge_this_iteration.append({
                                'event_id': neighbor_id,
                                'channel': neighbor_ch,
                                'start_time': neighbor_start,
                                'end_time': neighbor_end,
                                'gap': gap,
                                'gap_type': gap_type,
                                'connected_to_channel': ch
                            })
                            processed_channel_events.add(neighbor_id)
                    # else:
                        # print(f"          REJECTED: event {neighbor_id} ch{neighbor_ch} not connected to ch{ch} ({gap_type}, gap: {gap:.2f}s)")

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
            # Convert datetime objects to pandas Timestamps for safe formatting
            current_start_pd = pd.Timestamp(current_start)
            current_end_pd = pd.Timestamp(current_end)
            print(f"          UPDATED bounds after parallel merge: ({current_start_pd.strftime('%H:%M:%S')}-{current_end_pd.strftime('%H:%M:%S')})")
        else:
            print(f"        No qualifying events found for any temporal edge of channels {sorted(current_channels)}")
        if not found_new_merge:
            print(f"        No more mergeable neighbors found for any temporal edge of channels {sorted(current_channels)}")
        elif len(new_channels_this_iteration) == 0:
            print(f"        Temporal bounds changed but no new channels added - stopping to prevent infinite loop")
            found_new_merge = False
    
    # Assign merged event ID to all EODs in the final merged group
    print(f"      Final merged group after {iteration} iterations: {merged_group} -> merged_event_id {merged_event_counter}")
    # Convert datetime objects to pandas Timestamps for safe formatting
    current_start_pd = pd.Timestamp(current_start)
    current_end_pd = pd.Timestamp(current_end)
    print(f"      Final channels: {sorted(current_channels)}, final bounds: ({current_start_pd.strftime('%H:%M:%S')}-{current_end_pd.strftime('%H:%M:%S')})")
    
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
#%%
# =============================================================================
# STEP 7: EXTRACT EVENTS - STAGE 3 (2. SIZE AND AMP FILTERING)
# =============================================================================
print(f"\nStep 7: Stage 3 - Filtering merged events...")
print(f"  Removing events with <{min_eods_per_event} pulses OR no pulse ≥{min_amplitude} amplitude...")

# Post-merge filter: remove events based on size AND amplitude criteria

# Criterion 1: Size filter (at least min_eods_per_event EODs)
# Count EODs per merged event
event_sizes = merged_events.groupby('merged_event_id').size()
print(f"  Event sizes: min={event_sizes.min()}, max={event_sizes.max()}, mean={event_sizes.mean():.1f}")
valid_size_ids = event_sizes[event_sizes >= min_eods_per_event].index

# Criterion 2: Amplitude filter (at least one pulse with amplitude ≥ min_amplitude)
event_max_amplitudes = merged_events.groupby('merged_event_id')['eod_amplitude'].max()
valid_amplitude_ids = event_max_amplitudes[event_max_amplitudes >= min_amplitude].index

# Combined filter: channel events must pass BOTH criteria
valid_event_ids = valid_size_ids.intersection(valid_amplitude_ids)

# Calculate removal statistics
n_total = merged_events['merged_event_id'].nunique()
n_removed_size = n_total - len(valid_size_ids)
n_removed_amplitude = n_total - len(valid_amplitude_ids)
n_removed_total = n_total - len(valid_event_ids)

print(f"  Events passing size filter: {len(valid_event_ids)} / {len(event_sizes)}")
print(f"    Original channel events: {n_total}")
print(f"    Failed size criterion (<{min_eods_per_event} pulses): {n_removed_size}")
print(f"    Failed amplitude criterion (max amplitude <{min_amplitude}): {n_removed_amplitude}")
print(f"    Remaining after combined filter: {len(valid_event_ids)}")
print(f"    Total removed: {n_removed_total}")

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
#%%
# =============================================================================
# STEP 8: LOAD WAVEFORMS FOR FINAL EVENTS (MEMORY-OPTIMIZED)
# =============================================================================
print(f"\nStep 8: Loading waveforms for final events...")
print(f"  Loading waveforms only for {len(event_table)} EODs in final {event_table['event_id'].nunique()} events")

if len(files_with_waveforms) > 0:
    # Create mapping of which EODs we need from each file AND their target positions
    filtered_eods_by_file = {}
    eod_position_mapping = {}  # Maps (file_idx, original_row) to position in event_table
    
    for order_idx, (_, eod_row) in enumerate(event_table.iterrows()):
        file_idx = eod_row['file_index']
        original_row = eod_row['original_row_in_file']
        
        if file_idx not in filtered_eods_by_file:
            filtered_eods_by_file[file_idx] = []
        filtered_eods_by_file[file_idx].append(original_row)
        eod_position_mapping[(file_idx, original_row)] = order_idx
    
    print(f"  Need to load waveforms from {len(filtered_eods_by_file)} files")
    
    # Pre-allocate waveform list with None placeholders
    all_filtered_waveforms = [None] * len(event_table)
    
    for file_idx, needed_rows in filtered_eods_by_file.items():
        file_info = file_metadata[file_idx]
        
        if not file_info['has_waveforms']:
            print(f"    Skipping {file_info['base_name']} - no waveform files")
            continue
            
        print(f"    Loading {len(needed_rows)} waveforms from {file_info['base_name']}")
        
        # Load ALL waveforms from this file
        waveforms = load_variable_length_waveforms(file_info['waveform_base_path'])
        
        # Extract waveforms and place them at correct positions in pre-allocated list
        for row_idx in needed_rows:
            if row_idx < len(waveforms):
                # Get the correct position in the event_table order
                target_position = eod_position_mapping[(file_idx, row_idx)]
                all_filtered_waveforms[target_position] = waveforms[row_idx]
            else:
                print(f"      Warning: Row {row_idx} not found in waveforms (file has {len(waveforms)} waveforms)")
            
        # Clean up large temporary variables immediately after use
        del waveforms
        
        # Force garbage collection after processing each file to free memory
        gc.collect()
        print(f"      Memory freed after processing {file_info['base_name']}")
    
    # Check for any None values (missing waveforms) and filter them out
    valid_waveforms = [w for w in all_filtered_waveforms if w is not None]
    missing_count = len(all_filtered_waveforms) - len(valid_waveforms)
    
    if missing_count > 0:
        print(f"      Warning: {missing_count} waveforms could not be loaded")
        # Filter out corresponding EODs from event_table
        valid_indices = [i for i, w in enumerate(all_filtered_waveforms) if w is not None]
        event_table = event_table.iloc[valid_indices].reset_index(drop=True)
        all_filtered_waveforms = valid_waveforms
    
    if all_filtered_waveforms:
        print(f"  Loaded {len(all_filtered_waveforms)} waveforms for final events")
        print(f"  Waveform-EOD order matches: each waveform corresponds to same row in event_table")
    else:
        print("  No waveforms loaded")
        
    # Clean up temporary waveform loading variables
    del filtered_eods_by_file, eod_position_mapping
    
    # Force garbage collection after all waveform loading is complete
    gc.collect()
    print("  Memory cleanup completed after optimized waveform loading")
    print_memory_usage("After optimized waveform loading")
    
    # Verification: Check that waveform_length matches actual waveform lengths
    print(f"  Verifying waveform-EOD correspondence...")
    if len(all_filtered_waveforms) > 0 and len(event_table) > 0:
        l_r = []
        mismatches = 0
        for i, l in enumerate(event_table['waveform_length']):
            if i < len(all_filtered_waveforms):
                actual_length = len(all_filtered_waveforms[i])
                ratio = l / actual_length
                l_r.append(ratio)
                if abs(ratio - 1.0) > 0.001:  # Allow small floating point differences
                    mismatches += 1
            else:
                l_r.append(float('nan'))
                mismatches += 1
        
        print(f"    Checked {len(l_r)} waveform-EOD pairs for final events")
        if len(l_r) > 0:
            print(f"    Length ratio statistics: min={min(l_r):.4f}, max={max(l_r):.4f}, mean={sum(l_r)/len(l_r):.4f}")
            print(f"    Mismatches (ratio != 1.0): {mismatches}")
            
            if mismatches == 0:
                print("    ✓ SUCCESS: All final event waveforms correspond to correct EOD entries!")
                
                # Create enhanced waveform-metadata structure with robust indexing
                print("  Creating enhanced waveform-metadata structure...")
                waveform_metadata_table = []
                for i, (_, eod_row) in enumerate(event_table.iterrows()):
                    if i < len(all_filtered_waveforms):
                        waveform_metadata_table.append({
                            'global_index': i,  # Index in all_filtered_waveforms
                            'event_id': eod_row['event_id'],
                            'channel_event_id': eod_row['channel_event_id'],
                            'file_index': eod_row['file_index'],
                            'original_row_in_file': eod_row['original_row_in_file'],
                            'eod_channel': eod_row['eod_channel'],
                            'timestamp_dt': eod_row['timestamp_dt'],
                            'waveform_length': eod_row['waveform_length'],
                            'actual_waveform_length': len(all_filtered_waveforms[i])
                        })
                
                waveform_metadata_df = pd.DataFrame(waveform_metadata_table)
                
                # Create combined_waveforms structure with metadata
                combined_waveforms = {
                    'waveforms': all_filtered_waveforms,
                    'metadata_df': waveform_metadata_df
                }
                
                print(f"    Created metadata table with {len(waveform_metadata_table)} entries")
                print("    ✓ Enhanced structure ready for individual event processing")
                
            else:
                print(f"    ⚠ WARNING: {mismatches} waveforms don't match their EOD entries")
                print("    Creating basic structure despite mismatches...")
                combined_waveforms = {'waveforms': all_filtered_waveforms}
        
else:
    print("  No files with waveforms found")
    all_filtered_waveforms = []
    combined_waveforms = None
#%%
# =============================================================================
# STEP 9: CREATE EVENT SUMMARY (BEFORE INDIVIDUAL PROCESSING)
# =============================================================================
print(f"\nStep 9: Creating event summary...")

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
# STEP 10: PROCESS INDIVIDUAL EVENTS (AUDIO EXTRACTION + INDIVIDUAL SAVES)
# =============================================================================
print(f"\nStep 10: Processing individual events (audio extraction and individual saves)...")

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
                print(f"        Extracting waveforms for event {event_id}...")
                event_waveforms = []
                event_waveform_metadata = []
                
                # Use enhanced metadata structure for robust waveform-EOD correspondence
                if 'metadata_df' in combined_waveforms:
                    # Enhanced approach: use metadata table to find correct waveforms
                    metadata_df = combined_waveforms['metadata_df']
                    
                    for _, eod_row in event_eods.iterrows():
                        # Find matching metadata entry based on multiple criteria
                        matching_meta = metadata_df[
                            (metadata_df['event_id'] == eod_row['event_id']) &
                            (metadata_df['channel_event_id'] == eod_row['channel_event_id']) &
                            (metadata_df['eod_channel'] == eod_row['eod_channel']) &
                            (metadata_df['timestamp_dt'] == eod_row['timestamp_dt'])
                        ]
                        
                        if len(matching_meta) == 1:
                            # Found unique match
                            global_idx = matching_meta.iloc[0]['global_index']
                            if global_idx < len(combined_waveforms['waveforms']):
                                waveform = combined_waveforms['waveforms'][global_idx]
                                metadata = {'global_index': global_idx, 'metadata_row': matching_meta.iloc[0].to_dict()}
                                
                                event_waveforms.append(waveform)
                                event_waveform_metadata.append(metadata)
                                
                                # Verify length match
                                expected_length = eod_row['waveform_length']
                                if len(waveform) != expected_length:
                                    print(f"          Warning: Length mismatch for EOD at timestamp {eod_row['timestamp_dt']} (expected {expected_length}, got {len(waveform)})")
                            else:
                                print(f"          Warning: Global index {global_idx} out of range for waveforms array")
                        elif len(matching_meta) == 0:
                            print(f"          Warning: No metadata match found for EOD at timestamp {eod_row['timestamp_dt']}")
                        else:
                            print(f"          Warning: Multiple metadata matches found for EOD at timestamp {eod_row['timestamp_dt']}")
                    
                    print(f"          Successfully extracted {len(event_waveforms)}/{len(event_eods)} waveforms using metadata table")
                    
                else:
                    # Fallback approach: direct indexing (original problematic method)
                    print(f"          Warning: No metadata table available, using fallback indexing")
                    for _, eod_row in event_eods.iterrows():
                        # Try to find the corresponding waveform using pandas index
                        waveform_idx = eod_row.name  # Use pandas index (problematic!)
                        if waveform_idx < len(combined_waveforms['waveforms']):
                            waveform = combined_waveforms['waveforms'][waveform_idx]
                            metadata = {'fallback_index': waveform_idx}
                            
                            event_waveforms.append(waveform)
                            event_waveform_metadata.append(metadata)

                # Final verification: Check that all waveforms match expected lengths
                valid_waveforms = []
                verification_passed = True
                for i, waveform in enumerate(event_waveforms):
                    if i < len(event_eods):
                        expected_length = event_eods.iloc[i]['waveform_length']
                        if len(waveform) == expected_length:
                            valid_waveforms.append(waveform)
                        else:
                            print(f"        ⚠ VERIFICATION FAILED: Waveform {i} length mismatch (expected {expected_length}, got {len(waveform)})")
                            verification_passed = False
                    else:
                        print(f"        ⚠ VERIFICATION FAILED: More waveforms than EOD entries")
                        verification_passed = False

                if verification_passed:
                    print(f"        ✓ VERIFICATION PASSED: All {len(event_waveforms)} waveforms match expected lengths")
                else:
                    print(f"        ✗ VERIFICATION FAILED: Using valid waveforms only ({len(valid_waveforms)}/{len(event_waveforms)})")
                    event_waveforms = valid_waveforms

                if len(event_waveforms) > 0:
                    event_waveform_base_path = Path(output_folder) / f"event_{event_id:03d}_waveforms"
                    save_variable_length_waveforms(event_waveforms, str(event_waveform_base_path))
                    print(f"        Saved waveforms: {event_waveform_base_path.name}_concatenated.npz")
                else:
                    print(f"        Warning: No valid waveforms found for event {event_id}")
            else:
                print(f"        No waveforms available for event {event_id}")
            
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
                            
                            # Create real-time x-coordinates for downsampled data
                            time_indices = np.arange(0, len(event_data), plot_step)[:len(data_diff)]
                            time_offsets = pd.to_timedelta(time_indices / sample_rate, unit='s')
                            x_coords = event_start_time + time_offsets
                            plt.plot(x_coords, data_diff + i * offset_diff, linewidth=0.5, label=f'Ch{i}-{i+1}')

                            # Plot detected pulses for this channel
                            ch_eods = event_eods[event_eods['eod_channel'] == i]
                            if len(ch_eods) > 0:
                                # Plot peaks (red)
                                if 'peak_idx_event_snippet' in ch_eods.columns:
                                    peak_sample_indices = ch_eods['peak_idx_event_snippet'].values.astype(np.int64)
                                    valid_peaks = (peak_sample_indices >= 0) & (peak_sample_indices < len(event_data)) & (~np.isnan(peak_sample_indices))
                                    if np.any(valid_peaks):
                                        valid_peak_samples = peak_sample_indices[valid_peaks]
                                        peak_timestamps = event_start_time + pd.to_timedelta(valid_peak_samples / sample_rate, unit='s')
                                        peak_diffs = np.diff(event_data[valid_peak_samples, i:i+2], axis=1).flatten()
                                        plt.plot(peak_timestamps, peak_diffs + i * offset_diff, 
                                                'o', markersize=3, color='red', alpha=0.8, label='Peaks' if i == 0 else "")
                                        
                                # Plot troughs (blue)
                                if 'trough_idx_event_snippet' in ch_eods.columns:
                                    trough_sample_indices = ch_eods['trough_idx_event_snippet'].values.astype(np.int64)
                                    valid_troughs = (trough_sample_indices >= 0) & (trough_sample_indices < len(event_data)) & (~np.isnan(trough_sample_indices))
                                    if np.any(valid_troughs):
                                        valid_trough_samples = trough_sample_indices[valid_troughs]
                                        trough_timestamps = event_start_time + pd.to_timedelta(valid_trough_samples / sample_rate, unit='s')
                                        trough_diffs = np.diff(event_data[valid_trough_samples, i:i+2], axis=1).flatten()
                                        plt.plot(trough_timestamps, trough_diffs + i * offset_diff, 
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
                            # Get min/max sample for this channel event and convert to timestamps
                            if 'midpoint_idx_event_snippet' in ce_eods.columns:
                                min_sample = int(np.nanmin(ce_eods['midpoint_idx_event_snippet']))
                                max_sample = int(np.nanmax(ce_eods['midpoint_idx_event_snippet']))
                                # Convert sample indices to timestamps
                                min_timestamp = event_start_time + pd.to_timedelta(min_sample / sample_rate, unit='s')
                                max_timestamp = event_start_time + pd.to_timedelta(max_sample / sample_rate, unit='s')
                                width_seconds = (max_timestamp - min_timestamp).total_seconds()
                            else:
                                min_timestamp = event_start_time
                                max_timestamp = event_start_time + pd.to_timedelta(len(event_data) / sample_rate, unit='s')
                                width_seconds = (max_timestamp - min_timestamp).total_seconds()
                            
                            # Draw a colored rectangle for this channel event
                            # Convert timestamps to matplotlib-compatible format for Rectangle
                            min_timestamp_mpl = mdates.date2num(min_timestamp.to_pydatetime())
                            width_days = width_seconds / 86400.0  # Convert seconds to days for matplotlib
                            
                            rect = mpatches.Rectangle((min_timestamp_mpl, ch * offset_diff - 0.5 * offset_diff),
                                                    width_days,
                                                    offset_diff,
                                                    linewidth=2, edgecolor=channel_event_colors[iteration], facecolor='none', alpha=0.7, zorder=10)
                            plt.gca().add_patch(rect)
                            
                            # Add event number annotation in the top-left corner of the rectangle
                            text_timestamp_mpl = min_timestamp_mpl + width_days * 0.05  # 5% from left edge in matplotlib time units
                            text_y = ch * offset_diff + 0.3 * offset_diff  # Upper part of the channel
                            plt.text(text_timestamp_mpl, text_y, str(iteration), 
                                    fontsize=12, fontweight='bold', 
                                    color=channel_event_colors[iteration], 
                                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor=channel_event_colors[iteration], alpha=0.8),
                                    ha='center', va='center', zorder=15)
                            
                            # Draw a line at the mean midpoint
                            if 'midpoint_idx_event_snippet' in ce_eods.columns:
                                mean_sample = int(np.nanmean(ce_eods['midpoint_idx_event_snippet']))
                                mean_timestamp = event_start_time + pd.to_timedelta(mean_sample / sample_rate, unit='s')
                                mean_timestamp_mpl = mdates.date2num(mean_timestamp.to_pydatetime())
                                plt.plot([mean_timestamp_mpl], [ch * offset_diff], marker='s', color=channel_event_colors[iteration], markersize=10, zorder=11)

                        # Draw lines connecting the mean midpoints of the merged channel events (showing merge path)
                        midpoints = []
                        for cid in merged_channel_event_ids:
                            ce_eods = event_eods[event_eods['channel_event_id'] == cid]
                            ch = ce_eods['eod_channel'].iloc[0]
                            if 'midpoint_idx_event_snippet' in ce_eods.columns:
                                mean_sample = int(np.nanmean(ce_eods['midpoint_idx_event_snippet']))
                                mean_timestamp = event_start_time + pd.to_timedelta(mean_sample / sample_rate, unit='s')
                                mean_timestamp_mpl = mdates.date2num(mean_timestamp.to_pydatetime())
                            else:
                                mean_timestamp_mpl = mdates.date2num(event_start_time.to_pydatetime())
                            midpoints.append((mean_timestamp_mpl, ch * offset_diff))
                        midpoints = sorted(midpoints, key=lambda x: x[0])
                        if len(midpoints) > 1:
                            plt.plot([m[0] for m in midpoints], [m[1] for m in midpoints], color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=12)

                        plt.ylim(bottom=None, top=(event_data.shape[1]-1.5)*offset_diff)
                        if plot_step > 1:
                            plt.title(f'Event {event_id} - Differential EOD Detections (downsampled {plot_step}x)\nDuration: {event_duration:.1f}s - Colored boxes: merged channel events')
                        else:
                            plt.title(f'Event {event_id} - Differential EOD Detections\nDuration: {event_duration:.1f}s - Colored boxes: merged channel events')
                        plt.legend(loc='upper right')
                        plt.xlabel('Time')
                        plt.ylabel('Voltage (stacked by channel)')
                        
                        # Format x-axis for better timestamp readability
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                        plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, int(event_duration/10))))
                        plt.xticks(rotation=45)

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
# STEP 11: SAVE FINAL COMBINED RESULTS
# =============================================================================
print(f"\nStep 11: Saving final combined results...")

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
    'parameter': ['max_ipi_seconds', 'min_eods_per_event', 'max_merge_gap_seconds', 'sample_rate', 'file_duration', 'margin', 'min_channel_event_size', 'min_amplitude', 'pre_merge_filtering', 'clustering_enabled', 'dbscan_eps', 'dbscan_min_samples'],
    'value': [max_ipi_seconds, min_eods_per_event, max_merge_gap_seconds, sample_rate, file_duration, margin, min_channel_event_size, min_amplitude, pre_merge_filtering, clustering_enabled, dbscan_eps, dbscan_min_samples]
})

params_path = output_path / "session_events_extraction_params.csv"
params_df.to_csv(params_path, index=False)
print(f"  Saved parameters: {params_path}")

print(f"\nNote: Individual event data was saved during processing in Step 9:")
print(f"  - event_XXX_eod_table.csv: EOD data for each individual event")
print(f"  - event_XXX_waveforms_concatenated.npz: Waveforms for each event (if available)")
print(f"  - event_XXX_YYYYMMDDTHHMMSS.wav: Raw audio data for each event")

# =============================================================================
# STEP 12: CREATE TIMELINE PLOT
# =============================================================================
print(f"\nStep 12: Creating event timeline...")

if len(event_summary) > 0:
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
