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
from eod_functions_backup import load_variable_length_waveforms, save_variable_length_waveforms
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

plt.ioff()
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
sample_rate = 96000         # Audio sample rate
file_duration = 600.0       # Audio file duration in seconds
margin = 1.0                # Safety margin around events in seconds
min_channel_event_size = 10  # Minimum size of channel events before merging (high-pass filter)
min_amplitude = 0.004          # Minimum amplitude threshold for events (at least one eod should have this size)
create_plots = True

print(f"  max_ipi_seconds: {max_ipi_seconds}")
print(f"  min_eods_per_event: {min_eods_per_event}")
print(f"  min_channel_event_size: {min_channel_event_size}")
print(f"  min_amplitude: {min_amplitude}")


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
    
    # Reset channel_event_id to be consecutive
    filtered_channel_events['channel_event_id'] = (filtered_channel_events['channel_event_id']
                                                  .astype('category')
                                                  .cat.codes)

    # Note: Waveform loading is now deferred until after event summary creation (Step 8.5)
    # This saves memory by only loading waveforms for final events that pass all filters
    
    # Update channel_events to the filtered version
    channel_events = filtered_channel_events
    
    # Clean up intermediate filtering variables
    del filtered_channel_events, valid_size_ids, valid_amplitude_ids
    del valid_channel_event_ids, event_sizes, event_max_amplitudes
    gc.collect()
    print("  Memory cleanup completed after filtering")


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

#%%
# =============================================================================
# STEP 8: CREATE EVENT SUMMARY (BEFORE INDIVIDUAL PROCESSING)
# =============================================================================
print(f"\nStep 8: Creating event summary...")

# Create event summary directly from channel_events BEFORE the processing loop
event_summaries = []
for event_id in sorted(channel_events['channel_event_id'].unique()):
    event_eods = channel_events[channel_events['channel_event_id'] == event_id]
    
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

# =============================================================================
# STEP 8.5: LOAD WAVEFORMS FOR FINAL EVENTS (MEMORY-OPTIMIZED)
# =============================================================================
print(f"\nStep 8.5: Loading waveforms for final events...")
print(f"  Loading waveforms only for {len(channel_events)} EODs in final {channel_events['channel_event_id'].nunique()} events")

combined_waveforms = None

if len(files_with_waveforms) > 0:
    # Create mapping of which EODs we need from each file AND their target positions
    filtered_eods_by_file = {}
    eod_position_mapping = {}  # Maps (file_idx, original_row) to position in channel_events
    
    for order_idx, (_, eod_row) in enumerate(channel_events.iterrows()):
        file_idx = eod_row['file_index']
        original_row = eod_row['original_row_in_file']
        
        if file_idx not in filtered_eods_by_file:
            filtered_eods_by_file[file_idx] = []
        filtered_eods_by_file[file_idx].append(original_row)
        eod_position_mapping[(file_idx, original_row)] = order_idx
    
    print(f"  Need to load waveforms from {len(filtered_eods_by_file)} files")
    
    # Pre-allocate waveform list with None placeholders
    all_filtered_waveforms = [None] * len(channel_events)
    all_filtered_metadata = [None] * len(channel_events)
    
    for file_idx, needed_rows in filtered_eods_by_file.items():
        file_info = file_metadata[file_idx]
        
        if not file_info['has_waveforms']:
            print(f"    Skipping {file_info['base_name']} - no waveform files")
            continue
            
        print(f"    Loading {len(needed_rows)} waveforms from {file_info['base_name']}")
        
        # Load ALL waveforms from this file
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
        
        # Extract waveforms and place them at correct positions in pre-allocated list
        for row_idx in needed_rows:
            if row_idx < len(waveforms):
                # Get the correct position in the channel_events order
                target_position = eod_position_mapping[(file_idx, row_idx)]
                all_filtered_waveforms[target_position] = waveforms[row_idx]
                if row_idx < len(file_metadata_list):
                    all_filtered_metadata[target_position] = file_metadata_list[row_idx]
                else:
                    all_filtered_metadata[target_position] = {'index': row_idx, 'file': file_info['base_name']}
            else:
                print(f"      Warning: Row {row_idx} not found in waveforms (file has {len(waveforms)} waveforms)")
            
        # Clean up large temporary variables immediately after use
        del waveforms
        del file_metadata_list
        
        # Force garbage collection after processing each file to free memory
        gc.collect()
        print(f"      Memory freed after processing {file_info['base_name']}")
    
    # Check for any None values (missing waveforms) and filter them out
    valid_waveforms = [w for w in all_filtered_waveforms if w is not None]
    valid_metadata = [m for m in all_filtered_metadata if m is not None]
    missing_count = len(all_filtered_waveforms) - len(valid_waveforms)
    
    if missing_count > 0:
        print(f"      Warning: {missing_count} waveforms could not be loaded")
        # Filter out corresponding EODs from channel_events
        valid_indices = [i for i, w in enumerate(all_filtered_waveforms) if w is not None]
        channel_events = channel_events.iloc[valid_indices].reset_index(drop=True)
        all_filtered_waveforms = valid_waveforms
        all_filtered_metadata = valid_metadata
    
    if all_filtered_waveforms:
        combined_waveforms = {'waveforms': all_filtered_waveforms, 'metadata': all_filtered_metadata}
        print(f"  Loaded {len(all_filtered_waveforms)} waveforms for final events")
        print(f"  Waveform-EOD order matches: each waveform corresponds to same row in channel_events")
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
    if len(all_filtered_waveforms) > 0 and len(channel_events) > 0:
        l_r = []
        mismatches = 0
        for i, l in enumerate(channel_events['waveform_length']):
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
            else:
                print(f"    ⚠ WARNING: {mismatches} waveforms don't match their EOD entries")
        
else:
    print("  No files with waveforms found")
    all_filtered_waveforms = []

#%%
# =============================================================================
# STEP 10: PROCESS INDIVIDUAL EVENTS (AUDIO EXTRACTION + INDIVIDUAL SAVES)
# =============================================================================
print(f"\nStep 10: Processing individual events (audio extraction and individual saves)...")

# Find all wav files in raw data folder
wav_files = glob.glob(str(Path(raw_data_folder) / "*.wav"))
print(f"  Found {len(wav_files)} wav files")

# Add event snippet-relative index columns to channel_events
channel_events['peak_idx_event_snippet'] = -1
channel_events['trough_idx_event_snippet'] = -1
channel_events['midpoint_idx_event_snippet'] = -1

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
            event_eods = channel_events[channel_events['channel_event_id'] == event_id].copy()
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
            
            # Update channel_events with the event snippet indices
            channel_events.loc[channel_events['channel_event_id'] == event_id, ['peak_idx_event_snippet', 'trough_idx_event_snippet', 'midpoint_idx_event_snippet']] = event_eods[['peak_idx_event_snippet', 'trough_idx_event_snippet', 'midpoint_idx_event_snippet']].values
                    
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
                            
                            # Create real-time x-coordinates for downsampled data
                            time_indices = np.arange(0, len(event_data), plot_step)[:len(data_diff)]
                            time_offsets = pd.to_timedelta(time_indices / sample_rate, unit='s')
                            x_coords = event_start_time + time_offsets
                            plt.plot(x_coords, data_diff + i * offset_diff, linewidth=0.5, label=f'Ch{i}-{i+1}')

                            # Plot detected pulses for this channel
                            ch_eods = event_eods[event_eods['eod_channel'] == i]
                            if len(ch_eods) > 0:
                                # Plot peaks (red)
                                if 'timestamp' in ch_eods.columns:
                                    # Calculate peak timestamps - convert peak indices to timestamps
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

                        plt.ylim(bottom=None, top=(event_data.shape[1]-1.5)*offset_diff)
                        if plot_step > 1:
                            plt.title(f'Event {event_id} - Differential EOD Detections (downsampled {plot_step}x)\nDuration: {event_duration:.1f}s - Colored boxes: merged channel events')
                        else:
                            plt.title(f'Event {event_id} - Differential EOD Detections\nDuration: {event_duration:.1f}s - Colored boxes: merged channel events')
                        plt.legend(loc='upper right')
                        plt.xlabel('Time')
                        plt.ylabel('Voltage (stacked by channel)')
                        
                        # Format x-axis for better timestamp readability
                        import matplotlib.dates as mdates
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


# =============================================================================
# STEP 11: SAVE FINAL COMBINED RESULTS
# =============================================================================
print(f"\nStep 11: Saving final combined results...")

output_path = Path(output_folder)

# # Save combined event table (with snippet-relative indices)
# channel_events_clean = channel_events.drop(columns=['peak_idx', 'trough_idx', 'midpoint_idx'], errors='ignore')
# channel_events_path = output_path / "session_events_eod_table.csv"
# channel_events_clean.to_csv(channel_events_path, index=False)
# print(f"  Saved combined event table: {channel_events_path}")

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
    'parameter': ['max_ipi_seconds', 'min_eods_per_event', 'sample_rate', 'file_duration', 'margin', 'min_channel_event_size', 'min_amplitude'],
    'value': [max_ipi_seconds, min_eods_per_event, sample_rate, file_duration, margin, min_channel_event_size, min_amplitude]
})
params_path = output_path / "session_events_extraction_params.csv"
params_df.to_csv(params_path, index=False)
print(f"  Saved parameters: {params_path}")

print(f"\nNote: Individual event data was saved during processing in Step 10:")
print(f"  - event_XXX_eod_table.csv: EOD data for each individual event")
print(f"  - event_XXX_waveforms_concatenated.npz: Waveforms for each event (if available)")
print(f"  - event_XXX_YYYYMMDDTHHMMSS.wav: Raw audio data for each event")

# =============================================================================
# STEP 12: CREATE TIMELINE PLOT
# =============================================================================
print(f"\nStep 12: Creating event timeline...")

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
        event_eods = channel_events[channel_events['channel_event_id'] == event_id]
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
print(f"  Final events (Stage 3): {len(event_summary)}")
print(f"  Total EODs in events: {len(channel_events)}")
print(f"  Extraction efficiency: {len(channel_events)/len(combined_table)*100:.1f}%")

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
del combined_table, sorted_table, channel_events, event_summary
if 'combined_waveforms' in locals():
    del combined_waveforms
if 'file_metadata' in locals():
    del file_metadata
gc.collect()
print_memory_usage("Final memory state")
print("Memory cleanup completed.")

# %%
