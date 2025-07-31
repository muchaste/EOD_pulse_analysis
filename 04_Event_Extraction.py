"""
Event Extraction from EOD Data

This script extracts meaningful fish encounter events from EOD data across multiple files.
An event is defined as a sequence of EODs that occur within a specified time window,
with quality filters for minimum event size and signal quality.

This prepares the data for downstream clustering analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import tkinter as tk
from tkinter import filedialog
import warnings
import os
import gc
import audioio as aio
warnings.filterwarnings('ignore')

# Import shared functions from eod_functions module
from eod_functions import (
    load_session_data, extract_events, analyze_session_for_events,
    save_event_results
)

print("EOD Session-Level Event Extraction")
print("=" * 40)

# Get input and output folders
root = tk.Tk()
root.withdraw()

session_folder = filedialog.askdirectory(
    title="Select folder with EOD table/waveform files from a recording session"
)

output_folder = filedialog.askdirectory(
    title="Select folder to save event extraction results"
)

raw_data_folder = filedialog.askdirectory(
    title="Select folder with raw recording files (optional, can be empty)"
)

# Parameters (can be adjusted based on your data)
parameters = {
    'max_ipi_seconds': 5.0,  # Maximum inter-pulse interval
    'min_eods_per_event': 30
}

try:
    # Load session data
    combined_table, combined_waveforms = load_session_data(session_folder)
    
    # Optional: Analyze session to help choose parameters
    print("\nAnalyzing session data...")
    analysis = analyze_session_for_events(session_folder, parameters['max_ipi_seconds'])
    if analysis:
        print(f"Session analysis:")
        print(f"  Total EODs: {analysis['total_eods']}")
        print(f"  Potential events (IPI < {parameters['max_ipi_seconds']}s): {analysis['potential_events']}")
        print(f"  Events with ≥20 EODs: {analysis['events_with_20_plus_eods']}")
        print(f"  Events with ≥10 EODs: {analysis['events_with_10_plus_eods']}")
        print(f"  IPI 90th percentile: {analysis['ipi_percentiles']['90th']:.2f}s")
    
    # Extract events
    print("\n" + "="*30)
    event_table, event_labels, event_summary = extract_events(
        combined_table, 
        max_ipi_seconds=parameters['max_ipi_seconds'], 
        min_eods_per_event=parameters['min_eods_per_event']
    )
    
    if len(event_table) > 0:
        
        # Sample raw data for each event
        print("\nSampling raw data for events...")

        # Find all wav files in raw data folder
        wav_files = glob.glob(str(Path(raw_data_folder) / "*.wav"))
        if wav_files:
            # Extract timestamps from filenames
            timecol = [pd.to_datetime(os.path.basename(fname).split('-')[1][0:-4], format='%Y%m%dT%H%M%S') for fname in wav_files]
            file_times_df = pd.DataFrame({'filename': wav_files, 'file_timestamp': timecol})
            
            # Add raw data filenames to event summary
            event_summary['raw_event_data_filename'] = ''
            event_summary['margin_complete'] = True  # Track if full margin was applied
            
            for idx, event in event_summary.iterrows():
                event_start = pd.to_datetime(event['start_time'])
                event_end = pd.to_datetime(event['end_time'])
                
                # Find files that overlap with event (with 2s safety margin)
                margin = pd.Timedelta(seconds=2)
                event_start_margin = event_start - margin
                event_end_margin = event_end + margin
                
                # Find overlapping files
                overlapping_files = []
                for _, file_row in file_times_df.iterrows():
                    file_time = file_row['file_timestamp']
                    # Assume 10-minute files (adjust if different)
                    file_end = file_time + pd.Timedelta(minutes=10)
                    
                    if (file_time <= event_end_margin) and (file_end >= event_start_margin):
                        overlapping_files.append(file_row)  # Store entire row with both filename and timestamp
                
                if overlapping_files:
                    # Sample and save raw data
                    event_id = event['event_id']
                    output_filename = f"event_{event_id:03d}_{event_start.strftime('%Y%m%dT%H%M%S')}.wav"
                    output_path = Path(output_folder) / output_filename
                    
                    try:
                        # Load only needed portions from overlapping files
                        all_data = []
                        sample_rate = None
                        
                        for i, file_row in enumerate(sorted(overlapping_files, key=lambda x: x['file_timestamp'])):
                            wav_file = file_row['filename']
                            file_time = file_row['file_timestamp']
                            
                            with aio.AudioLoader(wav_file) as sf:
                                if sample_rate is None:
                                    sample_rate = sf.rate
                                
                                # Calculate file time boundaries
                                file_end_time = file_time + pd.Timedelta(minutes=10)  # Assume 10-minute files
                                
                                # Calculate what portion of this file we need
                                file_start_sample = 0
                                file_end_sample = len(sf)
                                
                                # Calculate desired start time for this file
                                desired_start_time = max(event_start_margin, file_time)
                                desired_end_time = min(event_end_margin, file_end_time)
                                
                                # Convert to sample indices
                                if desired_start_time > file_time:
                                    start_offset = (desired_start_time - file_time).total_seconds()
                                    file_start_sample = max(0, int(start_offset * sample_rate))
                                
                                if desired_end_time < file_end_time:
                                    end_offset = (desired_end_time - file_time).total_seconds()
                                    file_end_sample = min(len(sf), int(end_offset * sample_rate))
                                
                                # Only read if we have valid samples to read
                                if file_start_sample < file_end_sample:
                                    data_chunk = sf[file_start_sample:file_end_sample, :]
                                    all_data.append(data_chunk)
                        
                        if all_data:
                            # Concatenate data
                            event_data = np.concatenate(all_data, axis=0)
                            
                            # Check if margins were fully applied
                            actual_duration = len(event_data) / sample_rate
                            requested_duration = (event_end_margin - event_start_margin).total_seconds()
                            margin_complete = abs(actual_duration - requested_duration) < 0.1  # 0.1s tolerance
                            
                            # Save as wav file
                            aio.write_audio(str(output_path), event_data, sample_rate)
                            
                            # TODO: plots still suck
                            # Plot the event data (differential channels) with detected pulses
                            # Get EODs for this specific event
                            event_eods = event_table[event_table['event_id'] == event_id].copy()
                            
                            if len(event_eods) > 0:
                                # Calculate offset for stacking differential channels
                                offset_diff = np.percentile(np.abs(event_data), 95) * 1.5
                                
                                plt.figure(figsize=(20, 8))
                                for i in range(event_data.shape[1]-1):
                                    # Calculate differential signal
                                    data_diff = np.diff(event_data[:, i:i+2], axis=1).flatten()
                                    
                                    # Find EODs detected on this differential channel
                                    if 'eod_channel' in event_eods.columns:
                                        ch_eods = event_eods[event_eods['eod_channel'] == i]
                                    else:
                                        # If no channel info, use all EODs (fallback)
                                        ch_eods = event_eods
                                    
                                    # Plot differential signal (downsample if too large)
                                    step = max(1, len(data_diff) // 15000000)  # Limit to ~15M points per channel
                                    x_coords = np.arange(0, len(data_diff), step)
                                    plt.plot(x_coords, data_diff[::step] + i * offset_diff, linewidth=0.5, label=f'Ch{i}-{i+1}')
                                    
                                    # Plot detected pulses for this channel
                                    if len(ch_eods) > 0:
                                        # Convert EOD timestamps to sample indices within the event audio
                                        event_start_time = pd.to_datetime(event['start_time'])
                                        audio_start_time = event_start_time - pd.Timedelta(seconds=2)
                                        
                                        # Get timestamp column
                                        if 'timestamp_dt' in ch_eods.columns:
                                            eod_times = ch_eods['timestamp_dt']
                                        elif 'timestamp' in ch_eods.columns:
                                            eod_times = pd.to_datetime(ch_eods['timestamp'])
                                        else:
                                            continue  # Skip if no timestamp available
                                        
                                        # Convert to sample indices
                                        eod_sample_indices = ((eod_times - audio_start_time).dt.total_seconds() * sample_rate).astype(int)
                                        
                                        # Ensure indices are within bounds
                                        valid_indices = (eod_sample_indices >= 0) & (eod_sample_indices < len(data_diff))
                                        if np.any(valid_indices):
                                            valid_samples = eod_sample_indices[valid_indices]
                                            
                                            # Plot peaks (red) and troughs (blue)
                                            plt.plot(valid_samples, 
                                                   data_diff[valid_samples] + i * offset_diff, 
                                                   'o', markersize=2, color='red', alpha=0.8)
                                            
                                            # For troughs, use same positions (simplified)
                                            plt.plot(valid_samples, 
                                                   data_diff[valid_samples] + i * offset_diff, 
                                                   'o', markersize=2, color='blue', alpha=0.8)
                                
                                plt.ylim(bottom=None, top=(event_data.shape[1]-1.5)*offset_diff)
                                plt.title(f'Event {event_id} - Differential EOD Detections - Red=Peaks, Blue=Troughs (n={len(event_eods)} EODs)')
                                plt.legend(loc='upper right')
                                plt.xlabel('Sample')
                                plt.ylabel('Voltage (stacked by channel)')
                                
                                # Save plot
                                plot_filename = f'event_{event_id:03d}_differential_detection.png'
                                plot_path = Path(output_folder) / plot_filename
                                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                                plt.close()
                                
                                print(f"  Created differential plot: {plot_filename}")
                                
                                # Clean up plot variables
                                del data_diff
                            

                            # Update event summary with filename and margin info
                            event_summary.loc[idx, 'raw_event_data_filename'] = output_filename
                            event_summary.loc[idx, 'margin_complete'] = margin_complete
                            
                            margin_note = "" if margin_complete else " (margin truncated)"
                            print(f"  Event {event_id}: saved {output_filename} ({actual_duration:.1f}s){margin_note}")
                            
                            # Free memory
                            del event_data, all_data
                            gc.collect()
                    
                    except Exception as e:
                        print(f"  Warning: Could not sample raw data for event {event_id}: {e}")
                        event_summary.loc[idx, 'raw_event_data_filename'] = 'ERROR'
                else:
                    print(f"  Warning: No raw data files found for event {event['event_id']}")
                    event_summary.loc[idx, 'raw_event_data_filename'] = 'NOT_FOUND'
        else:
            print("  No wav files found in session folder - skipping raw data sampling")
            event_summary['raw_event_data_filename'] = 'NO_WAV_FILES'
    
        print(f"\nEvent extraction summary:")
        print(f"  Total events: {len(event_summary)}")
        print(f"  Total EODs in events: {len(event_table)}")
        print(f"  EODs filtered out: {np.sum(event_labels == -1)}")
        print(f"  Extraction efficiency: {len(event_table)/len(combined_table)*100:.1f}%")
        
        if len(event_summary) > 0:
            print(f"  Event duration: {event_summary['duration_seconds'].min():.1f} - {event_summary['duration_seconds'].max():.1f} seconds")
            print(f"  EODs per event: {event_summary['n_eods'].min()} - {event_summary['n_eods'].max()}")
        
        # Save results
        print("\n" + "="*30)
        stats = save_event_results(event_table, event_summary, combined_waveforms, event_labels, output_folder)

        # Save event extraction parameters as .csv
        params_df = pd.DataFrame.from_dict(parameters, orient='index', columns=['value'])
        params_df.to_csv(Path(output_folder) / "session_events_extraction_params.csv")

        # Create plots
        print("\nCreating event timeline...")
        output_path = Path(output_folder)
        
        if len(event_summary) > 0:
            # Event timeline plot
            plt.figure(figsize=(15, 6))
            
            # Convert to datetime for plotting
            start_times = pd.to_datetime(event_summary['start_time'])
            durations = event_summary['duration_seconds']
            n_eods = event_summary['n_eods']
            
            # Create timeline plot
            for i, (start, duration, n_eod) in enumerate(zip(start_times, durations, n_eods)):
                plt.barh(i, duration/60, left=(start - start_times.min()).total_seconds()/60, 
                        height=0.8, alpha=0.7, label=f'{n_eod} EODs' if i < 5 else "")
            
            plt.xlabel('Time (minutes)')
            plt.ylabel('Event ID')
            plt.title(f'Event Timeline ({len(event_summary)} events)')
            plt.grid(True, alpha=0.3)
            if len(event_summary) <= 5:
                plt.legend()
            plt.savefig(output_path / 'event_timeline.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Created event timeline plot")

        print(f"Event plots saved to {output_folder}")
        
        print("\nEvent extraction completed successfully!")
        print("\nOutput files:")
        print("  - session_events_eod_table.csv: EOD data for valid events")
        print("  - session_events_waveforms_*: Variable-length waveform data (efficient format)")
        print("  - session_events_waveforms.csv: Waveform data (legacy CSV format)")
        print("  - session_events_summary.csv: Summary statistics for each event")
        print("  - session_events_extraction_stats.csv: Overall extraction statistics")
        print("  - event_timeline.png: Timeline visualization of all events")
        print("  - event_*_differential_detection.png: Individual differential plots for each event")
        print("  - event_*.wav: Raw audio data for each event (with 2s margins)")
        print("\nNext step: Use 04_Session_Clustering.py to cluster the extracted events")
    else:
        print("No valid events found! Try adjusting parameters:")
        print(f"- Reduce min_eods_per_event (currently {parameters['min_eods_per_event']})")
        print(f"- Increase max_ipi_seconds (currently {parameters['max_ipi_seconds']})")

    
except Exception as e:
    print(f"Error during event extraction: {e}")
    import traceback
    traceback.print_exc()

