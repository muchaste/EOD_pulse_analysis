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
warnings.filterwarnings('ignore')

def load_session_data(session_folder):
    """
    Load all EOD tables and waveforms from a recording session.
    
    Parameters
    ----------
    session_folder : str
        Path to folder containing *_eod_table.csv and *_eod_waveforms.csv files
    
    Returns
    -------
    combined_table : pd.DataFrame
        Combined EOD event table from all files
    combined_waveforms : np.array
        Combined waveform data (n_events, n_samples)
    file_origins : np.array
        File index for each event (for tracking origin)
    """
    # Find all EOD table files
    table_files = glob.glob(str(Path(session_folder) / "*_eod_table.csv"))
    waveform_files = glob.glob(str(Path(session_folder) / "*_eod_waveforms.csv"))
    
    if len(table_files) == 0:
        raise ValueError(f"No EOD table files found in {session_folder}")
    
    print(f"Loading data from {len(table_files)} files...")
    
    combined_tables = []
    combined_waveforms = []
    file_origins = []
    
    for i, (table_file, waveform_file) in enumerate(zip(sorted(table_files), sorted(waveform_files))):
        # Load event table
        table = pd.read_csv(table_file)
        if len(table) > 0:
            table['file_index'] = i
            table['filename'] = Path(table_file).stem.replace('_eod_table', '')
            combined_tables.append(table)
            
            # Load waveforms
            waveforms = pd.read_csv(waveform_file).values
            combined_waveforms.append(waveforms)
            file_origins.extend([i] * len(waveforms))
            
            print(f"  {Path(table_file).name}: {len(table)} events")
    
    if len(combined_tables) == 0:
        raise ValueError("No events found in any files")
    
    # Combine all data
    combined_table = pd.concat(combined_tables, ignore_index=True)
    combined_waveforms = np.vstack(combined_waveforms)
    file_origins = np.array(file_origins)
    
    print(f"Total events loaded: {len(combined_table)}")
    return combined_table, combined_waveforms, file_origins

def extract_events(combined_table, max_ipi_seconds=5.0, min_eods_per_event=20, min_snr=2.0):
    """
    Extract events (fish encounters) from EOD data by temporal clustering.
    
    An event is defined as a sequence of EODs where consecutive EODs are separated
    by no more than max_ipi_seconds, with at least min_eods_per_event total EODs
    and at least one EOD meeting the SNR threshold.
    
    Parameters
    ----------
    combined_table : pd.DataFrame
        Combined EOD event data with timestamp column
    max_ipi_seconds : float
        Maximum inter-pulse interval to consider EODs as part of same event
    min_eods_per_event : int
        Minimum number of EODs required per event
    min_snr : float
        Minimum SNR required for at least one EOD in the event
    
    Returns
    -------
    event_table : pd.DataFrame
        Table with event information and filtered EODs
    event_labels : np.array
        Event ID for each EOD (-1 if not part of valid event)
    event_summary : pd.DataFrame
        Summary statistics for each event
    """
    if len(combined_table) == 0:
        return combined_table.copy(), np.array([]), pd.DataFrame()
    
    print("Extracting events from temporal patterns...")
    
    # Sort by timestamp
    sorted_table = combined_table.sort_values('timestamp').copy()
    sorted_table['timestamp_dt'] = pd.to_datetime(sorted_table['timestamp'])
    
    # Calculate inter-pulse intervals
    time_diffs = sorted_table['timestamp_dt'].diff().dt.total_seconds()
    
    # Identify event boundaries (gaps > max_ipi_seconds)
    event_boundaries = np.where(time_diffs > max_ipi_seconds)[0]
    
    # Create event segments
    event_starts = np.concatenate([[0], event_boundaries])
    event_ends = np.concatenate([event_boundaries, [len(sorted_table)]])
    
    print(f"Found {len(event_starts)} potential events before filtering...")
    
    # Filter events based on criteria
    valid_events = []
    event_labels = np.full(len(sorted_table), -1)
    event_summaries = []
    
    for event_id, (start_idx, end_idx) in enumerate(zip(event_starts, event_ends)):
        event_eods = sorted_table.iloc[start_idx:end_idx]
        
        # Check minimum number of EODs
        if len(event_eods) < min_eods_per_event:
            continue
        
        # Check SNR requirement (if SNR column exists)
        has_good_snr = True
        if 'snr' in event_eods.columns:
            max_snr = event_eods['snr'].max()
            has_good_snr = max_snr >= min_snr
        elif 'eod_amplitude' in event_eods.columns:
            # Use amplitude as proxy for SNR if no SNR column
            # Assume higher amplitude indicates better SNR
            max_amplitude = event_eods['eod_amplitude'].max()
            median_amplitude = combined_table['eod_amplitude'].median()
            has_good_snr = max_amplitude >= median_amplitude * 1.5
        
        if not has_good_snr:
            continue
        
        # Valid event - assign labels
        event_labels[start_idx:end_idx] = len(valid_events)
        valid_events.append((start_idx, end_idx))
        
        # Calculate event summary statistics
        duration = (event_eods['timestamp_dt'].max() - event_eods['timestamp_dt'].min()).total_seconds()
        mean_ipi = time_diffs.iloc[start_idx+1:end_idx].mean() if len(event_eods) > 1 else 0
        
        summary = {
            'event_id': len(valid_events) - 1,
            'start_time': event_eods['timestamp_dt'].min(),
            'end_time': event_eods['timestamp_dt'].max(),
            'duration_seconds': duration,
            'n_eods': len(event_eods),
            'mean_ipi_seconds': mean_ipi,
            'median_ipi_seconds': time_diffs.iloc[start_idx+1:end_idx].median() if len(event_eods) > 1 else 0,
            'mean_amplitude': event_eods['eod_amplitude'].mean(),
            'max_amplitude': event_eods['eod_amplitude'].max(),
            'mean_width_ms': event_eods['pulse_width'].mean() * 1000,
            'n_files': event_eods['file_index'].nunique() if 'file_index' in event_eods.columns else 1,
            'file_names': ','.join(event_eods['filename'].unique()) if 'filename' in event_eods.columns else 'unknown'
        }
        
        if 'snr' in event_eods.columns:
            summary['max_snr'] = event_eods['snr'].max()
            summary['mean_snr'] = event_eods['snr'].mean()
        
        event_summaries.append(summary)
    
    print(f"Extracted {len(valid_events)} valid events")
    print(f"Total EODs in events: {np.sum(event_labels >= 0)} / {len(combined_table)}")
    
    # Create filtered table with only EODs in valid events
    valid_mask = event_labels >= 0
    event_table = sorted_table[valid_mask].copy()
    event_table['event_id'] = event_labels[valid_mask]
    
    # Create event summary DataFrame
    event_summary = pd.DataFrame(event_summaries)
    
    # Reorder event_labels to match original table order
    # Map back to original indices
    original_event_labels = np.full(len(combined_table), -1)
    sorted_indices = sorted_table.index.values
    original_event_labels[sorted_indices] = event_labels
    
    return event_table, original_event_labels, event_summary

def analyze_session_for_events(session_folder, max_ipi_seconds=5.0):
    """
    Analyze session data to help choose appropriate event extraction parameters.
    
    Parameters
    ----------
    session_folder : str
        Path to folder containing EOD files
    max_ipi_seconds : float
        Maximum IPI to consider for event grouping
    
    Returns
    -------
    analysis : dict
        Statistics about potential events
    """
    combined_table, _, _ = load_session_data(session_folder)
    
    if len(combined_table) == 0:
        return {}
    
    # Sort by timestamp
    sorted_table = combined_table.sort_values('timestamp')
    sorted_table['timestamp_dt'] = pd.to_datetime(sorted_table['timestamp'])
    
    # Calculate inter-pulse intervals
    time_diffs = sorted_table['timestamp_dt'].diff().dt.total_seconds()
    
    # Identify potential events
    event_boundaries = np.where(time_diffs > max_ipi_seconds)[0]
    event_starts = np.concatenate([[0], event_boundaries])
    event_ends = np.concatenate([event_boundaries, [len(sorted_table)]])
    
    event_lengths = event_ends - event_starts
    
    analysis = {
        'total_eods': len(combined_table),
        'potential_events': len(event_starts),
        'event_lengths': {
            'min': event_lengths.min(),
            'max': event_lengths.max(),
            'mean': event_lengths.mean(),
            'median': np.median(event_lengths)
        },
        'events_with_20_plus_eods': np.sum(event_lengths >= 20),
        'events_with_10_plus_eods': np.sum(event_lengths >= 10),
        'events_with_5_plus_eods': np.sum(event_lengths >= 5),
        'ipi_percentiles': {
            '50th': np.percentile(time_diffs.dropna(), 50),
            '90th': np.percentile(time_diffs.dropna(), 90),
            '95th': np.percentile(time_diffs.dropna(), 95),
            '99th': np.percentile(time_diffs.dropna(), 99)
        }
    }
    
    return analysis

def save_event_results(event_table, event_summary, combined_waveforms, event_labels, output_folder):
    """
    Save event extraction results to files.
    
    Parameters
    ----------
    event_table : pd.DataFrame
        Filtered EOD data (only EODs in valid events)
    event_summary : pd.DataFrame
        Summary statistics for each event
    combined_waveforms : np.array
        All waveform data
    event_labels : np.array
        Event ID for each EOD (-1 if not part of valid event)
    output_folder : str
        Path to save results
    """
    output_path = Path(output_folder)
    
    # Save event table
    event_table.to_csv(output_path / 'session_events_eod_table.csv', index=False)
    
    # Save event summary
    event_summary.to_csv(output_path / 'session_events_summary.csv', index=False)
    
    # Save waveforms for events only
    event_mask = event_labels >= 0
    event_waveforms = combined_waveforms[event_mask]
    pd.DataFrame(event_waveforms).to_csv(output_path / 'session_events_waveforms.csv', index=False)
    
    # Save event extraction parameters and statistics
    stats = {
        'total_eods_loaded': len(event_labels),
        'eods_in_events': np.sum(event_mask),
        'eods_filtered_out': np.sum(~event_mask),
        'total_events': len(event_summary),
        'extraction_efficiency': np.sum(event_mask) / len(event_labels) if len(event_labels) > 0 else 0
    }
    
    if len(event_summary) > 0:
        stats.update({
            'min_event_duration_sec': event_summary['duration_seconds'].min(),
            'max_event_duration_sec': event_summary['duration_seconds'].max(),
            'mean_event_duration_sec': event_summary['duration_seconds'].mean(),
            'min_eods_per_event': event_summary['n_eods'].min(),
            'max_eods_per_event': event_summary['n_eods'].max(),
            'mean_eods_per_event': event_summary['n_eods'].mean()
        })
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_path / 'session_events_extraction_stats.csv', index=False)
    
    print(f"Event extraction results saved to {output_folder}")
    return stats

def create_event_plots(event_table, event_summary, output_folder):
    """
    Create visualization plots for event extraction results.
    
    Parameters
    ----------
    event_table : pd.DataFrame
        Filtered EOD data
    event_summary : pd.DataFrame
        Event summary statistics
    output_folder : str
        Path to save plots
    """
    output_path = Path(output_folder)
    
    if len(event_summary) == 0:
        print("No events to plot")
        return
    
    # Plot 1: Event timeline
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
    
    # Plot 2: Event quality metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Duration vs number of EODs
    axes[0,0].scatter(event_summary['duration_seconds']/60, event_summary['n_eods'], alpha=0.6)
    axes[0,0].set_xlabel('Duration (minutes)')
    axes[0,0].set_ylabel('Number of EODs')
    axes[0,0].set_title('Event Duration vs EOD Count')
    
    # Mean IPI distribution
    axes[0,1].hist(event_summary['mean_ipi_seconds'], bins=20, alpha=0.7)
    axes[0,1].set_xlabel('Mean IPI (seconds)')
    axes[0,1].set_ylabel('Number of Events')
    axes[0,1].set_title('Distribution of Mean Inter-Pulse Intervals')
    
    # Amplitude distribution
    axes[1,0].hist(event_summary['mean_amplitude'], bins=20, alpha=0.7)
    axes[1,0].set_xlabel('Mean Amplitude')
    axes[1,0].set_ylabel('Number of Events')
    axes[1,0].set_title('Distribution of Mean Amplitudes')
    
    # Width distribution
    axes[1,1].hist(event_summary['mean_width_ms'], bins=20, alpha=0.7)
    axes[1,1].set_xlabel('Mean Width (ms)')
    axes[1,1].set_ylabel('Number of Events')
    axes[1,1].set_title('Distribution of Mean Pulse Widths')
    
    plt.tight_layout()
    plt.savefig(output_path / 'event_quality_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Event size distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(event_summary['n_eods'], bins=20, alpha=0.7)
    plt.xlabel('Number of EODs per Event')
    plt.ylabel('Number of Events')
    plt.title('Event Size Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(event_summary['duration_seconds']/60, bins=20, alpha=0.7)
    plt.xlabel('Event Duration (minutes)')
    plt.ylabel('Number of Events')
    plt.title('Event Duration Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path / 'event_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Event plots saved to {output_folder}")

def main():
    """
    Main function for event extraction from session data.
    """
    print("EOD Session-Level Event Extraction")
    print("=" * 40)
    
    # Get input and output folders
    root = tk.Tk()
    root.withdraw()
    
    session_folder = filedialog.askdirectory(
        title="Select folder with EOD table/waveform files from a recording session"
    )
    if not session_folder:
        return
    
    output_folder = filedialog.askdirectory(
        title="Select folder to save event extraction results"
    )
    if not output_folder:
        return
    
    # Parameters (can be adjusted based on your data)
    max_ipi_seconds = 5.0
    min_eods_per_event = 20
    min_snr = 2.0
    
    try:
        # Load session data
        combined_table, combined_waveforms, file_origins = load_session_data(session_folder)
        
        # Optional: Analyze session to help choose parameters
        print("\nAnalyzing session data...")
        analysis = analyze_session_for_events(session_folder, max_ipi_seconds)
        if analysis:
            print(f"Session analysis:")
            print(f"  Total EODs: {analysis['total_eods']}")
            print(f"  Potential events (IPI < {max_ipi_seconds}s): {analysis['potential_events']}")
            print(f"  Events with ≥20 EODs: {analysis['events_with_20_plus_eods']}")
            print(f"  Events with ≥10 EODs: {analysis['events_with_10_plus_eods']}")
            print(f"  IPI 90th percentile: {analysis['ipi_percentiles']['90th']:.2f}s")
        
        # Extract events
        print("\n" + "="*30)
        event_table, event_labels, event_summary = extract_events(
            combined_table, 
            max_ipi_seconds=max_ipi_seconds, 
            min_eods_per_event=min_eods_per_event, 
            min_snr=min_snr
        )
        
        if len(event_table) == 0:
            print("No valid events found! Try adjusting parameters:")
            print(f"- Reduce min_eods_per_event (currently {min_eods_per_event})")
            print(f"- Increase max_ipi_seconds (currently {max_ipi_seconds})")
            print(f"- Reduce min_snr (currently {min_snr})")
            return
        
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
        
        # Create plots
        create_event_plots(event_table, event_summary, output_folder)
        
        print("\nEvent extraction completed successfully!")
        print("\nOutput files:")
        print("  - session_events_eod_table.csv: EOD data for valid events")
        print("  - session_events_waveforms.csv: Waveform data for valid events")
        print("  - session_events_summary.csv: Summary statistics for each event")
        print("  - session_events_extraction_stats.csv: Overall extraction statistics")
        print("  - Various plots: timeline, quality metrics, distributions")
        print("\nNext step: Use 04_Session_Clustering.py to cluster the extracted events")
        
    except Exception as e:
        print(f"Error during event extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
