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

# Import shared functions from eod_functions module
from eod_functions import (
    load_variable_length_waveforms, save_variable_length_waveforms,
    load_session_data, extract_events, analyze_session_for_events,
    save_event_results, create_event_plots
)











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
        print("  - session_events_waveforms_*: Variable-length waveform data (efficient format)")
        print("  - session_events_waveforms.csv: Waveform data (legacy CSV format)")
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
