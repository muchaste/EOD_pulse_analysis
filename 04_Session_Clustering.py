"""
Session-Level EOD Clustering for Species and Individual Assignment

This script takes pre-extracted event data and applies hierarchical clustering
to identify species and individuals within each event.

Input: Output from 03_Event_Extraction.py
Output: Species and individual assignments for each EOD

Based on the thunderfish.pulses.extract_pulsefish() clustering approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')

# Import shared functions from eod_functions module
from eod_functions import (
    load_variable_length_waveforms, calculate_waveform_stats,
    load_event_data, bgm_clustering, extract_waveform_features,
    cluster_session_eods, save_clustering_results, create_clustering_plots
)













def main():
    """
    Main function for session-level clustering.
    """
    print("EOD Session-Level Clustering")
    print("=" * 40)
    
    # Get input and output folders
    root = tk.Tk()
    root.withdraw()
    
    input_folder = filedialog.askdirectory(
        title="Select folder with event extraction results (from 03_Event_Extraction.py)"
    )
    if not input_folder:
        return
    
    output_folder = filedialog.askdirectory(
        title="Select folder to save clustering results"
    )
    if not output_folder:
        return
    
    try:
        # Load event data
        event_table, event_waveforms, event_summary = load_event_data(input_folder)
        
        if len(event_table) == 0:
            print("No event data found!")
            return
        
        # Perform clustering on event data
        print("\n" + "="*30)
        species_labels, individual_labels, clustering_info = cluster_session_eods(
            event_table, event_waveforms, event_summary, min_cluster_size=10
        )
        
        # Save results
        print("\n" + "="*30)
        results_table = save_clustering_results(
            event_table, event_summary, species_labels, individual_labels, 
            clustering_info, output_folder
        )
        
        # Create plots
        create_clustering_plots(
            event_table, event_waveforms, event_summary, species_labels, 
            individual_labels, output_folder
        )
        
        print("\nClustering completed successfully!")
        print("\nOutput files:")
        print("  - session_eod_clusters.csv: EOD data with cluster assignments")
        print("  - event_summary_with_clusters.csv: Event-level summary with dominant species/individuals")
        print("  - clustering_summary.csv: Overall clustering statistics")
        print("  - species_summary.csv: Species-level summary")
        print("  - Various plots: species clusters, waveforms, timeline, summary")
        
    except Exception as e:
        print(f"Error during clustering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
