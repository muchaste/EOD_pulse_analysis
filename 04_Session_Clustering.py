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

def load_event_data(input_folder):
    """
    Load extracted event data from 03_Event_Extraction.py output.
    
    Parameters
    ----------
    input_folder : str
        Path to folder containing event extraction results
    
    Returns
    -------
    event_table : pd.DataFrame
        EOD data for valid events
    event_waveforms : np.array
        Waveform data for valid events
    event_summary : pd.DataFrame
        Summary statistics for each event
    """
    input_path = Path(input_folder)
    
    # Load event data
    event_table_file = input_path / 'session_events_eod_table.csv'
    event_waveforms_file = input_path / 'session_events_waveforms.csv'
    event_summary_file = input_path / 'session_events_summary.csv'
    
    if not event_table_file.exists():
        raise FileNotFoundError(f"Event table not found: {event_table_file}")
    
    if not event_waveforms_file.exists():
        raise FileNotFoundError(f"Event waveforms not found: {event_waveforms_file}")
    
    if not event_summary_file.exists():
        raise FileNotFoundError(f"Event summary not found: {event_summary_file}")
    
    # Load data
    event_table = pd.read_csv(event_table_file)
    event_waveforms = pd.read_csv(event_waveforms_file).values
    event_summary = pd.read_csv(event_summary_file)
    
    print(f"Loaded event data:")
    print(f"  Events: {len(event_summary)}")
    print(f"  EODs: {len(event_table)}")
    print(f"  Waveform shape: {event_waveforms.shape}")
    
    return event_table, event_waveforms, event_summary

def bgm_clustering(data, n_components=10, merge_threshold=0.1, use_log=False):
    """
    Bayesian Gaussian Mixture clustering (adapted from thunderfish).
    
    Parameters
    ----------
    data : 1D array
        Data to cluster
    n_components : int
        Maximum number of Gaussian components
    merge_threshold : float
        Threshold for merging similar clusters
    use_log : bool
        Whether to use log-transformed data
    
    Returns
    -------
    labels : 1D array
        Cluster labels for each data point
    """
    if len(data) < n_components:
        return np.zeros(len(data), dtype=int)
    
    # Prepare data
    X = data.reshape(-1, 1)
    if use_log:
        X = np.log(X + 1e-10)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit BGM
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        max_iter=200,
        n_init=3,
        random_state=42
    )
    
    labels = bgm.fit_predict(X_scaled)
    
    # Merge similar clusters based on merge_threshold
    # (Simplified version - could be more sophisticated)
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        means = []
        for label in unique_labels:
            means.append(np.mean(data[labels == label]))
        
        means = np.array(means)
        # Simple merging: if two cluster means are within merge_threshold, merge them
        for i in range(len(means)):
            for j in range(i+1, len(means)):
                if abs(means[i] - means[j]) / max(means[i], means[j]) < merge_threshold:
                    labels[labels == unique_labels[j]] = unique_labels[i]
    
    return labels

def extract_waveform_features(waveforms, n_components=5):
    """
    Extract PCA features from normalized waveforms.
    
    Parameters
    ----------
    waveforms : 2D array
        Waveform data (n_events, n_samples)
    n_components : int
        Number of PCA components
    
    Returns
    -------
    features : 2D array
        PCA features (n_events, n_components)
    pca : PCA object
        Fitted PCA transformer
    """
    # Remove zero-variance waveforms
    valid_mask = np.var(waveforms, axis=1) > 1e-10
    if np.sum(valid_mask) == 0:
        return np.zeros((len(waveforms), n_components)), None
    
    valid_waveforms = waveforms[valid_mask]
    
    # Standardize each waveform
    scaler = StandardScaler()
    waveforms_scaled = scaler.fit_transform(valid_waveforms.T).T
    
    # PCA
    pca = PCA(n_components=min(n_components, waveforms_scaled.shape[1]))
    features_valid = pca.fit_transform(waveforms_scaled)
    
    # Create full feature array
    features = np.zeros((len(waveforms), features_valid.shape[1]))
    features[valid_mask] = features_valid
    
    return features, pca

def cluster_session_eods(event_table, event_waveforms, event_summary, min_cluster_size=10):
    """
    Hierarchical clustering approach on event-level data (similar to extract_pulsefish).
    
    Parameters
    ----------
    event_table : pd.DataFrame
        Filtered EOD data (only EODs in valid events)
    event_waveforms : np.array
        Waveform data corresponding to event_table
    event_summary : pd.DataFrame
        Summary statistics for each event
    min_cluster_size : int
        Minimum events per cluster
    
    Returns
    -------
    species_labels : np.array
        Species cluster labels for each EOD
    individual_labels : np.array
        Individual ID labels within species for each EOD
    clustering_info : dict
        Information about clustering process
    """
    n_events = len(event_table)
    if n_events < min_cluster_size:
        return np.zeros(n_events), np.zeros(n_events), {}
    
    print(f"Starting hierarchical clustering on {len(event_summary)} events with {n_events} EODs...")
    
    # Step 1: Cluster by pulse width
    widths = event_table['pulse_width'].values * 1000  # Convert to ms
    width_labels = bgm_clustering(widths, n_components=5, merge_threshold=0.2)
    
    print(f"Width clustering: {len(np.unique(width_labels))} clusters")
    
    # Step 2: Within each width cluster, cluster by amplitude
    n_events = len(event_table)
    all_species_labels = np.full(n_events, -1)
    all_individual_labels = np.full(n_events, -1)
    clustering_info = {}
    
    species_counter = 0
    individual_counter = 0
    
    for width_cluster in np.unique(width_labels):
        width_mask = width_labels == width_cluster
        if np.sum(width_mask) < min_cluster_size:
            continue
            
        # Get data for this width cluster
        cluster_amplitudes = event_table.loc[event_table.index[width_mask], 'eod_amplitude'].values
        cluster_waveforms = event_waveforms[width_mask]
        
        # Amplitude clustering
        amplitude_labels = bgm_clustering(cluster_amplitudes, n_components=8, 
                                        merge_threshold=0.15, use_log=True)
        
        print(f"  Width cluster {width_cluster}: {len(np.unique(amplitude_labels))} amplitude clusters")
        
        # Step 3: Within each amplitude cluster, cluster by waveform shape
        for amp_cluster in np.unique(amplitude_labels):
            amp_mask = amplitude_labels == amp_cluster
            combined_mask = width_mask.copy()
            combined_mask[width_mask] = amp_mask
            
            if np.sum(combined_mask) < min_cluster_size:
                continue
            
            # Get waveforms for this amplitude cluster
            shape_waveforms = event_waveforms[combined_mask]
            
            # Extract PCA features
            features, pca = extract_waveform_features(shape_waveforms, n_components=5)
            
            if features.shape[1] == 0:
                continue
            
            # DBSCAN clustering on PCA features
            # Parameters need tuning based on your data
            dbscan = DBSCAN(eps=0.5, min_samples=max(3, min_cluster_size//3))
            shape_labels = dbscan.fit_predict(features)
            
            # Assign species labels (each shape cluster is a species)
            for shape_cluster in np.unique(shape_labels):
                if shape_cluster == -1:  # Noise cluster
                    continue
                    
                shape_mask = shape_labels == shape_cluster
                final_mask = combined_mask.copy()
                final_mask[combined_mask] = shape_mask
                
                if np.sum(final_mask) < min_cluster_size:
                    continue
                
                # Assign species label
                all_species_labels[final_mask] = species_counter
                
                # Step 4: Individual assignment within species
                # Use subtle waveform differences and temporal patterns
                species_waveforms = event_waveforms[final_mask]
                species_timestamps = event_table.loc[event_table.index[final_mask], 'timestamp'].values
                
                # Individual clustering (simplified - could be more sophisticated)
                # Use fine-grained PCA features
                ind_features, _ = extract_waveform_features(species_waveforms, n_components=10)
                
                # Add temporal features (pulse rate patterns)
                if len(species_timestamps) > 5:
                    # Calculate local pulse rates
                    sorted_times = pd.to_datetime(species_timestamps).sort_values()
                    if len(sorted_times) > 1:
                        ipis = np.diff(sorted_times.astype(np.int64)) / 1e9  # Convert to seconds
                        median_ipi = np.median(ipis)
                        # Could add more temporal features here
                
                # Individual DBSCAN (more sensitive parameters)
                ind_dbscan = DBSCAN(eps=0.3, min_samples=max(2, min_cluster_size//5))
                individual_clusters = ind_dbscan.fit_predict(ind_features)
                
                # Assign individual labels
                for ind_cluster in np.unique(individual_clusters):
                    if ind_cluster == -1:
                        # Assign singleton individuals for noise
                        noise_mask = individual_clusters == -1
                        noise_indices = np.where(final_mask)[0][noise_mask]
                        for idx in noise_indices:
                            all_individual_labels[idx] = individual_counter
                            individual_counter += 1
                    else:
                        ind_mask = individual_clusters == ind_cluster
                        ind_indices = np.where(final_mask)[0][ind_mask]
                        all_individual_labels[ind_indices] = individual_counter
                        individual_counter += 1
                
                species_counter += 1
    
    clustering_info = {
        'n_species': len(np.unique(all_species_labels[all_species_labels >= 0])),
        'n_individuals': len(np.unique(all_individual_labels[all_individual_labels >= 0])),
        'n_unclustered': np.sum(all_species_labels == -1)
    }
    
    print(f"Clustering complete:")
    print(f"  Species found: {clustering_info['n_species']}")
    print(f"  Individuals found: {clustering_info['n_individuals']}")
    print(f"  Unclustered events: {clustering_info['n_unclustered']}")
    
    return all_species_labels, all_individual_labels, clustering_info

def save_clustering_results(event_table, event_summary, species_labels, individual_labels, 
                          clustering_info, output_folder):
    """
    Save clustering results to files.
    """
    # Add clustering results to table
    results_table = event_table.copy()
    results_table['species_cluster'] = species_labels
    results_table['individual_id'] = individual_labels
    
    # Save main results
    output_path = Path(output_folder)
    results_table.to_csv(output_path / 'session_eod_clusters.csv', index=False)
    
    # Save event summary with cluster information
    event_summary_with_clusters = event_summary.copy()
    if len(event_summary) > 0:
        # Add dominant species and individual for each event
        event_species = []
        event_individuals = []
        for event_id in event_summary['event_id']:
            event_mask = results_table['event_id'] == event_id
            if np.sum(event_mask) > 0:
                event_species_labels = results_table.loc[event_mask, 'species_cluster'].values
                event_individual_labels = results_table.loc[event_mask, 'individual_id'].values
                
                # Get most common species and individual
                valid_species = event_species_labels[event_species_labels >= 0]
                valid_individuals = event_individual_labels[event_individual_labels >= 0]
                
                dominant_species = np.bincount(valid_species).argmax() if len(valid_species) > 0 else -1
                dominant_individual = np.bincount(valid_individuals).argmax() if len(valid_individuals) > 0 else -1
                
                event_species.append(dominant_species)
                event_individuals.append(dominant_individual)
            else:
                event_species.append(-1)
                event_individuals.append(-1)
        
        event_summary_with_clusters['dominant_species'] = event_species
        event_summary_with_clusters['dominant_individual'] = event_individuals
    
    event_summary_with_clusters.to_csv(output_path / 'event_summary_with_clusters.csv', index=False)
    
    # Save clustering summary
    summary = pd.DataFrame([clustering_info])
    summary.to_csv(output_path / 'clustering_summary.csv', index=False)
    
    # Save species summary
    species_summary = []
    if len(event_table) > 0:
        for species in np.unique(species_labels[species_labels >= 0]):
            species_mask = species_labels == species
            individuals = np.unique(individual_labels[species_mask])
            individuals = individuals[individuals >= 0]
            
            # Get events for this species
            species_events = results_table.loc[species_mask, 'event_id'].unique()
            
            species_summary.append({
                'species_cluster': species,
                'n_eods': np.sum(species_mask),
                'n_events': len(species_events),
                'n_individuals': len(individuals),
                'individual_ids': ','.join(map(str, individuals)),
                'event_ids': ','.join(map(str, species_events)),
                'mean_amplitude': np.mean(event_table.loc[event_table.index[species_mask], 'eod_amplitude']),
                'mean_width_ms': np.mean(event_table.loc[event_table.index[species_mask], 'pulse_width']) * 1000
            })
    
    species_df = pd.DataFrame(species_summary)
    species_df.to_csv(output_path / 'species_summary.csv', index=False)
    
    print(f"Clustering results saved to {output_folder}")
    return results_table

def create_clustering_plots(event_table, event_waveforms, event_summary, species_labels, 
                          individual_labels, output_folder):
    """
    Create summary plots of clustering results.
    """
    output_path = Path(output_folder)
    
    # Plot 1: Species clusters in width-amplitude space
    plt.figure(figsize=(12, 8))
    
    valid_mask = species_labels >= 0
    if np.sum(valid_mask) > 0:
        widths = event_table.loc[event_table.index[valid_mask], 'pulse_width'].values * 1000
        amplitudes = event_table.loc[event_table.index[valid_mask], 'eod_amplitude'].values
        colors = species_labels[valid_mask]
        
        scatter = plt.scatter(widths, amplitudes, c=colors, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Species Cluster')
        plt.xlabel('Pulse Width (ms)')
        plt.ylabel('EOD Amplitude')
        plt.title('Species Clusters in Width-Amplitude Space')
        plt.savefig(output_path / 'species_clusters_2d.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Example waveforms for each species
    species_list = np.unique(species_labels[species_labels >= 0])
    if len(species_list) > 0:
        n_cols = min(4, len(species_list))
        n_rows = (len(species_list) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(4*n_cols, 3*n_rows))
        
        for i, species in enumerate(species_list):
            species_mask = species_labels == species
            species_waveforms = event_waveforms[species_mask]
            
            plt.subplot(n_rows, n_cols, i+1)
            
            # Plot up to 20 example waveforms
            n_examples = min(20, len(species_waveforms))
            for j in range(n_examples):
                plt.plot(species_waveforms[j], alpha=0.3, color='blue', linewidth=0.5)
            
            # Plot mean waveform
            mean_waveform = np.mean(species_waveforms, axis=0)
            plt.plot(mean_waveform, color='red', linewidth=2, label='Mean')
            
            plt.title(f'Species {species} (n={np.sum(species_mask)})')
            plt.xlabel('Sample')
            plt.ylabel('Normalized Amplitude')
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'species_waveforms.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Event-species assignment
    if len(event_summary) > 0 and len(species_list) > 0:
        plt.figure(figsize=(12, 6))
        
        # Get dominant species for each event
        event_species = []
        for event_id in event_summary['event_id']:
            event_mask = event_table['event_id'] == event_id
            if np.sum(event_mask) > 0:
                event_species_labels = species_labels[event_mask]
                valid_species = event_species_labels[event_species_labels >= 0]
                dominant_species = np.bincount(valid_species).argmax() if len(valid_species) > 0 else -1
                event_species.append(dominant_species)
            else:
                event_species.append(-1)
        
        event_species = np.array(event_species)
        
        # Plot event timeline colored by species
        start_times = pd.to_datetime(event_summary['start_time'])
        durations = event_summary['duration_seconds']
        
        for i, (start, duration, species) in enumerate(zip(start_times, durations, event_species)):
            color = plt.cm.tab10(species) if species >= 0 else 'gray'
            plt.barh(i, duration/60, left=(start - start_times.min()).total_seconds()/60, 
                    height=0.8, color=color, alpha=0.7)
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Event ID')
        plt.title('Event Timeline Colored by Dominant Species')
        plt.savefig(output_path / 'event_species_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Clustering summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Species distribution
    if len(species_list) > 0:
        species_counts = [np.sum(species_labels == s) for s in species_list]
        axes[0].bar(species_list, species_counts)
        axes[0].set_xlabel('Species Cluster')
        axes[0].set_ylabel('Number of EODs')
        axes[0].set_title('EODs per Species')
    
    # Individual distribution
    individual_list = np.unique(individual_labels[individual_labels >= 0])
    if len(individual_list) > 0:
        individual_counts = [np.sum(individual_labels == i) for i in individual_list]
        axes[1].bar(individual_list, individual_counts)
        axes[1].set_xlabel('Individual ID')
        axes[1].set_ylabel('Number of EODs')
        axes[1].set_title('EODs per Individual')
    
    # Clustering efficiency
    n_total = len(species_labels)
    n_clustered = np.sum(species_labels >= 0)
    n_unclustered = n_total - n_clustered
    
    axes[2].pie([n_clustered, n_unclustered], labels=['Clustered', 'Unclustered'], 
               autopct='%1.1f%%', startangle=90)
    axes[2].set_title('Clustering Efficiency')
    
    plt.tight_layout()
    plt.savefig(output_path / 'clustering_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Clustering plots saved to {output_folder}")

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
