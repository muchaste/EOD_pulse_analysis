"""
Event Classification Script

1. Cluster all EODs by length and shape
2. Assign species to clusters using trained classifiers
3. Estimate fish number per shape/species cluster
4. Create visualization plots of species-colored detections on raw data, waveforms per cluster
5. Save classified events and plots


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import audioio as aio
# import json
import gc
import warnings
import pickle
import re  # For regex pattern matching
from sklearn.mixture import BayesianGaussianMixture # Cluster EODs by length using bayesian Gaussian Mixture Model
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import pywt
from scipy import stats
from pulse_functions import load_fixed_length_waveforms, normalize_waveforms
warnings.filterwarnings('ignore')


print("="*60)
print("EOD EVENT CLASSIFICATION")
print("="*60)

# =============================================================================
# FILE SELECTION
# =============================================================================

# Set up GUI for file selection
root = tk.Tk()
root.withdraw()


# Either/or messagebox for feature-based or waveform-based classification
feature_classification = messagebox.askquestion("Select Classification Method",
                                               "Use feature-based classification (if no, waveform-based)?")

# Select trained classifier
print("\nSelect the trained classifier file...")
classifier_file = filedialog.askopenfilename(
    title="Select Trained Classifier File",
    filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
)

if not classifier_file:
    print("No classifier file selected. Exiting...")
    exit()

print(f"Selected classifier: {classifier_file}")

# Select folder with event EOD tables
print("\nSelect folder containing event EOD tables...")
events_folder = filedialog.askdirectory(
    title="Select Folder with Event EOD Tables"
)

if not events_folder:
    print("No events folder selected. Exiting...")
    exit()

print(f"Selected events folder: {events_folder}")

# Select output directory
print("\nSelect output directory for classified results...")
output_path = filedialog.askdirectory(
    title="Select Output Directory for Classification Results"
)

if not output_path:
    print("No output directory selected. Exiting...")
    exit()

print(f"Selected output directory: {output_path}")

#%%
# =============================================================================
# LOAD CLASSIFIER
# =============================================================================
print("\nLoading trained classifier...")
print("\n" + "="*60)
print("LOADING CLASSIFIER")
print("="*60)

try:
    print(f"Loading classifier from: {classifier_file}")
    with open(classifier_file, 'rb') as f:
        model_package = pickle.load(f)

    # Extract components from model package
    classifier = model_package['model']
    label_encoder = model_package['label_encoder']
    performance = model_package['performance']
    classifier_name = type(classifier).__name__
    test_accuracy = performance['test_accuracy']
    train_accuracy = performance['train_accuracy']
    cv_mean = performance['cv_mean']
    cv_std = performance['cv_std']
    auc_score = performance['auc_score']

    if feature_classification == 'yes':
        print("Using feature-based classification.")
        scaler = model_package['scaler']
        feature_columns = model_package.get('feature_columns', ['eod_amplitude_ratio', 'eod_width_us', 'fft_freq_max'])

    else:
        print("Using waveform-based classification.")
        scaler = model_package['wavelet_scaler']  # Wavelet-specific scaler
        wavelet_type_raw = model_package['wavelet_type']
        feature_count = model_package['feature_count']
        n_features = model_package['feature_count']
except Exception as e:
    print(f"✗ Error loading classifier: {e}")
    print("Please ensure the classifier file contains the required components:")
    print("  model, wavelet_scaler, label_encoder, wavelet_type, feature_count, performance")
    exit()

print(f"✓ Classifier loaded successfully: {classifier_name}")
print(f"  Test Accuracy: {test_accuracy:.2f}")
print(f"  Train Accuracy: {train_accuracy:.2f}")
print(f"  CV Mean Accuracy: {cv_mean:.2f} ± {cv_std:.2f}")
print(f"  AUC Score: {auc_score:.2f}")
print("="*60)
#%%
# =============================================================================
# PROCESS EACH EVENT FILE
# =============================================================================

# List event EOD files in the selected folder
# find all .csv files that follow this pattern:
# L2-20231011T191303_event_1_20231011T191303.csv, L2-20231011T191303_event_3_20231011T191311.csv, etc.

# Option 1: Simple pattern matching (current approach)
event_files = [f for f in os.listdir(events_folder) if f.endswith('.csv') and '_event_' in f and not f.startswith('all_')]

# Option 2: More robust regex pattern matching (recommended for complex patterns)
# event_pattern = re.compile(r'^[^_]+_\d{8}T\d{6}_event_\d+_\d{8}T\d{6}\.csv$')
# event_files = [f for f in os.listdir(events_folder) if event_pattern.match(f)]

# Option 3: If you need to filter by specific logger IDs or date ranges:
# event_pattern = re.compile(r'^L2-\d{8}T\d{6}_event_\d+_\d{8}T\d{6}\.csv$')
# event_files = [f for f in os.listdir(events_folder) if event_pattern.match(f)]
audio_files = [f for f in os.listdir(events_folder) if f.endswith('.wav')]
# waveform_files = [f for f in os.listdir(events_folder) if f.endswith('.npz')]

if not event_files:
    print("No event EOD files found in the selected folder. Exiting...")
    exit()

# length_param = 'waveform_length'
length_param = 'eod_width_us'

for event_file in event_files:
    print(f"\nProcessing event file: {event_file}")
    event_path = os.path.join(events_folder, event_file)
    
    # Load event EOD data
    event_data = pd.read_csv(event_path)
    print(f"Loaded {len(event_data)} EODs from {event_file}")

    # Check for required columns
    required_columns = [length_param, 'eod_amplitude_ratio', 'fft_freq_max']
    if not all(col in event_data.columns for col in required_columns):
        print(f"✗ Missing required columns in {event_file}. Required: {required_columns}")
        continue

    # Load corresponding raw audio file
    # Filename example: event_file = event_000_eod_table.csv; audio_file = event_000_20231011T191303.wav
    base_name = event_file.replace('.csv', '') # event_000

    # find base_name in audio_files
    if not any(f.startswith(base_name) for f in audio_files):
        print(f"✗ No corresponding audio file found for {base_name}. Skipping...")
        continue
    audio_file = [f for f in audio_files if f.startswith(base_name)][0]
    audio_file = os.path.join(events_folder, audio_file)
    audio_data, sample_rate = aio.load_audio(audio_file)
    print(f"Loaded audio data from {audio_file} with sample rate {sample_rate} Hz")

    # Load waveforms
    waveform_file = base_name + '_waveforms'
    waveform_file = os.path.join(events_folder, waveform_file)
    event_waveforms = load_fixed_length_waveforms(waveform_file)
    if event_waveforms is None or len(event_waveforms) == 0:
        print(f"✗ No waveforms found for {base_name}. Skipping...")
        continue
    print(f"Loaded {len(event_waveforms)} waveforms for event {base_name}")
    
    # Pre-filter: small events/low frequency are treated as 1 cluster
    # Calculate mean instant frequency and overall frequency
    event_data['timestamp'] = pd.to_datetime(event_data['timestamp']) 
    # Calculate inter-pulse intervals for this channel
    time_diffs = event_data['timestamp'].diff().dt.total_seconds()
    mean_inst_f = np.mean(1/time_diffs)
    

    if len(event_data) < 100 or mean_inst_f < 30:
        print(f"Pre-filtering event {base_name}: assigning to cluster 0")
        event_data['length_cluster'] = 0
        labels = np.zeros(len(event_data), dtype=int)
        n_length_clusters = 1
    else:
        print(f"Clustering event {base_name} by EOD length...")
        # Cluster by length using Bayesian Gaussian Mixture Model
        eod_widths = event_data[length_param].values

        # Auto-determine optimal number of clusters
        max_clusters = min(10, len(eod_widths) // 10)  # At least 10 points per cluster
        max_clusters = max(1, max_clusters)  # At least 1 cluster
        max_iter = 200
        n_init = 5
        use_log = True

        # Prepare data
        if use_log and np.all(eod_widths > 0):
            data_for_clustering = stats.zscore(np.log(eod_widths)).reshape(-1, 1)
        else:
            data_for_clustering = stats.zscore(eod_widths).reshape(-1, 1)

        if len(eod_widths) < 10:
            print(f"Too few EODs ({len(eod_widths)}) for clustering. Assigning all to one cluster.")
            labels = np.zeros(len(eod_widths), dtype=int)
            BGM_model = None
        else:
            # Find optimal number of components using BIC
            best_lower_bound = -np.inf
            best_model = None
            best_n_components = 1
            
            for n_components in range(1, max_clusters + 1):
                model = BayesianGaussianMixture(
                    n_components=n_components, 
                    max_iter=max_iter, 
                    n_init=n_init,
                    weight_concentration_prior=1,  
                    covariance_type='spherical',
                    covariance_prior=3*1e6/sample_rate,           # Allow for covariance of at least 3 samples
                    random_state=42
                )
                model.fit(data_for_clustering)
                lower_bound = model.lower_bound_

                if lower_bound > best_lower_bound:
                    best_lower_bound = lower_bound
                    best_model = model
                    best_n_components = n_components
            
            BGM_model = best_model
            labels = BGM_model.predict(data_for_clustering)

            print(f"Selected {best_n_components} clusters (lower_bound={best_lower_bound:.1f})")

            if not BGM_model.converged_:
                print('!!! Gaussian mixture did not converge !!!')

        event_data['length_cluster'] = labels
        n_length_clusters = len(np.unique(labels))

    print(f"Identified {n_length_clusters} length clusters.")

    # After clustering, filter out extreme outliers (n_length_clusters == 1) or small clusters (n_length_clusters > 1)
    if n_length_clusters == 1:
        print("All events assigned to a single cluster. Filtering out extreme outliers.")
        # Outlier = mean ± 5 standard deviations
        mean_length = np.mean(event_data[length_param])
        std_length = np.std(event_data[length_param])
        length_filter_mask = (event_data[length_param] < mean_length - 5 * std_length) | (event_data[length_param] > mean_length + 5 * std_length)
        print(f"One length cluster, filtering out {np.sum(length_filter_mask)} outliers")
    else:
        print("Multiple clusters found. Filtering out small clusters.")
        cluster_counts = event_data['length_cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < 10].index  # Clusters with fewer than 10 eods
        length_filter_mask = event_data['length_cluster'].isin(small_clusters)
        print(f"Filtering out {len(small_clusters)} small clusters: {small_clusters.tolist()} ({np.sum(length_filter_mask)} eods)")

    # Apply filtering
    event_data = event_data[~length_filter_mask]
    event_waveforms = [event_waveforms[i] for i in event_data.index]
    event_data.reset_index(drop=True, inplace=True)
    # Adjust n_length_clusters after filtering
    n_length_clusters = event_data['length_cluster'].nunique()
    # After filtering, some intermediate clusters might be removed, make list of clusters to iterate over
    length_clusters_to_iterate = event_data['length_cluster'].unique()

    # # Visualize length clusters: subplot 1: histogram, subplot 2: waveforms (one sub-subplot per cluster)
    # # Calculate subplot layout for clusters
    # n_cols = min(3, n_length_clusters)  # Maximum 3 columns
    # n_rows = (n_length_clusters + n_cols - 1) // n_cols  # Ceiling division
    
    # plt.figure(figsize=(15, 4 * n_rows))
    
    # # Subplot 1: Length histogram spanning all columns
    # plt.subplot(n_rows + 1, 1, 1)
    # # plt.hist(eod_widths, bins=n_length_clusters*10, color='gray', edgecolor='black', alpha=0.7)
    # colors = plt.cm.get_cmap('tab10', n_length_clusters)
    # for length_cluster in range(n_length_clusters):
    #     cluster_data = event_data[event_data['length_cluster'] == length_cluster]
    #     plt.hist(cluster_data[length_param], bins=10, color=colors(length_cluster), alpha=0.7, label=f'Length Cluster {length_cluster}')
    # plt.xlabel('EOD width (samples)')
    # plt.ylabel('Count')
    # plt.title('EOD Length Clusters')
    # plt.legend()
    
    # # Subplots 2+: Waveforms for each length_cluster
    # for length_cluster in length_clusters_to_iterate:
    #     plt.subplot(n_rows + 1, n_cols, length_cluster + n_cols + 1)
    #     length_cluster_indices = event_data[event_data['length_cluster'] == length_cluster].index
    #     length_cluster_waveforms = [event_waveforms[i] for i in length_cluster_indices]
    #     length_cluster_midpoints = event_data.loc[length_cluster_indices, 'snippet_midpoint_idx'].values
        
    #     for i, waveform in enumerate(length_cluster_waveforms):
    #         midpoint = length_cluster_midpoints[i]
    #         # Center waveform around midpoint
    #         time_axis = np.arange(len(waveform)) - midpoint
    #         plt.plot(time_axis, waveform, color=colors(length_cluster), alpha=0.3)
        
    #     plt.title(f'Cluster {length_cluster} Waveforms (n={len(length_cluster_waveforms)})')
    #     plt.xlabel('Samples from midpoint')
    #     plt.ylabel('Amplitude')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_path, f"{base_name}_length_clusters.png"))
    # plt.close()


    # =============================================================================
    # CLUSTER BY SHAPE WITHIN EACH LENGTH CLUSTER
    # =============================================================================

    print("Clustering by shape within each length cluster using DBSCAN...")
    for length_cluster in length_clusters_to_iterate:
        length_cluster_indices = event_data[event_data['length_cluster'] == length_cluster].index
        length_cluster_waveforms = [event_waveforms[i] for i in length_cluster_indices]
        length_cluster_data = event_data.loc[length_cluster_indices]
        
        if len(length_cluster_waveforms) < 2:
            # Not enough waveforms to cluster
            event_data.loc[length_cluster_indices, 'shape_cluster'] = 0
            continue
            
        # # Get peak and trough indices for alignment
        # length_cluster_peak_indices = event_data.loc[length_cluster_indices, 'snippet_p1_idx'].values
        # length_cluster_trough_indices = event_data.loc[length_cluster_indices, 'snippet_p2_idx'].values

        print(f"  Processing cluster {length_cluster} with {len(length_cluster_waveforms)} waveforms")

        # # STEP 1: Align waveforms to their midpoints (peak-trough midpoint)
        # aligned_waveforms = []
        # alignment_midpoints = []

        # for i, waveform in enumerate(length_cluster_waveforms):
        #     peak_idx = length_cluster_peak_indices[i]
        #     trough_idx = length_cluster_trough_indices[i]
        #     midpoint_idx = (peak_idx + trough_idx) // 2
        #     alignment_midpoints.append(midpoint_idx)
            
        #     # Create alignment by shifting waveform so midpoint is at center
        #     waveform_length = len(waveform)
        #     center_position = waveform_length // 2
        #     shift = midpoint_idx - center_position
            
        #     # Align waveform (roll to center the midpoint)
        #     if shift != 0:
        #         aligned_waveform = np.roll(waveform, -shift)
        #         # Handle edge effects by extrapolating rather than wrapping
        #         if shift > 0:  # Midpoint was to the right, now we have wrapped data at start
        #             aligned_waveform[:shift] = aligned_waveform[shift]  # Extend left
        #         else:  # Midpoint was to the left, wrapped data at end
        #             aligned_waveform[shift:] = aligned_waveform[shift-1]  # Extend right
        #     else:
        #         aligned_waveform = waveform.copy()
            
        #     aligned_waveforms.append(aligned_waveform)
        
        # # STEP 2: Determine optimal length (prune to common length)
        # lengths = [len(wf) for wf in aligned_waveforms]
        # target_length = int(np.percentile(lengths, 75))  # Use 75th percentile to avoid extreme outliers
        # target_length = max(target_length, 10)  # Ensure minimum meaningful length
        
        # print(f"    Target length for alignment: {target_length} samples")
        
        # # STEP 3: Prune/extend to target length (centered around midpoint)
        # normalized_waveforms = []
        
        # for aligned_waveform in aligned_waveforms:
        #     current_length = len(aligned_waveform)
        #     center_idx = current_length // 2
        #     half_target = target_length // 2
            
        #     # Calculate extraction bounds (centered)
        #     start_idx = max(0, center_idx - half_target)
        #     end_idx = min(current_length, center_idx + half_target)
            
        #     if end_idx - start_idx < target_length:
        #         # If we can't get target_length, pad symmetrically with edge values
        #         extracted = aligned_waveform[start_idx:end_idx]
        #         padding_needed = target_length - len(extracted)
        #         left_pad = padding_needed // 2
        #         right_pad = padding_needed - left_pad
                
        #         # Pad with edge values rather than zeros
        #         if len(extracted) > 0:
        #             padded_waveform = np.pad(extracted, (left_pad, right_pad), 
        #                                    mode='edge')  # Extend edge values
        #         else:
        #             padded_waveform = np.zeros(target_length)  # Fallback for empty
        #         normalized_waveforms.append(padded_waveform)
        #     else:
        #         # Extract centered portion
        #         extracted = aligned_waveform[start_idx:start_idx + target_length]
        #         normalized_waveforms.append(extracted)
        
        # # STEP 4: Additional normalization for clustering
        # # Ensure all waveforms are corrected for slope, have same peak amplitude and are zero-mean
        # final_waveforms = []
        # for waveform in normalized_waveforms:
        #     # Remove linear trend (detrending)
        #     slope = (waveform[-1] - waveform[0]) / (len(waveform) - 1)
        #     trend = slope * np.arange(len(waveform)) + waveform[0]
        #     waveform_detrended = waveform - trend

        #     # Remove DC offset
        #     waveform_centered = waveform_detrended - np.mean(waveform_detrended)
            
        #     # Normalize to unit peak-to-peak amplitude
        #     amplitude = np.max(waveform_centered) - np.min(waveform_centered)
        #     if amplitude > 0:
        #         waveform_normalized = waveform_centered / amplitude
        #     else:
        #         waveform_normalized = waveform_centered
        #     final_waveforms.append(waveform_normalized)
        
        final_waveforms = normalize_waveforms(length_cluster_waveforms, length_cluster_data['snippet_p1_idx'].values,
                                              length_cluster_data['snippet_p2_idx'].values, method='p1_unity')
        
        # STEP 5: Compute pairwise distances and cluster
        if len(final_waveforms) >= 2:
            distance_matrix = pairwise_distances(final_waveforms, metric='euclidean')
            
            # Adaptive eps based on data characteristics
            distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
            eps = np.percentile(distances, 20)  # Use 20th percentile as threshold
            
            dbscan = DBSCAN(eps=eps, min_samples=max(2, len(final_waveforms)//10), 
                           metric='precomputed')
            shape_labels = dbscan.fit_predict(distance_matrix)
            
            print(f"    Found {len(np.unique(shape_labels[shape_labels >= 0]))} shape clusters")
        else:
            shape_labels = np.zeros(len(final_waveforms), dtype=int)
        
        # Store shape cluster labels
        event_data.loc[length_cluster_indices, 'shape_cluster'] = shape_labels
        
        # Store normalized waveforms for later classification use
        if 'normalized_waveforms_by_cluster' not in locals():
            normalized_waveforms_by_cluster = {}
        normalized_waveforms_by_cluster[length_cluster] = final_waveforms

        # Filter out small shape clusters
        shape_cluster_counts = event_data.loc[length_cluster_indices, 'shape_cluster'].value_counts()
        valid_shape_clusters = shape_cluster_counts[shape_cluster_counts >= 5].index
        event_data.loc[length_cluster_indices, 'shape_cluster'] = event_data.loc[length_cluster_indices, 'shape_cluster'].where(event_data.loc[length_cluster_indices, 'shape_cluster'].isin(valid_shape_clusters), -1)
        event_waveforms = [event_waveforms[i] for i in event_data.index]
        event_data.reset_index(drop=True, inplace=True)


    # =============================================================================
    # SPECIES CLASSIFICATION USING CLUSTER REPRESENTATIVES
    # =============================================================================
    print("Classifying species using cluster representative EODs...")
    
    # Initialize species assignment columns
    event_data['predicted_species'] = 'Unknown'
    event_data['prediction_confidence'] = 0.0
    event_data['cluster_representative'] = False
    
    # Process each length cluster
    for length_cluster in length_clusters_to_iterate:
        length_cluster_data = event_data[event_data['length_cluster'] == length_cluster]
        
        if 'shape_cluster' not in length_cluster_data.columns:
            print(f"  Length cluster {length_cluster}: No shape clustering, skipping classification")
            continue
            
        # Get unique shape clusters (excluding noise points -1)
        shape_clusters = length_cluster_data['shape_cluster'].unique()
        shape_clusters = shape_clusters[shape_clusters >= 0]
        
        print(f"  Length cluster {length_cluster}: Processing {len(shape_clusters)} shape clusters")
        
        for shape_cluster in shape_clusters:
            # Get EODs in this length x shape cluster
            cluster_mask = (
                (event_data['length_cluster'] == length_cluster) & 
                (event_data['shape_cluster'] == shape_cluster)
            )
            cluster_indices = event_data[cluster_mask].index.tolist()
            cluster_eods = event_data[cluster_mask]
            
            if len(cluster_eods) == 0:
                continue
                
            print(f"    LC{length_cluster}.SC{shape_cluster}: {len(cluster_eods)} EODs")
            
            # Create representative EOD for this cluster
            if feature_classification == 'yes':
                # FEATURE-BASED CLASSIFICATION
                # Use median features as representative (more robust than mean)
                representative_features = cluster_eods[feature_columns].median().values.reshape(1, -1)
                
                # Classify the representative
                try:
                    # Scale features using the same scaler as training
                    representative_scaled = scaler.transform(representative_features)
                    
                    # Predict species and confidence
                    predicted_class = classifier.predict(representative_scaled)[0]
                    prediction_proba = classifier.predict_proba(representative_scaled)[0]
                    confidence = np.max(prediction_proba)
                    
                    # Decode species label
                    predicted_species = label_encoder.inverse_transform([predicted_class])[0]
                    
                    print(f"      Feature-based prediction: {predicted_species} (confidence: {confidence:.3f})")
                    
                except Exception as e:
                    print(f"      Classification failed: {e}")
                    predicted_species = 'Unknown'
                    confidence = 0.0
                    
            else:
                # WAVEFORM-BASED CLASSIFICATION
                # Use already aligned and normalized waveforms from shape clustering
                cluster_normalized_waveforms = []
                
                if 'normalized_waveforms_by_cluster' in locals() and length_cluster in normalized_waveforms_by_cluster:
                    # Get the normalized waveforms for this specific length cluster
                    length_cluster_final_waveforms = normalized_waveforms_by_cluster[length_cluster]
                    length_cluster_indices = event_data[event_data['length_cluster'] == length_cluster].index.tolist()
                    
                    # Map cluster indices to position in final waveforms array
                    for idx in cluster_indices:
                        if idx in length_cluster_indices:
                            relative_idx = length_cluster_indices.index(idx)
                            if relative_idx < len(length_cluster_final_waveforms):
                                cluster_normalized_waveforms.append(length_cluster_final_waveforms[relative_idx])
                
                if len(cluster_normalized_waveforms) == 0:
                    predicted_species = 'Unknown'
                    confidence = 0.0
                else:
                    # Create representative waveform (median for robustness)
                    representative_waveform = np.median(cluster_normalized_waveforms, axis=0)
                    
                    try:
                        # Extract wavelet features from representative
                        
                        # Fix wavelet type naming (remove 'dwt_' prefix if present)
                        wavelet_type = wavelet_type_raw
                        if wavelet_type.startswith('dwt_') or wavelet_type.startswith('DWT_'):
                            wavelet_type = wavelet_type[4:]  # Remove 'dwt_' prefix
                        
                        print(f"      Using wavelet: {wavelet_type}")
                        
                        # Wavelet decomposition
                        coeffs = pywt.wavedec(representative_waveform, wavelet_type, level=4)
                        
                        # Extract features (same as training)
                        wavelet_features = []
                        for coeff in coeffs:
                            if len(coeff) > 0:
                                wavelet_features.extend([
                                    np.mean(coeff), np.std(coeff), 
                                    np.max(coeff), np.min(coeff)
                                ])
                        
                        # Pad or truncate to match training feature count
                        if len(wavelet_features) > n_features:
                            wavelet_features = wavelet_features[:n_features]
                        elif len(wavelet_features) < n_features:
                            wavelet_features.extend([0.0] * (n_features - len(wavelet_features)))
                        
                        representative_features = np.array(wavelet_features).reshape(1, -1)
                        
                        # Scale and classify
                        representative_scaled = scaler.transform(representative_features)
                        predicted_class = classifier.predict(representative_scaled)[0]
                        prediction_proba = classifier.predict_proba(representative_scaled)[0]
                        confidence = np.max(prediction_proba)
                        predicted_species = label_encoder.inverse_transform([predicted_class])[0]
                        
                        print(f"      Waveform-based prediction: {predicted_species} (confidence: {confidence:.3f})")
                        
                    except Exception as e:
                        print(f"      Wavelet feature extraction failed: {e}")
                        predicted_species = 'Unknown'
                        confidence = 0.0
            
            # Assign prediction to all EODs in this cluster
            event_data.loc[cluster_mask, 'predicted_species'] = predicted_species
            event_data.loc[cluster_mask, 'prediction_confidence'] = confidence
            
            # Mark the first EOD as cluster representative for reference
            if len(cluster_indices) > 0:
                event_data.loc[cluster_indices[0], 'cluster_representative'] = True
    
    # Classification summary
    species_counts = event_data['predicted_species'].value_counts()
    print(f"\nClassification Summary for {base_name}:")
    for species, count in species_counts.items():
        mean_confidence = event_data[event_data['predicted_species'] == species]['prediction_confidence'].mean()
        print(f"  {species}: {count} EODs (avg confidence: {mean_confidence:.3f})")
    
    # Save classified results
    classified_output = os.path.join(output_path, f"{base_name}_classified_eod_table.csv")
    event_data.to_csv(classified_output, index=False)
    print(f"Saved classified results to: {classified_output}")
    
    
    
    # =============================================================================
    # COMPREHENSIVE EVENT VISUALIZATION
    # =============================================================================
   
    # Calculate total shape clusters across all length clusters
    print("Creating comprehensive visualization...")
    length_cluster_info = {}
    
    for length_cluster in length_clusters_to_iterate:
        cluster_data = event_data[event_data['length_cluster'] == length_cluster]
        if 'shape_cluster' in cluster_data.columns:
            shape_clusters = cluster_data['shape_cluster'].unique()
            shape_clusters = shape_clusters[shape_clusters >= 0]  # Remove noise (-1)
            length_cluster_info[length_cluster] = {
                'n_eods': len(cluster_data),
                'shape_clusters': shape_clusters,
                'n_shape_clusters': len(shape_clusters)
            }
        else:
            length_cluster_info[length_cluster] = {
                'n_eods': len(cluster_data),
                'shape_clusters': [],
                'n_shape_clusters': 0
            }
    
    # Dynamic layout calculation
    max_shape_clusters_per_length = max([info['n_shape_clusters'] for info in length_cluster_info.values()]) if length_cluster_info else 1
    max_shape_clusters_per_length = max(1, max_shape_clusters_per_length)
    
    # Calculate grid: length clusters as columns, shape clusters as rows within each
    n_length_cols = min(4, n_length_clusters)  # Max 4 length clusters per row
    total_rows = 2 + max_shape_clusters_per_length  # Top summary + length hist + shape rows
    
    fig = plt.figure(figsize=(4 * n_length_cols, 3 * total_rows))
    gs = fig.add_gridspec(total_rows, n_length_cols, hspace=0.4, wspace=0.3)
    
    # ROW 1: Overall length distribution
    ax_length = fig.add_subplot(gs[0, :])
    colors = plt.cm.get_cmap('tab10', n_length_clusters)
    for length_cluster in range(n_length_clusters):
        cluster_data = event_data[event_data['length_cluster'] == length_cluster]
        ax_length.hist(cluster_data[length_param], bins=15, color=colors(length_cluster), 
                      alpha=0.7, label=f'LC{length_cluster} (n={len(cluster_data)})')
    ax_length.set_xlabel('EOD Width (μs)')
    ax_length.set_ylabel('Count')
    ax_length.set_title(f'Event {base_name}: Length Clusters (Total: {len(event_data)} EODs)')
    ax_length.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # ROW 2: Summary statistics
    ax_summary = fig.add_subplot(gs[1, :])
    ax_summary.axis('off')
    summary_text = f"""LENGTH CLUSTERS: {n_length_clusters} clusters found
TOTAL EODs: {len(event_data)}
MEAN INST. FREQ: {mean_inst_f:.1f} Hz"""
    
    # Add length cluster details
    for lc, info in length_cluster_info.items():
        summary_text += f"\n  LC{lc}: {info['n_eods']} EODs → {info['n_shape_clusters']} shape clusters"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    # ROWS 3+: Shape clusters organized by length cluster
    for length_cluster in length_clusters_to_iterate:
        col = length_cluster % n_length_cols
        info = length_cluster_info[length_cluster]
        
        if info['n_shape_clusters'] == 0:
            # No shape clustering performed
            row = 2
            ax = fig.add_subplot(gs[row, col])
            ax.text(0.5, 0.5, f'Length Cluster {length_cluster}\n{info["n_eods"]} EODs\nNo shape clustering', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors(length_cluster), alpha=0.3))
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Plot each shape cluster
        for shape_idx, shape_cluster in enumerate(info['shape_clusters']):
            row = 2 + shape_idx
            ax = fig.add_subplot(gs[row, col])
            
            # Get waveforms for this shape cluster
            shape_data = event_data[
                (event_data['length_cluster'] == length_cluster) & 
                (event_data['shape_cluster'] == shape_cluster)
            ]
            
            if len(shape_data) > 0:
                # Get corresponding waveforms
                shape_indices = shape_data.index.tolist()
                n_examples = min(20, len(shape_indices))
                
                for i, idx in enumerate(shape_indices[:n_examples]):
                    if idx < len(final_waveforms):
                        waveform = final_waveforms[idx]
                        # Center around midpoint for display
                        if 'snippet_midpoint_idx' in event_data.columns:
                            midpoint = event_data.loc[idx, 'snippet_midpoint_idx']
                            time_axis = np.arange(len(waveform)) - midpoint
                        else:
                            time_axis = np.arange(len(waveform)) - len(waveform)//2
                        ax.plot(time_axis, waveform, color=colors(length_cluster), alpha=0.4, linewidth=0.8)
                
                ax.set_title(f'LC{length_cluster}.SC{shape_cluster}\n({len(shape_data)} EODs)', fontsize=9)
                ax.set_xlabel('Samples from midpoint', fontsize=8)
                ax.set_ylabel('Amplitude', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.savefig(os.path.join(output_path, f"{base_name}_comprehensive_clustering.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    
    # =============================================================================
    # SPECIES-COLORED EVENT VISUALIZATION 
    # =============================================================================
    print("Creating species-colored event visualization...")
    
    # Get unique species for color mapping
    unique_species = event_data['predicted_species'].unique()
    n_species = len(unique_species)
    
    # Create species color map
    species_colors = plt.cm.get_cmap('tab10', n_species)
    species_color_dict = {species: species_colors(i) for i, species in enumerate(unique_species)}
    
    # Check event duration for plotting strategy
    audio_duration = len(audio_data) / sample_rate
    max_plot_duration = 300  # 5 minutes
    
    if audio_duration > max_plot_duration:
        plot_step = 3  # 3x downsampling for long events
        print(f"  Long event detected ({audio_duration:.1f}s) - applying 3x downsampling")
    else:
        plot_step = 1  # Original resolution
        print(f"  Normal event ({audio_duration:.1f}s) - using original resolution")
    
    # Calculate offset for stacking differential channels
    max_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 1.0
    offset_diff = max_amplitude * 1.5
    
    plt.figure(figsize=(20, 10))
    
    try:
        # Plot differential signals for each channel pair
        for i in range(audio_data.shape[1]-1):
            # Calculate differential signal with downsampling
            data_diff = np.diff(audio_data[::plot_step, i:i+2], axis=1).flatten()
            
            # Create time coordinates
            time_indices = np.arange(0, len(audio_data), plot_step)[:len(data_diff)]
            time_seconds = time_indices / sample_rate
            
            plt.plot(time_seconds, data_diff + i * offset_diff, 
                    linewidth=0.5, color='gray', alpha=0.7, label=f'Ch{i}-{i+1}')
            
            # Plot detected EODs colored by predicted species
            ch_eods = event_data[event_data['eod_channel'] == i]
            
            if len(ch_eods) > 0:
                # Group EODs by species for efficient plotting
                for species in unique_species:
                    species_eods = ch_eods[ch_eods['predicted_species'] == species]
                    
                    if len(species_eods) == 0:
                        continue
                    
                    # Plot midpoints (main detection markers)
                    if 'midpoint_idx' in species_eods.columns:
                        midpoint_indices = species_eods['midpoint_idx'].values
                        valid_midpoints = (midpoint_indices >= 0) & (midpoint_indices < len(audio_data))
                        
                        if np.any(valid_midpoints):
                            valid_midpoint_indices = midpoint_indices[valid_midpoints]
                            midpoint_times = valid_midpoint_indices / sample_rate
                            
                            # Get differential signal values at midpoints
                            midpoint_diffs = np.diff(audio_data[valid_midpoint_indices.astype(int), i:i+2], axis=1).flatten()
                            
                            plt.scatter(midpoint_times, midpoint_diffs + i * offset_diff,
                                      c=[species_color_dict[species]], s=30, alpha=0.8,
                                      label=f'{species}' if i == 0 else "", 
                                      edgecolors='black', linewidths=0.5)
                    
                    # Plot peaks and troughs with species colors (smaller markers)
                    if 'p1_idx' in species_eods.columns:
                        peak_indices = species_eods['p1_idx'].values
                        valid_peaks = (peak_indices >= 0) & (peak_indices < len(audio_data))
                        
                        if np.any(valid_peaks):
                            valid_peak_indices = peak_indices[valid_peaks]
                            peak_times = valid_peak_indices / sample_rate
                            peak_diffs = np.diff(audio_data[valid_peak_indices.astype(int), i:i+2], axis=1).flatten()
                            
                            plt.scatter(peak_times, peak_diffs + i * offset_diff,
                                      c=[species_color_dict[species]], s=15, alpha=0.6,
                                      marker='^', edgecolors='black', linewidths=0.3)
                    
                    if 'p2_idx' in species_eods.columns:
                        trough_indices = species_eods['p2_idx'].values
                        valid_troughs = (trough_indices >= 0) & (trough_indices < len(audio_data))
                        
                        if np.any(valid_troughs):
                            valid_trough_indices = trough_indices[valid_troughs]
                            trough_times = valid_trough_indices / sample_rate
                            trough_diffs = np.diff(audio_data[valid_trough_indices.astype(int), i:i+2], axis=1).flatten()
                            
                            plt.scatter(trough_times, trough_diffs + i * offset_diff,
                                      c=[species_color_dict[species]], s=15, alpha=0.6,
                                      marker='v', edgecolors='black', linewidths=0.3)
            
            # Clean up large arrays immediately
            del data_diff
        
        # Formatting
        plt.ylim(bottom=None, top=(audio_data.shape[1]-1.5)*offset_diff)
        
        title_suffix = f" (downsampled {plot_step}x)" if plot_step > 1 else ""
        plt.title(f'Event {base_name}: Species Classification Results{title_suffix}\n'
                 f'Duration: {audio_duration:.1f}s - Colors: predicted species', fontsize=14)
        
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Differential Voltage (stacked by channel)', fontsize=12)
        
        # Create legend with species counts and confidence
        legend_elements = []
        for species in unique_species:
            species_count = len(event_data[event_data['predicted_species'] == species])
            avg_confidence = event_data[event_data['predicted_species'] == species]['prediction_confidence'].mean()
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=species_color_dict[species],
                                            markersize=10, markeredgecolor='black',
                                            label=f'{species} (n={species_count}, conf={avg_confidence:.2f})'))
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        species_plot_path = os.path.join(output_path, f"{base_name}_species_classification.png")
        plt.savefig(species_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved species-colored event plot: {species_plot_path}")
        
    except Exception as e:
        print(f"  ERROR creating species plot: {e}")
        plt.close()
        
        # Create a simple info plot instead
        try:
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.7, f'Event {base_name}\nSpecies Classification Plot Creation Failed', 
                    ha='center', va='center', fontsize=16, 
                    bbox=dict(boxstyle="round,pad=1", facecolor="lightcoral", alpha=0.7))
            
            # Add species summary
            summary_text = f"Classification Summary:\n"
            for species in unique_species:
                count = len(event_data[event_data['predicted_species'] == species])
                avg_conf = event_data[event_data['predicted_species'] == species]['prediction_confidence'].mean()
                summary_text += f"  {species}: {count} EODs (conf: {avg_conf:.2f})\n"
            summary_text += f"\nError: {str(e)}"
            
            plt.text(0.5, 0.3, summary_text, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            
            error_plot_path = os.path.join(output_path, f"{base_name}_species_error.png")
            plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved error info plot: {error_plot_path}")
            
        except:
            print(f"  Could not create error plot either")
    
    # Clean up memory
    del audio_data
    if 'data_diff' in locals():
        del data_diff
    gc.collect()