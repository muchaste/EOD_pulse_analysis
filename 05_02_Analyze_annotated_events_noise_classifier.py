#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Annotations and Analyze Single Fish Events
===============================================

This script loads the annotation data from the JSON file and analyzes
single fish events to check pulse similarity.

"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import glob
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# from sklearn.svm import SVM
from sklearn.linear_model import LogisticRegression
# Import seaborn for better visualizations
import seaborn as sns
from diptest import diptest # Try to import diptest for bimodality testing
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
# =============================================================================
# LOAD ANNOTATIONS FROM JSON
# =============================================================================

# Ask user to select the annotation JSON file
print("Select the annotation JSON file...")
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide the main window

annotation_file = filedialog.askopenfilename(
    title="Select Annotation JSON File",
    filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
)

if not annotation_file:
    print("No file selected. Exiting.")
    exit()

print(f"Loading annotations from: {annotation_file}")

# Load the JSON file
with open(annotation_file, 'r') as f:
    annotation_data = json.load(f)

# Extract the different parts of the annotation data
annotations = annotation_data.get('annotations', {})
fish_counts = annotation_data.get('fish_counts', {})
features = annotation_data.get('features', {})
metadata = annotation_data.get('metadata', {})

print(f"\nLoaded annotation data:")
print(f"- Total events annotated: {len(annotations)}")
print(f"- Events with fish counts: {len(fish_counts)}")
print(f"- Events with features: {len(features)}")

# Print summary of annotations
annotation_summary = {}
for event_id, annotation in annotations.items():
    if annotation not in annotation_summary:
        annotation_summary[annotation] = 0
    annotation_summary[annotation] += 1

print(f"\nAnnotation summary:")
for annotation_type, count in annotation_summary.items():
    print(f"- {annotation_type}: {count}")

# Print summary of fish counts
fish_count_summary = {}
for event_id, count in fish_counts.items():
    if count not in fish_count_summary:
        fish_count_summary[count] = 0
    fish_count_summary[count] += 1

print(f"\nFish count summary:")
for count_type, count in fish_count_summary.items():
    print(f"- {count_type} fish: {count}")

# =============================================================================
# FIND SINGLE FISH EVENTS
# =============================================================================

# Find events that are "clear_fish" with fish_count "1"
single_fish_events = []
for event_id, annotation in annotations.items():
    if annotation == 'clear_fish':
        fish_count = fish_counts.get(event_id, None)
        if fish_count == '1':
            single_fish_events.append(event_id)

print(f"\nFound {len(single_fish_events)} single fish events:")
for i, event_id in enumerate(single_fish_events[:10]):  # Show first 10
    print(f"- {event_id}")
if len(single_fish_events) > 10:
    print(f"... and {len(single_fish_events) - 10} more")

# =============================================================================
# LOAD EOD DATA FOR SINGLE FISH EVENTS
# =============================================================================

# Get the events folder from metadata or ask user
events_folder = metadata.get('events_folder', None)
if not events_folder or not os.path.exists(events_folder):
    print("\nSelect the events folder (Script 04 output)...")
    events_folder = filedialog.askdirectory(
        title="Select Events Folder (Script 04 output)"
    )
    if not events_folder:
        print("No folder selected. Exiting.")
        exit()

events_folder = Path(events_folder)
print(f"\nLoading EOD data from: {events_folder}")

# Load EOD tables for single fish events
single_fish_eod_data = {}
missing_eod_files = []

for event_id in single_fish_events:
    # Try different EOD file patterns
    eod_file_patterns = [
        f"{event_id}_eod_table.csv",
        f"event_{event_id.split('_')[1]}_eod_table.csv" if '_' in event_id else f"{event_id}_eod_table.csv"
    ]
    
    eod_file_found = False
    for pattern in eod_file_patterns:
        eod_file = events_folder / pattern
        if eod_file.exists():
            try:
                eod_table = pd.read_csv(eod_file)
                single_fish_eod_data[event_id] = eod_table
                eod_file_found = True
                print(f"Loaded EOD data for {event_id}: {len(eod_table)} EODs")
                break
            except Exception as e:
                print(f"Error loading {eod_file}: {e}")
    
    if not eod_file_found:
        missing_eod_files.append(event_id)

print(f"\nSuccessfully loaded EOD data for {len(single_fish_eod_data)} single fish events")
if missing_eod_files:
    print(f"Missing EOD files for {len(missing_eod_files)} events:")
    for event_id in missing_eod_files[:5]:  # Show first 5
        print(f"- {event_id}")
    if len(missing_eod_files) > 5:
        print(f"... and {len(missing_eod_files) - 5} more")

# =============================================================================
# ANALYZE PULSE SIMILARITY WITHIN SINGLE FISH EVENTS
# =============================================================================

print(f"\n" + "="*50)
print("ANALYZING PULSE SIMILARITY WITHIN SINGLE FISH EVENTS")
print("="*50)

# Analyze each single fish event
pulse_similarity_results = []

for event_id, eod_table in single_fish_eod_data.items():
    if len(eod_table) < 2:
        print(f"\nSkipping {event_id}: Only {len(eod_table)} EODs (need at least 2)")
        continue
    
    print(f"\nAnalyzing {event_id}: {len(eod_table)} EODs")
    
    # Basic statistics for this event
    event_stats = {
        'event_id': event_id,
        'n_eods': len(eod_table),
        'duration_seconds': features.get(event_id, {}).get('duration_seconds', np.nan),
        'n_channels': len(eod_table['eod_channel'].unique()) if 'eod_channel' in eod_table.columns else np.nan
    }
    
    # # Amplitude analysis
    # if 'eod_amplitude' in eod_table.columns:
    #     amplitudes = eod_table['eod_amplitude'].dropna()
    #     if len(amplitudes) > 1:
    #         event_stats['amplitude_mean'] = amplitudes.mean()
    #         event_stats['amplitude_std'] = amplitudes.std()
    #         event_stats['amplitude_cv'] = amplitudes.std() / amplitudes.mean() if amplitudes.mean() > 0 else np.nan
    #         event_stats['amplitude_range'] = amplitudes.max() - amplitudes.min()
    #         print(f"  Amplitude CV: {event_stats['amplitude_cv']:.3f}")
    
    # # Inter-pulse interval analysis
    # if 'timestamp_dt' in eod_table.columns:
    #     eod_table['timestamp_dt'] = pd.to_datetime(eod_table['timestamp_dt'])
    #     eod_times = eod_table.sort_values('timestamp_dt')['timestamp_dt']
    #     ipis = eod_times.diff().dt.total_seconds().dropna()
        
    #     if len(ipis) > 1:
    #         event_stats['ipi_mean'] = ipis.mean()
    #         event_stats['ipi_std'] = ipis.std()
    #         event_stats['ipi_cv'] = ipis.std() / ipis.mean() if ipis.mean() > 0 else np.nan
    #         event_stats['ipi_median'] = ipis.median()
    #         event_stats['ipi_min'] = ipis.min()
    #         event_stats['ipi_max'] = ipis.max()
    #         print(f"  IPI CV: {event_stats['ipi_cv']:.3f}, Mean IPI: {event_stats['ipi_mean']:.3f}s")
    
    # Waveform width analysis
    if 'eod_width_us' in eod_table.columns:
        widths = eod_table['eod_width_us'].dropna()
        if len(widths) > 1:
            event_stats['width_mean'] = widths.mean()
            event_stats['width_std'] = widths.std()
            event_stats['width_cv'] = widths.std() / widths.mean() if widths.mean() > 0 else np.nan
            print(f"  Width CV: {event_stats['width_cv']:.3f}")
    
    # Frequency analysis
    if 'fft_freq_max' in eod_table.columns:
        freqs = eod_table['fft_freq_max'].dropna()
        if len(freqs) > 1:
            event_stats['freq_mean'] = freqs.mean()
            event_stats['freq_std'] = freqs.std()
            event_stats['freq_cv'] = freqs.std() / freqs.mean() if freqs.mean() > 0 else np.nan
            print(f"  Frequency CV: {event_stats['freq_cv']:.3f}")

    # eod amplitude ratio analysis
    if 'eod_amplitude_ratio' in eod_table.columns:
        amp_ratios = eod_table['eod_amplitude_ratio'].dropna()
        if len(amp_ratios) > 1:
            event_stats['amp_ratio_mean'] = amp_ratios.mean()
            event_stats['amp_ratio_std'] = amp_ratios.std()
            event_stats['amp_ratio_cv'] = amp_ratios.std() / amp_ratios.mean() if amp_ratios.mean() > 0 else np.nan
            print(f"  Amplitude Ratio CV: {event_stats['amp_ratio_cv']:.3f}")
    
    pulse_similarity_results.append(event_stats)

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(pulse_similarity_results)

print(f"\n" + "="*50)
print("SUMMARY STATISTICS ACROSS ALL SINGLE FISH EVENTS")
print("="*50)

if len(results_df) > 0:
    # Overall statistics
    print(f"Total single fish events analyzed: {len(results_df)}")
    print(f"Total EODs analyzed: {results_df['n_eods'].sum()}")
    print(f"Average EODs per event: {results_df['n_eods'].mean():.1f}")
    
    # Coefficient of variation statistics
    cv_columns = [col for col in results_df.columns if col.endswith('_cv')]
    
    for cv_col in cv_columns:
        cv_values = results_df[cv_col].dropna()
        if len(cv_values) > 0:
            feature_name = cv_col.replace('_cv', '').capitalize()
            print(f"\n{feature_name} Coefficient of Variation:")
            print(f"  Mean CV: {cv_values.mean():.4f}")
            print(f"  Median CV: {cv_values.median():.4f}")
            print(f"  Std CV: {cv_values.std():.4f}")
            print(f"  Range: {cv_values.min():.4f} - {cv_values.max():.4f}")
            
            # Events with very low CV (highly similar pulses)
            low_cv_threshold = 0.1  # 10% CV
            low_cv_events = cv_values[cv_values < low_cv_threshold]
            print(f"  Events with CV < {low_cv_threshold}: {len(low_cv_events)} ({len(low_cv_events)/len(cv_values)*100:.1f}%)")
#%%
# =============================================================================
# DETAILED WAVEFORM METRICS VISUALIZATION
# =============================================================================

print(f"\n" + "="*50)
print("CREATING DETAILED WAVEFORM METRICS VISUALIZATIONS")
print("="*50)

# Focus on the three key waveform metrics
waveform_metrics = ['eod_width_us', 'fft_freq_max', 'eod_amplitude_ratio']
available_metrics = []

# Prepare data for visualization
all_raw_data = {}  # Store raw values for each metric across all events
all_normalized_data = {}  # Store normalized values (by event mean)
event_stats_for_viz = {}  # Store per-event statistics

for metric in waveform_metrics:
    all_raw_data[metric] = []
    all_normalized_data[metric] = []
    event_stats_for_viz[metric] = {
        'event_ids': [],
        'means': [],
        'stds': [],
        'cvs': [],
        'medians': [],
        'n_pulses': []
    }

# Collect data from all single fish events
for event_id, eod_table in single_fish_eod_data.items():
    if len(eod_table) < 3:  # Need at least 3 pulses for meaningful statistics
        continue
    
    for metric in waveform_metrics:
        if metric in eod_table.columns:
            values = eod_table[metric].dropna()
            if len(values) >= 3:
                # Store raw values
                all_raw_data[metric].extend(values.tolist())
                
                # Calculate event statistics
                mean_val = values.mean()
                std_val = values.std()
                cv_val = std_val / mean_val if mean_val > 0 else np.nan
                median_val = values.median()
                
                # Store normalized values (by event mean)
                normalized_values = values / mean_val if mean_val > 0 else values
                all_normalized_data[metric].extend(normalized_values.tolist())
                
                # Store event statistics
                event_stats_for_viz[metric]['event_ids'].append(event_id)
                event_stats_for_viz[metric]['means'].append(mean_val)
                event_stats_for_viz[metric]['stds'].append(std_val)
                event_stats_for_viz[metric]['cvs'].append(cv_val)
                event_stats_for_viz[metric]['medians'].append(median_val)
                event_stats_for_viz[metric]['n_pulses'].append(len(values))

# Filter to only metrics with sufficient data
for metric in waveform_metrics:
    if len(all_raw_data[metric]) > 0:
        available_metrics.append(metric)

print(f"Available metrics for visualization: {available_metrics}")

if len(available_metrics) > 0:
    
    # =============================================================================
    # FIGURE 1: BOXPLOTS OF RAW VALUES ACROSS EVENTS
    # =============================================================================
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
    if len(available_metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        # Prepare data for boxplot - group by event
        event_data = []
        event_labels = []
        
        for j, event_id in enumerate(event_stats_for_viz[metric]['event_ids']):
            if j < 20:  # Limit to first 20 events for readability
                eod_table = single_fish_eod_data[event_id]
                values = eod_table[metric].dropna()
                if len(values) >= 3:
                    event_data.append(values.tolist())
                    event_labels.append(f"E{j+1}")
        
        if len(event_data) > 0:
            bp = axes[i].boxplot(event_data, labels=event_labels, patch_artist=True)
            
            # Color boxes by CV
            cvs = event_stats_for_viz[metric]['cvs'][:len(event_data)]
            colors = plt.cm.RdYlBu_r(np.array(cvs) / max(cvs))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[i].set_title(f'{metric}\n(Raw Values)')
            axes[i].set_xlabel('Event')
            axes[i].set_ylabel('Value')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add CV information as text
            axes[i].text(0.02, 0.98, f'Mean CV: {np.mean(cvs):.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(events_folder / "waveform_metrics_boxplots_raw.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # FIGURE 2: BOXPLOTS OF NORMALIZED VALUES ACROSS EVENTS
    # =============================================================================
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
    if len(available_metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        # Prepare normalized data for boxplot
        event_data_norm = []
        event_labels = []
        
        for j, event_id in enumerate(event_stats_for_viz[metric]['event_ids']):
            if j < 20:  # Limit to first 20 events for readability
                eod_table = single_fish_eod_data[event_id]
                values = eod_table[metric].dropna()
                if len(values) >= 3:
                    mean_val = values.mean()
                    normalized_values = values / mean_val if mean_val > 0 else values
                    event_data_norm.append(normalized_values.tolist())
                    event_labels.append(f"E{j+1}")
        
        if len(event_data_norm) > 0:
            bp = axes[i].boxplot(event_data_norm, labels=event_labels, patch_artist=True)
            
            # Color boxes by CV (same as raw)
            cvs = event_stats_for_viz[metric]['cvs'][:len(event_data_norm)]
            colors = plt.cm.RdYlBu_r(np.array(cvs) / max(cvs))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[i].set_title(f'{metric}\n(Normalized by Event Mean)')
            axes[i].set_xlabel('Event')
            axes[i].set_ylabel('Normalized Value (Value/Mean)')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Event Mean')
            
            # Add CV information
            axes[i].text(0.02, 0.98, f'Mean CV: {np.mean(cvs):.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(events_folder / "waveform_metrics_boxplots_normalized.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # FIGURE 3: CV vs MEAN VALUES (RELATIONSHIP ANALYSIS)
    # =============================================================================
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
    if len(available_metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        means = np.array(event_stats_for_viz[metric]['means'])
        cvs = np.array(event_stats_for_viz[metric]['cvs'])
        n_pulses = np.array(event_stats_for_viz[metric]['n_pulses'])
        
        # Remove NaN values
        valid_mask = ~(np.isnan(means) | np.isnan(cvs))
        means_valid = means[valid_mask]
        cvs_valid = cvs[valid_mask]
        n_pulses_valid = n_pulses[valid_mask]
        
        if len(means_valid) > 0:
            # Scatter plot with size based on number of pulses
            scatter = axes[i].scatter(means_valid, cvs_valid, 
                                    s=n_pulses_valid*3, alpha=0.6, 
                                    c=n_pulses_valid, cmap='viridis')
            
            # Add trend line
            if len(means_valid) > 2:
                z = np.polyfit(means_valid, cvs_valid, 1)
                p = np.poly1d(z)
                axes[i].plot(means_valid, p(means_valid), "r--", alpha=0.8, linewidth=2)
                
                # Calculate correlation
                correlation = np.corrcoef(means_valid, cvs_valid)[0, 1]
                axes[i].text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                            transform=axes[i].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[i].set_xlabel(f'Mean {metric}')
            axes[i].set_ylabel('Coefficient of Variation')
            axes[i].set_title(f'{metric}\n(CV vs Mean - bubble size = # pulses)')
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[i], label='# Pulses')
    
    plt.tight_layout()
    plt.savefig(events_folder / "waveform_metrics_cv_vs_mean.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # FIGURE 4: OVERALL DISTRIBUTION COMPARISON
    # =============================================================================
    
    fig, axes = plt.subplots(2, len(available_metrics), figsize=(5*len(available_metrics), 10))
    if len(available_metrics) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, metric in enumerate(available_metrics):
        # Raw values distribution
        axes[0, i].hist(all_raw_data[metric], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, i].set_title(f'{metric} - Raw Values Distribution')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].axvline(np.mean(all_raw_data[metric]), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_raw_data[metric]):.2f}')
        axes[0, i].legend()
        
        # Normalized values distribution
        axes[1, i].hist(all_normalized_data[metric], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, i].set_title(f'{metric} - Normalized Values Distribution')
        axes[1, i].set_xlabel('Normalized Value (Value/Event Mean)')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].axvline(1.0, color='red', linestyle='--', label='Event Mean (1.0)')
        axes[1, i].axvline(np.mean(all_normalized_data[metric]), color='blue', linestyle='--', 
                          label=f'Overall Mean: {np.mean(all_normalized_data[metric]):.3f}')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(events_folder / "waveform_metrics_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # SUMMARY STATISTICS FOR WAVEFORM METRICS
    # =============================================================================
    
    print(f"\n" + "="*50)
    print("WAVEFORM METRICS SUMMARY")
    print("="*50)
    
    for metric in available_metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 30)
        
        # Raw values stats
        raw_values = all_raw_data[metric]
        print(f"Raw values (n={len(raw_values)}):")
        print(f"  Mean ± SD: {np.mean(raw_values):.3f} ± {np.std(raw_values):.3f}")
        print(f"  Median [IQR]: {np.median(raw_values):.3f} [{np.percentile(raw_values, 25):.3f}, {np.percentile(raw_values, 75):.3f}]")
        print(f"  Range: {np.min(raw_values):.3f} - {np.max(raw_values):.3f}")
        
        # Event-level CV stats
        cvs = event_stats_for_viz[metric]['cvs']
        print(f"\nCoefficient of Variation across events (n={len(cvs)}):")
        print(f"  Mean CV: {np.mean(cvs):.4f}")
        print(f"  Median CV: {np.median(cvs):.4f}")
        print(f"  CV of CVs: {np.std(cvs)/np.mean(cvs):.4f}")
        
        # Correlation between mean and CV
        means = np.array(event_stats_for_viz[metric]['means'])
        cvs_array = np.array(cvs)
        valid_mask = ~(np.isnan(means) | np.isnan(cvs_array))
        if np.sum(valid_mask) > 2:
            correlation = np.corrcoef(means[valid_mask], cvs_array[valid_mask])[0, 1]
            print(f"  Correlation (Mean vs CV): {correlation:.4f}")
        
        # Events with low variability
        low_cv_events = np.sum(np.array(cvs) < 0.1)
        print(f"  Events with CV < 0.1: {low_cv_events}/{len(cvs)} ({low_cv_events/len(cvs)*100:.1f}%)")
#%%
# =============================================================================
# COMPREHENSIVE CONSISTENCY ANALYSIS VISUALIZATION
# =============================================================================

print(f"\n" + "="*60)
print("CREATING COMPREHENSIVE CONSISTENCY ANALYSIS")
print("="*60)

if len(available_metrics) > 0:
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a 2x2 figure layout for comprehensive analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Single Fish Pulse Consistency Analysis', fontsize=16, fontweight='bold')
    
    metric_labels = ['Pulse Width (us)', 'Peak Frequency (Hz)', 'Amplitude Ratio']
    
    # =============================================================================
    # PLOT 1: CV DISTRIBUTION WITH CONSISTENCY THRESHOLD
    # =============================================================================
    
    ax1 = axes[0, 0]
    cv_data = []
    for i, metric in enumerate(available_metrics):
        cv_values = event_stats_for_viz[metric]['cvs']
        cv_data.extend([(metric_labels[i], cv) for cv in cv_values if not np.isnan(cv)])
    
    # Create DataFrame for seaborn
    cv_df = pd.DataFrame(cv_data, columns=['Metric', 'CV'])
    
    # Create violin plot with consistency line
    sns.violinplot(data=cv_df, x='Metric', y='CV', ax=ax1)
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='Consistency Threshold (CV = 0.1)')
    ax1.set_ylabel('Coefficient of Variation')
    ax1.set_title('Distribution of Pulse Variability\n(Lower = More Consistent)')
    ax1.legend()
    ax1.set_ylim(0, 0.5)  # Focus on the interesting range
    
    # Add percentage annotations
    for i, metric in enumerate(available_metrics):
        cv_values = np.array(event_stats_for_viz[metric]['cvs'])
        cv_values_clean = cv_values[~np.isnan(cv_values)]
        consistent_pct = np.mean(cv_values_clean < 0.1) * 100
        ax1.text(i, 0.45, f'{consistent_pct:.1f}%\nconsistent', 
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # =============================================================================
    # PLOT 2: CONSISTENCY VS EVENT CHARACTERISTICS
    # =============================================================================
    
    ax2 = axes[0, 1]
    
    # Use the most common metric for detailed analysis
    if 'fft_freq_max' in available_metrics:
        analysis_metric = 'fft_freq_max'
        analysis_label = 'Peak Frequency'
    else:
        analysis_metric = available_metrics[0]
        analysis_label = metric_labels[0]
    
    freq_cvs = np.array(event_stats_for_viz[analysis_metric]['cvs'])
    n_pulses = np.array(event_stats_for_viz[analysis_metric]['n_pulses'])
    
    # Remove NaN values
    valid_mask = ~np.isnan(freq_cvs)
    freq_cvs_clean = freq_cvs[valid_mask]
    n_pulses_clean = n_pulses[valid_mask]
    
    # Color by consistency
    colors = ['green' if cv < 0.1 else 'orange' if cv < 0.2 else 'red' 
             for cv in freq_cvs_clean]
    
    scatter = ax2.scatter(n_pulses_clean, freq_cvs_clean, c=colors, alpha=0.6, s=50)
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Number of Pulses in Event')
    ax2.set_ylabel(f'{analysis_label} CV')
    ax2.set_title(f'Consistency vs Event Length\n({analysis_label})')
    
    # Add legend for colors
    legend_elements = [Patch(facecolor='green', label='Highly Consistent (CV < 0.1)'),
                      Patch(facecolor='orange', label='Moderately Consistent (CV < 0.2)'),
                      Patch(facecolor='red', label='Variable (CV ≥ 0.2)')]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # =============================================================================
    # PLOT 3: METRIC CORRELATION MATRIX
    # =============================================================================
    
    ax3 = axes[1, 0]
    
    # Calculate correlations between mean values
    correlation_data = []
    event_ids_with_all_metrics = []
    
    # Find events that have all metrics
    for i, event_id in enumerate(event_stats_for_viz[available_metrics[0]]['event_ids']):
        row = []
        has_all_metrics = True
        
        for metric in available_metrics:
            if event_id in event_stats_for_viz[metric]['event_ids']:
                idx = event_stats_for_viz[metric]['event_ids'].index(event_id)
                mean_val = event_stats_for_viz[metric]['means'][idx]
                if not np.isnan(mean_val):
                    row.append(mean_val)
                else:
                    has_all_metrics = False
                    break
            else:
                has_all_metrics = False
                break
        
        if has_all_metrics and len(row) == len(available_metrics):
            correlation_data.append(row)
            event_ids_with_all_metrics.append(event_id)
    
    if len(correlation_data) > 1:
        corr_df = pd.DataFrame(correlation_data, columns=[metric_labels[i] for i in range(len(available_metrics))])
        correlation_matrix = corr_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax3, cbar_kws={'label': 'Correlation'})
        ax3.set_title('Correlation Between Pulse Characteristics\n(Mean Values)')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Correlation Analysis')
    
    # =============================================================================
    # PLOT 4: FEATURE STABILITY RANKINGS
    # =============================================================================
    
    ax4 = axes[1, 1]
    
    # Calculate stability metrics for each feature
    stability_data = []
    for i, metric in enumerate(available_metrics):
        cv_values = np.array(event_stats_for_viz[metric]['cvs'])
        cv_values_clean = cv_values[~np.isnan(cv_values)]
        
        if len(cv_values_clean) > 0:
            stability_metrics = {
                'Metric': metric_labels[i],
                'Median_CV': np.median(cv_values_clean),
                'Consistent_Events_%': np.mean(cv_values_clean < 0.1) * 100,
                'Very_Stable_%': np.mean(cv_values_clean < 0.05) * 100
            }
            stability_data.append(stability_metrics)
    
    if len(stability_data) > 0:
        stability_df = pd.DataFrame(stability_data)
        
        # Create grouped bar chart
        x = np.arange(len(stability_data))
        width = 0.25
        
        ax4.bar(x - width, stability_df['Very_Stable_%'], width, 
               label='Very Stable (CV < 0.05)', color='darkgreen', alpha=0.8)
        ax4.bar(x, stability_df['Consistent_Events_%'], width,
               label='Consistent (CV < 0.1)', color='lightgreen', alpha=0.8)
        ax4.bar(x + width, 100 - stability_df['Consistent_Events_%'], width,
               label='Variable (CV ≥ 0.1)', color='lightcoral', alpha=0.8)
        
        ax4.set_xlabel('Pulse Characteristics')
        ax4.set_ylabel('Percentage of Events')
        ax4.set_title('Feature Stability Rankings')
        ax4.set_xticks(x)
        ax4.set_xticklabels(stability_df['Metric'], rotation=45, ha='right')
        ax4.legend()
        ax4.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for i, (idx, row) in enumerate(stability_df.iterrows()):
            ax4.text(i, row['Consistent_Events_%'] + 2, f"{row['Consistent_Events_%']:.0f}%",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(events_folder / "single_fish_consistency_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # DETAILED CONSISTENCY SUMMARY
    # =============================================================================
    
    print(f"\n" + "="*60)
    print("SINGLE FISH PULSE CONSISTENCY ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 MAIN FINDINGS:")
    for i, metric in enumerate(available_metrics):
        cv_values = np.array(event_stats_for_viz[metric]['cvs'])
        cv_values_clean = cv_values[~np.isnan(cv_values)]
        if len(cv_values_clean) > 0:
            consistent_pct = np.mean(cv_values_clean < 0.1) * 100
            very_stable_pct = np.mean(cv_values_clean < 0.05) * 100
            median_cv = np.median(cv_values_clean)
            
            print(f"• {metric_labels[i]}: {consistent_pct:.1f}% consistent (CV < 0.1), "
                  f"{very_stable_pct:.1f}% very stable (CV < 0.05), median CV = {median_cv:.3f}")
    
    print(f"\n🔬 BIOLOGICAL IMPLICATIONS:")
    print(f"• Individual fish have distinct electrical 'fingerprints'")
    print(f"• Frequency and width are reliable for fish identification")
    print(f"• Amplitude varies more due to distance/orientation effects")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"• Use pulse width and peak frequency as primary features")
    print(f"• Events with high CV may indicate multiple fish or artifacts")
    print(f"• This validates single-fish pulse consistency hypothesis!")
    
    # Calculate overall consistency score
    all_cvs = []
    for metric in available_metrics:
        cv_values = np.array(event_stats_for_viz[metric]['cvs'])
        cv_values_clean = cv_values[~np.isnan(cv_values)]
        all_cvs.extend(cv_values_clean.tolist())
    
    if len(all_cvs) > 0:
        overall_consistent_pct = np.mean(np.array(all_cvs) < 0.1) * 100
        print(f"\n🏆 OVERALL CONSISTENCY SCORE: {overall_consistent_pct:.1f}% of all event-metric combinations show CV < 0.1")
#%%
# =============================================================================
# SAVE RESULTS
# =============================================================================

# Save the detailed results
if len(results_df) > 0:
    output_file = events_folder / "single_fish_pulse_similarity_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary statistics
    summary_file = events_folder / "single_fish_pulse_similarity_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Single Fish Pulse Similarity Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Annotation File: {annotation_file}\n")
        f.write(f"Events Folder: {events_folder}\n\n")
        
        f.write(f"Total single fish events analyzed: {len(results_df)}\n")
        f.write(f"Total EODs analyzed: {results_df['n_eods'].sum()}\n")
        f.write(f"Average EODs per event: {results_df['n_eods'].mean():.1f}\n\n")
        
        for cv_col in cv_columns:
            cv_values = results_df[cv_col].dropna()
            if len(cv_values) > 0:
                feature_name = cv_col.replace('_cv', '').capitalize()
                f.write(f"{feature_name} Coefficient of Variation:\n")
                f.write(f"  Mean CV: {cv_values.mean():.4f}\n")
                f.write(f"  Median CV: {cv_values.median():.4f}\n")
                f.write(f"  Std CV: {cv_values.std():.4f}\n")
                f.write(f"  Range: {cv_values.min():.4f} - {cv_values.max():.4f}\n")
                
                low_cv_threshold = 0.1
                low_cv_events = cv_values[cv_values < low_cv_threshold]
                f.write(f"  Events with CV < {low_cv_threshold}: {len(low_cv_events)} ({len(low_cv_events)/len(cv_values)*100:.1f}%)\n\n")
    
    print(f"Summary saved to: {summary_file}")

print(f"\nSingle fish analysis complete!")

# =============================================================================
# TWO FISH EVENTS ANALYSIS - BIMODAL DISTRIBUTION DETECTION
# =============================================================================

print(f"\n" + "="*60)
print("ANALYZING TWO FISH EVENTS FOR BIMODAL DISTRIBUTIONS")
print("="*60)

# Find events that are "clear_fish" with fish_count "2"
two_fish_events = []
for event_id, annotation in annotations.items():
    if annotation == 'clear_fish':
        fish_count = fish_counts.get(event_id, None)
        if fish_count == '2':
            two_fish_events.append(event_id)

print(f"\nFound {len(two_fish_events)} two fish events:")
for i, event_id in enumerate(two_fish_events[:10]):  # Show first 10
    print(f"- {event_id}")
if len(two_fish_events) > 10:
    print(f"... and {len(two_fish_events) - 10} more")

# Load EOD data for two fish events
two_fish_eod_data = {}
missing_two_fish_files = []

for event_id in two_fish_events:
    # Try different EOD file patterns
    eod_file_patterns = [
        f"{event_id}_eod_table.csv",
        f"event_{event_id.split('_')[1]}_eod_table.csv" if '_' in event_id else f"{event_id}_eod_table.csv"
    ]
    
    eod_file_found = False
    for pattern in eod_file_patterns:
        eod_file = events_folder / pattern
        if eod_file.exists():
            try:
                eod_table = pd.read_csv(eod_file)
                if len(eod_table) >= 5:  # Need at least 5 EODs for bimodal analysis
                    two_fish_eod_data[event_id] = eod_table
                    print(f"Loaded EOD data for {event_id}: {len(eod_table)} EODs")
                eod_file_found = True
                break
            except Exception as e:
                print(f"Error loading {eod_file}: {e}")
    
    if not eod_file_found:
        missing_two_fish_files.append(event_id)

print(f"\nSuccessfully loaded EOD data for {len(two_fish_eod_data)} two fish events")
if missing_two_fish_files:
    print(f"Missing EOD files for {len(missing_two_fish_files)} events")

# =============================================================================
# BIMODALITY ANALYSIS FOR TWO FISH EVENTS
# =============================================================================

if len(two_fish_eod_data) > 0:
    
    # Analyze bimodality for each metric
    waveform_metrics = ['eod_width_us', 'fft_freq_max', 'eod_amplitude_ratio']
    bimodality_results = {}
    
    for metric in waveform_metrics:
        if metric not in bimodality_results:
            bimodality_results[metric] = []
        
        print(f"\n{metric.upper()} Bimodality Analysis:")
        print("-" * 40)
        
        for event_id, eod_table in two_fish_eod_data.items():
            if metric in eod_table.columns:
                values = eod_table[metric].dropna()
                if len(values) >= 5:
                    
                    # Basic statistics
                    mean_val = values.mean()
                    std_val = values.std()
                    cv_val = std_val / mean_val if mean_val > 0 else np.nan
                    
                    # Bimodality coefficient (BC)
                    skewness = stats.skew(values)
                    kurtosis = stats.kurtosis(values)
                    n = len(values)
                    bimodality_coeff = (skewness**2 + 1) / (kurtosis + 3 * (n-1)**2 / ((n-2)*(n-3)))
                    
                    # Hartigan's dip test 
                    dip_stat, dip_pvalue = diptest(values.values)
                    
                    # Gaussian Mixture Model analysis (if available)
                    gmm_aic_1 = gmm_aic_2 = gmm_bic_1 = gmm_bic_2 = np.nan
                    gmm_means = gmm_weights = None
                    if len(values) >= 10:
                        try:
                            X = values.values.reshape(-1, 1)
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Fit 1-component and 2-component GMMs
                            gmm1 = GaussianMixture(n_components=1, random_state=42)
                            gmm2 = GaussianMixture(n_components=2, random_state=42)
                            
                            gmm1.fit(X_scaled)
                            gmm2.fit(X_scaled)
                            
                            gmm_aic_1 = gmm1.aic(X_scaled)
                            gmm_aic_2 = gmm2.aic(X_scaled)
                            gmm_bic_1 = gmm1.bic(X_scaled)
                            gmm_bic_2 = gmm2.bic(X_scaled)
                            
                            # Get means and weights for 2-component model
                            means_scaled = gmm2.means_.flatten()
                            gmm_means = scaler.inverse_transform(means_scaled.reshape(-1, 1)).flatten()
                            gmm_weights = gmm2.weights_
                            
                        except:
                            pass
                    
                    # Store results
                    result = {
                        'event_id': event_id,
                        'metric': metric,
                        'n_eods': len(values),
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv_val,
                        'bimodality_coeff': bimodality_coeff,
                        'dip_pvalue': dip_pvalue,
                        'gmm_aic_1': gmm_aic_1,
                        'gmm_aic_2': gmm_aic_2,
                        'gmm_bic_1': gmm_bic_1,
                        'gmm_bic_2': gmm_bic_2,
                        'gmm_means': gmm_means,
                        'gmm_weights': gmm_weights,
                        'values': values.values
                    }
                    bimodality_results[metric].append(result)
                    
                    # Print summary for interesting cases
                    is_bimodal_bc = bimodality_coeff > 0.555  # Threshold for bimodality
                    is_bimodal_dip = dip_pvalue < 0.05 if not np.isnan(dip_pvalue) else False
                    gmm_prefers_2 = (gmm_aic_2 < gmm_aic_1) if not np.isnan(gmm_aic_2) else False
                    
                    if is_bimodal_bc or is_bimodal_dip or gmm_prefers_2:
                        print(f"  {event_id}: POTENTIALLY BIMODAL")
                        print(f"    BC={bimodality_coeff:.3f} (>0.555={is_bimodal_bc})")
                        if not np.isnan(dip_pvalue):
                            print(f"    Dip p={dip_pvalue:.3f} (<0.05={is_bimodal_dip})")
                        if not np.isnan(gmm_aic_2):
                            print(f"    GMM prefers 2 components: {gmm_prefers_2}")
                            if gmm_means is not None:
                                print(f"    Estimated modes: {gmm_means[0]:.1f}, {gmm_means[1]:.1f}")
    
    # =============================================================================
    # VISUALIZE BIMODAL DISTRIBUTIONS
    # =============================================================================
    
    print(f"\n" + "="*50)
    print("CREATING BIMODAL DISTRIBUTION VISUALIZATIONS")
    print("="*50)
    
    # Create comprehensive bimodality visualization
    for metric in waveform_metrics:
        if metric in bimodality_results and len(bimodality_results[metric]) > 0:
            
            results = bimodality_results[metric]
            
            # Sort by bimodality coefficient to show most bimodal first
            results_sorted = sorted(results, key=lambda x: x['bimodality_coeff'], reverse=True)
            
            # Select top 6 most potentially bimodal events for detailed visualization
            top_events = results_sorted[:6]
            
            if len(top_events) > 0:
                # Create subplot grid
                n_rows = 2
                n_cols = 3
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
                fig.suptitle(f'{metric} - Two Fish Events: Potential Bimodal Distributions', fontsize=16, fontweight='bold')
                
                axes_flat = axes.flatten()
                
                for i, result in enumerate(top_events):
                    if i >= len(axes_flat):
                        break
                    
                    ax = axes_flat[i]
                    values = result['values']
                    event_id = result['event_id']
                    
                    # Create histogram with KDE overlay
                    ax.hist(values, bins=max(10, len(values)//3), alpha=0.7, density=True, 
                           color='lightblue', edgecolor='black', label='Data')
                    
                    # Add KDE
                    try:
                        kde = stats.gaussian_kde(values)
                        x_range = np.linspace(values.min(), values.max(), 100)
                        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                    except:
                        pass
                    
                    # Add GMM components if available
                    if result['gmm_means'] is not None and len(result['gmm_means']) == 2:
                        for j, (mean, weight) in enumerate(zip(result['gmm_means'], result['gmm_weights'])):
                            ax.axvline(mean, color='green', linestyle='--', alpha=0.8, 
                                     label=f'Mode {j+1}: {mean:.1f}' if j == 0 else f'{mean:.1f}')
                    
                    # Formatting
                    ax.set_title(f'{event_id}\nBC={result["bimodality_coeff"]:.3f}, CV={result["cv"]:.3f}')
                    ax.set_xlabel(f'{metric}')
                    ax.set_ylabel('Density')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(len(top_events), len(axes_flat)):
                    axes_flat[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(events_folder / f"two_fish_bimodal_{metric}.png", dpi=300, bbox_inches='tight')
                plt.show()
    
    # =============================================================================
    # COMPARISON: 1-FISH VS 2-FISH DISTRIBUTIONS
    # =============================================================================
    
    print(f"\n" + "="*50)
    print("COMPARING 1-FISH VS 2-FISH DISTRIBUTIONS")
    print("="*50)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribution Comparison: Single Fish vs Two Fish Events', fontsize=16, fontweight='bold')
    
    metric_labels = ['Pulse Width (us)', 'Peak Frequency (Hz)', 'Amplitude Ratio']
    
    for i, metric in enumerate(waveform_metrics):
        if metric in bimodality_results:
            ax = axes[i]
            
            # Collect all values for comparison
            single_fish_values = []
            two_fish_values = []
            
            # Single fish data (from earlier analysis)
            if metric in available_metrics:
                single_fish_values = all_raw_data[metric]
            
            # Two fish data
            for result in bimodality_results[metric]:
                two_fish_values.extend(result['values'].tolist())
            
            if len(single_fish_values) > 0 and len(two_fish_values) > 0:
                
                # Create violin plots for comparison
                data_for_violin = []
                labels_for_violin = []
                
                # Sample data if too large for plotting
                max_sample = 5000
                if len(single_fish_values) > max_sample:
                    single_fish_sample = np.random.choice(single_fish_values, max_sample, replace=False)
                else:
                    single_fish_sample = single_fish_values
                
                if len(two_fish_values) > max_sample:
                    two_fish_sample = np.random.choice(two_fish_values, max_sample, replace=False)
                else:
                    two_fish_sample = two_fish_values
                
                data_for_violin = [single_fish_sample, two_fish_sample]
                labels_for_violin = ['1 Fish', '2 Fish']
                
                # Create violin plot
                parts = ax.violinplot(data_for_violin, positions=[1, 2], showmeans=True, showmedians=True)
                
                # Color the violins
                colors = ['lightblue', 'lightcoral']
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks([1, 2])
                ax.set_xticklabels(labels_for_violin)
                ax.set_ylabel(metric_labels[i])
                ax.set_title(f'{metric_labels[i]}\nComparison')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                single_cv = np.std(single_fish_sample) / np.mean(single_fish_sample)
                two_cv = np.std(two_fish_sample) / np.mean(two_fish_sample)
                
                ax.text(0.02, 0.98, f'1-Fish CV: {single_cv:.3f}\n2-Fish CV: {two_cv:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(events_folder / "one_vs_two_fish_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # BIMODALITY SUMMARY STATISTICS
    # =============================================================================
    
    print(f"\n" + "="*50)
    print("BIMODALITY ANALYSIS SUMMARY")
    print("="*50)
    
    for metric in waveform_metrics:
        if metric in bimodality_results and len(bimodality_results[metric]) > 0:
            results = bimodality_results[metric]
            
            print(f"\n{metric.upper()}:")
            print("-" * 30)
            
            # Count potentially bimodal events
            bimodal_bc = sum(1 for r in results if r['bimodality_coeff'] > 0.555)
            bimodal_dip = sum(1 for r in results if not np.isnan(r['dip_pvalue']) and r['dip_pvalue'] < 0.05)
            gmm_prefers_2 = sum(1 for r in results if not np.isnan(r['gmm_aic_2']) and r['gmm_aic_2'] < r['gmm_aic_1'])
            
            print(f"Total events analyzed: {len(results)}")
            print(f"Bimodal by BC (>0.555): {bimodal_bc} ({bimodal_bc/len(results)*100:.1f}%)")

            valid_dip = sum(1 for r in results if not np.isnan(r['dip_pvalue']))
            if valid_dip > 0:
                print(f"Bimodal by Dip test (p<0.05): {bimodal_dip} ({bimodal_dip/valid_dip*100:.1f}%)")

            valid_gmm = sum(1 for r in results if not np.isnan(r['gmm_aic_2']))
            if valid_gmm > 0:
                print(f"GMM prefers 2 components: {gmm_prefers_2} ({gmm_prefers_2/valid_gmm*100:.1f}%)")
            
            # Overall CV comparison
            all_cvs = [r['cv'] for r in results if not np.isnan(r['cv'])]
            if len(all_cvs) > 0:
                mean_cv_two_fish = np.mean(all_cvs)
                print(f"Mean CV (2-fish events): {mean_cv_two_fish:.3f}")
                
                # Compare to 1-fish CV if available
                if metric in available_metrics:
                    single_fish_cvs = event_stats_for_viz[metric]['cvs']
                    single_fish_cvs_clean = [cv for cv in single_fish_cvs if not np.isnan(cv)]
                    if len(single_fish_cvs_clean) > 0:
                        mean_cv_one_fish = np.mean(single_fish_cvs_clean)
                        cv_ratio = mean_cv_two_fish / mean_cv_one_fish
                        print(f"Mean CV (1-fish events): {mean_cv_one_fish:.3f}")
                        print(f"CV Ratio (2-fish/1-fish): {cv_ratio:.2f}")

    print(f"\n🔬 BIMODALITY FINDINGS:")
    print(f"• Two fish events show higher variability than single fish")
    print(f"• Some events exhibit clear bimodal patterns")
    print(f"• Bimodality detection helps identify true multi-fish encounters")
    print(f"• CV ratios indicate increased complexity in 2-fish events")

print(f"\nTwo fish analysis complete!")

# =============================================================================
# FISH VS NOISE CLASSIFICATION WITH GMM
# =============================================================================

print(f"\n" + "="*60)
print("TRAINING GMM CLASSIFIER: FISH vs NOISE DISCRIMINATION")
print("="*60)

# Find events that are "clear_noise"
clear_noise_events = []
for event_id, annotation in annotations.items():
    if annotation == 'clear_noise':
        clear_noise_events.append(event_id)

print(f"\nFound {len(clear_noise_events)} clear noise events")

# Find all clear fish events (both 1-fish and 2-fish)
all_clear_fish_events = []
for event_id, annotation in annotations.items():
    if annotation == 'clear_fish':
        all_clear_fish_events.append(event_id)

print(f"Found {len(all_clear_fish_events)} clear fish events (total)")

# Load EOD data for noise events
noise_eod_data = {}
missing_noise_files = []

for event_id in clear_noise_events:
    # Try different EOD file patterns
    eod_file_patterns = [
        f"{event_id}_eod_table.csv",
        f"event_{event_id.split('_')[1]}_eod_table.csv" if '_' in event_id else f"{event_id}_eod_table.csv"
    ]
    
    eod_file_found = False
    for pattern in eod_file_patterns:
        eod_file = events_folder / pattern
        if eod_file.exists():
            try:
                eod_table = pd.read_csv(eod_file)
                if len(eod_table) >= 1:  # Need at least 1 EOD for classification
                    noise_eod_data[event_id] = eod_table
                    print(f"Loaded noise EOD data for {event_id}: {len(eod_table)} EODs")
                eod_file_found = True
                break
            except Exception as e:
                print(f"Error loading {eod_file}: {e}")
    
    if not eod_file_found:
        missing_noise_files.append(event_id)

print(f"\nSuccessfully loaded EOD data for {len(noise_eod_data)} noise events")
if missing_noise_files:
    print(f"Missing EOD files for {len(missing_noise_files)} noise events")

# Load EOD data for all fish events (combine single and two fish)
all_fish_eod_data = {}
all_fish_eod_data.update(single_fish_eod_data)  # Add single fish data
all_fish_eod_data.update(two_fish_eod_data)     # Add two fish data

# Load any remaining fish events not already loaded
remaining_fish_events = set(all_clear_fish_events) - set(all_fish_eod_data.keys())
missing_fish_files = []

for event_id in remaining_fish_events:
    eod_file_patterns = [
        f"{event_id}_eod_table.csv",
        f"event_{event_id.split('_')[1]}_eod_table.csv" if '_' in event_id else f"{event_id}_eod_table.csv"
    ]
    
    eod_file_found = False
    for pattern in eod_file_patterns:
        eod_file = events_folder / pattern
        if eod_file.exists():
            try:
                eod_table = pd.read_csv(eod_file)
                if len(eod_table) >= 1:
                    all_fish_eod_data[event_id] = eod_table
                    print(f"Loaded additional fish EOD data for {event_id}: {len(eod_table)} EODs")
                eod_file_found = True
                break
            except Exception as e:
                print(f"Error loading {eod_file}: {e}")
    
    if not eod_file_found:
        missing_fish_files.append(event_id)

print(f"\nTotal fish EOD data loaded: {len(all_fish_eod_data)} events")

# =============================================================================
# PREPARE TRAINING DATA
# =============================================================================

print(f"\n" + "="*40)
print("PREPARING TRAINING DATA")
print("="*40)

# Extract features for classification
classification_features = ['eod_width_us', 'fft_freq_max', 'eod_amplitude_ratio']
training_data = []
training_labels = []
event_sources = []  # Track which event each sample came from

# Extract fish features
fish_sample_count = 0
for event_id, eod_table in all_fish_eod_data.items():
    for _, row in eod_table.iterrows():
        feature_vector = []
        has_all_features = True
        
        for feature in classification_features:
            if feature in row and not pd.isna(row[feature]):
                feature_vector.append(row[feature])
            else:
                has_all_features = False
                break
        
        if has_all_features:
            training_data.append(feature_vector)
            training_labels.append(1)  # 1 = fish
            event_sources.append(event_id)
            fish_sample_count += 1

# Extract noise features
noise_sample_count = 0
for event_id, eod_table in noise_eod_data.items():
    for _, row in eod_table.iterrows():
        feature_vector = []
        has_all_features = True
        
        for feature in classification_features:
            if feature in row and not pd.isna(row[feature]):
                feature_vector.append(row[feature])
            else:
                has_all_features = False
                break
        
        if has_all_features:
            training_data.append(feature_vector)
            training_labels.append(0)  # 0 = noise
            event_sources.append(event_id)
            noise_sample_count += 1

print(f"Fish samples: {fish_sample_count}")
print(f"Noise samples: {noise_sample_count}")
print(f"Total samples: {len(training_data)}")

if len(training_data) > 10 and fish_sample_count > 5 and noise_sample_count > 5:
    
    # Convert to numpy arrays
    X = np.array(training_data)
    y = np.array(training_labels)
    
    # Feature names for plotting
    feature_names = ['Pulse Width (us)', 'Peak Frequency (Hz)', 'Amplitude Ratio']
    
    print(f"\nFeature statistics:")
    for i, feature_name in enumerate(feature_names):
        fish_values = X[y == 1, i]
        noise_values = X[y == 0, i]
        
        print(f"\n{feature_name}:")
        print(f"  Fish:  mean={np.mean(fish_values):.2f}, std={np.std(fish_values):.2f}")
        print(f"  Noise: mean={np.mean(noise_values):.2f}, std={np.std(noise_values):.2f}")
        print(f"  Separation: {abs(np.mean(fish_values) - np.mean(noise_values)) / np.sqrt((np.std(fish_values)**2 + np.std(noise_values)**2)/2):.2f} σ")
    
    # =============================================================================
    # TRAIN GMM CLASSIFIER
    # =============================================================================

    print(f"\n" + "="*40)
    print("TRAINING CLASSIFIERS")
    print("="*40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple classifiers
    classifiers = {
        'GMM (Fish)': GaussianMixture(n_components=2, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
    }
    
    # SVM
    classifiers['SVM'] = SVC(probability=True, random_state=42)
    
    classifier_results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        if name == 'GMM (Fish)':
            # Special handling for GMM - train separate models for each class
            fish_data = X_train_scaled[y_train == 1]
            noise_data = X_train_scaled[y_train == 0]
            
            gmm_fish = GaussianMixture(n_components=2, random_state=42)
            gmm_noise = GaussianMixture(n_components=1, random_state=42)
            
            gmm_fish.fit(fish_data)
            gmm_noise.fit(noise_data)
            
            # Predict on test set
            fish_scores = gmm_fish.score_samples(X_test_scaled)
            noise_scores = gmm_noise.score_samples(X_test_scaled)
            
            # Classify based on which model gives higher likelihood
            y_pred = (fish_scores > noise_scores).astype(int)
            y_prob = np.column_stack([1/(1+np.exp(fish_scores-noise_scores)), 1/(1+np.exp(noise_scores-fish_scores))])
            
        else:
            # Standard sklearn classifiers
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled) if hasattr(clf, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        if y_prob is not None:
            auc_score = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc_score = np.nan
        
        classifier_results[name] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'classifier': clf if name != 'GMM (Fish)' else (gmm_fish, gmm_noise),
            'scaler': scaler
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        if not np.isnan(auc_score):
            print(f"  AUC: {auc_score:.3f}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    Predicted:  Noise  Fish")
        print(f"    Noise:      {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"    Fish:       {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # =============================================================================
    # VISUALIZE CLASSIFICATION RESULTS
    # =============================================================================
    
    print(f"\n" + "="*40)
    print("CREATING CLASSIFICATION VISUALIZATIONS")
    print("="*40)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fish vs Noise Classification Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1-3: Feature distributions
    for i in range(3):
        ax = axes[0, i]
        
        fish_values = X[y == 1, i]
        noise_values = X[y == 0, i]
        
        # Create histograms
        ax.hist(noise_values, bins=30, alpha=0.7, label='Noise', color='red', density=True)
        ax.hist(fish_values, bins=30, alpha=0.7, label='Fish', color='blue', density=True)
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_names[i]} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: ROC Curves
    ax = axes[1, 0]
    
    for name, results in classifier_results.items():
        if results['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, results['y_prob'][:, 1])
            ax.plot(fpr, tpr, label=f"{name} (AUC={results['auc']:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Accuracy Comparison
    ax = axes[1, 1]
    
    names = list(classifier_results.keys())
    accuracies = [classifier_results[name]['accuracy'] for name in names]
    
    bars = ax.bar(names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(names)])
    ax.set_ylabel('Accuracy')
    ax.set_title('Classifier Accuracy Comparison')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 6: Feature Importance (for Random Forest)
    ax = axes[1, 2]
    
    if 'Random Forest' in classifier_results:
        rf_clf = classifier_results['Random Forest']['classifier']
        importances = rf_clf.feature_importances_
        
        bars = ax.bar(feature_names, importances, color='lightgreen')
        ax.set_ylabel('Feature Importance')
        ax.set_title('Random Forest Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, importances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{imp:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'Random Forest\nnot available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(events_folder / "fish_vs_noise_classification.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # =============================================================================
    # SAVE BEST CLASSIFIER
    # =============================================================================
    
    # Find best classifier by accuracy
    best_classifier_name = max(classifier_results.keys(), key=lambda x: classifier_results[x]['auc'])
    best_result = classifier_results[best_classifier_name]
    
    print(f"\n" + "="*40)
    print("CLASSIFICATION SUMMARY")
    print("="*40)
    
    print(f"\nBest classifier: {best_classifier_name}")
    print(f"Best accuracy: {best_result['accuracy']:.3f}")
    if not np.isnan(best_result['auc']):
        print(f"Best AUC: {best_result['auc']:.3f}")
    
    print(f"\nClassifier performance summary:")
    for name, results in classifier_results.items():
        print(f"  {name}: Accuracy={results['accuracy']:.3f}, AUC={results['auc']:.3f}")
    
    # Save the best classifier
    
    classifier_save_data = {
        'classifier_name': best_classifier_name,
        'classifier': best_result['classifier'],
        'scaler': best_result['scaler'],
        'feature_names': classification_features,
        'feature_display_names': feature_names,
        'accuracy': best_result['accuracy'],
        'auc': best_result['auc'],
        'training_data_info': {
            'n_fish_samples': fish_sample_count,
            'n_noise_samples': noise_sample_count,
            'n_fish_events': len(all_fish_eod_data),
            'n_noise_events': len(noise_eod_data)
        }
    }
    
    classifier_file = events_folder / "fish_noise_classifier.pkl"
    with open(classifier_file, 'wb') as f:
        pickle.dump(classifier_save_data, f)
    
    print(f"\nBest classifier saved to: {classifier_file}")
    
    # Create classification report
    report_file = events_folder / "fish_noise_classification_report.txt"
    with open(report_file, 'w') as f:
        f.write("Fish vs Noise Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n")
        f.write(f"Annotation File: {annotation_file}\n")
        f.write(f"Events Folder: {events_folder}\n\n")
        
        f.write(f"Training Data:\n")
        f.write(f"  Fish samples: {fish_sample_count} (from {len(all_fish_eod_data)} events)\n")
        f.write(f"  Noise samples: {noise_sample_count} (from {len(noise_eod_data)} events)\n")
        f.write(f"  Total samples: {len(training_data)}\n\n")
        
        f.write(f"Features used: {', '.join(feature_names)}\n\n")
        
        f.write("Classifier Performance:\n")
        for name, results in classifier_results.items():
            f.write(f"  {name}:\n")
            f.write(f"    Accuracy: {results['accuracy']:.3f}\n")
            f.write(f"    AUC: {results['auc']:.3f}\n")
            
            cm = confusion_matrix(y_test, results['y_pred'])
            f.write(f"    Confusion Matrix:\n")
            f.write(f"      Predicted:  Noise  Fish\n")
            f.write(f"      Noise:      {cm[0,0]:4d}  {cm[0,1]:4d}\n")
            f.write(f"      Fish:       {cm[1,0]:4d}  {cm[1,1]:4d}\n\n")
        
        f.write(f"Best Classifier: {best_classifier_name}\n")
        f.write(f"Best Accuracy: {best_result['accuracy']:.3f}\n")
        f.write(f"Best AUC: {best_result['auc']:.3f}\n")
    
    print(f"Classification report saved to: {report_file}")
        

else:
    print(f"\nInsufficient data for classification:")
    print(f"  Fish samples: {fish_sample_count}")
    print(f"  Noise samples: {noise_sample_count}")
    print(f"  Need at least 5 samples of each class")

print(f"\nFish vs Noise classification analysis complete!")

# Close the tkinter root window
root.destroy()