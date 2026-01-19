"""
# 02_Pulse_extraction_control.py
# Control Recording EOD Pulse Extraction (Script 02)

This script extracts EOD pulses from control/catalogue recordings

"""


import thunderfish.pulses as pulses
import matplotlib.pyplot as plt
import audioio as aio
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import gc
import glob
import datetime as dt
import os
import json

# Import consolidated EOD functions
from eod_functions_backup import (
    save_variable_length_waveforms,
    extract_pulse_snippets,
    # extract_pulse_snippets_control,
    filter_waveforms
)

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Individual Fish Folders")
output_path = filedialog.askdirectory(title="Select Folder to Store Analysis Results")

# Find individual fish folders
individual_folders = [d for d in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(d)]

if not individual_folders:
    print("No individual folders found!")
    exit()

print(f"Found {len(individual_folders)} individual fish folders")

# Extract species information from folder names
individual_info = []
for folder in individual_folders:
    folder_name = os.path.basename(folder)
    species_code = folder_name[:2].upper()
    individual_id = folder_name
    individual_info.append({
        'folder_path': folder,
        'individual_id': individual_id,
        'species_code': species_code
    })

# Group by species for summary
species_groups = {}
for info in individual_info:
    species = info['species_code']
    if species not in species_groups:
        species_groups[species] = []
    species_groups[species].append(info)

print(f"Species found:")
for species, individuals in species_groups.items():
    individual_names = [ind['individual_id'] for ind in individuals]
    print(f"  {species}: {individual_names}")

# Load first individual's first file for parameter setup
first_folder = individual_info[0]['folder_path']
wav_files = glob.glob(os.path.join(first_folder, "*.wav"))
if not wav_files:
    print("No WAV files found in first folder!")
    exit()

data, rate = aio.load_audio(wav_files[0])
data = data[:,0]  # Use only the first channel

# Option to import parameters from diagnostic tool
import_params = messagebox.askyesno("Import Parameters", 
                                   "Do you want to import parameters from a diagnostic tool JSON file?")

if import_params:
    param_file = filedialog.askopenfilename(
        title="Select Parameter File", 
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    try:
        with open(param_file, 'r') as f:
            imported_params = json.load(f)
        print(f"Imported parameters from: {param_file}")
        
        # Use imported parameters
        parameters = imported_params
        
    except Exception as e:
        messagebox.showerror("Import Error", f"Failed to import parameters:\n{str(e)}")
        print(f"Parameter import failed: {e}")
        # Fall back to default parameters
        parameters = {'thresh':0.004,  # Reduced threshold for pulse detection to catch low-amplitude fish
                    'min_rel_slope_diff':0.25,
                    'min_width_us':30,  # Minimum pulse width in microseconds
                    'max_width_us':1000,  # Maximum pulse width in microseconds
                    'width_fac_detection':7.0,
                    'interp_factor':3,  # Interpolation factor for waveform extraction
                    'amplitude_ratio_min':0.2,  # Minimum peak-to-peak amplitude ratio
                    'amplitude_ratio_max':4,     # Maximum peak-to-peak amplitude ratio
                    'save_filtered_out':True, # Option to save filtered-out pulses for quality control
                    'peak_fft_freq_min':50,
                    'peak_fft_freq_max':10000,
                    'clip_threshold':1.6,  # Threshold for clipping detection (1.6 is conservative, adjust as needed)
                    'top_amplitude_percent':50  # Only keep top x% highest amplitude EODs
                    }
else:
    # Default control recording parameters
    parameters = {'thresh':0.004,  # Reduced threshold for pulse detection to catch low-amplitude fish
                'min_rel_slope_diff':0.25,
                'min_width_us':30,  # Minimum pulse width in microseconds
                'max_width_us':1000,  # Maximum pulse width in microseconds
                'width_fac_detection':7.0,
                'interp_factor':3,  # Interpolation factor for waveform extraction
                'amplitude_ratio_min':0.2,  # Minimum peak-to-peak amplitude ratio
                'amplitude_ratio_max':4,     # Maximum peak-to-peak amplitude ratio
                'save_filtered_out':True, # Option to save filtered-out pulses for quality control
                'peak_fft_freq_min':50,
                'peak_fft_freq_max':10000,
                'clip_threshold':1.6,  # Threshold for clipping detection (1.6 is conservative, adjust as needed)
                'top_amplitude_percent':50  # Only keep top x% highest amplitude EODs
                }

# Plot raw data
plt.figure(figsize=(20, 6))
plt.plot(data)
plt.title('Raw Data from %s' % individual_info[0]['individual_id'])
plt.hlines(y=parameters['thresh'], xmin=0, xmax=len(data), colors='red', linestyles='dashed', label='Threshold')
plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
plt.show(block=False)


print(parameters)
print("change parameters? (1/0)")
change_params = int(input())
while change_params:
    print("input parameter name")
    ch_par = input()
    print("input parameter value")
    ch_par_value = float(input())
    parameters[ch_par] = ch_par_value
    print(parameters)
    print("done? (1/0)")
    done = int(input())
    if done:
        change_params = 0

# Convert to DataFrame for saving
parameters_df = pd.DataFrame({k: [v] for k, v in parameters.items()})
parameters_df.to_csv('%s\\analysis_parameters.csv' % output_path, index=False)
plt.close()


# Initialize collections for all data
all_eod_tables = []
all_waveform_data = []
all_clipped_tables = []
all_clipped_waveforms = []

# Process each individual folder
for n, individual in enumerate(individual_info):
    folder_path = individual['folder_path']
    individual_id = individual['individual_id']
    species_code = individual['species_code']
    
    print(f"Processing individual {n+1}/{len(individual_info)}: {individual_id}")

    # Memory management: force garbage collection before each individual
    gc.collect()
    
    # Find all WAV files in this individual's folder
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
    if not wav_files:
        print(f"  No WAV files found in {individual_id}, skipping...")
        continue
    
    print(f"  Found {len(wav_files)} WAV files")
    
    # Load and concatenate all WAV files for this individual
    concatenated_data = []
    file_boundaries = []  # Track where each file starts in concatenated data
    current_start = 0
    
    for wav_file in sorted(wav_files):
        file_data, file_rate = aio.load_audio(wav_file)
        file_data = file_data[:, 0]  # Use only the first channel
        
        if len(concatenated_data) == 0:
            rate = file_rate  # Use rate from first file
        elif file_rate != rate:
            print(f"  Warning: Sample rate mismatch in {os.path.basename(wav_file)} ({file_rate} vs {rate})")
            # Resample if needed (basic approach)
            from scipy import signal as scipy_signal
            file_data = scipy_signal.resample(file_data, int(len(file_data) * rate / file_rate))
        
        file_boundaries.append({
            'filename': os.path.basename(wav_file),
            'start_sample': current_start,
            'end_sample': current_start + len(file_data),
            'duration_s': len(file_data) / rate
        })
        
        concatenated_data.extend(file_data)
        current_start += len(file_data)
    
    data = np.array(concatenated_data)
    print(f"  Total concatenated duration: {len(data)/rate:.1f}s from {len(wav_files)} files")
    
    # Find peaks
    pulse_widths = []
    
    # Detect pulses
    peaks, troughs, _ , pulse_widths = \
        pulses.detect_pulses(data, rate, 
                                thresh = parameters['thresh'], 
                                min_rel_slope_diff=parameters['min_rel_slope_diff'],
                                min_width=parameters['min_width_us']/1e6,  # in seconds
                                max_width=parameters['max_width_us']/1e6,  # in seconds
                                width_fac=parameters['width_fac_detection'],
                                verbose=2,
                                return_data=False)
    
    midpoints = (peaks + troughs) // 2  # Midpoint indices of pulses

        
    # Extract variable-width snippets and analyze them
    if len(midpoints) > 0:
        print(f"  Analyzing {len(midpoints)} unique events...")

        # Analyze snippets with variable widths

        eod_waveforms, eod_amps, eod_widths, ch_amps, ch_cor_coeffs, eod_chan, is_differential,\
            snippet_peak_idc, snippet_trough_idc, snippet_midpoint_idc,\
            final_peak_idc, final_trough_idc, final_midpoint_idc,\
            pulse_orientation, amplitude_ratios, waveform_lengths, fft_peak_freqs = \
            extract_pulse_snippets(
            data, midpoints, peaks, troughs, thresh = parameters_df['thresh'][0],
            width_factor = parameters_df['width_fac_detection'][0], interp_factor = parameters_df['interp_factor'][0], rate = rate,
            source = '1ch_diff', return_differential = True
            )
        # eod_waveforms, eod_amps, eod_widths, snippet_peak_idc, snippet_trough_idc, snippet_midpoint_idc, \
        #     final_peak_idc, final_trough_idc, final_midpoint_idc, pulse_orientation, amplitude_ratios, waveform_lengths, fft_peak_freqs = \
        #     extract_pulse_snippets_control(data, parameters_df, rate=rate, midpoints=midpoints, peaks=peaks, troughs=troughs, widths=pulse_widths,
        #                             center_on_zero_crossing=False)  # Skip centering for storage efficiency
        


        # DUPLICATE DETECTION: Remove events with identical peak/trough indices
        # This handles multi-phasic EODs where low threshold detects both p1-t1 and p2-t2,
        # but _process_waveform_common re-detects the same dominant p2-t2 in both snippets
        print(f"  Before duplicate removal: {len(eod_waveforms)} events")
        
        # Create unique identifier for each event based on peak and trough indices
        # Use tuple of (peak_idx, trough_idx) as the key
        unique_events = {}
        duplicate_indices = []
        
        for i in range(len(final_peak_idc)):
            event_key = (final_peak_idc[i], final_trough_idc[i])
            
            if event_key in unique_events:
                # Found duplicate - keep the one with better characteristics
                existing_idx = unique_events[event_key]
                
                # Comparison criteria (in order of importance):
                # 1. Smaller waveform length (more precise extraction)
                # 2. Higher amplitude (better signal quality)
                # 3. Better amplitude ratio (closer to expected range)
                
                current_length = waveform_lengths[i]
                existing_length = waveform_lengths[existing_idx]
                current_amp = eod_amps[i]
                existing_amp = eod_amps[existing_idx]
                current_ratio = amplitude_ratios[i]
                existing_ratio = amplitude_ratios[existing_idx]
                
                # Decide which one to keep
                keep_current = False
                
                if current_length != existing_length:
                    # Prefer smaller length (more precise extraction)
                    keep_current = current_length < existing_length
                elif current_amp != existing_amp:
                    # If lengths are equal, prefer higher amplitude
                    keep_current = current_amp > existing_amp
                else:
                    # If both length and amplitude are equal, prefer better amplitude ratio
                    # (closer to the middle of acceptable range)
                    ratio_min = parameters['amplitude_ratio_min']
                    ratio_max = parameters['amplitude_ratio_max']
                    ratio_mid = (ratio_min + ratio_max) / 2
                    
                    current_ratio_distance = abs(current_ratio - ratio_mid)
                    existing_ratio_distance = abs(existing_ratio - ratio_mid)
                    keep_current = current_ratio_distance < existing_ratio_distance
                
                if keep_current:
                    # Mark the existing one for removal and update the dictionary
                    duplicate_indices.append(existing_idx)
                    unique_events[event_key] = i
                    # print(f"    Duplicate at indices ({final_peak_idc[i]}, {final_trough_idc[i]}): "
                    #       f"keeping event {i} (len={current_length}) over {existing_idx} (len={existing_length})")
                else:
                    # Mark the current one for removal
                    duplicate_indices.append(i)
                    # print(f"    Duplicate at indices ({final_peak_idc[i]}, {final_trough_idc[i]}): "
                    #       f"keeping event {existing_idx} (len={existing_length}) over {i} (len={current_length})")
            else:
                # First occurrence of this peak/trough pair
                unique_events[event_key] = i
        
        # Remove duplicates from all arrays and lists
        if duplicate_indices:
            print(f"    Removing {len(duplicate_indices)} duplicate events")
            
            # Create mask for non-duplicate indices
            non_duplicate_mask = np.ones(len(eod_waveforms), dtype=bool)
            non_duplicate_mask[duplicate_indices] = False
            
            # Filter all arrays and lists
            eod_waveforms = [eod_waveforms[i] for i in range(len(eod_waveforms)) if non_duplicate_mask[i]]
            eod_amps = eod_amps[non_duplicate_mask]
            eod_widths = eod_widths[non_duplicate_mask]
            snippet_peak_idc = snippet_peak_idc[non_duplicate_mask]
            snippet_trough_idc = snippet_trough_idc[non_duplicate_mask]
            snippet_midpoint_idc = snippet_midpoint_idc[non_duplicate_mask]
            final_peak_idc = final_peak_idc[non_duplicate_mask]
            final_trough_idc = final_trough_idc[non_duplicate_mask]
            final_midpoint_idc = final_midpoint_idc[non_duplicate_mask]
            pulse_orientation = pulse_orientation[non_duplicate_mask]
            amplitude_ratios = amplitude_ratios[non_duplicate_mask]
            waveform_lengths = waveform_lengths[non_duplicate_mask]
            fft_peak_freqs = fft_peak_freqs[non_duplicate_mask]
            
            print(f"    After duplicate removal: {len(eod_waveforms)} events")
        else:
            print(f"    No duplicates found")

        keep_indices, filtered_features = filter_waveforms(
            eod_waveforms, eod_widths, amplitude_ratios, fft_peak_freqs, rate*parameters['interp_factor'],
            dur_min=parameters['min_width_us'], 
            dur_max=parameters['max_width_us'],
            pp_r_min=parameters['amplitude_ratio_min'], 
            pp_r_max=parameters['amplitude_ratio_max'],
            fft_freq_min=parameters['peak_fft_freq_min'], 
            fft_freq_max=parameters['peak_fft_freq_max'],
            return_features=True
        )

        # Create complete DataFrame with all extracted data first (without fft_freq_max)
        complete_eod_table = pd.DataFrame({
            'species_code': [species_code] * len(eod_waveforms),
            'individual_id': [individual_id] * len(eod_waveforms),
            'midpoint_idx': final_midpoint_idc,
            'peak_idx': final_peak_idc,
            'trough_idx': final_trough_idc,
            'snippet_peak_idx': snippet_peak_idc,
            'snippet_trough_idx': snippet_trough_idc,
            'snippet_midpoint_idx': snippet_midpoint_idc,
            'eod_amplitude': eod_amps,
            'eod_width_us': eod_widths,
            'eod_amplitude_ratio': amplitude_ratios,
            'pulse_orientation': pulse_orientation,
            'waveform_length': waveform_lengths,
            'fft_freq_max': fft_peak_freqs
            })
        
        # Apply amplitude-based filtering: only keep top X% highest amplitude EODs
        # First, filter by the initial criteria (duration, ratio, frequency)
        initially_filtered_table = complete_eod_table.iloc[keep_indices].copy()
        
        # Detect clipped pulses (amplitude >= clip_threshold) among the initially filtered events
        initially_filtered_clipped_mask = initially_filtered_table['eod_amplitude'] >= parameters['clip_threshold']
        
        # Get non-clipped events from the initially filtered table
        non_clipped_filtered_table = initially_filtered_table[~initially_filtered_clipped_mask].copy()
        
        if len(non_clipped_filtered_table) > 0:
            # Check if we have enough non-clipped events to apply percentage filtering
            if len(non_clipped_filtered_table) >= 50:
                # Calculate amplitude threshold for top X% of non-clipped events
                amplitude_threshold_percentile = 100 - parameters['top_amplitude_percent']
                amplitude_threshold = np.percentile(non_clipped_filtered_table['eod_amplitude'], amplitude_threshold_percentile)
                
                # Create mask for high-amplitude events among non-clipped
                high_amplitude_mask = non_clipped_filtered_table['eod_amplitude'] >= amplitude_threshold
                
                print(f"  Amplitude filtering: keeping top {parameters['top_amplitude_percent']}% of {len(non_clipped_filtered_table)} non-clipped events")
                print(f"    Amplitude threshold: {amplitude_threshold:.4f}")
            else:
                # Take all non-clipped events if fewer than 50
                high_amplitude_mask = np.ones(len(non_clipped_filtered_table), dtype=bool)
                amplitude_threshold = non_clipped_filtered_table['eod_amplitude'].min() if len(non_clipped_filtered_table) > 0 else 0.0
                
                print(f"  Amplitude filtering: taking all {len(non_clipped_filtered_table)} non-clipped events (< 50 available)")
                print(f"    Minimum amplitude: {amplitude_threshold:.4f}")
            
            # Get the indices in the original keep_indices array that correspond to high-amplitude non-clipped events
            non_clipped_indices_in_keep = np.where(~initially_filtered_clipped_mask)[0]
            selected_non_clipped_indices = non_clipped_indices_in_keep[high_amplitude_mask]
            high_amplitude_indices = keep_indices[selected_non_clipped_indices]
            
            print(f"    Events after amplitude filtering: {len(high_amplitude_indices)}")
        else:
            high_amplitude_indices = np.array([])
            amplitude_threshold = 0.0
            print(f"  No non-clipped events available for amplitude filtering")
        
        # Detect all clipped pulses (amplitude >= clip_threshold) in original data
        all_clipped_mask = eod_amps >= parameters['clip_threshold']
        
        # Create filtering masks using high-amplitude indices
        keep_mask = np.isin(np.arange(len(eod_waveforms)), high_amplitude_indices)
        normal_mask = keep_mask & ~all_clipped_mask
        clipped_high_amp_mask = keep_mask & all_clipped_mask
        
        # Filter DataFrames and waveforms using masks
        eod_table = complete_eod_table[normal_mask].copy()
        clipped_table = complete_eod_table[all_clipped_mask].copy()  # Use ALL clipped, not just high-amplitude
        
        # Add fft_freq_max to the filtered tables
        # filtered_features['fft_freq'] has the same length as keep_indices
        if len(filtered_features) > 0 and 'fft_freq' in filtered_features:
            # Create full-length array with zeros, then fill in the filtered values
            full_fft_freq = np.zeros(len(eod_waveforms))
            full_fft_freq[keep_indices] = filtered_features['fft_freq']
            
            # Add to the filtered tables
            eod_table['fft_freq_max'] = full_fft_freq[normal_mask]
            clipped_table['fft_freq_max'] = full_fft_freq[all_clipped_mask]
        else:
            # Add zeros if no filtered features
            eod_table['fft_freq_max'] = np.zeros(len(eod_table))
            clipped_table['fft_freq_max'] = np.zeros(len(clipped_table))
        
        # Filter waveforms using the same masks
        filtered_eod_waveforms = [eod_waveforms[i] for i in np.where(normal_mask)[0]]
        clipped_eod_waveforms = [eod_waveforms[i] for i in np.where(all_clipped_mask)[0]]
        
        print(f"  Final EODs after all filtering: {len(high_amplitude_indices)} out of {len(eod_waveforms)} total")
        print(f"    Normal high-amplitude events: {len(eod_table)}")
        print(f"    All clipped events (amp >= {parameters['clip_threshold']}): {len(clipped_table)}")
                
        # Save individual results
        eod_table.to_csv(f'{output_path}\\{individual_id}_eod_table.csv', index=False)
        if len(clipped_table) > 0:
            clipped_table.to_csv(f'{output_path}\\{individual_id}_clipped_eod_table.csv', index=False)
        
        # Add to master collections
        all_eod_tables.append(eod_table)
        if len(clipped_table) > 0:
            all_clipped_tables.append(clipped_table)
        
        # Store waveform data with metadata
        if len(filtered_eod_waveforms) > 0:
            waveform_metadata = save_variable_length_waveforms(
                filtered_eod_waveforms, 
                f'{output_path}\\{individual_id}_eod_waveforms'
            )
            
            # Add to master waveform collection
            all_waveform_data.append({
                'species_code': species_code,
                'individual_id': individual_id,
                'waveforms': filtered_eod_waveforms,
                'metadata': waveform_metadata
            })
        else:
            # Create empty files for consistency
            np.savez_compressed(f'{output_path}\\{individual_id}_eod_waveforms_concatenated.npz', data=np.array([]))
            import json
            with open(f'{output_path}\\{individual_id}_eod_waveforms_metadata.json', 'w') as f:
                json.dump({'lengths': [], 'start_indices': [], 'total_waveforms': 0, 'total_samples': 0}, f, separators=(',', ':'))
        
        # Save clipped waveforms separately
        if len(clipped_eod_waveforms) > 0:
            clipped_waveform_metadata = save_variable_length_waveforms(
                clipped_eod_waveforms, 
                f'{output_path}\\{individual_id}_clipped_eod_waveforms'
            )
            
            # Add to master clipped waveform collection
            all_clipped_waveforms.append({
                'species_code': species_code,
                'individual_id': individual_id,
                'waveforms': clipped_eod_waveforms,
                'metadata': clipped_waveform_metadata
            })
        
        print(f"  Saved {len(eod_table)} normal EOD events and {len(clipped_table)} clipped events")
        
        # Multi-panel summary plot
        if len(midpoints) > 0:  # Show plot if we have any detections at all
            
            # Create figure with custom layout
            fig = plt.figure(figsize=(20, 12))
            
            # Define grid layout: top panel spans full width, bottom has 3 panels
            gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
            
            # Top panel: Raw data with detections (spans all columns)
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(data, 'k-', linewidth=0.5, alpha=0.7)
            
            # Plot all initially filtered events (before amplitude filtering)
            if len(keep_indices) > 0:
                all_filtered_peaks = final_peak_idc[keep_indices]
                all_filtered_troughs = final_trough_idc[keep_indices]
                ax1.plot(all_filtered_peaks, 
                        data[all_filtered_peaks], 'o', markersize=3, color='red', alpha=0.6, label='All filtered')
                ax1.plot(all_filtered_troughs, 
                        data[all_filtered_troughs], 'o', markersize=3, color='blue', alpha=0.6)
            
            # Highlight the high-amplitude events (top 30%) in yellow
            if len(eod_table) > 0:
                ax1.plot(eod_table['peak_idx'].values, 
                        data[eod_table['peak_idx'].values], 'o', markersize=4, color='yellow', alpha=0.8, 
                        label=f'Selected high-amplitude peaks')
                ax1.plot(eod_table['trough_idx'].values, 
                        data[eod_table['trough_idx'].values], 'o', markersize=4, color='orange', alpha=0.8, 
                        label=f'Selected high-amplitude troughs')
            
            # Update title based on filtering approach
            if 'non_clipped_filtered_table' in locals() and len(non_clipped_filtered_table) >= 50:
                title_suffix = f"Top {parameters['top_amplitude_percent']}%: {len(eod_table)}"
            else:
                title_suffix = f"All non-clipped: {len(eod_table)}"
            
            ax1.set_title(f'{individual_id} - EOD Detections (Total: {len(keep_indices)}, {title_suffix})', fontsize=14)
            ax1.set_xlabel('Sample')
            ax1.set_ylabel('Voltage')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Prepare normalized waveforms for analysis
            if len(filtered_eod_waveforms) > 0:
                # Waveforms are already normalized, just center them by their midpoint
                centered_waveforms = []
                for i, wf in enumerate(filtered_eod_waveforms):
                    # Get the midpoint index for this waveform (from the corresponding DataFrame row)
                    midpoint_idx = eod_table.iloc[i]['snippet_midpoint_idx']
                    
                    # Center the waveform by shifting so midpoint is at index 0
                    # Create indices relative to midpoint
                    wf_indices = np.arange(len(wf)) - midpoint_idx
                    centered_waveforms.append((wf_indices, wf))
                
                # Find the range needed to contain all waveforms when centered
                all_min_idx = min([min(indices) for indices, _ in centered_waveforms])
                all_max_idx = max([max(indices) for indices, _ in centered_waveforms])
                
                # Create a common index array
                common_indices = np.arange(all_min_idx, all_max_idx + 1)
                
                # Interpolate all waveforms to the common index grid for averaging
                aligned_waveforms = []
                for indices, wf in centered_waveforms:
                    # Interpolate to common grid, using NaN for extrapolated regions
                    aligned_wf = np.interp(common_indices, indices, wf, left=np.nan, right=np.nan)
                    aligned_waveforms.append(aligned_wf)
                
                # Calculate average waveform (ignoring NaN values)
                if aligned_waveforms:
                    aligned_array = np.array(aligned_waveforms)
                    average_waveform = np.nanmean(aligned_array, axis=0)
                    waveform_std = np.nanstd(aligned_array, axis=0)
                    # Store common indices for plotting
                    avg_indices = common_indices
                else:
                    average_waveform = np.array([])
                    waveform_std = np.array([])
                    avg_indices = np.array([])
            else:
                centered_waveforms = []
                average_waveform = np.array([])
                waveform_std = np.array([])
                avg_indices = np.array([])
            
            # Middle left: Normalized waveforms overlay
            ax2 = fig.add_subplot(gs[1, 0])
            if len(filtered_eod_waveforms) > 0 and centered_waveforms:
                # Plot individual waveforms with low alpha (centered by midpoint)
                for indices, wf in centered_waveforms:
                    ax2.plot(indices, wf, 'k-', alpha=0.1, linewidth=0.5)
                
                # Plot average waveform as bold line
                if len(average_waveform) > 0:
                    ax2.plot(avg_indices, average_waveform, 'r-', linewidth=3, label='Average')
                    ax2.fill_between(avg_indices, 
                                average_waveform - waveform_std,
                                average_waveform + waveform_std,
                                alpha=0.3, color='red', label='±1 SD')
                
                ax2.set_title('Centered Normalized Waveforms')
                ax2.set_xlabel('Sample (relative to midpoint)')
                ax2.set_ylabel('Normalized Amplitude')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Midpoint')
            else:
                ax2.text(0.5, 0.5, 'No waveforms available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Centered Normalized Waveforms')
            
            # Middle middle: Power Spectral Densities
            ax3 = fig.add_subplot(gs[1, 1])
            if len(filtered_eod_waveforms) > 0 and centered_waveforms:
                # Calculate PSDs for individual waveforms
                psds = []
                freqs = None
                
                # Use a fixed nperseg for consistent frequency resolution
                waveform_lengths = [len(wf) for _, wf in centered_waveforms]
                nperseg = min(128, min(waveform_lengths) // 2)
                if nperseg < 8:  # Ensure minimum segments for meaningful PSD
                    nperseg = 8
                
                for indices, wf in centered_waveforms:
                    if len(wf) > nperseg:
                        # Use scipy.signal.welch for PSD with fixed nperseg
                        from scipy import signal as scipy_signal
                        f, psd = scipy_signal.welch(wf, fs=rate*parameters['interp_factor'], nperseg=nperseg)
                        if freqs is None:
                            freqs = f
                        psds.append(psd)
                
                if psds and freqs is not None and len(psds) > 0:
                    # Convert to array (all PSDs now have same length)
                    psds_array = np.array(psds)
                    
                    # Plot individual PSDs with low alpha
                    for psd in psds:
                        ax3.loglog(freqs, psd, 'k-', alpha=0.1, linewidth=0.5)
                    
                    # Calculate and plot average PSD
                    avg_psd = np.mean(psds_array, axis=0)
                    ax3.loglog(freqs, avg_psd, 'r-', linewidth=3, label='Average')
                
                ax3.set_title('Power Spectral Densities')
                ax3.set_xlabel('Frequency (Hz)')
                ax3.set_ylabel('Power')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No PSDs available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Power Spectral Densities')


            # Add summary statistics as text
            total_initially_filtered = len(keep_indices) if 'keep_indices' in locals() else 0
            non_clipped_count = len(non_clipped_filtered_table) if 'non_clipped_filtered_table' in locals() else 0
            stats_text = f"Total initial detections: {len(midpoints)}\n"
            stats_text += f"After quality filtering: {total_initially_filtered}\n"
            stats_text += f"Non-clipped after filtering: {non_clipped_count}\n"
            stats_text += f"Selected high-amplitude: {len(eod_table)}\n"
            if len(eod_table) > 0:
                if non_clipped_count >= 50:
                    stats_text += f"Top {parameters['top_amplitude_percent']}% threshold: {amplitude_threshold:.4f}\n"
                else:
                    stats_text += f"All non-clipped taken (< 50)\n"
                stats_text += f"Mean amplitude: {eod_table['eod_amplitude'].mean():.4f}\n"
                stats_text += f"Mean width: {eod_table['eod_width_us'].mean():.1f} μs\n"
                stats_text += f"Mean amp ratio: {eod_table['eod_amplitude_ratio'].mean():.2f}\n"
                stats_text += f"Mean fft peak: {eod_table['fft_freq_max'].mean():.2f}\n"

            # Place stats in middle right panel
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')

            # Bottom left: Amplitude ratios histogram
            ax5 = fig.add_subplot(gs[2, 0])
            if len(eod_table) > 0:
                amplitude_ratios = eod_table['eod_amplitude_ratio'].values
                ax5.hist(amplitude_ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                mean_amp_ratio = amplitude_ratios.mean()
                ax5.axvline(mean_amp_ratio, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_amp_ratio:.2f}')
                ax5.set_title('Amplitude Ratios')
                ax5.set_xlabel('Ratio')
                ax5.set_ylabel('Count')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Amplitude Ratios')
                
            # Bottom middle: Waveform lengths histogram
            ax6 = fig.add_subplot(gs[2, 1])
            if len(eod_table) > 0:
                # Convert to microseconds for display
                eod_widths_us = eod_table['eod_width_us'].values
                ax6.hist(eod_widths_us, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                mean_width = eod_widths_us.mean()
                ax6.axvline(mean_width, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_width:.1f} μs')
                ax6.set_title('EOD Widths')
                ax6.set_xlabel('Width (μs)')
                ax6.set_ylabel('Count')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('EOD Widths')
            
            # Bottom right: peak FFT frequencies histogram
            ax7 = fig.add_subplot(gs[2, 2])
            if len(eod_table) > 0:
                fft_peaks = eod_table['fft_freq_max'].values
                ax7.hist(fft_peaks, bins=20, alpha=0.7, color='salmon', edgecolor='black')
                mean_peak_freq = fft_peaks.mean()
                ax7.axvline(mean_peak_freq, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_peak_freq:.1f} Hz')
                ax7.set_title('Peak FFT Frequencies')
                ax7.set_xlabel('Frequency (Hz)')
                ax7.set_ylabel('Count')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            else:
                ax7.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Peak FFT Frequencies')

            plt.suptitle(f'EOD Analysis Summary - {individual_id}', fontsize=16)
            plt.savefig(f'{output_path}\\{individual_id}_analysis_summary.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Clear data from memory
            del data
            gc.collect()
    else:
        print(f"  No midpoints detected in {individual_id}, skipping...")
        # Clear data from memory
        del data
        gc.collect()
    
# After processing all individuals, create and save master files
print("\nCreating master files...")

if all_eod_tables:
    # Combine all normal EOD tables
    master_eod_table = pd.concat(all_eod_tables, ignore_index=True)
    master_eod_table.to_csv(f'{output_path}\\master_eod_table.csv', index=False)
    print(f"Master EOD table saved: master_eod_table.csv ({len(master_eod_table)} events)")
    
    # Combine all normal waveforms
    if all_waveform_data:
        all_combined_waveforms = []
        for wf_data in all_waveform_data:
            all_combined_waveforms.extend(wf_data['waveforms'])
        
        if all_combined_waveforms:
            master_waveform_metadata = save_variable_length_waveforms(
                all_combined_waveforms, 
                f'{output_path}\\master_eod_waveforms'
            )
            print(f"Master waveforms saved: master_eod_waveforms_concatenated.npz ({len(all_combined_waveforms)} waveforms)")

if all_clipped_tables:
    # Combine all clipped EOD tables
    master_clipped_table = pd.concat(all_clipped_tables, ignore_index=True)
    master_clipped_table.to_csv(f'{output_path}\\master_clipped_eod_table.csv', index=False)
    print(f"Master clipped table saved: master_clipped_eod_table.csv ({len(master_clipped_table)} events)")
    
    # Combine all clipped waveforms
    if all_clipped_waveforms:
        all_combined_clipped_waveforms = []
        for wf_data in all_clipped_waveforms:
            all_combined_clipped_waveforms.extend(wf_data['waveforms'])
        
        if all_combined_clipped_waveforms:
            master_clipped_waveform_metadata = save_variable_length_waveforms(
                all_combined_clipped_waveforms, 
                f'{output_path}\\master_clipped_eod_waveforms'
            )
            print(f"Master clipped waveforms saved: master_clipped_eod_waveforms_concatenated.npz ({len(all_combined_clipped_waveforms)} waveforms)")

# Print final summary
total_individuals = len(individual_folders)
total_normal_events = len(master_eod_table) if all_eod_tables else 0
total_clipped_events = len(master_clipped_table) if all_clipped_tables else 0

print(f"\n{'='*60}")
print("PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Processing parameters:")
print(f"  Detection threshold: {parameters['thresh']}")
print(f"  Top amplitude percentile: {parameters['top_amplitude_percent']}%")
print(f"Total individuals processed: {total_individuals}")
print(f"Total high-amplitude EOD events: {total_normal_events}")
print(f"Total clipped events: {total_clipped_events}")

if all_eod_tables:
    species_counts = master_eod_table['species_code'].value_counts()
    print(f"Species distribution:")
    for species, count in species_counts.items():
        print(f"  {species}: {count} events")

print(f"Output directory: {output_path}")
