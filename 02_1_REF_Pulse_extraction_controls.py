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
from pulse_functions import (
    save_fixed_length_waveforms,
    extract_pulse_snippets,
    filter_waveforms,
    normalize_waveforms
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
        parameters = {'thresh':0.1,  # Reduced threshold for pulse detection to catch low-amplitude fish
                    'min_rel_slope_diff':0.25,
                    'min_width_us':30,  # Minimum pulse width in microseconds
                    'max_width_us':1000,  # Maximum pulse width in microseconds
                    'width_fac_detection':7.0,
                    'interp_factor':1,  # Interpolation factor for waveform extraction
                    'amplitude_ratio_min':0.2,  # Minimum peak-to-peak amplitude ratio
                    'amplitude_ratio_max':4,     # Maximum peak-to-peak amplitude ratio
                    'save_filtered_out':True, # Option to save filtered-out pulses for quality control
                    'peak_fft_freq_min':50,
                    'peak_fft_freq_max':10000,
                    'clip_threshold':1.4,  # Threshold for clipping detection (1.6 is conservative, adjust as needed)
                    'top_amplitude_percent':50,  # Only keep top x% highest amplitude EODs
                    'return_diff': True
                    }
else:
    # Default control recording parameters
    parameters = {'thresh':0.1,  # Reduced threshold for pulse detection to catch low-amplitude fish
                'min_rel_slope_diff':0.25,
                'min_width_us':30,  # Minimum pulse width in microseconds
                'max_width_us':1000,  # Maximum pulse width in microseconds
                'width_fac_detection':7.0,
                'interp_factor':1,  # Interpolation factor for waveform extraction
                'amplitude_ratio_min':0.2,  # Minimum peak-to-peak amplitude ratio
                'amplitude_ratio_max':4,     # Maximum peak-to-peak amplitude ratio
                'save_filtered_out':True, # Option to save filtered-out pulses for quality control
                'peak_fft_freq_min':50,
                'peak_fft_freq_max':10000,
                'clip_threshold':1.4,  # Threshold for clipping detection (1.6 is conservative, adjust as needed)
                'top_amplitude_percent':50,  # Only keep top x% highest amplitude EODs
                'return_diff': True
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
                                verbose=0,
                                return_data=False)
    
    midpoints = (peaks + troughs) // 2  # Midpoint indices of pulses

        
    # Extract snippets and analyze them
    if len(midpoints) > 0:
        print(f"  Analyzing {len(midpoints)} unique events...")

        (
            eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
            snippet_p1_idc, snippet_p2_idc, raw_p1_idc, raw_p2_idc, 
            pulse_orientations, amp_ratios, fft_peak_freqs, peak_locations
        ) = extract_pulse_snippets(
            data, peaks, troughs, rate = rate, length = 2000,
            source = '1ch_diff', return_differential = parameters_df['return_diff'][0]
        )

        # Use basic threshold filtering
        keep_indices, filtered_features, filteredout_features = filter_waveforms(
            eod_snippets, eod_widths, amp_ratios, fft_peak_freqs, rate,
            dur_min=parameters_df['min_width_us'][0], 
            dur_max=parameters_df['max_width_us'][0],
            pp_r_min=parameters_df['amplitude_ratio_min'][0], 
            pp_r_max=parameters_df['amplitude_ratio_max'][0],
            fft_freq_min=parameters_df['peak_fft_freq_min'][0], 
            fft_freq_max=parameters_df['peak_fft_freq_max'][0],
            return_features=True, return_filteredout_features=True
        )

        # Create complete DataFrame with all extracted data first
        raw_midpoint_idc = (raw_p1_idc + raw_p2_idc) // 2  # Recalculate midpoints from raw p1/p2
        snippet_midpoint_idc = (snippet_p1_idc + snippet_p2_idc) // 2  # Recalculate midpoints from raw p1/p2
        complete_eod_table = pd.DataFrame({
            'species_code': [species_code] * len(eod_snippets),
            'individual_id': [individual_id] * len(eod_snippets),
            'midpoint_idx': raw_midpoint_idc,
            'relative_time_s': raw_midpoint_idc / rate,
            'p1_idx': raw_p1_idc,
            'p2_idx': raw_p2_idc,
            'eod_channel': eod_chan,
            'snippet_p1_idx': snippet_p1_idc,
            'snippet_p2_idx': snippet_p2_idc,
            'snippet_midpoint_idx': snippet_midpoint_idc,
            'eod_amplitude': eod_amps,
            'eod_width_us': eod_widths,
            'eod_amplitude_ratio': amp_ratios,
            'pulse_orientation': pulse_orientations,
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
        keep_mask = np.isin(np.arange(len(eod_snippets)), high_amplitude_indices)
        normal_mask = keep_mask & ~all_clipped_mask
        clipped_high_amp_mask = keep_mask & all_clipped_mask
        
        # Filter DataFrames and waveforms using masks
        eod_table = complete_eod_table[normal_mask].copy()
        clipped_table = complete_eod_table[all_clipped_mask].copy()  # Use ALL clipped, not just high-amplitude
        
        # Add fft_freq_max to the filtered tables
        # filtered_features['fft_freq'] has the same length as keep_indices
        if len(filtered_features) > 0 and 'fft_freq' in filtered_features:
            # Create full-length array with zeros, then fill in the filtered values
            full_fft_freq = np.zeros(len(eod_snippets))
            full_fft_freq[keep_indices] = filtered_features['fft_freq']
            
            # Add to the filtered tables
            eod_table['fft_freq_max'] = full_fft_freq[normal_mask]
            clipped_table['fft_freq_max'] = full_fft_freq[all_clipped_mask]
        else:
            # Add zeros if no filtered features
            eod_table['fft_freq_max'] = np.zeros(len(eod_table))
            clipped_table['fft_freq_max'] = np.zeros(len(clipped_table))
        
        # Filter waveforms using the same masks
        filtered_eod_waveforms = [eod_snippets[i] for i in np.where(normal_mask)[0]]
        clipped_eod_waveforms = [eod_snippets[i] for i in np.where(all_clipped_mask)[0]]
        
        print(f"  Final EODs after all filtering: {len(high_amplitude_indices)} out of {len(eod_snippets)} total")
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
            # Save original waveforms
            waveform_metadata = save_fixed_length_waveforms(
                filtered_eod_waveforms, 
                f'{output_path}\\{individual_id}_eod_waveforms'
            )
            
            # Save normalized waveforms (if normalization was successful)
            if 'normalized_waveforms' in locals() and len(normalized_waveforms) > 0:
                try:
                    normalized_waveform_metadata = save_fixed_length_waveforms(
                        normalized_waveforms, 
                        f'{output_path}\\{individual_id}_eod_waveforms_normalized'
                    )
                    print(f"    Saved {len(normalized_waveforms)} normalized waveforms for {individual_id}")
                except Exception as e:
                    print(f"    Warning: Could not save normalized waveforms for {individual_id}: {e}")
            
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
            clipped_waveform_metadata = save_fixed_length_waveforms(
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
                all_filtered_p1 = raw_p1_idc[keep_indices]
                all_filtered_p2 = raw_p2_idc[keep_indices]
                ax1.plot(all_filtered_p1, 
                        data[all_filtered_p1], 'o', markersize=3, color='red', alpha=0.6, label='All filtered')
                ax1.plot(all_filtered_p2, 
                        data[all_filtered_p2], 'o', markersize=3, color='blue', alpha=0.6)

            # Highlight the high-amplitude events (top 30%) in yellow
            if len(eod_table) > 0:
                ax1.plot(eod_table['p1_idx'].values, 
                        data[eod_table['p1_idx'].values], 'o', markersize=4, color='yellow', alpha=0.8, 
                        label=f'Selected high-amplitude peaks')
                ax1.plot(eod_table['p2_idx'].values, 
                        data[eod_table['p2_idx'].values], 'o', markersize=4, color='orange', alpha=0.8, 
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
                print(f"  Normalizing {len(filtered_eod_waveforms)} waveforms for visualization...")
                
                # Get P1/P2 indices for the filtered waveforms
                filtered_snippet_p1_idc = snippet_p1_idc[normal_mask]
                filtered_snippet_p2_idc = snippet_p2_idc[normal_mask]
                
                # Normalize waveforms using p1_unity method (best for visualization)
                normalized_waveforms = normalize_waveforms(
                    filtered_eod_waveforms, 
                    filtered_snippet_p1_idc, 
                    filtered_snippet_p2_idc,
                    method='p1_unity',  # P1 = +1, head-positive orientation
                    baseline_correction=True
                )
                
                print(f"    Successfully normalized {len(normalized_waveforms)} waveforms")
                
                # Calculate average waveform directly from normalized waveforms
                if len(normalized_waveforms) > 0:
                    # Find the longest waveform to set common length
                    max_length = max(len(wf) for wf in normalized_waveforms)
                    
                    # Align all waveforms by padding shorter ones
                    aligned_waveforms = []
                    for wf in normalized_waveforms:
                        if len(wf) == max_length:
                            aligned_waveforms.append(wf)
                        else:
                            # Pad shorter waveforms with zeros at the end
                            padded = np.pad(wf, (0, max_length - len(wf)), 'constant')
                            aligned_waveforms.append(padded)
                    
                    # Calculate average and std
                    aligned_array = np.array(aligned_waveforms)
                    average_waveform = np.mean(aligned_array, axis=0)
                    waveform_std = np.std(aligned_array, axis=0)
                    avg_indices = np.arange(len(average_waveform))
                else:
                    average_waveform = np.array([])
                    waveform_std = np.array([])
                    avg_indices = np.array([])
            else:
                normalized_waveforms = []
                average_waveform = np.array([])
                waveform_std = np.array([])
                avg_indices = np.array([])
            
            # Middle left: Normalized waveforms overlay
            ax2 = fig.add_subplot(gs[1, 0])
            if len(normalized_waveforms) > 0:
                # Plot individual normalized waveforms with low alpha
                for wf in normalized_waveforms:
                    ax2.plot(wf, 'k-', alpha=0.1, linewidth=0.5)
                
                # Plot average waveform as bold line
                if len(average_waveform) > 0:
                    ax2.plot(avg_indices, average_waveform, 'r-', linewidth=3, label='Average')
                    ax2.fill_between(avg_indices, 
                                average_waveform - waveform_std,
                                average_waveform + waveform_std,
                                alpha=0.3, color='red', label='±1 SD')
                
                ax2.set_title('P1-Normalized Waveforms (P1=+1)')
                ax2.set_xlabel('Sample')
                ax2.set_ylabel('Normalized Amplitude')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No waveforms available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('P1-Normalized Waveforms (P1=+1)')
            
            # Middle middle: Power Spectral Densities
            ax3 = fig.add_subplot(gs[1, 1])
            if len(normalized_waveforms) > 0:
                # Calculate PSDs for normalized waveforms
                psds = []
                freqs = None
                
                # Use a fixed nperseg for consistent frequency resolution
                waveform_lengths = [len(wf) for wf in normalized_waveforms]
                nperseg = min(128, min(waveform_lengths) // 2) if waveform_lengths else 64
                if nperseg < 8:  # Ensure minimum segments for meaningful PSD
                    nperseg = 8
                
                for wf in normalized_waveforms:
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
            master_waveform_metadata = save_fixed_length_waveforms(
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
            master_clipped_waveform_metadata = save_fixed_length_waveforms(
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
