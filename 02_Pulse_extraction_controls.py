import thunderfish.pulses as pulses
import matplotlib.pyplot as plt
import audioio as aio
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
import glob
import datetime as dt
import os

# Import consolidated EOD functions
from eod_functions import (
    save_variable_length_waveforms,
    extract_pulse_snippets_control,
    remove_noise_artifacts,
    plot_waveform_comparison,
    compare_table_features,
    filter_waveforms
)

def main():
    # Set directories
    root = tk.Tk()
    root.withdraw()
    input_path = filedialog.askdirectory(title="Select Folder with Individual Fish Folders")
    output_path = filedialog.askdirectory(title="Select Folder to Store Analysis Results")

    # Find individual fish folders
    individual_folders = [d for d in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(d)]

    if not individual_folders:
        print("No individual folders found!")
        return

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
        return

    data, rate = aio.load_audio(wav_files[0])
    data = data[:,0]  # Use only the first channel

    # Pulse extraction parameters
    parameters = {'thresh':0.03,  # Threshold for pulse detection
                'min_rel_slope_diff':0.25,
                'min_width_s':3e-05,  # Minimum pulse width in seconds
                'max_width_s':0.001,  # Maximum pulse width in seconds
                'width_fac_detection':10.0,
                'width_fac_extraction':10.0,  # Factor for variable-width extraction
                'verbose':0,
                'return_data':False,
                # Additional filtering parameters
                'amplitude_ratio_min':0.1,  # Minimum peak-to-peak amplitude ratio
                'amplitude_ratio_max':3,     # Maximum peak-to-peak amplitude ratio
                'save_filtered_out':True, # Option to save filtered-out pulses for quality control
                'noise_removal':False,
                'max_freq_content':0.9,  # Allow some high freq for species differences
                'min_snr':1.5,
                'max_ipi_ratio':100.0,
                'peak_fft_freq_min':500,
                'peak_fft_freq_max':20000,
                'clip_threshold':1.6  # Threshold for clipping detection (1.6 is conservative, adjust as needed)
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

    parameters = pd.DataFrame({k: [v] for k, v in parameters.items()})
    parameters.to_csv('%s\\analysis_parameters.csv' % output_path, index=False)
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
                                    thresh = parameters['thresh'][0], 
                                    min_rel_slope_diff=parameters['min_rel_slope_diff'][0],
                                    min_width=parameters['min_width_s'][0],  # in seconds
                                    max_width=parameters['max_width_s'][0],  # in seconds
                                    width_fac=parameters['width_fac_detection'][0],
                                    verbose=parameters['verbose'][0],
                                    return_data=parameters['return_data'][0])
        
        midpoints = (peaks + troughs) // 2  # Midpoint indices of pulses

            
        # Extract variable-width snippets and analyze them
        if len(midpoints) > 0:
            print(f"  Analyzing {len(midpoints)} unique events...")

            # Analyze snippets with variable widths
            eod_waveforms, eod_amps, eod_widths, snippet_peak_idc, snippet_trough_idc, snippet_midpoint_idc, \
                final_peak_idc, final_trough_idc, final_midpoint_idc, pulse_orientation, amplitude_ratios, waveform_lengths = \
                extract_pulse_snippets_control(data, rate, midpoints, peaks, troughs, pulse_widths,
                                        width_factor=parameters['width_fac_extraction'][0], 
                                        interp_factor=1, center_on_zero_crossing=False)  # Skip centering for storage efficiency

            keep_indices, filtered_features = filter_waveforms(
                eod_waveforms, eod_widths, amplitude_ratios, rate,
                dur_min=parameters['min_width_s'][0]*1e6, 
                dur_max=parameters['max_width_s'][0]*1e6,
                pp_r_min=parameters['amplitude_ratio_min'][0], 
                pp_r_max=parameters['amplitude_ratio_max'][0],
                fft_freq_min=parameters['peak_fft_freq_min'][0], 
                fft_freq_max=parameters['peak_fft_freq_max'][0],
                return_features=True
            )    

            # Indices of all events after duplicate removal
            all_indices = np.arange(len(eod_waveforms))
            # Indices of filtered-out events
            filtered_out_indices = np.setdiff1d(all_indices, keep_indices)
            
            # Detect clipped pulses (amplitude >= clip_threshold)
            clipped_mask = eod_amps >= parameters['clip_threshold'][0]
            clipped_indices = np.where(clipped_mask)[0]
            
            # Separate clipped from normal events
            normal_keep_indices = keep_indices[~np.isin(keep_indices, clipped_indices)]
            clipped_keep_indices = keep_indices[np.isin(keep_indices, clipped_indices)]
            
            print(f"  EODs after filtering: {len(keep_indices)} out of {len(eod_waveforms)}")
            print(f"    Normal events: {len(normal_keep_indices)}")
            print(f"    Clipped events (amp >= {parameters['clip_threshold'][0]}): {len(clipped_keep_indices)}")

            # Process normal (non-clipped) events
            if len(normal_keep_indices) > 0:
                # Filter all arrays for normal events
                filtered_eod_waveforms = [eod_waveforms[i] for i in normal_keep_indices]
                filtered_eod_amps = eod_amps[normal_keep_indices]
                filtered_eod_widths = eod_widths[normal_keep_indices]
                filtered_snippet_peak_idc = snippet_peak_idc[normal_keep_indices]
                filtered_snippet_trough_idc = snippet_trough_idc[normal_keep_indices]
                filtered_snippet_midpoint_idc = snippet_midpoint_idc[normal_keep_indices]
                filtered_final_peak_idc = final_peak_idc[normal_keep_indices]
                filtered_final_trough_idc = final_trough_idc[normal_keep_indices]
                filtered_final_midpoints = final_midpoint_idc[normal_keep_indices]
                filtered_pulse_orientation = pulse_orientation[normal_keep_indices]
                filtered_amplitude_ratios = amplitude_ratios[normal_keep_indices]
                filtered_waveform_lengths = waveform_lengths[normal_keep_indices]
                filtered_fft_peaks = np.array([filtered_features['fft_freq'][i] for i, orig_idx in enumerate(keep_indices) if orig_idx in normal_keep_indices])
            else:
                # No normal events
                filtered_eod_waveforms = []
                filtered_eod_amps = np.empty(0)
                filtered_eod_widths = np.empty(0)
                filtered_snippet_peak_idc = np.empty(0)
                filtered_snippet_trough_idc = np.empty(0)
                filtered_snippet_midpoint_idc = np.empty(0)
                filtered_final_peak_idc = np.empty(0)
                filtered_final_trough_idc = np.empty(0)
                filtered_final_midpoints = np.empty(0)
                filtered_pulse_orientation = np.empty(0)
                filtered_amplitude_ratios = np.empty(0)
                filtered_waveform_lengths = np.empty(0)
                filtered_fft_peaks = np.empty(0)

            # Process clipped events
            if len(clipped_keep_indices) > 0:
                # Filter all arrays for clipped events
                clipped_eod_waveforms = [eod_waveforms[i] for i in clipped_keep_indices]
                clipped_eod_amps = eod_amps[clipped_keep_indices]
                clipped_eod_widths = eod_widths[clipped_keep_indices]
                clipped_snippet_peak_idc = snippet_peak_idc[clipped_keep_indices]
                clipped_snippet_trough_idc = snippet_trough_idc[clipped_keep_indices]
                clipped_snippet_midpoint_idc = snippet_midpoint_idc[clipped_keep_indices]
                clipped_final_peak_idc = final_peak_idc[clipped_keep_indices]
                clipped_final_trough_idc = final_trough_idc[clipped_keep_indices]
                clipped_final_midpoints = final_midpoint_idc[clipped_keep_indices]
                clipped_pulse_orientation = pulse_orientation[clipped_keep_indices]
                clipped_amplitude_ratios = amplitude_ratios[clipped_keep_indices]
                clipped_waveform_lengths = waveform_lengths[clipped_keep_indices]
                clipped_fft_peaks = np.array([filtered_features['fft_freq'][i] for i, orig_idx in enumerate(keep_indices) if orig_idx in clipped_keep_indices])
            else:
                # No clipped events
                clipped_eod_waveforms = []
                clipped_eod_amps = np.empty(0)
                clipped_eod_widths = np.empty(0)
                clipped_snippet_peak_idc = np.empty(0)
                clipped_snippet_trough_idc = np.empty(0)
                clipped_snippet_midpoint_idc = np.empty(0)
                clipped_final_peak_idc = np.empty(0)
                clipped_final_trough_idc = np.empty(0)
                clipped_final_midpoints = np.empty(0)
                clipped_pulse_orientation = np.empty(0)
                clipped_amplitude_ratios = np.empty(0)
                clipped_waveform_lengths = np.empty(0)
                clipped_fft_peaks = np.empty(0)

            # Create results DataFrames with species and individual info
            
            # Normal events table
            eod_table = pd.DataFrame({
                'species_code': [species_code] * len(filtered_final_peak_idc),
                'individual_id': [individual_id] * len(filtered_final_peak_idc),
                'peak_idx': filtered_final_peak_idc,
                'trough_idx': filtered_final_trough_idc,
                'midpoint_idx': filtered_final_midpoints,
                'snippet_peak_idx': filtered_snippet_peak_idc,
                'snippet_trough_idx': filtered_snippet_trough_idc,
                'snippet_midpoint_idx': filtered_snippet_midpoint_idc,
                'eod_amplitude': filtered_eod_amps,
                'eod_width_uS': filtered_eod_widths,
                'pulse_orientation': filtered_pulse_orientation,
                'amplitude_ratio': filtered_amplitude_ratios,
                'waveform_length': filtered_waveform_lengths,
                'fft_freq_max': filtered_fft_peaks
            })
            
            # Clipped events table
            clipped_table = pd.DataFrame({
                'species_code': [species_code] * len(clipped_final_peak_idc),
                'individual_id': [individual_id] * len(clipped_final_peak_idc),
                'peak_idx': clipped_final_peak_idc,
                'trough_idx': clipped_final_trough_idc,
                'midpoint_idx': clipped_final_midpoints,
                'snippet_peak_idx': clipped_snippet_peak_idc,
                'snippet_trough_idx': clipped_snippet_trough_idc,
                'snippet_midpoint_idx': clipped_snippet_midpoint_idc,
                'eod_amplitude': clipped_eod_amps,
                'eod_width_uS': clipped_eod_widths,
                'pulse_orientation': clipped_pulse_orientation,
                'amplitude_ratio': clipped_amplitude_ratios,
                'waveform_length': clipped_waveform_lengths,
                'fft_freq_max': clipped_fft_peaks
            })
                    
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
            if len(filtered_final_midpoints) > 0:
                
                # Create figure with custom layout
                fig = plt.figure(figsize=(20, 12))
                
                # Define grid layout: top panel spans full width, bottom has 3 panels
                gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
                
                # Top panel: Raw data with detections (spans all columns)
                ax1 = fig.add_subplot(gs[0, :])
                ax1.plot(data, 'k-', linewidth=0.5, alpha=0.7)
                
                # Plot filtered (accepted) events
                if len(filtered_final_peak_idc) > 0:
                    ax1.plot(filtered_final_peak_idc, 
                            data[filtered_final_peak_idc], 'o', markersize=4, color='red', alpha=0.8, label='Peaks')
                    ax1.plot(filtered_final_trough_idc, 
                            data[filtered_final_trough_idc], 'o', markersize=4, color='blue', alpha=0.8, label='Troughs')
                
                ax1.set_title(f'{individual_id} - EOD Detections (n: {len(filtered_eod_amps)})', fontsize=14)
                ax1.set_xlabel('Sample')
                ax1.set_ylabel('Voltage')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Prepare normalized waveforms for analysis
                if len(filtered_eod_waveforms) > 0:
                    # Waveforms are already normalized, just center them by their midpoint
                    centered_waveforms = []
                    for i, wf in enumerate(filtered_eod_waveforms):
                        # Get the midpoint index for this waveform
                        midpoint_idx = filtered_snippet_midpoint_idc[i]
                        
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
                            f, psd = scipy_signal.welch(wf, fs=rate, nperseg=nperseg)
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
                stats_text = f"Total detections: {len(filtered_eod_amps)}\n"
                if len(filtered_eod_amps) > 0:
                    stats_text += f"Mean amplitude: {np.mean(filtered_eod_amps):.4f}\n"
                    stats_text += f"Mean width: {np.mean(filtered_eod_widths):.1f} μs\n"
                    stats_text += f"Mean amp ratio: {np.mean(filtered_amplitude_ratios):.2f}\n"
                    stats_text += f"Mean fft peak: {np.mean(filtered_fft_peaks):.2f}\n"

                # Place stats in middle right panel
                ax4 = fig.add_subplot(gs[1, 2])
                ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis('off')

                # Bottom left: Amplitude ratios histogram
                ax5 = fig.add_subplot(gs[2, 0])
                if len(filtered_amplitude_ratios) > 0:
                    ax5.hist(filtered_amplitude_ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    mean_amp_ratio = np.mean(filtered_amplitude_ratios)
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
                if len(filtered_eod_widths) > 0:
                    # Convert to microseconds for display
                    eod_widths_us = filtered_eod_widths
                    ax6.hist(eod_widths_us, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                    mean_width = np.mean(eod_widths_us)
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
                if len(filtered_fft_peaks) > 0:
                    ax7.hist(filtered_fft_peaks, bins=20, alpha=0.7, color='salmon', edgecolor='black')
                    mean_peak_freq = np.mean(filtered_fft_peaks)
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
        print(f"Total individuals processed: {total_individuals}")
        print(f"Total normal EOD events: {total_normal_events}")
        print(f"Total clipped events: {total_clipped_events}")
        
        if all_eod_tables:
            species_counts = master_eod_table['species_code'].value_counts()
            print(f"Species distribution:")
            for species, count in species_counts.items():
                print(f"  {species}: {count} events")
        
        print(f"Output directory: {output_path}")

if __name__ == "__main__":
    main()