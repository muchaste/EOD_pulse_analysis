# import thunderfish.eventdetection as ed

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

# Import consolidated EOD functions
from eod_functions import (
    save_variable_length_waveforms,
    calculate_storage_efficiency,
    remove_noise_artifacts,
    extract_pulse_snippets,
    plot_waveform_comparison,
    compare_table_features,
    filter_waveforms
)

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Logger Files")
output_path = filedialog.askdirectory(title="Select Folder to Store Analysis Results")

# List all .wav files
filelist = glob.glob(input_path + '/*.wav', recursive=True)

# Load first file
data, rate = aio.load_audio(filelist[0])
data = data[:,0]  # Use only the first channel

# Pulse extraction parameters
parameters = {'thresh':0.004,  # Threshold for pulse detection
              'min_rel_slope_diff':0.25,
              'min_width_s':3e-05,  # Minimum pulse width in seconds
              'max_width_s':0.001,  # Maximum pulse width in seconds
              'width_fac_detection':5.0,
              'width_fac_extraction':5.0,  # Factor for variable-width extraction
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
              'peak_fft_freq_max':20000
              }

# Plot raw data
plt.figure(figsize=(20, 6))
plt.plot(data)
plt.title('Raw Data from %s' % filelist[0].split('\\')[-1])
plt.hlines(y=parameters['thresh'][0], xmin=0, xmax=len(data), colors='red', linestyles='dashed', label='Threshold')
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


# Process each file
for n, filepath in enumerate(filelist):
    fname = filepath.split('\\')[-1]
    print(f"Processing file {n+1}/{len(filelist)}: {fname}")

    # Memory management: force garbage collection before each file
    gc.collect()
    
    # Load file
    data, rate = aio.load_audio(filepath)
    data = data[:, 0]  # Use only the first channel
    
    # Find peaks
    # pulse_heights = []
    pulse_widths = []
    pd_log_dict = {}
    # Detect pulses
    peaks, troughs, _ , pulse_widths = \
        pulses.detect_pulses(data[:, i], rate, 
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
        print(f"Analyzing {len(midpoints)} unique events...")

        # Analyze snippets with variable widths
        eod_waveforms, eod_amps, eod_widths, ch_amps,  ch_cor_coeffs, eod_chan, is_differential, final_peak_idc, final_trough_idc, final_midpoint_idc, original_pulse_orientation, amplitude_ratios, waveform_lengths = \
            extract_pulse_snippets(data, rate, unique_midpoints, unique_peaks, unique_troughs, 
                                       unique_widths, width_factor=parameters['width_fac_extraction'][0], 
                                       interp_factor=1, center_on_zero_crossing=False, return_diff=True)  # Skip centering for storage efficiency
        
        # 1. Create filter mask (only differential waveforms + amplitude ratio)
        # basic_filter_mask = (amplitude_ratios >= parameters['amplitude_ratio_min'][0]) & \
        #                    (amplitude_ratios <= parameters['amplitude_ratio_max'][0]) & \
        #                    (eod_widths <= parameters['max_width_s'][0]*1e6) & \
        #                    (eod_widths >= parameters['min_width_s'][0]*1e6)  # Ensure width is within limits
        # basic_keep_indices = np.where(basic_filter_mask)[0]

        basic_keep_indices = filter_waveforms(
            eod_waveforms, eod_widths, amplitude_ratios, rate,
            dur_min=parameters['min_width_s'][0]*1e6, 
            dur_max=parameters['max_width_s'][0]*1e6,
            pp_r_min=parameters['amplitude_ratio_min'][0], 
            pp_r_max=parameters['amplitude_ratio_max'][0],
            fft_freq_min=parameters['peak_fft_freq_min'][0], 
            fft_freq_max=parameters['peak_fft_freq_max'][0]
        )[0]        

        # 2. Apply noise removal to basic filtered events
        if len(basic_keep_indices) > 0:
            # Get variable-length waveforms for basic filtered events
            basic_filtered_waveforms = [eod_waveforms[i] for i in basic_keep_indices]
            basic_filtered_timestamps = final_midpoint_idc[basic_keep_indices] / rate
            
            if parameters['noise_removal'][0]:
                # Remove noise artifacts (now handles variable-length waveforms properly)
                noise_clean_mask = remove_noise_artifacts(
                    basic_filtered_waveforms,  # Pass list of variable-length waveforms
                    basic_filtered_timestamps, 
                    rate,
                    max_freq_content=parameters['max_freq_content'][0],  # Allow some high freq for species differences
                    min_snr=parameters['min_snr'][0],               # Minimum SNR for noise removal
                    max_ipi_ratio=parameters['max_ipi_ratio'][0]  # Maximum IPI ratio for noise removal
                )
                
                # Combine filters: basic + noise removal
                final_keep_indices = basic_keep_indices[noise_clean_mask]
                # keep_mask = np.zeros(len(eod_waveforms), dtype=bool)
                # keep_mask[final_keep_indices] = True
                keep_indices = final_keep_indices
            else:
                # keep_mask = basic_filter_mask
                keep_indices = basic_keep_indices
        else:
            # keep_mask = basic_filter_mask
            keep_indices = basic_keep_indices

        # Indices of all events after duplicate removal
        all_indices = np.arange(len(eod_waveforms))
        # Indices of filtered-out events
        filtered_out_indices = np.setdiff1d(all_indices, keep_indices)
        
        print(f"EODs after differential + amplitude ratio + noise filtering: {len(keep_indices)} out of {len(eod_waveforms)}")
        # print(f"  - Differential events: {np.sum(is_differential == 1)}")
        # print(f"  - Single-ended events (excluded): {np.sum(is_differential == 0)}")
        # print(f"  - Amplitude ratio and width passed: {np.sum((amplitude_ratios >= parameters['amplitude_ratio_min'][0]) & (amplitude_ratios <= parameters['amplitude_ratio_max'][0]) & (eod_widths//rate <= parameters['eod_width_max'][0]) & (eod_widths//rate >= parameters['eod_width_min'][0]))}")
        print(f"  - Amplitude ratio and width passed: {len(basic_keep_indices)}")
        if len(basic_keep_indices) > 0:
            print(f"  - Noise artifacts removed: {len(basic_keep_indices) - len(keep_indices)}")
        
        # 2. Filter all features and waveforms
        if len(keep_indices) > 0:
            # Filter all arrays
            filtered_eod_waveforms = [eod_waveforms[i] for i in keep_indices]  # List of variable-length waveforms
            filtered_eod_amps = eod_amps[keep_indices]
            filtered_eod_widths = eod_widths[keep_indices]
            filtered_ch_amps = ch_amps[keep_indices]
            filtered_ch_cor_coeffs = ch_cor_coeffs[keep_indices]
            filtered_eod_chan = eod_chan[keep_indices]
            # filtered_is_differential = is_differential[keep_indices]
            filtered_final_peak_idc = final_peak_idc[keep_indices]
            filtered_final_trough_idc = final_trough_idc[keep_indices]
            filtered_final_midpoints = final_midpoint_idc[keep_indices]
            # filtered_unique_peaks = unique_peaks[keep_indices]
            # filtered_unique_troughs = unique_troughs[keep_indices]
            filtered_unique_widths = unique_widths[keep_indices]
            filtered_original_pulse_orientation = original_pulse_orientation[keep_indices]
            filtered_waveform_lengths = waveform_lengths[keep_indices]
            
            # Create features DataFrame for filtered events
            durations_us = np.abs(filtered_final_peak_idc - filtered_final_trough_idc) / rate * 1e6
            filtered_features = pd.DataFrame({
                'pp_dur_us': durations_us,
                'pp_ratio': amplitude_ratios[keep_indices],
            })
            
            # print(f"After filtering: {len(filtered_original_pulse_orientation)} events with orientations: {np.unique(filtered_original_pulse_orientation, return_counts=True)}")
        else:
            # No events passed filtering - create empty arrays with correct structure
            print("No events passed amplitude ratio filtering. Creating empty results.")
            filtered_eod_waveforms = []  # Empty list for variable-length waveforms
            filtered_eod_amps = np.empty(0)
            filtered_eod_widths = np.empty(0)
            filtered_ch_amps = np.empty((0, n_channels))
            filtered_ch_cor_coeffs = np.empty((0, n_channels - 1))
            filtered_eod_chan = np.empty(0, dtype=int)
            # filtered_is_differential = np.empty(0, dtype=int)
            filtered_final_peak_idc = np.empty(0, dtype=int)
            filtered_final_trough_idc = np.empty(0, dtype=int)
            filtered_final_midpoints = np.empty(0, dtype=int)
            filtered_unique_peaks = np.empty(0, dtype=int)
            filtered_unique_troughs = np.empty(0, dtype=int)
            filtered_unique_widths = np.empty(0)
            filtered_original_pulse_orientation = np.empty(0, dtype='<U2')
            filtered_waveform_lengths = np.empty(0, dtype=int)
            filtered_features = pd.DataFrame(columns=['pp_dur_us', 'pp_ratio'])
        
        # 3. Create results DataFrame
        print(f"Creating event table with {len(filtered_final_midpoints)} events...")
        
        # Create timestamps for each event
        eod_timestamps = []
        for i in range(len(filtered_final_midpoints)):
            eod_timestamps.append(file_set['timestamp'][n] + dt.timedelta(seconds=filtered_final_midpoints[i]/rate))
        
        eod_table = pd.DataFrame({
            'timestamp': eod_timestamps,
            'midpoint_idx': filtered_final_midpoints,
            'peak_idx': filtered_final_peak_idc,
            'trough_idx': filtered_final_trough_idc,
            'eod_channel': filtered_eod_chan,
            # 'is_differential': filtered_is_differential,
            'eod_amplitude': filtered_eod_amps,
            'eod_width_uS': filtered_eod_widths,
            # 'pulse_width': filtered_unique_widths,
            'pulse_orientation': filtered_original_pulse_orientation,
            # 'original_peak_idx': filtered_unique_peaks,
            # 'original_trough_idx': filtered_unique_troughs
        })
        
        # Add amplitude information for each channel
        for ch in range(n_channels):
            eod_table[f'amplitude_ch{ch}'] = filtered_ch_amps[:, ch] if len(filtered_ch_amps) > 0 else []

        # Add correlation coefficients
        for ch in range(n_channels - 1):
            eod_table[f'correlation_ch{ch}_{ch+1}'] = filtered_ch_cor_coeffs[:, ch] if len(filtered_ch_cor_coeffs) > 0 else []

        # Add waveform length information
        eod_table['waveform_length'] = filtered_waveform_lengths if len(filtered_waveform_lengths) > 0 else []
        
        # Add extracted features
        eod_table = eod_table.join(filtered_features)
        
        # Save results (always save, even if empty)
        eod_table.to_csv(f'{output_path}\\{fname[:-4]}_eod_table.csv', index=False)
        
        # Save waveforms using efficient variable-length storage
        if len(filtered_eod_waveforms) > 0:
            # Calculate storage efficiency
            storage_metrics = calculate_storage_efficiency(filtered_eod_waveforms)
            
            # Save using new efficient format
            waveform_metadata = save_variable_length_waveforms(
                filtered_eod_waveforms, 
                f'{output_path}\\{fname[:-4]}_eod_waveforms'
            )
            
            # print(f"Saved {len(filtered_eod_waveforms)} variable-length waveforms efficiently")
            # print(f"  Total samples: {waveform_metadata['total_samples']:,}")
            # print(f"  Length range: {min(waveform_metadata['lengths'])} - {max(waveform_metadata['lengths'])} samples")
            # print(f"  Storage efficiency: {storage_metrics['efficiency_percent']:.1f}% space saved")
            # print(f"  Compression ratio: {storage_metrics['compression_ratio']:.2f}:1")
            # print(f"  Disk space saved: {storage_metrics['space_saved_mb']:.2f} MB")
            
            # Optionally save a compatibility CSV with zero-padded waveforms for legacy code
            # Only generate if explicitly requested to save disk space
            generate_legacy_csv = False  # Set to True if needed for compatibility
            if generate_legacy_csv:
                max_len = max(waveform_metadata['lengths'])
                padded_waveforms = np.zeros((len(filtered_eod_waveforms), max_len))
                for i, wf in enumerate(filtered_eod_waveforms):
                    padded_waveforms[i, :len(wf)] = wf
                pd.DataFrame(padded_waveforms).to_csv(f'{output_path}\\{fname[:-4]}_eod_waveforms_padded.csv', index=False)
                print(f"  Legacy CSV saved: {fname[:-4]}_eod_waveforms_padded.csv ({storage_metrics['padded_size_mb']:.2f} MB)")
        else:
            # Create empty files for consistency
            np.savez_compressed(f'{output_path}\\{fname[:-4]}_eod_waveforms_concatenated.npz', data=np.array([]))
            import json
            with open(f'{output_path}\\{fname[:-4]}_eod_waveforms_metadata.json', 'w') as f:
                json.dump({'lengths': [], 'start_indices': [], 'total_waveforms': 0, 'total_samples': 0}, f, separators=(',', ':'))
        
        print(f"Saved {len(eod_table)} filtered EOD events to {fname[:-4]}_eod_table.csv")
        
        # Control plots
        if len(filtered_final_midpoints) > 0:
            # Only differential events are kept after filtering
            # differential_idc = np.arange(len(filtered_is_differential))  # All events are differential now
            
            # Plot: Differential detections with differential data
            # if len(differential_idc) > 0:
            eod_idc = np.arange(len(filtered_eod_waveforms))
            data_diff = np.diff(data, axis=1)
            offset_diff = np.max(filtered_eod_amps) * 1.5
            
            plt.figure(figsize=(20, 8))
            for i in range(data_diff.shape[1]):
                # Find events detected on this differential channel
                ch_diff_idc = np.where(filtered_eod_chan == i)[0]
                actual_diff_idc = eod_idc[ch_diff_idc]
                
                # Plot only every nth sample for large datasets to save memory
                step = max(1, len(data_diff) // 15000000)  # Limit to ~15 mio points per channel
                x_coords = np.arange(0, len(data_diff), step)
                plt.plot(x_coords, data_diff[::step, i] + i * offset_diff, linewidth=0.5, label=f'Ch{i}-{i+1}')
                
                # Plot filtered (accepted) events
                if len(actual_diff_idc) > 0:
                    plt.plot(filtered_final_peak_idc[actual_diff_idc], 
                            data_diff[filtered_final_peak_idc[actual_diff_idc], i] + i * offset_diff, 
                            'o', markersize=1, color='red')
                    plt.plot(filtered_final_trough_idc[actual_diff_idc], 
                            data_diff[filtered_final_trough_idc[actual_diff_idc], i] + i * offset_diff, 
                            'o', markersize=1, color='blue')
                
                # Plot filtered-out events in grey
                if len(filtered_out_indices) > 0:
                    # Find filtered-out events detected on this differential channel
                    filteredout_eod_chan = eod_chan[filtered_out_indices]
                    filteredout_final_peak_idc = final_peak_idc[filtered_out_indices]
                    filteredout_final_trough_idc = final_trough_idc[filtered_out_indices]
                    
                    filteredout_ch_diff_idc = np.where(filteredout_eod_chan == i)[0]
                    if len(filteredout_ch_diff_idc) > 0:
                        plt.plot(filteredout_final_peak_idc[filteredout_ch_diff_idc], 
                                data_diff[filteredout_final_peak_idc[filteredout_ch_diff_idc], i] + i * offset_diff, 
                                'o', markersize=1, color='grey', alpha=0.6)
                        plt.plot(filteredout_final_trough_idc[filteredout_ch_diff_idc], 
                                data_diff[filteredout_final_trough_idc[filteredout_ch_diff_idc], i] + i * offset_diff, 
                                'o', markersize=1, color='grey', alpha=0.6)
            
            plt.ylim(bottom=None, top=(data_diff.shape[1]-0.5)*offset_diff)
            plt.title(f'{fname} - Differential EOD Detections - Red=Peaks, Blue=Troughs, Grey=Filtered Out (n={len(eod_idc)} kept, {len(filtered_out_indices)} filtered)')
            plt.legend(loc='upper right')
            plt.xlabel('Sample')
            plt.ylabel('Voltage')
            plt.savefig(f'{output_path}\\{fname[:-4]}_differential_detection_plot.png', dpi=150, bbox_inches='tight')
            plt.show(block=False)
            # plt.close()
            
            # Clear differential data from memory
            del data_diff
            gc.collect()
            
            # Summary analysis plots (simplified for memory efficiency)
            plt.figure(figsize=(12, 8))  # Reduced size
            
            # Check if we have any data to plot
            if len(filtered_eod_waveforms) == 0:
                plt.text(0.5, 0.5, 'No filtered events to display', ha='center', va='center', 
                        transform=plt.gcf().transFigure, fontsize=16)
                plt.savefig(f'{output_path}\\{fname[:-4]}_analysis_summary.png', dpi=150, bbox_inches='tight')
                plt.close()
                continue
            
            # Plot some example waveforms (limit to 25 for memory)
            plt.subplot(2, 3, 1)
            n_examples = min(25, len(filtered_eod_waveforms))
            for i in range(n_examples):
                plt.plot(filtered_eod_waveforms[i], alpha=0.4, linewidth=0.5)
            plt.title(f'Example EOD Waveforms (n={n_examples})')
            plt.xlabel('Sample')
            plt.ylabel('Normalized Amplitude')
            
            # Plot amplitude distribution
            plt.subplot(2, 3, 2)
            plt.hist(filtered_eod_amps, bins=20, alpha=0.7)  # Fewer bins
            plt.title(f'EOD Amplitude Distribution (n={len(filtered_eod_amps)})')
            plt.xlabel('Amplitude')
            plt.ylabel('Count')
            
            # Plot channel usage (only differential channels now)
            plt.subplot(2, 3, 3)
            if len(filtered_eod_waveforms) > 0:
                diff_channel_counts = np.bincount(filtered_eod_chan, minlength=n_channels-1)
                plt.bar(range(len(diff_channel_counts)), diff_channel_counts, alpha=0.7, 
                        label=f'Differential ({len(filtered_eod_chan)})', width=0.8)
                plt.title('Differential Channel Usage')
                plt.xlabel('Channel Pair Index')
                plt.ylabel('Count')
                plt.xticks(range(len(diff_channel_counts)), [f'{i}-{i+1}' for i in range(len(diff_channel_counts))], rotation=45)
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Differential Channel Usage')
                plt.xlabel('Channel Pair Index')
                plt.ylabel('Count')
            
            # Plot detection type distribution (only differential now)
            plt.subplot(2, 3, 4)
            if len(filtered_eod_waveforms) > 0:
                plt.bar([0], [len(filtered_eod_waveforms)], alpha=0.7, label='Differential')
                plt.title('Detection Type (All Differential)')
                plt.xlabel('Detection Type')
                plt.ylabel('Count')
                plt.xticks([0], ['Differential'])
                
                # Add count label on bar
                plt.text(0, len(filtered_eod_waveforms) + max(1, len(filtered_eod_waveforms)*0.01), 
                        str(len(filtered_eod_waveforms)), ha='center', va='bottom')
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Detection Type')
                plt.xlabel('Detection Type')
                plt.ylabel('Count')
            
            # Plot duration distribution (from detect_pulses, not filtering)
            plt.subplot(2, 3, 5)
            plt.hist(filtered_features['pp_dur_us'], bins=20, alpha=0.7)
            plt.title(f'Pulse Duration Distribution (n={len(filtered_features)})')
            plt.xlabel('Duration (μs)')
            plt.ylabel('Count')
            
            # Plot pulse orientation distribution
            plt.subplot(2, 3, 6)
            if len(filtered_original_pulse_orientation) > 0:
                orientation_counts = pd.Series(filtered_original_pulse_orientation).value_counts()
                bars = plt.bar(orientation_counts.index, orientation_counts.values, alpha=0.7)
                plt.title('Pulse Orientation')
                plt.xlabel('Orientation')
                plt.ylabel('Count')
                
                # Add count labels on bars
                for bar, count in zip(bars, orientation_counts.values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(orientation_counts.values)*0.01, 
                            str(int(count)), ha='center', va='bottom')
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Pulse Orientation')
                plt.xlabel('Orientation')
                plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(f'{output_path}\\{fname[:-4]}_analysis_summary.png', dpi=150, bbox_inches='tight')  # Reduced DPI
            plt.close()

        
        # --- Save filtered-out events for QC if requested ---
        if len(filtered_out_indices) > 0:
            filteredout_eod_waveforms = [eod_waveforms[i] for i in filtered_out_indices]
            filteredout_ch_amps = ch_amps[filtered_out_indices]
            filteredout_eod_amp = eod_amps[filtered_out_indices]
            filteredout_eod_widths = eod_widths[filtered_out_indices]
            filteredout_ch_cor_coeffs = ch_cor_coeffs[filtered_out_indices]
            filteredout_eod_chan = eod_chan[filtered_out_indices]
            filteredout_is_differential = is_differential[filtered_out_indices]
            filteredout_final_peak_idc = final_peak_idc[filtered_out_indices]
            filteredout_final_trough_idc = final_trough_idc[filtered_out_indices]
            filteredout_unique_midpoints = unique_midpoints[filtered_out_indices]
            filteredout_unique_peaks = unique_peaks[filtered_out_indices]
            filteredout_unique_troughs = unique_troughs[filtered_out_indices]
            filteredout_unique_widths = unique_widths[filtered_out_indices]
            filteredout_original_pulse_orientation = original_pulse_orientation[filtered_out_indices]
            filteredout_waveform_lengths = waveform_lengths[filtered_out_indices]
            # Features
            filteredout_durations_us = np.abs(filteredout_final_peak_idc - filteredout_final_trough_idc) / rate * 1e6
            filteredout_features = pd.DataFrame({
                'pp_dur_us': filteredout_durations_us,
                'pp_ratio': amplitude_ratios[filtered_out_indices],
            })
            # Timestamps
            filteredout_eod_timestamps = [file_set['timestamp'][n] + dt.timedelta(seconds=filteredout_unique_midpoints[i]/rate) for i in range(len(filteredout_unique_midpoints))]
            # Table
            filteredout_eod_table = pd.DataFrame({
                'timestamp': filteredout_eod_timestamps,
                'midpoint_idx': filteredout_unique_midpoints,
                'peak_idx': filteredout_final_peak_idc,
                'trough_idx': filteredout_final_trough_idc,
                'eod_channel': filteredout_eod_chan,
                # 'is_differential': filteredout_is_differential,
                'eod_amplitude': filteredout_eod_amp,
                'eod_width_uS': filteredout_eod_widths,
                'pulse_orientation': filteredout_original_pulse_orientation,
                # 'original_peak_idx': filteredout_unique_peaks,
                # 'original_trough_idx': filteredout_unique_troughs
            })
            for ch in range(n_channels):
                filteredout_eod_table[f'amplitude_ch{ch}'] = filteredout_ch_amps[:, ch] if len(filteredout_ch_amps) > 0 else []
            for ch in range(n_channels - 1):
                filteredout_eod_table[f'correlation_ch{ch}_{ch+1}'] = filteredout_ch_cor_coeffs[:, ch] if len(filteredout_ch_cor_coeffs) > 0 else []
            filteredout_eod_table['waveform_length'] = filteredout_waveform_lengths if len(filteredout_waveform_lengths) > 0 else []
            filteredout_eod_table = filteredout_eod_table.join(filteredout_features)

            plot_waveform_comparison(filtered_eod_waveforms, filteredout_eod_waveforms, output_path, fname[:-4])
            compare_table_features(eod_table, filteredout_eod_table, output_path, fname[:-4])

            if parameters['save_filtered_out'][0]:
                # Save table and waveforms
                filteredout_eod_table.to_csv(f'{output_path}\\{fname[:-4]}_eod_table_filteredout.csv', index=False)
                if len(filteredout_eod_waveforms) > 0:
                    save_variable_length_waveforms(filteredout_eod_waveforms, f'{output_path}\\{fname[:-4]}_eod_waveforms_filteredout')
                print(f"Saved {len(filteredout_eod_waveforms)} filtered-out EOD events for QC.")
# ...existing code...