# import thunderfish.eventdetection as ed

import thunderfish.pulses as pulses
import matplotlib.pyplot as plt
import audioio as aio
from scipy.signal import find_peaks, correlate, windows
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
import glob
import datetime as dt
# from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

def save_variable_length_waveforms(waveforms_list, output_path):
    """
    Save variable-length waveforms efficiently without zero-padding.
    Optimized to minimize disk space and I/O operations.
    
    Parameters
    ----------
    waveforms_list : list of 1D arrays
        List of waveforms with potentially different lengths
    output_path : str
        Path to save the waveforms
    
    Returns
    -------
    metadata : dict
        Information about the saved waveforms for reconstruction
    """
    # Filter out empty waveforms and track original indices
    non_empty_waveforms = []
    non_empty_indices = []
    lengths = []
    
    for i, wf in enumerate(waveforms_list):
        if len(wf) > 0:
            non_empty_waveforms.append(wf)
            non_empty_indices.append(i)
            lengths.append(len(wf))
        else:
            lengths.append(0)
    
    if non_empty_waveforms:
        # Calculate start indices for non-empty waveforms
        start_indices = np.cumsum([0] + lengths[:-1] if lengths else [0])
        
        # Concatenate all non-empty waveforms
        concatenated = np.concatenate(non_empty_waveforms)
        
        # Use float32 instead of float64 for space savings (if precision allows)
        if concatenated.dtype == np.float64:
            # Check if we can safely convert to float32
            max_val = np.max(np.abs(concatenated))
            if max_val < 1e6:  # Safe range for float32
                concatenated = concatenated.astype(np.float32)
        
        # Save concatenated data with compression
        np.savez_compressed(output_path + '_concatenated.npz', data=concatenated)
    else:
        # Handle case with no valid waveforms
        np.savez_compressed(output_path + '_concatenated.npz', data=np.array([], dtype=np.float32))
        start_indices = np.array([])
    
    metadata = {
        'lengths': lengths,
        'start_indices': start_indices.tolist() if len(start_indices) > 0 else [],
        'non_empty_indices': non_empty_indices,
        'total_waveforms': len(waveforms_list),
        'total_samples': len(concatenated) if non_empty_waveforms else 0,
        'dtype': str(concatenated.dtype) if non_empty_waveforms else 'float32',
        'space_savings': {
            'original_padded_size_estimate': len(waveforms_list) * max(lengths) if lengths else 0,
            'actual_size': len(concatenated) if non_empty_waveforms else 0,
            'compression_ratio': (len(waveforms_list) * max(lengths)) / len(concatenated) if non_empty_waveforms and lengths else 1.0
        }
    }
    
    # Save metadata with compact JSON
    import json
    with open(output_path + '_metadata.json', 'w') as f:
        json.dump(metadata, f, separators=(',', ':'))  # Compact format
    
    return metadata

def load_variable_length_waveforms(base_path):
    """
    Load variable-length waveforms from efficient storage format.
    Optimized for memory efficiency and fast loading.
    
    Parameters
    ----------
    base_path : str
        Base path (without extensions) for the saved files
    
    Returns
    -------
    waveforms_list : list of 1D arrays
        List of reconstructed waveforms
    """
    import json
    
    # Load concatenated data and metadata
    try:
        data_file = np.load(base_path + '_concatenated.npz')
        concatenated = data_file['data']
    except:
        # Fallback to old format
        concatenated = np.load(base_path + '_concatenated.npy')
    
    with open(base_path + '_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Reconstruct individual waveforms
    waveforms_list = []
    lengths = metadata['lengths']
    
    # Handle case where some waveforms were empty
    if 'non_empty_indices' in metadata:
        non_empty_indices = metadata['non_empty_indices']
        concatenated_idx = 0
        
        for i in range(metadata['total_waveforms']):
            if lengths[i] == 0:
                waveforms_list.append(np.array([]))
            else:
                length = lengths[i]
                waveform = concatenated[concatenated_idx:concatenated_idx + length]
                waveforms_list.append(waveform)
                concatenated_idx += length
    else:
        # Original format - all waveforms were non-empty
        start_indices = metadata['start_indices']
        for i in range(metadata['total_waveforms']):
            start_idx = start_indices[i]
            length = lengths[i]
            waveform = concatenated[start_idx:start_idx + length]
            waveforms_list.append(waveform)
    
    return waveforms_list

def calculate_storage_efficiency(waveforms_list, padded_format_length=None):
    """
    Calculate storage efficiency metrics for variable-length vs zero-padded storage.
    
    Parameters
    ----------
    waveforms_list : list of 1D arrays
        List of variable-length waveforms
    padded_format_length : int, optional
        Length that would be used for zero-padded storage (max length if None)
    
    Returns
    -------
    metrics : dict
        Storage efficiency metrics
    """
    if not waveforms_list:
        return {'compression_ratio': 1.0, 'space_saved_mb': 0.0, 'efficiency_percent': 0.0}
    
    lengths = [len(wf) for wf in waveforms_list]
    max_length = max(lengths) if lengths else 0
    
    if padded_format_length is None:
        padded_format_length = max_length
    
    # Calculate sizes
    variable_length_samples = sum(lengths)
    padded_samples = len(waveforms_list) * padded_format_length
    
    # Assuming float32 (4 bytes per sample)
    bytes_per_sample = 4
    variable_size_mb = variable_length_samples * bytes_per_sample / (1024**2)
    padded_size_mb = padded_samples * bytes_per_sample / (1024**2)
    
    compression_ratio = padded_samples / variable_length_samples if variable_length_samples > 0 else 1.0
    space_saved_mb = padded_size_mb - variable_size_mb
    efficiency_percent = (space_saved_mb / padded_size_mb * 100) if padded_size_mb > 0 else 0.0
    
    return {
        'compression_ratio': compression_ratio,
        'space_saved_mb': space_saved_mb,
        'efficiency_percent': efficiency_percent,
        'variable_size_mb': variable_size_mb,
        'padded_size_mb': padded_size_mb,
        'total_waveforms': len(waveforms_list),
        'length_stats': {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': np.mean(lengths) if lengths else 0,
            'std': np.std(lengths) if lengths else 0
        }
    }

def analyze_waveform_fft_proper(waveform, rate, max_freq_content=0.8):
    """
    Proper FFT analysis without zero-padding artifacts.
    
    Parameters
    ----------
    waveform : 1D array
        Original waveform (no zero padding)
    rate : float
        Sampling rate
    max_freq_content : float
        Maximum allowed high-frequency content (0-1, where 1 = Nyquist)
    
    Returns
    -------
    is_noisy : bool
        True if waveform is likely noise (high frequency content)
    freq_stats : dict
        Frequency domain statistics
    """
    if len(waveform) < 10:
        return False, {}
    
    # Remove any zero-padding artifacts by trimming trailing zeros
    # (in case waveform still has padding from somewhere)
    non_zero_mask = waveform != 0
    if np.any(non_zero_mask):
        first_non_zero = np.argmax(non_zero_mask)
        last_non_zero = len(waveform) - 1 - np.argmax(non_zero_mask[::-1])
        waveform_clean = waveform[first_non_zero:last_non_zero+1]
    else:
        return True, {}  # All zeros = noise
    
    # Apply window to reduce spectral leakage (important for short signals)
    if len(waveform_clean) > 4:
        # Use Hann window for better frequency analysis
        window = windows.hann(len(waveform_clean))
        windowed_waveform = waveform_clean * window
    else:
        windowed_waveform = waveform_clean
    
    # FFT analysis on clean, windowed signal
    fft = np.fft.fft(windowed_waveform)
    freqs = np.fft.fftfreq(len(windowed_waveform), 1/rate)
    power_spectrum = np.abs(fft)**2
    
    # Calculate high-frequency content
    nyquist = rate / 2
    high_freq_mask = np.abs(freqs) > (max_freq_content * nyquist)
    high_freq_power = np.sum(power_spectrum[high_freq_mask])
    total_power = np.sum(power_spectrum)
    
    # Calculate other useful metrics
    freq_stats = {
        'high_freq_ratio': high_freq_power / total_power if total_power > 0 else 0,
        'dominant_freq': np.abs(freqs[np.argmax(power_spectrum[len(power_spectrum)//2:]) + len(power_spectrum)//2]),
        'spectral_centroid': np.sum(np.abs(freqs) * power_spectrum) / total_power if total_power > 0 else 0,
        'effective_length': len(waveform_clean),
        'window_applied': len(waveform_clean) > 4
    }
    
    # Conservative threshold: >35% high freq content suggests noise
    is_noisy = (freq_stats['high_freq_ratio'] > 0.35) if total_power > 0 else False
    
    return is_noisy, freq_stats

def remove_noise_artifacts(waveforms, timestamps, rate, 
                          max_freq_content=0.8, min_snr=2.0, max_ipi_ratio=50.0):
    """
    Remove obvious noise artifacts from EOD waveforms (per-file processing).
    Optimized for variable-length waveforms with improved efficiency.
    These filters target noise that is consistently identifiable regardless of fish species.
    
    Parameters
    ----------
    waveforms : list of 1D arrays
        Variable-length EOD waveforms
    timestamps : 1D array
        Timestamp for each waveform in seconds
    rate : float
        Sampling rate
    max_freq_content : float
        Maximum allowed high-frequency content (0-1, where 1 = Nyquist)
    min_snr : float
        Minimum signal-to-noise ratio
    max_ipi_ratio : float
        Maximum ratio of pulse width to inter-pulse interval
    
    Returns
    -------
    clean_mask : 1D boolean array
        Mask of waveforms to keep (True = keep, False = remove as noise)
    """
    n_events = len(waveforms)
    clean_mask = np.ones(n_events, dtype=bool)
    
    if n_events == 0:
        return clean_mask
    
    # Pre-calculate some stats to avoid redundant computation
    waveform_lengths = np.array([len(wf) for wf in waveforms])
    valid_waveforms_mask = waveform_lengths > 5  # Need minimum length for analysis
    
    # 1. Remove high-frequency artifacts (electrical noise) - OPTIMIZED FFT ANALYSIS
    for i in range(n_events):
        if valid_waveforms_mask[i]:
            waveform = waveforms[i]
            is_noisy, freq_stats = analyze_waveform_fft_proper(waveform, rate, max_freq_content)
            if is_noisy:
                clean_mask[i] = False
        else:
            # Waveforms too short are likely artifacts
            clean_mask[i] = False
    
    # 2. Remove low SNR waveforms (vectorized where possible)
    if np.sum(clean_mask) > 0:
        remaining_indices = np.where(clean_mask)[0]
        
        for idx in remaining_indices:
            waveform = waveforms[idx]
            if len(waveform) > 20:  # Need enough samples for reliable SNR estimation
                # Estimate SNR (signal variance vs noise variance)
                signal_var = np.var(waveform)
                # Estimate noise from high-frequency components using second derivative
                noise_est = np.median(np.abs(np.diff(waveform, n=2)))
                snr_est = signal_var / (noise_est**2 + 1e-10)
                if snr_est < min_snr:
                    clean_mask[idx] = False
    
    # 3. Remove unrealistic inter-pulse intervals (only if we have enough events)
    remaining_count = np.sum(clean_mask)
    if remaining_count > 5:
        clean_timestamps = timestamps[clean_mask]
        ipis = np.diff(clean_timestamps)
        
        if len(ipis) > 0:
            # Calculate median pulse characteristics for this file
            median_ipi = np.median(ipis)
            
            # Remove events with unrealistic IPI ratios
            remaining_indices = np.where(clean_mask)[0]
            for i, orig_idx in enumerate(remaining_indices[:-1]):  # Exclude last event
                if i < len(ipis):
                    if ipis[i] < median_ipi / max_ipi_ratio or ipis[i] > median_ipi * max_ipi_ratio:
                        clean_mask[orig_idx] = False
    
    return clean_mask

def extract_pulse_snippets(data, unique_midpoints, unique_peaks, unique_troughs, unique_widths, 
                                rate, width_factor=5.0, interp_factor=1, center_on_zero_crossing=False):
    """
    Extract and analyze EOD snippets with variable widths based on detected pulse widths.
    Optimized to store variable-length waveforms without zero-padding for maximum efficiency.
    
    Parameters
    ----------
    data : 2-D array
        The full recording data with channels in columns
    unique_midpoints : 1-D array
        Midpoint indices of unique events
    unique_peaks : 1-D array
        Peak indices of unique events
    unique_troughs : 1-D array
        Trough indices of unique events
    unique_widths : 1-D array
        Width (in seconds) of unique events
    rate : int
        Sampling rate
    width_factor : float
        Factor to multiply width to get snippet length
    interp_factor : int
        Interpolation factor for final waveforms
    center_on_zero_crossing : bool
        Whether to center waveforms on zero-crossing (False for storage efficiency)
    
    Returns
    -------
    eod_waveforms : list of 1-D arrays
        Variable-length EOD waveform snippets (no zero-padding)
    amps : 2-D array
        Max amplitudes across channels for each snippet
    eod_amp : 1-D array
        Amplitude of extracted waveform
    cor_coeffs : 2-D array
        Correlation coefficients between adjacent channels
    eod_chan : 1-D array
        Index of channel/channel-pair used for waveform extraction
    is_differential : 1-D array
        Indicator of differential vs single-ended waveform (1=differential, 0=single-ended)
    final_peak_idc : 1-D array
        Final peak indices in original data (for normalized head-positive orientation)
    final_trough_idc : 1-D array
        Final trough indices in original data (for normalized head-positive orientation)
    pulse_orientation : 1-D array
        Original pulse orientation before normalization ('HP' or 'HN')
    amplitude_ratios : 1-D array
        Peak-to-trough amplitude ratios for filtering
    waveform_lengths : 1-D array
        Length of each variable-length waveform
    """
    n_channels = data.shape[1]
    n_events = len(unique_midpoints)
    
    # Preallocate arrays
    eod_waveforms = []  # Store as list for variable lengths
    amps = np.zeros((n_events, n_channels))
    eod_amp = np.zeros(n_events)
    cor_coeffs = np.zeros((n_events, n_channels - 1))
    eod_chan = np.zeros(n_events, dtype=int)
    is_differential = np.ones(n_events, dtype=int)  # 1=differential, 0=single-ended
    final_peak_idc = np.zeros(n_events, dtype=int)
    final_trough_idc = np.zeros(n_events, dtype=int)
    pulse_orientation = np.array(['HP'] * n_events)  # Store original orientation
    amplitude_ratios = np.zeros(n_events)  # For amplitude ratio filtering
    waveform_lengths = np.zeros(n_events, dtype=int)  # Track actual lengths
    
    for i in range(n_events):
        # Calculate snippet length based on width - OPTIMIZED
        snippet_samples = int(unique_widths[i] * width_factor)
        snippet_samples = max(snippet_samples, 20)  # Minimum 20 samples
        
        # Extract snippet around midpoint - OPTIMIZED (reduce padding needs)
        center_idx = int(unique_midpoints[i])
        half_len = snippet_samples // 2
        
        # Calculate bounds and extract only necessary data (no pre-padding)
        start_idx = max(0, center_idx - half_len)
        end_idx = min(data.shape[0], center_idx + half_len)
        actual_length = end_idx - start_idx
        
        # Extract actual data without padding initially - MAKE A COPY to avoid modifying original data
        snippet = data[start_idx:end_idx, :].copy()
        
        # Only pad if absolutely necessary and keep track of padding
        padding_needed = snippet_samples - actual_length
        padding_start = max(0, -center_idx + half_len)  # How much padding at start
        padding_end = max(0, center_idx + half_len - data.shape[0])  # How much padding at end
        
        if padding_needed > 0:
            # Minimal padding - only what's needed
            if padding_start > 0:
                snippet = np.pad(snippet, ((padding_start, 0), (0, 0)), mode='constant')
            if padding_end > 0:
                snippet = np.pad(snippet, ((0, padding_end), (0, 0)), mode='constant')
        
        # Calculate amplitudes for each channel
        for j in range(n_channels):
            if snippet.shape[0] > 0:
                p_idx = np.argmax(snippet[:, j])
                t_idx = np.argmin(snippet[:, j])
                amps[i, j] = abs(snippet[p_idx, j] - snippet[t_idx, j])
        
        # Calculate correlation coefficients between adjacent channels
        if snippet.shape[0] > 1:  # Need at least 2 samples for correlation
            for j in range(n_channels - 1):
                if np.var(snippet[:, j]) > 0 and np.var(snippet[:, j+1]) > 0:
                    cor_coeffs[i, j] = np.corrcoef(snippet[:, j], snippet[:, j+1])[0, 1]
        
        # Compute differential signals only where needed
        if n_channels > 1:
            snippet_diff = np.diff(snippet, axis=1)
            amps_diff = np.zeros(n_channels - 1)
            
            for j in range(n_channels - 1):
                if snippet_diff.shape[0] > 0:
                    p_idx = np.argmax(snippet_diff[:, j])
                    t_idx = np.argmin(snippet_diff[:, j])
                    amps_diff[j] = abs(snippet_diff[p_idx, j] - snippet_diff[t_idx, j])
            
            # Find polarity flips (negative correlations)
            flips = np.where(cor_coeffs[i, :] < 0)[0]
            
            if len(flips) > 1:
                # Multiple flips: choose the one with largest amplitude difference
                eod_chan[i] = flips[np.argmax(amps_diff[flips])]
            elif len(flips) == 1:
                # Single flip: use it
                eod_chan[i] = flips[0]
            else:
                # No flip: use channel with largest single-ended amplitude
                best_channel = np.argmax(amps[i, :])
                eod_chan[i] = best_channel
                is_differential[i] = 0
                
            # Extract the final waveform - ALWAYS USE COPY to avoid modifying original data
            if len(flips) > 0:
                # Use differential signal
                eod_waveform = snippet_diff[:, eod_chan[i]].copy()
            else:
                # Use single-ended signal
                eod_waveform = snippet[:, eod_chan[i]].copy()
        else:
            # Single channel case
            eod_waveform = snippet[:, 0].copy()
            eod_chan[i] = 0
            is_differential[i] = 0
        
        # Find peak and trough in the selected waveform
        if len(eod_waveform) > 0:
            eod_peak_idx = np.argmax(eod_waveform)
            eod_trough_idx = np.argmin(eod_waveform)
            eod_amp[i] = abs(eod_waveform[eod_peak_idx] - eod_waveform[eod_trough_idx])
            
            # Calculate amplitude ratio for filtering
            max_val = np.max(eod_waveform)
            min_val = np.min(eod_waveform)
            amplitude_ratios[i] = abs(max_val / min_val) if min_val != 0 else np.inf
            
            # Determine pulse orientation based on peak and trough indices
            if eod_trough_idx < eod_peak_idx:
                pulse_orientation[i] = 'HN'  # Head-negative (trough before peak)
            else:
                pulse_orientation[i] = 'HP'  # Head-positive (peak before trough)
            
            # Normalize orientation (peak before trough) for consistent waveform analysis
            if eod_trough_idx < eod_peak_idx:
                eod_waveform *= -1
                eod_peak_idx, eod_trough_idx = eod_trough_idx, eod_peak_idx
            
            # Interpolate if needed
            if interp_factor > 1:
                interp_samples = len(eod_waveform) * interp_factor
                if len(eod_waveform) > 1:
                    interper = interp1d(np.arange(len(eod_waveform)), eod_waveform, kind='linear')
                    eod_waveform = interper(np.linspace(0, len(eod_waveform)-1, interp_samples))
            
            # Optional: Center the waveform on the zero-crossing between peak and trough
            # Skip this for storage efficiency unless specifically requested
            if center_on_zero_crossing and len(eod_waveform) > 2:
                zero_crossing_pos = eod_peak_idx  # fallback to peak if no zero-crossing found
                for j in range(eod_peak_idx, min(eod_trough_idx, len(eod_waveform)-1)):
                    if (eod_waveform[j] >= 0 and eod_waveform[j+1] <= 0) or \
                       (eod_waveform[j] <= 0 and eod_waveform[j+1] >= 0):
                        # Interpolate to find exact zero-crossing position
                        if eod_waveform[j+1] != eod_waveform[j]:
                            zero_crossing_pos = j - eod_waveform[j] / (eod_waveform[j+1] - eod_waveform[j])
                        else:
                            zero_crossing_pos = j
                        break
                
                center_pos = len(eod_waveform) // 2
                shift = center_pos - int(zero_crossing_pos)
                if abs(shift) < len(eod_waveform):  # Avoid excessive shifts
                    eod_waveform = np.roll(eod_waveform, shift)
            
            # Normalize amplitude
            max_abs = np.max(np.abs(eod_waveform))
            if max_abs > 0:
                eod_waveform /= max_abs
            
            # Store variable-length waveform and its length
            eod_waveforms.append(eod_waveform)
            waveform_lengths[i] = len(eod_waveform)
            
            # Store final indices (adjust for snippet position)
            final_peak_idc[i] = start_idx + eod_peak_idx
            final_trough_idc[i] = start_idx + eod_trough_idx
        else:
            # Empty waveform case
            eod_waveforms.append(np.array([]))
            waveform_lengths[i] = 0
            final_peak_idc[i] = unique_peaks[i]
            final_trough_idc[i] = unique_troughs[i]
    
    # Return variable-length waveforms as list (no zero-padding)
    return (eod_waveforms, amps, eod_amp, cor_coeffs, eod_chan, 
            is_differential, final_peak_idc, final_trough_idc, pulse_orientation, 
            amplitude_ratios, waveform_lengths)

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Logger Files")
output_path = filedialog.askdirectory(title="Select Folder to Store Analysis Results")
cal_file = filedialog.askopenfilename(title="Select File with Calibration Data")
cor_factors = np.array(pd.read_csv(cal_file))

# List all .wav files
filelist = glob.glob(input_path + '/*.wav', recursive=True)

# Sort filelist by time
timecol = [pd.to_datetime(fname.split('-')[1][0:-4], format='%Y%m%dT%H%M%S') for fname in filelist]
timecol = pd.to_datetime(timecol)
file_set = pd.DataFrame({'timestamp': timecol, 'filename': filelist})
file_set = file_set.sort_values(by=['timestamp'], ignore_index=True)

# Load 60 sec of first file
tmin, tmax = 0, 60
with aio.AudioLoader(file_set['filename'][0], 60) as sf:
    rate = sf.rate
    data = sf[int(tmin * rate):int(tmax * rate), :]
n_channels = data.shape[1]

# Calibrate with correction factor from .csv
thresh = 0
for i in range(n_channels):
    data[:, i] *= cor_factors[i, 1]
    sd = np.std(data[:, i])
    thresh = max(thresh, sd)
thresh *= 10

# Plot raw data
offset = np.max(abs(data))
plt.figure(figsize=(40, 12))
for i in range(n_channels):
    plt.plot(data[0:int(60 * rate - 1), i] + i * offset, label=str(i + 1))
    plt.hlines(y=thresh + i * offset, xmin=0, xmax=int(60 * rate - 1))
plt.legend(loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Voltage')
plt.savefig('%s\\%s_one_minute_raw.png' % (output_path, file_set['filename'][0].split('\\')[-1][:-4]))
plt.show(block=False)

# Pulse extraction parameters
parameters = {'thresh':thresh,
              'min_rel_slope_diff':0.25,
              'min_width_s':5e-05,  # Minimum pulse width in seconds
              'max_width_s':0.001,  # Maximum pulse width in seconds
              'width_fac':5.0,
              'verbose':0,
              'return_data':False,
              # Additional filtering parameters
              'amplitude_ratio_min':0.1,  # Minimum peak-to-peak amplitude ratio
              'amplitude_ratio_max':3     # Maximum peak-to-peak amplitude ratio
              }

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
for n, filepath in enumerate(file_set['filename']):
    fname = filepath.split('\\')[-1]
    print(f"Processing file {n+1}/{len(file_set)}: {fname}")
    
    # Memory management: force garbage collection before each file
    gc.collect()
    
    # Load file
    data, rate = aio.load_audio(filepath)
    n_channels = data.shape[1]
    
    # Calibrate with correction factor
    for i in range(n_channels):
        data[:, i] *= cor_factors[i, 1]
    
    # Find peaks in all channels
    peaks = []
    troughs = []
    eod_heights = []
    eod_widths = []
    pd_log_dict = {}
    for i in range(n_channels):
        # Detect pulses
        ch_peaks, ch_troughs, ch_eod_heights, ch_eod_widths = \
            pulses.detect_pulses(data[:, i], rate, 
                                 thresh = parameters['thresh'][0], 
                                 min_rel_slope_diff=parameters['min_rel_slope_diff'][0],
                                 min_width=parameters['min_width_s'][0],  # in seconds
                                 max_width=parameters['max_width_s'][0],  # in seconds
                                 width_fac=parameters['width_fac'][0],
                                 verbose=parameters['verbose'][0],
                                 return_data=parameters['return_data'][0])
        peaks.append(ch_peaks)
        troughs.append(ch_troughs)
        eod_heights.append(ch_eod_heights)
        eod_widths.append(ch_eod_widths)

    # --- Unify events across channels and keep corresponding peaks/troughs/widths ---
    # Collect (midpoint, peak, trough, width) tuples for all channels
    all_events = []
    for ch_peaks, ch_troughs, ch_widths in zip(peaks, troughs, eod_widths):
        n_pairs = min(len(ch_peaks), len(ch_troughs), len(ch_widths))
        for j in range(n_pairs):
            mp = (ch_peaks[j] + ch_troughs[j]) // 2
            all_events.append((mp, ch_peaks[j], ch_troughs[j], ch_widths[j]))

    # Sort by midpoint
    all_events.sort(key=lambda x: x[0])

    # Remove duplicates within 3 samples, keeping the first occurrence
    unique_events = []
    if all_events:
        last_mp = all_events[0][0]
        unique_events.append(all_events[0])
        for event in all_events[1:]:
            if event[0] - last_mp > 3:
                unique_events.append(event)
                last_mp = event[0]
        del all_events
        gc.collect()
    
    # Unpack to arrays for further analysis
    if unique_events:
        unique_midpoints = np.array([e[0] for e in unique_events])
        unique_peaks = np.array([e[1] for e in unique_events])
        unique_troughs = np.array([e[2] for e in unique_events])
        unique_widths = np.array([e[3] for e in unique_events])
    else:
        unique_midpoints = np.array([])
        unique_peaks = np.array([])
        unique_troughs = np.array([])
        unique_widths = np.array([])
        
    # Extract variable-width snippets and analyze them
    if len(unique_midpoints) > 0:
        print(f"Analyzing {len(unique_midpoints)} unique events...")
        
        # Analyze snippets with variable widths
        eod_waveforms, amps, eod_amp, cor_coeffs, eod_chan, is_differential, final_peak_idc, final_trough_idc, original_pulse_orientation, amplitude_ratios, waveform_lengths = \
            extract_pulse_snippets(data, unique_midpoints, unique_peaks, unique_troughs, 
                                       unique_widths, rate, width_factor=50.0, interp_factor=1, 
                                       center_on_zero_crossing=False)  # Skip centering for storage efficiency
        
        # 1. Create filter mask (only differential waveforms + amplitude ratio)
        basic_filter_mask = (amplitude_ratios >= parameters['amplitude_ratio_min'][0]) & \
                           (amplitude_ratios <= parameters['amplitude_ratio_max'][0]) & \
                           (is_differential == 1)
        basic_keep_indices = np.where(basic_filter_mask)[0]
        
        # 2. Apply noise removal to basic filtered events
        if len(basic_keep_indices) > 0:
            # Get variable-length waveforms for basic filtered events
            basic_filtered_waveforms = [eod_waveforms[i] for i in basic_keep_indices]
            basic_filtered_timestamps = unique_midpoints[basic_keep_indices] / rate
            
            # Remove noise artifacts (now handles variable-length waveforms properly)
            noise_clean_mask = remove_noise_artifacts(
                basic_filtered_waveforms,  # Pass list of variable-length waveforms
                basic_filtered_timestamps, 
                rate,
                max_freq_content=0.75,  # Allow some high freq for species differences
                min_snr=2.0,
                max_ipi_ratio=20.0
            )
            
            # Combine filters: basic + noise removal
            final_keep_indices = basic_keep_indices[noise_clean_mask]
            keep_mask = np.zeros(len(eod_waveforms), dtype=bool)
            keep_mask[final_keep_indices] = True
            keep_indices = final_keep_indices
        else:
            keep_mask = basic_filter_mask
            keep_indices = basic_keep_indices
        
        print(f"EODs after differential + amplitude ratio + noise filtering: {len(keep_indices)} out of {len(eod_waveforms)}")
        print(f"  - Differential events: {np.sum(is_differential == 1)}")
        print(f"  - Single-ended events (excluded): {np.sum(is_differential == 0)}")
        print(f"  - Amplitude ratio passed: {np.sum((amplitude_ratios >= parameters['amplitude_ratio_min'][0]) & (amplitude_ratios <= parameters['amplitude_ratio_max'][0]))}")
        if len(basic_keep_indices) > 0:
            print(f"  - Noise artifacts removed: {len(basic_keep_indices) - len(keep_indices)}")
        
        # 2. Filter all features and waveforms
        if len(keep_indices) > 0:
            # Filter all arrays
            filtered_eod_waveforms = [eod_waveforms[i] for i in keep_indices]  # List of variable-length waveforms
            filtered_amps = amps[keep_indices]
            filtered_eod_amp = eod_amp[keep_indices]
            filtered_cor_coeffs = cor_coeffs[keep_indices]
            filtered_eod_chan = eod_chan[keep_indices]
            filtered_is_differential = is_differential[keep_indices]
            filtered_final_peak_idc = final_peak_idc[keep_indices]
            filtered_final_trough_idc = final_trough_idc[keep_indices]
            filtered_unique_midpoints = unique_midpoints[keep_indices]
            filtered_unique_peaks = unique_peaks[keep_indices]
            filtered_unique_troughs = unique_troughs[keep_indices]
            filtered_unique_widths = unique_widths[keep_indices]
            filtered_original_pulse_orientation = original_pulse_orientation[keep_indices]
            filtered_waveform_lengths = waveform_lengths[keep_indices]
            
            # Create features DataFrame for filtered events
            durations_us = np.abs(filtered_final_peak_idc - filtered_final_trough_idc) / rate * 1e6
            filtered_features = pd.DataFrame({
                'pp_dur_us': durations_us,
                'pp_ratio': amplitude_ratios[keep_indices],
            })
            
            print(f"After filtering: {len(filtered_original_pulse_orientation)} events with orientations: {np.unique(filtered_original_pulse_orientation, return_counts=True)}")
        else:
            # No events passed filtering - create empty arrays with correct structure
            print("No events passed amplitude ratio filtering. Creating empty results.")
            filtered_eod_waveforms = []  # Empty list for variable-length waveforms
            filtered_amps = np.empty((0, n_channels))
            filtered_eod_amp = np.empty(0)
            filtered_cor_coeffs = np.empty((0, n_channels - 1))
            filtered_eod_chan = np.empty(0, dtype=int)
            filtered_is_differential = np.empty(0, dtype=int)
            filtered_final_peak_idc = np.empty(0, dtype=int)
            filtered_final_trough_idc = np.empty(0, dtype=int)
            filtered_unique_midpoints = np.empty(0, dtype=int)
            filtered_unique_peaks = np.empty(0, dtype=int)
            filtered_unique_troughs = np.empty(0, dtype=int)
            filtered_unique_widths = np.empty(0)
            filtered_original_pulse_orientation = np.empty(0, dtype='<U2')
            filtered_waveform_lengths = np.empty(0, dtype=int)
            filtered_features = pd.DataFrame(columns=['pp_dur_us', 'pp_ratio'])
        
        # 3. Create results DataFrame
        print(f"Creating event table with {len(filtered_unique_midpoints)} events...")
        
        # Create timestamps for each event
        eod_timestamps = []
        for i in range(len(filtered_unique_midpoints)):
            eod_timestamps.append(file_set['timestamp'][n] + dt.timedelta(seconds=filtered_unique_midpoints[i]/rate))
        
        eod_table = pd.DataFrame({
            'timestamp': eod_timestamps,
            'midpoint_idx': filtered_unique_midpoints,
            'peak_idx': filtered_final_peak_idc,
            'trough_idx': filtered_final_trough_idc,
            'eod_channel': filtered_eod_chan,
            'is_differential': filtered_is_differential,
            'eod_amplitude': filtered_eod_amp,
            'pulse_width': filtered_unique_widths,
            'pulse_orientation': filtered_original_pulse_orientation,
            'original_peak_idx': filtered_unique_peaks,
            'original_trough_idx': filtered_unique_troughs
        })
        
        # Add amplitude information for each channel
        for ch in range(n_channels):
            eod_table[f'amplitude_ch{ch}'] = filtered_amps[:, ch] if len(filtered_amps) > 0 else []
        
        # Add correlation coefficients
        for ch in range(n_channels - 1):
            eod_table[f'correlation_ch{ch}_{ch+1}'] = filtered_cor_coeffs[:, ch] if len(filtered_cor_coeffs) > 0 else []
        
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
            
            print(f"Saved {len(filtered_eod_waveforms)} variable-length waveforms efficiently")
            print(f"  Total samples: {waveform_metadata['total_samples']:,}")
            print(f"  Length range: {min(waveform_metadata['lengths'])} - {max(waveform_metadata['lengths'])} samples")
            print(f"  Storage efficiency: {storage_metrics['efficiency_percent']:.1f}% space saved")
            print(f"  Compression ratio: {storage_metrics['compression_ratio']:.2f}:1")
            print(f"  Disk space saved: {storage_metrics['space_saved_mb']:.2f} MB")
            
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
        
        # Enhanced quality control plots (memory optimized)
        if len(filtered_unique_midpoints) > 0:
            # Only differential events are kept after filtering
            differential_idc = np.arange(len(filtered_is_differential))  # All events are differential now
            
            # Plot: Differential detections with differential data
            if len(differential_idc) > 0:
                data_diff = np.diff(data, axis=1)
                offset_diff = np.max(filtered_eod_amp) * 1.5
                
                plt.figure(figsize=(20, 8))
                for i in range(data_diff.shape[1]):
                    # Find events detected on this differential channel
                    ch_diff_idc = np.where(filtered_eod_chan[differential_idc] == i)[0]
                    actual_diff_idc = differential_idc[ch_diff_idc]
                    
                    # Plot only every nth sample for large datasets to save memory
                    step = max(1, len(data_diff) // 15000000)  # Limit to ~15 mio points per channel
                    x_coords = np.arange(0, len(data_diff), step)
                    plt.plot(x_coords, data_diff[::step, i] + i * offset_diff, linewidth=0.5, label=f'Ch{i}-{i+1}')
                    
                    if len(actual_diff_idc) > 0:
                        plt.plot(filtered_final_peak_idc[actual_diff_idc], 
                                data_diff[filtered_final_peak_idc[actual_diff_idc], i] + i * offset_diff, 
                                'o', markersize=1, color='red')
                        plt.plot(filtered_final_trough_idc[actual_diff_idc], 
                                data_diff[filtered_final_trough_idc[actual_diff_idc], i] + i * offset_diff, 
                                'o', markersize=1, color='blue')
                
                plt.ylim(bottom=None, top=(data_diff.shape[1]-0.5)*offset_diff)
                plt.title(f'{fname} - Differential EOD Detections (n={len(differential_idc)}) - Red=Peaks, Blue=Troughs')
                plt.legend(loc='upper right')
                plt.xlabel('Sample')
                plt.ylabel('Voltage')
                plt.savefig(f'{output_path}\\{fname[:-4]}_differential_detection_plot.png', dpi=150, bbox_inches='tight')
                plt.close()
                
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
            plt.hist(filtered_eod_amp, bins=20, alpha=0.7)  # Fewer bins
            plt.title(f'EOD Amplitude Distribution (n={len(filtered_eod_amp)})')
            plt.xlabel('Amplitude')
            plt.ylabel('Count')
            
            # Plot channel usage (only differential channels now)
            plt.subplot(2, 3, 3)
            if len(filtered_eod_chan) > 0:
                diff_channel_counts = np.bincount(filtered_eod_chan, minlength=n_channels-1)
                plt.bar(range(len(diff_channel_counts)), diff_channel_counts, alpha=0.7, 
                        label=f'Differential ({len(filtered_eod_chan)})', width=0.8)
                plt.title(f'Differential Channel Usage')
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
            if len(filtered_is_differential) > 0:
                plt.bar([0], [len(filtered_is_differential)], alpha=0.7, label='Differential')
                plt.title('Detection Type (All Differential)')
                plt.xlabel('Detection Type')
                plt.ylabel('Count')
                plt.xticks([0], ['Differential'])
                
                # Add count label on bar
                plt.text(0, len(filtered_is_differential) + max(1, len(filtered_is_differential)*0.01), 
                        str(len(filtered_is_differential)), ha='center', va='bottom')
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
        
        else:
            print("No events passed filtering criteria.")
        
        # Clear large variables from memory
        del eod_waveforms, amps, eod_amp, cor_coeffs
        del filtered_eod_waveforms, filtered_amps, filtered_eod_amp, filtered_cor_coeffs
        gc.collect()
            
    else:
        print("No unique events found in this file.")
    
    # Clear data array for this file
    del data
    gc.collect()
    print(f"Completed file {n+1}/{len(file_set)}: {fname}")
    print(f"Memory freed for next file.\n")
    
    # ...continue with your analysis...