"""
EOD Analysis Functions Module
Consolidated functions for EOD pulse detection, processing, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.signal import find_peaks, correlate, windows
from scipy.interpolate import interp1d
from scipy import stats
import gc

# =============================================================================
# STORAGE FUNCTIONS
# =============================================================================

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
    try:
        # Load concatenated data and metadata
        try:
            data_file = np.load(base_path + '_concatenated.npz')
            concatenated = data_file['data']
        except:
            print(f"Warning: Could not load {base_path}_concatenated.npz")
            return []
        
        with open(base_path + '_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct individual waveforms
        waveforms_list = []
        lengths = metadata['lengths']
        
        # Handle case where some waveforms were empty
        if 'non_empty_indices' in metadata:
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
    except Exception as e:
        print(f"Error loading waveforms from {base_path}: {e}")
        return []

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

# =============================================================================
# FFT AND NOISE ANALYSIS FUNCTIONS
# =============================================================================

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

# =============================================================================
# WAVEFORM PROCESSING FUNCTIONS
# =============================================================================

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

# =============================================================================
# STATISTICS AND ANALYSIS FUNCTIONS
# =============================================================================

def calculate_waveform_stats(waveforms_list, label=""):
    """
    Calculate comprehensive statistics for a list of waveforms.
    
    Parameters
    ----------
    waveforms_list : list of 1D arrays
        List of waveforms to analyze
    label : str, optional
        Label for the waveforms (for display purposes)
    
    Returns
    -------
    stats_dict : dict
        Dictionary containing comprehensive statistics
    """
    if not waveforms_list:
        return {}
    
    # Filter out empty waveforms
    valid_waveforms = [wf for wf in waveforms_list if len(wf) > 0]
    
    if not valid_waveforms:
        return {}
    
    # Length statistics
    lengths = [len(wf) for wf in valid_waveforms]
    
    # Amplitude statistics
    max_amps = [np.max(np.abs(wf)) for wf in valid_waveforms]
    peak_to_trough = [np.max(wf) - np.min(wf) for wf in valid_waveforms]
    
    # Shape statistics
    skewness = [stats.skew(wf) if len(wf) > 3 else 0 for wf in valid_waveforms]
    kurtosis = [stats.kurtosis(wf) if len(wf) > 3 else 0 for wf in valid_waveforms]
    
    # Peak/trough position statistics
    peak_positions = []
    trough_positions = []
    for wf in valid_waveforms:
        if len(wf) > 0:
            peak_pos = np.argmax(wf) / len(wf)  # Relative position
            trough_pos = np.argmin(wf) / len(wf)  # Relative position
            peak_positions.append(peak_pos)
            trough_positions.append(trough_pos)
    
    stats_dict = {
        'count': len(valid_waveforms),
        'empty_count': len(waveforms_list) - len(valid_waveforms),
        'length_stats': {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'median': np.median(lengths),
            'q25': np.percentile(lengths, 25),
            'q75': np.percentile(lengths, 75)
        },
        'amplitude_stats': {
            'max_amp_mean': np.mean(max_amps),
            'max_amp_std': np.std(max_amps),
            'peak_to_trough_mean': np.mean(peak_to_trough),
            'peak_to_trough_std': np.std(peak_to_trough),
            'peak_to_trough_median': np.median(peak_to_trough)
        },
        'shape_stats': {
            'skewness_mean': np.mean(skewness),
            'skewness_std': np.std(skewness),
            'kurtosis_mean': np.mean(kurtosis),
            'kurtosis_std': np.std(kurtosis)
        },
        'position_stats': {
            'peak_pos_mean': np.mean(peak_positions),
            'peak_pos_std': np.std(peak_positions),
            'trough_pos_mean': np.mean(trough_positions),
            'trough_pos_std': np.std(trough_positions)
        }
    }
    
    return stats_dict
