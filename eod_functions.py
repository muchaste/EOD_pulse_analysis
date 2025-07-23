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
import glob


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

def analyze_waveform_fft(waveform, rate, max_freq_content=0.8):
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
                          max_freq_content=0.8, min_snr=2.0, max_ipi_ratio=50.0,
                          return_freq_stats=False):
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
    return_freq_stats : bool
        If True, return frequency stats (via analyze_waveform_fft)
    
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
            is_noisy, freq_stats = analyze_waveform_fft(waveform, rate, max_freq_content)
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
    if return_freq_stats:
        return clean_mask, freq_stats
    else:
        return clean_mask
    
def calc_fft_peak(signal, rate, lower_thresh=0, upper_thresh=100000, zero_padding_factor=100):
    """
    Calculate the frequency with the highest power in the signal using FFT with zero-padding.

    Parameters
    ----------
    signal : 1-D array
        The input signal to analyze.
    rate : int
        The sample rate of the signal.
    lower_thresh : float, optional
        The lower threshold frequency (default is 0).
    upper_thresh : float, optional
        The upper threshold frequency (default is 100000).
    zero_padding_factor : int, optional
        The factor by which to zero-pad the signal (default is 10).

    Returns
    -------
    peak_freq : float
        The frequency with the highest power in the signal.
    """
    # Zero-pad the signal
    padded_length = len(signal) * zero_padding_factor
    fft_spectrum = np.fft.fft(signal, n=padded_length)
    freqs = np.fft.fftfreq(padded_length, 1/rate)

    # Only consider positive frequencies and within specified range
    valid_idx = np.where((freqs > lower_thresh) & (freqs < upper_thresh))
    peak_freq = freqs[valid_idx][np.argmax(np.abs(fft_spectrum[valid_idx]))]

    return peak_freq

def filter_waveforms(eod_waveforms, eod_width, amplitude_ratios, rate,
                     dur_min=20, dur_max=300,
                     pp_r_min=0.1, pp_r_max=5,
                     fft_freq_min=1000, fft_freq_max=10000,
                     interp_factor=1, return_features=False,
                     return_params=False):
    """
    Filter variable-length waveforms based on duration, amplitude ratio, and FFT frequency.
    Updated to work with extract_pulse_snippets output format.

    Parameters
    ----------
    eod_waveforms : list of 1-D arrays
        List of variable-length EOD waveforms from extract_pulse_snippets.
    eod_width : 1-D array
        Width in microseconds between peak and trough from extract_pulse_snippets.
    amplitude_ratios : 1-D array
        Peak-to-trough amplitude ratios from extract_pulse_snippets.
    rate : int
        Sample rate of the original signal.
    dur_min : float, optional
        Minimum duration threshold in microseconds (default is 20).
    dur_max : float, optional
        Maximum duration threshold in microseconds (default is 300).
    pp_r_min : float, optional
        Minimum peak-to-peak ratio (default is 0.1).
    pp_r_max : float, optional
        Maximum peak-to-peak ratio (default is 5).
    fft_freq_min : float, optional
        Minimum FFT frequency threshold (default is 1000).
    fft_freq_max : float, optional
        Maximum FFT frequency threshold (default is 10000).
    interp_factor : int, optional
        Factor by which the signal is interpolated (default is 1).
    return_features : bool, optional
        Returns dataframe with waveform features if True.
    return_params : bool, optional
        Returns dataframe with filter parameters if True.

    Returns
    -------
    eod_waveforms_keep : list of 1-D arrays
        Filtered list of variable-length EOD waveforms.
    keep_indices : 1-D array
        Indices of the kept waveforms.
    """
    n_snippets = len(eod_waveforms)
    wf_rate = rate * interp_factor

    # Use the pre-calculated eod_width (already in microseconds)
    wf_durs = eod_width.copy()
    
    # Use the pre-calculated amplitude ratios
    wf_ratios = amplitude_ratios.copy()

    # Calculate FFT peak frequencies for each waveform
    fft_freqs = np.zeros(n_snippets)
    for i in range(n_snippets):
        if len(eod_waveforms[i]) > 0:  # Check for non-empty waveforms
            try:
                fft_freqs[i] = calc_fft_peak(eod_waveforms[i], wf_rate, zero_padding_factor=100)
            except:
                # If FFT calculation fails, set to a value that will be filtered out
                fft_freqs[i] = 0
        else:
            fft_freqs[i] = 0

    # Apply filters
    keep_mask = (
        (wf_durs >= dur_min) & (wf_durs <= dur_max) &
        (wf_ratios >= pp_r_min) & (wf_ratios <= pp_r_max) &
        (fft_freqs >= fft_freq_min) & (fft_freqs <= fft_freq_max)
    )

    keep_indices = np.where(keep_mask)[0]
    # eod_waveforms_keep = [eod_waveforms[i] for i in keep_indices]

    return_vars = [keep_indices]

    if return_features:
        features = pd.DataFrame({
            'pp_dur_us': wf_durs[keep_indices],
            'pp_ratio': wf_ratios[keep_indices],
            'fft_freq': fft_freqs[keep_indices]
        })
        return_vars.append(features)

    if return_params:
        params = pd.DataFrame({
            'dur_min': [dur_min],
            'dur_max': [dur_max],
            'pp_r_min': [pp_r_min],
            'pp_r_max': [pp_r_max],
            'fft_freq_min': [fft_freq_min],
            'fft_freq_max': [fft_freq_max]
        })
        return_vars.append(params)

    return return_vars


# =============================================================================
# WAVEFORM PROCESSING FUNCTIONS
# =============================================================================

def extract_pulse_snippets(data, rate, midpoints, peaks, troughs, widths, 
                            width_factor=5.0, interp_factor=1, center_on_zero_crossing=False, return_diff=False):
    """
    Extract and analyze EOD snippets with variable widths based on detected pulse widths.
    Optimized to store variable-length waveforms without zero-padding for maximum efficiency.
    
    Parameters
    ----------
    data : 2-D array
        The full recording data with channels in columns
    rate : int
        Sampling rate    
    midpoints : 1-D array
        Midpoint indices of unique events
    peaks : 1-D array
        Peak indices of unique events
    troughs : 1-D array
        Trough indices of unique events
    widths : 1-D array
        Width (in seconds) of unique events
    width_factor : float
        Factor to multiply width to get snippet length
    interp_factor : int
        Interpolation factor for final waveforms
    center_on_zero_crossing : bool
        Whether to center waveforms on zero-crossing (False for storage efficiency)
    return_diff : bool
        If True, only return differential events (is_differential=1), default False
    
    Returns
    -------
    eod_waveforms : list of 1-D arrays
        Variable-length EOD waveform snippets (no zero-padding)
    eod_amp : 1-D array
        Amplitude of extracted waveform
    eod_width : 1-D array
        Width in microseconds between peak and trough of the extracted waveform
    amps : 2-D array
        Max amplitudes across channels for each snippet
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
    n_events = len(midpoints)
    
    # Preallocate arrays
    eod_waveforms = []  # Store as list for variable lengths
    amps = np.zeros((n_events, n_channels))
    eod_amp = np.zeros(n_events)
    cor_coeffs = np.zeros((n_events, n_channels - 1))
    eod_chan = np.zeros(n_events, dtype=int)
    is_differential = np.ones(n_events, dtype=int)  # 1=differential, 0=single-ended
    final_peak_idc = np.zeros(n_events, dtype=int)
    final_trough_idc = np.zeros(n_events, dtype=int)
    final_midpoint_idc = np.zeros(n_events, dtype=int)
    pulse_orientation = np.array(['HP'] * n_events)  # Store original orientation
    amplitude_ratios = np.zeros(n_events)  # For amplitude ratio filtering
    waveform_lengths = np.zeros(n_events, dtype=int)  # Track actual lengths
    eod_width = np.zeros(n_events, dtype=int)  # Width between peak and trough
    
    for i in range(n_events):
        # Calculate snippet length based on width
        snippet_samples = int(widths[i] * width_factor)
        snippet_samples = max(snippet_samples, 20)  # Minimum 20 samples
        
        # Extract snippet around midpoint
        center_idx = int(midpoints[i])
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
            # max_abs = np.max(np.abs(eod_waveform))
            if max_val > 0:
                eod_waveform /= max_val

            # Store variable-length waveform and its length
            eod_waveforms.append(eod_waveform)
            waveform_lengths[i] = len(eod_waveform)
            
            # Calculate width between peak and trough in uS
            eod_width[i] = abs(eod_peak_idx - eod_trough_idx)*1e6//rate
            
            # Store final indices (adjust for snippet position)
            final_peak_idc[i] = start_idx + eod_peak_idx
            final_trough_idc[i] = start_idx + eod_trough_idx
            final_midpoint_idc[i] = final_peak_idc[i] + (final_trough_idc[i] - final_peak_idc[i]) // 2

        else:
            # Empty waveform case
            eod_waveforms.append(np.array([]))
            waveform_lengths[i] = 0
            eod_width[i] = 0
            final_peak_idc[i] = peaks[i]
            final_trough_idc[i] = troughs[i]
            final_midpoint_idc[i] = midpoints[i]
    
    # Filter for differential events only if requested
    if return_diff:
        diff_mask = is_differential == 1
        eod_waveforms = [eod_waveforms[i] for i in range(len(eod_waveforms)) if diff_mask[i]]
        amps = amps[diff_mask]
        eod_amp = eod_amp[diff_mask]
        cor_coeffs = cor_coeffs[diff_mask]
        eod_chan = eod_chan[diff_mask]
        is_differential = is_differential[diff_mask]
        final_peak_idc = final_peak_idc[diff_mask]
        final_trough_idc = final_trough_idc[diff_mask]
        final_midpoint_idc = final_midpoint_idc[diff_mask]
        pulse_orientation = pulse_orientation[diff_mask]
        amplitude_ratios = amplitude_ratios[diff_mask]
        waveform_lengths = waveform_lengths[diff_mask]
        eod_width = eod_width[diff_mask]
    
    # Return variable-length waveforms as list (no zero-padding)
    return (eod_waveforms, eod_amp, eod_width, amps, cor_coeffs, eod_chan, 
            is_differential, final_peak_idc, final_trough_idc, final_midpoint_idc,
            pulse_orientation, amplitude_ratios, waveform_lengths)

# =============================================================================
# EVENT EXTRACTION FUNCTIONS
# =============================================================================

def load_session_data(session_folder):
    """
    Load all EOD tables and waveforms from a recording session.
    
    Parameters
    ----------
    session_folder : str
        Path to folder containing *_eod_table.csv and *_eod_waveforms files
    
    Returns
    -------
    combined_table : pd.DataFrame
        Combined EOD event table from all files
    combined_waveforms : list of np.arrays
        Combined waveform data (variable-length waveforms)
    file_origins : np.array
        File index for each event (for tracking origin)
    """
    # Find all EOD table files
    table_files = glob.glob(str(Path(session_folder) / "*_eod_table.csv"))
    
    if len(table_files) == 0:
        raise ValueError(f"No EOD table files found in {session_folder}")
    
    print(f"Loading data from {len(table_files)} files...")
    
    combined_tables = []
    combined_waveforms = []
    file_origins = []
    
    for i, table_file in enumerate(sorted(table_files)):
        # Load event table
        table = pd.read_csv(table_file)
        if len(table) > 0:
            table['file_index'] = i
            table['filename'] = Path(table_file).stem.replace('_eod_table', '')
            combined_tables.append(table)
            
            # Load waveforms using new variable-length format
            waveform_base_path = str(Path(table_file).parent / Path(table_file).stem.replace('_eod_table', '_eod_waveforms'))
            waveforms = load_variable_length_waveforms(waveform_base_path)
            
            # Fallback to old CSV format if new format doesn't exist
            if not waveforms:
                old_waveform_file = str(Path(table_file).parent / Path(table_file).stem.replace('_eod_table', '_eod_waveforms.csv'))
                if Path(old_waveform_file).exists():
                    print(f"  Using legacy CSV format for {Path(table_file).name}")
                    waveforms_csv = pd.read_csv(old_waveform_file).values
                    # Convert to list of arrays for consistency
                    waveforms = [waveforms_csv[i, :] for i in range(waveforms_csv.shape[0])]
                else:
                    print(f"  Warning: No waveforms found for {Path(table_file).name}")
                    waveforms = []
            
            if len(waveforms) == len(table):
                combined_waveforms.extend(waveforms)
                file_origins.extend([i] * len(waveforms))
                print(f"  {Path(table_file).name}: {len(table)} events")
            else:
                print(f"  Warning: Mismatch in {Path(table_file).name} - {len(table)} events, {len(waveforms)} waveforms")
                # Skip this file or handle the mismatch
                combined_tables.pop()  # Remove the table we just added
    
    if len(combined_tables) == 0:
        raise ValueError("No events found in any files")
    
    # Combine all data
    combined_table = pd.concat(combined_tables, ignore_index=True)
    file_origins = np.array(file_origins)
    
    print(f"Total events loaded: {len(combined_table)}")
    print(f"Total waveforms loaded: {len(combined_waveforms)}")
    return combined_table, combined_waveforms, file_origins

def extract_events(combined_table, max_ipi_seconds=5.0, min_eods_per_event=20, min_snr=2.0):
    """
    Extract events (fish encounters) from EOD data by temporal clustering.
    
    An event is defined as a sequence of EODs where consecutive EODs are separated
    by no more than max_ipi_seconds, with at least min_eods_per_event total EODs
    and at least one EOD meeting the SNR threshold.
    
    Parameters
    ----------
    combined_table : pd.DataFrame
        Combined EOD event data with timestamp column
    max_ipi_seconds : float
        Maximum inter-pulse interval to consider EODs as part of same event
    min_eods_per_event : int
        Minimum number of EODs required per event
    min_snr : float
        Minimum SNR required for at least one EOD in the event
    
    Returns
    -------
    event_table : pd.DataFrame
        Table with event information and filtered EODs
    event_labels : np.array
        Event ID for each EOD (-1 if not part of valid event)
    event_summary : pd.DataFrame
        Summary statistics for each event
    """
    if len(combined_table) == 0:
        return combined_table.copy(), np.array([]), pd.DataFrame()
    
    print("Extracting events from temporal patterns...")
    
    # Sort by timestamp
    sorted_table = combined_table.sort_values('timestamp').copy()
    sorted_table['timestamp_dt'] = pd.to_datetime(sorted_table['timestamp'])
    
    # Calculate inter-pulse intervals
    time_diffs = sorted_table['timestamp_dt'].diff().dt.total_seconds()
    
    # Identify event boundaries (gaps > max_ipi_seconds)
    event_boundaries = np.where(time_diffs > max_ipi_seconds)[0]
    
    # Create event segments
    event_starts = np.concatenate([[0], event_boundaries])
    event_ends = np.concatenate([event_boundaries, [len(sorted_table)]])
    
    print(f"Found {len(event_starts)} potential events before filtering...")
    
    # Filter events based on criteria
    valid_events = []
    event_labels = np.full(len(sorted_table), -1)
    event_summaries = []
    
    for event_id, (start_idx, end_idx) in enumerate(zip(event_starts, event_ends)):
        event_eods = sorted_table.iloc[start_idx:end_idx]
        
        # Check minimum number of EODs
        if len(event_eods) < min_eods_per_event:
            continue
        
        # Check SNR requirement (if SNR column exists)
        has_good_snr = True
        if 'snr' in event_eods.columns:
            max_snr = event_eods['snr'].max()
            has_good_snr = max_snr >= min_snr
        elif 'eod_amplitude' in event_eods.columns:
            # Use amplitude as proxy for SNR if no SNR column
            # Assume higher amplitude indicates better SNR
            max_amplitude = event_eods['eod_amplitude'].max()
            median_amplitude = combined_table['eod_amplitude'].median()
            has_good_snr = max_amplitude >= median_amplitude * 1.5
        
        if not has_good_snr:
            continue
        
        # Valid event - assign labels
        event_labels[start_idx:end_idx] = len(valid_events)
        valid_events.append((start_idx, end_idx))
        
        # Calculate event summary statistics
        duration = (event_eods['timestamp_dt'].max() - event_eods['timestamp_dt'].min()).total_seconds()
        mean_ipi = time_diffs.iloc[start_idx+1:end_idx].mean() if len(event_eods) > 1 else 0
        
        summary = {
            'event_id': len(valid_events) - 1,
            'start_time': event_eods['timestamp_dt'].min(),
            'end_time': event_eods['timestamp_dt'].max(),
            'duration_seconds': duration,
            'n_eods': len(event_eods),
            'mean_ipi_seconds': mean_ipi,
            'median_ipi_seconds': time_diffs.iloc[start_idx+1:end_idx].median() if len(event_eods) > 1 else 0,
            'mean_amplitude': event_eods['eod_amplitude'].mean(),
            'max_amplitude': event_eods['eod_amplitude'].max(),
            'mean_width_ms': event_eods['pulse_width'].mean() * 1000,
            'n_files': event_eods['file_index'].nunique() if 'file_index' in event_eods.columns else 1,
            'file_names': ','.join(event_eods['filename'].unique()) if 'filename' in event_eods.columns else 'unknown'
        }
        
        if 'snr' in event_eods.columns:
            summary['max_snr'] = event_eods['snr'].max()
            summary['mean_snr'] = event_eods['snr'].mean()
        
        event_summaries.append(summary)
    
    print(f"Extracted {len(valid_events)} valid events")
    print(f"Total EODs in events: {np.sum(event_labels >= 0)} / {len(combined_table)}")
    
    # Create filtered table with only EODs in valid events
    valid_mask = event_labels >= 0
    event_table = sorted_table[valid_mask].copy()
    event_table['event_id'] = event_labels[valid_mask]
    
    # Create event summary DataFrame
    event_summary = pd.DataFrame(event_summaries)
    
    # Reorder event_labels to match original table order
    # Map back to original indices
    original_event_labels = np.full(len(combined_table), -1)
    sorted_indices = sorted_table.index.values
    original_event_labels[sorted_indices] = event_labels
    
    return event_table, original_event_labels, event_summary

def analyze_session_for_events(session_folder, max_ipi_seconds=5.0):
    """
    Analyze session data to help choose appropriate event extraction parameters.
    
    Parameters
    ----------
    session_folder : str
        Path to folder containing EOD files
    max_ipi_seconds : float
        Maximum IPI to consider for event grouping
    
    Returns
    -------
    analysis : dict
        Statistics about potential events
    """
    combined_table, _, _ = load_session_data(session_folder)
    
    if len(combined_table) == 0:
        return {}
    
    # Sort by timestamp
    sorted_table = combined_table.sort_values('timestamp')
    sorted_table['timestamp_dt'] = pd.to_datetime(sorted_table['timestamp'])
    
    # Calculate inter-pulse intervals
    time_diffs = sorted_table['timestamp_dt'].diff().dt.total_seconds()
    
    # Identify potential events
    event_boundaries = np.where(time_diffs > max_ipi_seconds)[0]
    event_starts = np.concatenate([[0], event_boundaries])
    event_ends = np.concatenate([event_boundaries, [len(sorted_table)]])
    
    event_lengths = event_ends - event_starts
    
    analysis = {
        'total_eods': len(combined_table),
        'potential_events': len(event_starts),
        'event_lengths': {
            'min': event_lengths.min(),
            'max': event_lengths.max(),
            'mean': event_lengths.mean(),
            'median': np.median(event_lengths)
        },
        'events_with_20_plus_eods': np.sum(event_lengths >= 20),
        'events_with_10_plus_eods': np.sum(event_lengths >= 10),
        'events_with_5_plus_eods': np.sum(event_lengths >= 5),
        'ipi_percentiles': {
            '50th': np.percentile(time_diffs.dropna(), 50),
            '90th': np.percentile(time_diffs.dropna(), 90),
            '95th': np.percentile(time_diffs.dropna(), 95),
            '99th': np.percentile(time_diffs.dropna(), 99)
        }
    }
    
    return analysis

def save_event_results(event_table, event_summary, combined_waveforms, event_labels, output_folder):
    """
    Save event extraction results to files.
    
    Parameters
    ----------
    event_table : pd.DataFrame
        Filtered EOD data (only EODs in valid events)
    event_summary : pd.DataFrame
        Summary statistics for each event
    combined_waveforms : list of np.arrays
        Variable-length waveform data
    event_labels : np.array
        Event ID for each EOD (-1 if not part of valid event)
    output_folder : str
        Path to save results
    """
    output_path = Path(output_folder)
    
    # Save event table
    event_table.to_csv(output_path / 'session_events_eod_table.csv', index=False)
    
    # Save event summary
    event_summary.to_csv(output_path / 'session_events_summary.csv', index=False)
    
    # Save waveforms for events only using new efficient format
    event_mask = event_labels >= 0
    event_waveforms = [combined_waveforms[i] for i in range(len(combined_waveforms)) if event_mask[i]]
    
    # Save in new variable-length format
    waveform_base_path = str(output_path / 'session_events_waveforms')
    metadata = save_variable_length_waveforms(event_waveforms, waveform_base_path)
    
    # Also save in legacy CSV format for backward compatibility
    # (Convert to fixed-length for CSV - pad to median length)
    if event_waveforms:
        lengths = [len(wf) for wf in event_waveforms if len(wf) > 0]
        if lengths:
            target_length = int(np.median(lengths))
            padded_waveforms = []
            for wf in event_waveforms:
                if len(wf) == 0:
                    padded_waveforms.append(np.zeros(target_length))
                elif len(wf) < target_length:
                    padded_wf = np.pad(wf, (0, target_length - len(wf)), mode='constant')
                    padded_waveforms.append(padded_wf)
                elif len(wf) > target_length:
                    padded_waveforms.append(wf[:target_length])
                else:
                    padded_waveforms.append(wf)
            
            padded_array = np.array(padded_waveforms)
            pd.DataFrame(padded_array).to_csv(output_path / 'session_events_waveforms.csv', index=False)
    
    # Save event extraction parameters and statistics
    stats = {
        'total_eods_loaded': len(event_labels),
        'eods_in_events': np.sum(event_mask),
        'eods_filtered_out': np.sum(~event_mask),
        'total_events': len(event_summary),
        'extraction_efficiency': np.sum(event_mask) / len(event_labels) if len(event_labels) > 0 else 0
    }
    
    if len(event_summary) > 0:
        stats.update({
            'min_event_duration_sec': event_summary['duration_seconds'].min(),
            'max_event_duration_sec': event_summary['duration_seconds'].max(),
            'mean_event_duration_sec': event_summary['duration_seconds'].mean(),
            'min_eods_per_event': event_summary['n_eods'].min(),
            'max_eods_per_event': event_summary['n_eods'].max(),
            'mean_eods_per_event': event_summary['n_eods'].mean()
        })
    
    # Add waveform storage information
    if event_waveforms:
        stats['waveform_storage'] = {
            'variable_length_format': True,
            'total_waveforms': len(event_waveforms),
            'space_savings_mb': metadata.get('space_savings', {}).get('space_saved_mb', 0)
        }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_path / 'session_events_extraction_stats.csv', index=False)
    
    print(f"Event extraction results saved to {output_folder}")
    return stats

def create_event_plots(event_table, event_summary, output_folder):
    """
    Create visualization plots for event extraction results.
    
    Parameters
    ----------
    event_table : pd.DataFrame
        Filtered EOD data
    event_summary : pd.DataFrame
        Event summary statistics
    output_folder : str
        Path to save plots
    """
    output_path = Path(output_folder)
    
    if len(event_summary) == 0:
        print("No events to plot")
        return
    
    # Plot 1: Event timeline
    plt.figure(figsize=(15, 6))
    
    # Convert to datetime for plotting
    start_times = pd.to_datetime(event_summary['start_time'])
    durations = event_summary['duration_seconds']
    n_eods = event_summary['n_eods']
    
    # Create timeline plot
    for i, (start, duration, n_eod) in enumerate(zip(start_times, durations, n_eods)):
        plt.barh(i, duration/60, left=(start - start_times.min()).total_seconds()/60, 
                height=0.8, alpha=0.7, label=f'{n_eod} EODs' if i < 5 else "")
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Event ID')
    plt.title(f'Event Timeline ({len(event_summary)} events)')
    plt.grid(True, alpha=0.3)
    if len(event_summary) <= 5:
        plt.legend()
    plt.savefig(output_path / 'event_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Event quality metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Duration vs number of EODs
    axes[0,0].scatter(event_summary['duration_seconds']/60, event_summary['n_eods'], alpha=0.6)
    axes[0,0].set_xlabel('Duration (minutes)')
    axes[0,0].set_ylabel('Number of EODs')
    axes[0,0].set_title('Event Duration vs EOD Count')
    
    # Mean IPI distribution
    axes[0,1].hist(event_summary['mean_ipi_seconds'], bins=20, alpha=0.7)
    axes[0,1].set_xlabel('Mean IPI (seconds)')
    axes[0,1].set_ylabel('Number of Events')
    axes[0,1].set_title('Distribution of Mean Inter-Pulse Intervals')
    
    # Amplitude distribution
    axes[1,0].hist(event_summary['mean_amplitude'], bins=20, alpha=0.7)
    axes[1,0].set_xlabel('Mean Amplitude')
    axes[1,0].set_ylabel('Number of Events')
    axes[1,0].set_title('Distribution of Mean Amplitudes')
    
    # Width distribution
    axes[1,1].hist(event_summary['mean_width_ms'], bins=20, alpha=0.7)
    axes[1,1].set_xlabel('Mean Width (ms)')
    axes[1,1].set_ylabel('Number of Events')
    axes[1,1].set_title('Distribution of Mean Pulse Widths')
    
    plt.tight_layout()
    plt.savefig(output_path / 'event_quality_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Event size distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(event_summary['n_eods'], bins=20, alpha=0.7)
    plt.xlabel('Number of EODs per Event')
    plt.ylabel('Number of Events')
    plt.title('Event Size Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(event_summary['duration_seconds']/60, bins=20, alpha=0.7)
    plt.xlabel('Event Duration (minutes)')
    plt.ylabel('Number of Events')
    plt.title('Event Duration Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path / 'event_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Event plots saved to {output_folder}")

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
    event_waveforms : list of np.arrays
        Variable-length waveform data for valid events
    event_summary : pd.DataFrame
        Summary statistics for each event
    """
    input_path = Path(input_folder)
    
    # Load event data
    event_table_file = input_path / 'session_events_eod_table.csv'
    event_summary_file = input_path / 'session_events_summary.csv'
    
    if not event_table_file.exists():
        raise FileNotFoundError(f"Event table not found: {event_table_file}")
    
    if not event_summary_file.exists():
        raise FileNotFoundError(f"Event summary not found: {event_summary_file}")
    
    # Load data
    event_table = pd.read_csv(event_table_file)
    event_summary = pd.read_csv(event_summary_file)
    
    # Load waveforms using new variable-length format
    waveform_base_path = str(input_path / 'session_events_waveforms')
    event_waveforms = load_variable_length_waveforms(waveform_base_path)
    
    # Fallback to old CSV format if new format doesn't exist
    if not event_waveforms:
        old_waveform_file = input_path / 'session_events_waveforms.csv'
        if old_waveform_file.exists():
            print("Using legacy CSV format for waveforms")
            waveforms_csv = pd.read_csv(old_waveform_file).values
            # Convert to list of arrays for consistency
            event_waveforms = [waveforms_csv[i, :] for i in range(waveforms_csv.shape[0])]
        else:
            raise FileNotFoundError(f"Event waveforms not found: {waveform_base_path} or {old_waveform_file}")
    
    print(f"Loaded event data:")
    print(f"  Events: {len(event_summary)}")
    print(f"  EODs: {len(event_table)}")
    print(f"  Waveforms: {len(event_waveforms)}")
    
    # Check for length mismatch
    if len(event_waveforms) != len(event_table):
        print(f"Warning: Mismatch between table ({len(event_table)}) and waveforms ({len(event_waveforms)})")
        # Truncate to shorter length
        min_len = min(len(event_table), len(event_waveforms))
        event_table = event_table.iloc[:min_len]
        event_waveforms = event_waveforms[:min_len]
        print(f"  Truncated to {min_len} entries")
    
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import BayesianGaussianMixture
    
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
    Extract PCA features from variable-length normalized waveforms.
    
    Parameters
    ----------
    waveforms : list of np.arrays
        Variable-length waveform data
    n_components : int
        Number of PCA components
    
    Returns
    -------
    features : 2D array
        PCA features (n_events, n_components)
    pca : PCA object
        Fitted PCA transformer
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from scipy.interpolate import interp1d
    
    # Filter out empty waveforms
    valid_waveforms = [wf for wf in waveforms if len(wf) > 0]
    if len(valid_waveforms) == 0:
        return np.zeros((len(waveforms), n_components)), None
    
    # Normalize all waveforms to same length for PCA
    lengths = [len(wf) for wf in valid_waveforms]
    target_length = int(np.median(lengths))
    
    # Interpolate all waveforms to target length
    normalized_waveforms = []
    for wf in valid_waveforms:
        if len(wf) > 3:  # Need at least 4 points for interpolation
            x_old = np.linspace(0, 1, len(wf))
            x_new = np.linspace(0, 1, target_length)
            f = interp1d(x_old, wf, kind='linear', fill_value='extrapolate')
            wf_interp = f(x_new)
            normalized_waveforms.append(wf_interp)
        else:
            # Too short for interpolation - pad with zeros
            if len(wf) > 0:
                padded = np.zeros(target_length)
                padded[:len(wf)] = wf
                normalized_waveforms.append(padded)
            else:
                normalized_waveforms.append(np.zeros(target_length))
    
    if len(normalized_waveforms) == 0:
        return np.zeros((len(waveforms), n_components)), None
    
    waveforms_array = np.array(normalized_waveforms)
    
    # Remove zero-variance waveforms
    valid_mask = np.var(waveforms_array, axis=1) > 1e-10
    if np.sum(valid_mask) == 0:
        return np.zeros((len(waveforms), n_components)), None
    
    valid_waveforms_array = waveforms_array[valid_mask]
    
    # Standardize each waveform
    scaler = StandardScaler()
    waveforms_scaled = scaler.fit_transform(valid_waveforms_array.T).T
    
    # PCA
    pca = PCA(n_components=min(n_components, waveforms_scaled.shape[1]))
    features_valid = pca.fit_transform(waveforms_scaled)
    
    # Create full feature array mapping back to original waveform indices
    features = np.zeros((len(waveforms), features_valid.shape[1]))
    
    # Map valid waveforms back to original indices
    valid_indices = []
    for i, wf in enumerate(waveforms):
        if len(wf) > 0:
            valid_indices.append(i)
    
    valid_indices = np.array(valid_indices)
    if len(valid_indices) > 0:
        features[valid_indices[valid_mask]] = features_valid
    
    return features, pca

def cluster_session_eods(event_table, event_waveforms, event_summary, min_cluster_size=10):
    """
    Hierarchical clustering approach on event-level data (similar to extract_pulsefish).
    
    Parameters
    ----------
    event_table : pd.DataFrame
        Filtered EOD data (only EODs in valid events)
    event_waveforms : list of np.arrays
        Variable-length waveform data corresponding to event_table
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
    from sklearn.cluster import DBSCAN
    
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
        cluster_waveforms = [event_waveforms[i] for i in range(len(event_waveforms)) if width_mask[i]]
        
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
            shape_waveforms = [event_waveforms[i] for i in range(len(event_waveforms)) if combined_mask[i]]
            
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
                species_waveforms = [event_waveforms[i] for i in range(len(event_waveforms)) if final_mask[i]]
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
    from scipy.interpolate import interp1d
    
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
            species_waveforms = [event_waveforms[j] for j in range(len(event_waveforms)) if species_mask[j]]
            
            plt.subplot(n_rows, n_cols, i+1)
            
            # Plot up to 20 example waveforms
            n_examples = min(20, len(species_waveforms))
            for j in range(n_examples):
                wf = species_waveforms[j]
                if len(wf) > 0:
                    plt.plot(wf, alpha=0.3, color='blue', linewidth=0.5)
            
            # Plot mean waveform (normalize lengths first)
            if len(species_waveforms) > 0:
                # Get non-empty waveforms
                non_empty_waveforms = [wf for wf in species_waveforms if len(wf) > 0]
                if non_empty_waveforms:
                    # Normalize all to same length
                    lengths = [len(wf) for wf in non_empty_waveforms]
                    target_length = int(np.median(lengths))
                    
                    normalized_waveforms = []
                    for wf in non_empty_waveforms:
                        if len(wf) > 3:  # Need at least 4 points for interpolation
                            x_old = np.linspace(0, 1, len(wf))
                            x_new = np.linspace(0, 1, target_length)
                            f = interp1d(x_old, wf, kind='linear', fill_value='extrapolate')
                            wf_interp = f(x_new)
                            normalized_waveforms.append(wf_interp)
                    
                    if normalized_waveforms:
                        mean_waveform = np.mean(normalized_waveforms, axis=0)
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



# =============================================================================
# STATISTICS AND ANALYSIS FUNCTIONS
# =============================================================================

def calculate_waveform_stats(waveforms_list, label=""):

    """
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

# =============================================================================
# QC FUNCTIONS
# =============================================================================

def plot_waveform_comparison(accepted_waveforms, filtered_waveforms, output_path, file_prefix, save_fig=True, show_fig=False):
    """Create comprehensive comparison plots."""
    
    # Filter out empty waveforms
    acc_valid = [wf for wf in accepted_waveforms if len(wf) > 0]
    filt_valid = [wf for wf in filtered_waveforms if len(wf) > 0]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overlay plot of example waveforms
    plt.subplot(3, 4, 1)
    n_examples = min(100, len(acc_valid))  # Show up to 100 examples
    for i in range(n_examples):
        plt.plot(acc_valid[i], 'b-', alpha=0.1, linewidth=0.5)
    plt.title(f'Accepted Waveforms (n={len(acc_valid)})\nShowing {n_examples} examples')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 2)
    n_examples = min(100, len(filt_valid))  # Show up to 100 examples
    for i in range(n_examples):
        plt.plot(filt_valid[i], 'r-', alpha=0.1, linewidth=0.5)
    plt.title(f'Filtered-Out Waveforms (n={len(filt_valid)})\nShowing {n_examples} examples')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 2. Length distribution comparison
    plt.subplot(3, 4, 3)
    acc_lengths = [len(wf) for wf in acc_valid]
    filt_lengths = [len(wf) for wf in filt_valid]
    
    bins = np.linspace(0, max(max(acc_lengths) if acc_lengths else 0, 
                             max(filt_lengths) if filt_lengths else 0), 50)
    
    plt.hist(acc_lengths, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_lengths)})', 
             color='blue', density=True)
    plt.hist(filt_lengths, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_lengths)})', 
             color='red', density=True)
    plt.title('Waveform Length Distribution')
    plt.xlabel('Length (samples)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Amplitude distribution comparison
    plt.subplot(3, 4, 4)
    acc_amps = [np.max(wf) - np.min(wf) for wf in acc_valid]
    filt_amps = [np.max(wf) - np.min(wf) for wf in filt_valid]
    
    bins = np.linspace(0, max(max(acc_amps) if acc_amps else 0, 
                             max(filt_amps) if filt_amps else 0), 50)
    
    plt.hist(acc_amps, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_amps)})', 
             color='blue', density=True)
    plt.hist(filt_amps, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_amps)})', 
             color='red', density=True)
    plt.title('Peak-to-Trough Amplitude Distribution')
    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Average waveform shapes
    plt.subplot(3, 4, 5)
    if acc_valid:
        # Normalize all waveforms to same length for averaging
        target_length = int(np.median([len(wf) for wf in acc_valid]))
        normalized_acc = []
        for wf in acc_valid[:200]:  # Use first 200 for averaging
            if len(wf) > 5:
                x_old = np.linspace(0, 1, len(wf))
                x_new = np.linspace(0, 1, target_length)
                wf_interp = np.interp(x_new, x_old, wf)
                normalized_acc.append(wf_interp)
        
        if normalized_acc:
            mean_acc = np.mean(normalized_acc, axis=0)
            std_acc = np.std(normalized_acc, axis=0)
            x = np.arange(target_length)
            plt.plot(x, mean_acc, 'b-', linewidth=2, label='Mean')
            plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, 
                           alpha=0.3, color='blue', label='±1 STD')
    
    plt.title('Average Accepted Waveform Shape')
    plt.xlabel('Normalized Sample')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 6)
    if filt_valid:
        # Normalize all waveforms to same length for averaging
        target_length = int(np.median([len(wf) for wf in filt_valid]))
        normalized_filt = []
        for wf in filt_valid[:200]:  # Use first 200 for averaging
            if len(wf) > 5:
                x_old = np.linspace(0, 1, len(wf))
                x_new = np.linspace(0, 1, target_length)
                wf_interp = np.interp(x_new, x_old, wf)
                normalized_filt.append(wf_interp)
        
        if normalized_filt:
            mean_filt = np.mean(normalized_filt, axis=0)
            std_filt = np.std(normalized_filt, axis=0)
            x = np.arange(target_length)
            plt.plot(x, mean_filt, 'r-', linewidth=2, label='Mean')
            plt.fill_between(x, mean_filt - std_filt, mean_filt + std_filt, 
                           alpha=0.3, color='red', label='±1 STD')
    
    plt.title('Average Filtered-Out Waveform Shape')
    plt.xlabel('Normalized Sample')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Skewness comparison
    plt.subplot(3, 4, 7)
    acc_skew = [stats.skew(wf) for wf in acc_valid if len(wf) > 3]
    filt_skew = [stats.skew(wf) for wf in filt_valid if len(wf) > 3]
    
    bins = np.linspace(-3, 3, 50)
    plt.hist(acc_skew, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_skew)})', 
             color='blue', density=True)
    plt.hist(filt_skew, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_skew)})', 
             color='red', density=True)
    plt.title('Skewness Distribution')
    plt.xlabel('Skewness')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Kurtosis comparison
    plt.subplot(3, 4, 8)
    acc_kurt = [stats.kurtosis(wf) for wf in acc_valid if len(wf) > 3]
    filt_kurt = [stats.kurtosis(wf) for wf in filt_valid if len(wf) > 3]
    
    bins = np.linspace(-2, 10, 50)
    plt.hist(acc_kurt, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_kurt)})', 
             color='blue', density=True)
    plt.hist(filt_kurt, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_kurt)})', 
             color='red', density=True)
    plt.title('Kurtosis Distribution')
    plt.xlabel('Kurtosis')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7-8. Peak and trough position distributions
    plt.subplot(3, 4, 9)
    acc_peak_pos = [np.argmax(wf) / len(wf) for wf in acc_valid if len(wf) > 0]
    filt_peak_pos = [np.argmax(wf) / len(wf) for wf in filt_valid if len(wf) > 0]
    
    bins = np.linspace(0, 1, 50)
    plt.hist(acc_peak_pos, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_peak_pos)})', 
             color='blue', density=True)
    plt.hist(filt_peak_pos, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_peak_pos)})', 
             color='red', density=True)
    plt.title('Peak Position Distribution')
    plt.xlabel('Relative Peak Position')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 10)
    acc_trough_pos = [np.argmin(wf) / len(wf) for wf in acc_valid if len(wf) > 0]
    filt_trough_pos = [np.argmin(wf) / len(wf) for wf in filt_valid if len(wf) > 0]
    
    bins = np.linspace(0, 1, 50)
    plt.hist(acc_trough_pos, bins=bins, alpha=0.7, label=f'Accepted (n={len(acc_trough_pos)})', 
             color='blue', density=True)
    plt.hist(filt_trough_pos, bins=bins, alpha=0.7, label=f'Filtered (n={len(filt_trough_pos)})', 
             color='red', density=True)
    plt.title('Trough Position Distribution')
    plt.xlabel('Relative Trough Position')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Length vs Amplitude scatter plot
    plt.subplot(3, 4, 11)
    acc_lengths = [len(wf) for wf in acc_valid]
    acc_amps = [np.max(wf) - np.min(wf) for wf in acc_valid]
    
    # Subsample for plotting if too many points
    if len(acc_lengths) > 5000:
        indices = np.random.choice(len(acc_lengths), 5000, replace=False)
        acc_lengths = [acc_lengths[i] for i in indices]
        acc_amps = [acc_amps[i] for i in indices]
    
    plt.scatter(acc_lengths, acc_amps, alpha=0.5, color='blue', s=1, label='Accepted')
    
    filt_lengths = [len(wf) for wf in filt_valid]
    filt_amps = [np.max(wf) - np.min(wf) for wf in filt_valid]
    
    # Subsample for plotting if too many points
    if len(filt_lengths) > 5000:
        indices = np.random.choice(len(filt_lengths), 5000, replace=False)
        filt_lengths = [filt_lengths[i] for i in indices]
        filt_amps = [filt_amps[i] for i in indices]
    
    plt.scatter(filt_lengths, filt_amps, alpha=0.5, color='red', s=1, label='Filtered')
    plt.title('Length vs Amplitude')
    plt.xlabel('Length (samples)')
    plt.ylabel('Peak-to-Trough Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Summary statistics text
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    summary_text = f"""
    FILTERING SUMMARY:
    
    Total Pulses: {len(accepted_waveforms) + len(filtered_waveforms)}
    Accepted: {len(accepted_waveforms)} ({len(accepted_waveforms)/(len(accepted_waveforms)+len(filtered_waveforms))*100:.1f}%)
    Filtered: {len(filtered_waveforms)} ({len(filtered_waveforms)/(len(accepted_waveforms)+len(filtered_waveforms))*100:.1f}%)
    
    ACCEPTED WAVEFORMS:
    Length: {np.mean([len(wf) for wf in acc_valid]):.1f} ± {np.std([len(wf) for wf in acc_valid]):.1f}
    Amplitude: {np.mean([np.max(wf)-np.min(wf) for wf in acc_valid]):.3f} ± {np.std([np.max(wf)-np.min(wf) for wf in acc_valid]):.3f}
    
    FILTERED WAVEFORMS:
    Length: {np.mean([len(wf) for wf in filt_valid]):.1f} ± {np.std([len(wf) for wf in filt_valid]):.1f}
    Amplitude: {np.mean([np.max(wf)-np.min(wf) for wf in filt_valid]):.3f} ± {np.std([np.max(wf)-np.min(wf) for wf in filt_valid]):.3f}
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{output_path}\\{file_prefix}_QC_comparison.png', dpi=300, bbox_inches='tight')
    if show_fig:
        plt.show()

def compare_table_features(accepted_table, filtered_table, output_path, file_prefix, save_fig=True, show_fig=False):
    """Compare tabular features between accepted and filtered events."""
    
    if accepted_table.empty and filtered_table.empty:
        print("No data to compare")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Amplitude ratio comparison
    if 'pp_ratio' in accepted_table.columns and 'pp_ratio' in filtered_table.columns:
        axes[0, 0].hist(accepted_table['pp_ratio'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[0, 0].hist(filtered_table['pp_ratio'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[0, 0].set_title('Amplitude Ratio Distribution')
        axes[0, 0].set_xlabel('Peak-to-Peak Ratio')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Duration comparison
    if 'pp_dur_us' in accepted_table.columns and 'pp_dur_us' in filtered_table.columns:
        axes[0, 1].hist(accepted_table['pp_dur_us'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[0, 1].hist(filtered_table['pp_dur_us'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[0, 1].set_title('Pulse Duration Distribution')
        axes[0, 1].set_xlabel('Duration (μs)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # # 3. Differential vs single-ended
    # if 'is_differential' in accepted_table.columns and 'is_differential' in filtered_table.columns:
    #     acc_diff = accepted_table['is_differential'].value_counts()
    #     filt_diff = filtered_table['is_differential'].value_counts()
        
    #     x = np.arange(len(acc_diff))
    #     width = 0.35
        
    #     axes[0, 2].bar(x - width/2, acc_diff.values, width, label='Accepted', color='blue', alpha=0.7)
    #     axes[0, 2].bar(x + width/2, filt_diff.values, width, label='Filtered', color='red', alpha=0.7)
    #     axes[0, 2].set_title('Differential vs Single-ended')
    #     axes[0, 2].set_xlabel('Type (0=Single, 1=Differential)')
    #     axes[0, 2].set_ylabel('Count')
    #     axes[0, 2].legend()
    #     axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Amplitude by channel
    if 'eod_amplitude' in accepted_table.columns and 'eod_amplitude' in filtered_table.columns:
        axes[1, 0].hist(accepted_table['eod_amplitude'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[1, 0].hist(filtered_table['eod_amplitude'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[1, 0].set_title('EOD Amplitude Distribution')
        axes[1, 0].set_xlabel('Amplitude')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Pulse width distribution
    if 'eod_width_uS' in accepted_table.columns and 'eod_width_uS' in filtered_table.columns:
        axes[1, 1].hist(accepted_table['eod_width_uS'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[1, 1].hist(filtered_table['eod_width_uS'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[1, 1].set_title('Pulse Width Distribution')
        axes[1, 1].set_xlabel('Width (us)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Waveform length distribution
    if 'waveform_length' in accepted_table.columns and 'waveform_length' in filtered_table.columns:
        axes[1, 2].hist(accepted_table['waveform_length'], bins=50, alpha=0.7, 
                       label=f'Accepted (n={len(accepted_table)})', color='blue', density=True)
        axes[1, 2].hist(filtered_table['waveform_length'], bins=50, alpha=0.7, 
                       label=f'Filtered (n={len(filtered_table)})', color='red', density=True)
        axes[1, 2].set_title('Waveform Length Distribution')
        axes[1, 2].set_xlabel('Length (samples)')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{output_path}\\{file_prefix}_QC_table_features.png', dpi=300, bbox_inches='tight')
    if show_fig:
        plt.show()