"""
EOD Analysis Functions Module
Functions for EOD pulse detection, processing, and analysis.

"""

import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from scipy.signal import find_peaks, correlate, windows, find_peaks
from scipy.interpolate import interp1d
from scipy import stats
import glob
from thunderfish import pulses
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.patches as mpatches


def unify_across_channels(peaks, troughs, pulse_widths):
    """
    Unify pulse detections across multiple channels, removing duplicates.

    Parameters:
    -----------
    peaks : list of arrays
        Peak indices for each channel
    troughs : list of arrays
        Trough indices for each channel
    pulse_widths : list of arrays
        Pulse widths for each channel
        
    Returns:
    --------
    tuple : (unique_midpoints, unique_peaks, unique_troughs, unique_widths)
        Arrays of unified pulse parameters
    """
    # Collect all pulses from all channels
    all_pulses = []
    for ch_peaks, ch_troughs, ch_widths in zip(peaks, troughs, pulse_widths):
        n_pairs = min(len(ch_peaks), len(ch_troughs))
        for j in range(n_pairs):
            mp = (ch_peaks[j] + ch_troughs[j]) // 2
            all_pulses.append((mp, ch_peaks[j], ch_troughs[j], ch_widths[j]))

    # Sort by midpoint and remove duplicates
    all_pulses.sort(key=lambda x: x[0])
    unique_pulses = []

    if all_pulses:
        last_mp = all_pulses[0][0]
        unique_pulses.append(all_pulses[0])
        for pulse in all_pulses[1:]:
            if pulse[0] - last_mp > 3:
                unique_pulses.append(pulse)
                last_mp = pulse[0]

    # Unpack to arrays for further analysis
    if unique_pulses:
        unique_midpoints = np.array([e[0] for e in unique_pulses])
        unique_peaks = np.array([e[1] for e in unique_pulses])
        unique_troughs = np.array([e[2] for e in unique_pulses])
        unique_widths = np.array([e[3] for e in unique_pulses])
    else:
        unique_midpoints = np.array([])
        unique_peaks = np.array([])
        unique_troughs = np.array([])
        unique_widths = np.array([])

    return unique_midpoints, unique_peaks, unique_troughs, unique_widths

def extract_pulse_snippets(data, peaks, troughs, rate, length,
                           source, return_differential=True, use_pca=False,
                           pca_component=0, pca_interp_points=100):
    """
    Extract and analyze EOD snippets with variable widths based on detected pulse widths.
    For 2-D multi-channel data (finds polarity flips and extracts differential signals OR
    uses PCA-based noise reduction with spatial interpolation).
    Optimized to store variable-length waveforms without zero-padding for maximum efficiency.
    
    Parameters
    ----------
    data : 2-D array
        The full recording data with channels in columns
    peaks : 1-D array
        Peak indices of unique pulses
    troughs : 1-D array
        Trough indices of unique pulses
    rate : int
        Sampling rate    
    length : int
        Length (in microseconds) of waveform snippets to extract around each pulse midpoint
    source : str
        Source of data:
        - '1ch_diff' : single-channel differential data (control recordings)
        - 'multich_linear' : multi-channel data with linear electrode arrangement (field recordings)
    return_differential : bool
        Whether to keep only differential pulses (default True, ignored if use_pca=True)
    use_pca : bool
        If True, use PCA-based extraction with spatial interpolation instead of differential.
        This method reduces noise by projecting multi-channel data onto principal components
        and reconstructing the signal with spatial interpolation (Zlenko et al., 2024).
        Default False for backward compatibility.
    pca_component : int
        Which principal component to use for signal reconstruction (default 0 = first component).
        Only used when use_pca=True.
    pca_interp_points : int
        Number of interpolation points for 1D spatial interpolation along electrode array.
        Default 100 provides high spatial resolution for locating signal maxima.
        Only used when use_pca=True.
    
    Returns
    -------
    eod_waveforms : list of 1-D arrays
        Variable-length EOD waveform snippets (no zero-padding).
        If use_pca=True: PCA-reconstructed waveforms with spatial interpolation.
        If use_pca=False: Differential waveforms from adjacent channel pairs.
    eod_amps : 1-D array
        Amplitude of extracted waveform
    eod_widths : 1-D array
        Width in microseconds between peak and trough of the extracted waveform
    eod_chan : 1-D array
        Index of channel/channel-pair used for waveform extraction.
        For PCA extraction: electrode with highest spatial amplitude (0 to N-1).
    is_differential : 1-D array
        Indicator of extraction method:
        - 1 = differential waveform (standard method)
        - 0 = single-ended waveform
        - 2 = PCA-based extraction with spatial interpolation
    snippet_p1_idc : 1-D array
        Indices of first dominant peak in the snippet
    snippet_p2_idc : 1-D array
        Indices of second dominant peak in the snippet
    final_p1_idc : 1-D array
        Indices of first dominant peak in original data (for normalized head-positive orientation)
    final_p2_idc : 1-D array
        Indices of second dominant peak in original data (for normalized head-positive orientation)
    pulse_orientation : 1-D array
        Original pulse orientation before normalization ('HP' or 'HN')
    amplitude_ratios : 1-D array
        Peak-to-trough amplitude ratios for filtering
    fft_peak_freqs : 1-D array
        Peak FFT frequency for each waveform
    peak_locations : 1-D array
        Interpolated continuous location of peak amplitude along electrode array.
        Range: 0.0 (first electrode) to N-1 (last electrode) with sub-electrode precision.
        For PCA: Based on cubic spline interpolation of spatial amplitude profile.
        For differential: Based on amplitude-weighted average between channel pairs.
        For single-channel: Always 0.0.
    """

    if source == '1ch_diff':
        return_differential = True
        print("    Single-channel differential data source detected...")
        n_channels = 1
        # Single channel case
        n_pulses = len(peaks)
        eod_chans = np.zeros(n_pulses, dtype=int)
        is_differential = np.ones(n_pulses, dtype=int)  # All differentially recorded
    elif source == 'multich_linear':
        n_channels = data.shape[1]
        print("    Multi-channel linear data source detected...")
        
        # PCA extraction requires multi-channel data
        if use_pca and n_channels < 2:
            print("    Warning: PCA extraction requires multi-channel data. Falling back to standard extraction.")
            use_pca = False
        
        if use_pca:
            print(f"    Using PCA-based extraction with spatial interpolation from {n_channels}-channel data...")
            print(f"    PCA settings: component={pca_component}, interpolation points={pca_interp_points}")
            # For PCA, we don't need differential channel selection
            n_pulses = len(peaks)
            eod_chans = np.zeros(n_pulses, dtype=int)  # Will be updated with peak electrode
            is_differential = np.full(n_pulses, 2, dtype=int)  # 2 = PCA method
        elif n_channels > 1:
            print(f"    Extracting differential waveforms from {n_channels}-channel data...")
            # Find differential channel with polarity flip
            eod_chans, is_differential, _, _ = _select_differential_channel_pointwise(
                data, n_channels, peaks, troughs)
            n_pulses = len(eod_chans)
        else:
            print("    Single-channel data detected, skipping differential channel selection...")
            # Single channel case
            n_pulses = len(peaks)
            eod_chans = np.zeros(n_pulses, dtype=int)
            is_differential = np.zeros(n_pulses, dtype=int)  # All single-ended


    if return_differential and not use_pca:
        print("    Filtering to keep only differential pulses...")
        # Filter non-differential pulses
        keep_mask = is_differential == 1
    else:
        keep_mask = np.array([True] * len(peaks))

    filtered_peak_idc = peaks[keep_mask]
    filtered_trough_idc = troughs[keep_mask]
    filtered_eod_chans = eod_chans[keep_mask]
    n_pulses_diff = len(filtered_peak_idc)

    if return_differential and not use_pca:
        print(f"    Found {n_pulses_diff} differential pulses out of {n_pulses} total")
    elif use_pca:
        print(f"    Processing {n_pulses_diff} pulses with PCA extraction")
        

    # Extract snippets and process waveforms
    eod_waveforms = []
    snippet_p1_idc = []
    snippet_p2_idc = []
    final_p1_idc = []
    final_p2_idc = []
    eod_amps = []
    pulse_orientations = []
    amplitude_ratios = []
    eod_widths = []
    fft_peak_freqs = []
    pca_variance_explained = []  # Track PCA quality
    peak_locations = []  # Track interpolated peak locations (for both PCA and differential)
    re_extract = False
    
    for i in range(n_pulses_diff):
        peak_idx = filtered_peak_idc[i].copy()
        trough_idx = filtered_trough_idc[i].copy()

        # Determine center and length for snippet extraction
        center_idx = (peak_idx + trough_idx) // 2
        wf_length = int(length * rate / 1e6)
        start_idx = max(0, center_idx - wf_length // 2)
        end_idx = min(data.shape[0], center_idx + wf_length // 2)

        # Extract snippet based on extraction method
        if use_pca and source == 'multich_linear':
            # PCA extraction: get all channels
            snippet_multichannel = data[start_idx:end_idx, :]
            
            # Apply PCA-based extraction with spatial interpolation
            try:
                snippet, spatial_amps, var_explained, peak_ch, peak_loc = _extract_pca_waveform(
                    snippet_multichannel, 
                    pca_component=pca_component,
                    interp_points=pca_interp_points,
                    apply_common_average=True
                )
                pca_variance_explained.append(var_explained)
                peak_locations.append(peak_loc)
                # Update the channel assignment to the electrode with strongest signal
                filtered_eod_chans[i] = peak_ch
            except Exception as e:
                # If PCA fails, skip this pulse
                print(f"    Warning: PCA extraction failed for pulse {i}: {e}")
                continue
                
        elif source == '1ch_diff':
            # Single channel differential
            snippet = data[start_idx:end_idx].flatten()
            # Single channel has no spatial information
            peak_locations.append(0.0)
        elif source == 'multich_linear':
            # Standard differential extraction between adjacent channels
            snippet = np.diff(data[start_idx:end_idx, filtered_eod_chans[i]:filtered_eod_chans[i]+2]).flatten()
            # Estimate interpolated location for differential
            diff_location = _estimate_differential_peak_location(
                data, peak_idx, trough_idx, filtered_eod_chans[i], n_channels
            )
            peak_locations.append(diff_location)

        if snippet.shape[0] == 0:
            # If snippet extraction failed, append default values
            eod_waveforms.append(np.array([]))
            # start_idc.append(0)
            snippet_p1_idc.append(0)
            snippet_p2_idc.append(0)
            eod_amps.append(0.0)
            pulse_orientations.append('HP')
            amplitude_ratios.append(0.0)
            eod_widths.append(0)
            fft_peak_freqs.append(0.0)

            continue

        # Re-detect peaks/troughs but constrain search area
        search_window = len(snippet) // 2  # Search within 50% of snippet length
        center_pos = len(snippet) // 2
        
        # Find peaks in constrained region around center
        search_start = max(0, center_pos - search_window)
        search_end = min(len(snippet), center_pos + search_window)
        search_region = snippet[search_start:search_end]
        snippet_peak_idx = np.argmax(search_region)
        snippet_trough_idx = np.argmin(search_region)

        # Update snippet indices based on re-detected peaks/troughs
        if snippet_peak_idx != peak_idx - start_idx: # in this case, the peak has shifted
            peak_diff = snippet_peak_idx - (peak_idx - start_idx)
            filtered_peak_idc[i] += peak_diff
            re_extract = True    

        if snippet_trough_idx != trough_idx - start_idx: # in this case, the trough has shifted
            trough_diff = snippet_trough_idx - (trough_idx - start_idx)
            filtered_trough_idc[i] += trough_diff
            re_extract = True

        if re_extract:
            # Re-extract snippet with updated peak/trough indices
            peak_idx = filtered_peak_idc[i].copy()
            trough_idx = filtered_trough_idc[i].copy()

            # Determine center and length for snippet extraction
            center_idx = (peak_idx + trough_idx) // 2
            start_idx = max(0, center_idx - wf_length // 2)
            end_idx = min(data.shape[0], center_idx + wf_length // 2)

            # Extract snippet based on method
            if use_pca and source == 'multich_linear':
                snippet_multichannel = data[start_idx:end_idx, :]
                try:
                    snippet, spatial_amps, var_explained, peak_ch, peak_loc = _extract_pca_waveform(
                        snippet_multichannel, 
                        pca_component=pca_component,
                        interp_points=pca_interp_points,
                        apply_common_average=True
                    )
                    pca_variance_explained[-1] = var_explained  # Update the last entry
                    peak_locations[-1] = peak_loc  # Update the last entry
                    filtered_eod_chans[i] = peak_ch  # Update channel assignment
                except Exception as e:
                    print(f"    Warning: PCA re-extraction failed for pulse {i}: {e}")
                    continue
            elif source == '1ch_diff':
                snippet = data[start_idx:end_idx].flatten()
                # Update location (single channel)
                peak_locations[-1] = 0.0
            elif source == 'multich_linear':
                snippet = np.diff(data[start_idx:end_idx, filtered_eod_chans[i]:filtered_eod_chans[i]+2]).flatten()
                # Update interpolated location for differential
                diff_location = _estimate_differential_peak_location(
                    data, peak_idx, trough_idx, filtered_eod_chans[i], n_channels
                )
                peak_locations[-1] = diff_location

            # Update snippet-relative peak/trough indices after re-extraction
            snippet_peak_idx = peak_idx - start_idx
            snippet_trough_idx = trough_idx - start_idx

            re_extract = False

        # Calculate width between peak and trough
        eod_width = abs(snippet_peak_idx - snippet_trough_idx) * 1e6 // rate

        # Calculate peak-to-trough amplitude
        eod_amp = abs(snippet[snippet_peak_idx] - snippet[snippet_trough_idx])
    
        # Determine pulse orientation based on peak and trough indices
        if snippet_trough_idx < snippet_peak_idx:
            pulse_orientation = 'HN'  # Head-negative (trough before peak)
        else:
            pulse_orientation = 'HP'  # Head-positive (peak before trough)

        # Calculate amplitude ratio and determine p1 and p2
        if pulse_orientation == 'HP':
            snippet_p1_idc.append(snippet_peak_idx)
            snippet_p2_idc.append(snippet_trough_idx)
            final_p1_idc.append(filtered_peak_idc[i])
            final_p2_idc.append(filtered_trough_idc[i])
            amplitude_ratio = abs(snippet[snippet_peak_idx] / snippet[snippet_trough_idx]) if snippet[snippet_trough_idx] != 0 else np.inf
        else:
            snippet_p1_idc.append(snippet_trough_idx)
            snippet_p2_idc.append(snippet_peak_idx)
            final_p1_idc.append(filtered_trough_idc[i])
            final_p2_idc.append(filtered_peak_idc[i])
            amplitude_ratio = abs(snippet[snippet_trough_idx] / snippet[snippet_peak_idx]) if snippet[snippet_peak_idx] != 0 else np.inf

        # amplitude_ratio = abs(snippet[snippet_peak_idx] / snippet[snippet_trough_idx]) if snippet[snippet_trough_idx] != 0 else np.inf

        # Calculate FFT peak frequency for the processed waveform
        if len(snippet) > 0:
            try:
                fft_peak_snippet = calc_fft_peak(snippet, rate, zero_padding_factor=100)
            except:
                # If FFT calculation fails, set to 0
                fft_peak_snippet = 0.0
        else:
            fft_peak_snippet = 0.0

        # Append results
        eod_waveforms.append(snippet)
        eod_amps.append(eod_amp)
        pulse_orientations.append(pulse_orientation)
        amplitude_ratios.append(amplitude_ratio)
        eod_widths.append(eod_width)
        fft_peak_freqs.append(fft_peak_snippet)

    # Filter out any empty waveforms
    if len(eod_waveforms) == 0:
        # No valid waveforms found
        if use_pca:
            print("    No valid PCA-extracted waveforms found.")
        else:
            print("    No valid differential waveforms found.")
        return ([], np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    else:
        # Convert lists to arrays before returning
        eod_amps = np.array(eod_amps)
        eod_widths = np.array(eod_widths)
        eod_chan = np.array(filtered_eod_chans)  # Use filtered channel list
        
        # Set is_differential based on extraction method
        if use_pca:
            is_differential_filtered = np.full(len(eod_waveforms), 2, dtype=int)  # 2 = PCA method
            if pca_variance_explained:
                mean_var_explained = np.mean(pca_variance_explained)
                print(f"    PCA extraction complete: mean variance explained = {mean_var_explained:.3f}")
        else:
            is_differential_filtered = np.array([1] * len(eod_waveforms))  # All remaining are differential
        
        snippet_p1_idc = np.array(snippet_p1_idc)
        snippet_p2_idc = np.array(snippet_p2_idc)
        final_p1_idc = np.array(final_p1_idc)
        final_p2_idc = np.array(final_p2_idc)
        pulse_orientations = np.array(pulse_orientations)
        amplitude_ratios = np.array(amplitude_ratios)
        fft_peak_freqs = np.array(fft_peak_freqs)
        
        # Convert peak locations to array (populated for both PCA and differential extraction)
        if peak_locations:
            peak_locations_array = np.array(peak_locations)
            mean_peak_loc = np.mean(peak_locations_array)
            if use_pca:
                print(f"    PCA spatial info: mean peak location = {mean_peak_loc:.2f} (electrode units)")
            else:
                print(f"    Differential spatial info: mean peak location = {mean_peak_loc:.2f} (electrode units)")
        else:
            peak_locations_array = np.array([])  # Empty fallback

    # Return variable-length waveforms as list (no zero-padding)
    return (eod_waveforms, eod_amps, eod_widths, eod_chan, 
            is_differential_filtered, snippet_p1_idc, snippet_p2_idc,
            final_p1_idc, final_p2_idc, 
            pulse_orientations, amplitude_ratios, fft_peak_freqs, peak_locations_array)
    

def _select_differential_channel_pointwise(data, n_channels, peaks, troughs):
    """
    Select the best differential channel from multi-channel data.
    Modified version - this one only uses peak and trough (instead of waveform snippet)
    
    Parameters
    ----------
    data : 2-D array
        Multi-channel data
    n_channels : int
        Number of channels
    peaks : 1-D array
        Peak indices of pulses
    troughs : 1-D array
        Trough indices of pulses
    
    Returns
    -------
    eod_chan : 1-D array
        Selected channel index
    is_differential : 1-D array
        1 if differential, 0 if single-ended
    amps : 1-D array
        Amplitudes for each channel
    cor_coeffs : 2-D array
        Correlation coefficients between adjacent channels
    """
    n_pulses = len(peaks)
    amps = np.zeros((n_pulses, n_channels))  # Initialize 2D array
    cor_coeffs = np.zeros((n_pulses, n_channels - 1))
    cor_coeffs_diff = np.zeros((n_pulses, n_channels - 2))
    eod_chan = np.zeros(n_pulses, dtype=int)
    is_differential = np.zeros(n_pulses, dtype=int)

    for i in range(n_pulses):
        if peaks[i] < 0 or troughs[i] < 0:
            continue  # Skip invalid indices
        
        # Extract 2-point snippet = data at peak and trough
        snippet = data[[peaks[i], troughs[i]], :]  # Peak first, trough second

        # Calculate amplitudes for each channel using correct peak/trough rows
        amps[i, :] = abs(np.diff(snippet, axis = 0)[0])  # Store in 2D array

        if snippet.shape[0] > 1:  # Need at least 2 samples for correlation
            for j in range(n_channels - 1):
                if np.var(snippet[:, j]) > 0 and np.var(snippet[:, j+1]) > 0:
                    cor_coeffs[i, j] = np.corrcoef(snippet[:, j], snippet[:, j+1])[0, 1]
    
        # Compute differential signals only where needed
        if n_channels > 1:
            snippet_diff = np.diff(snippet, axis=1)

            # Calculate amplitudes for each differential channel
            amps_diff = abs(np.diff(snippet_diff, axis = 0))[0]
            
            # Calculate correlation coefficients for differential channels
            if snippet_diff.shape[0] > 1:  # Need at least 2 samples for correlation
                for j in range(n_channels - 2):
                    if np.var(snippet_diff[:, j]) > 0 and np.var(snippet_diff[:, j+1]) > 0:
                        cor_coeffs_diff[i, j] = np.corrcoef(snippet_diff[:, j], snippet_diff[:, j+1])[0, 1]

            # Find polarity flips (negative correlations)
            flips = np.where(cor_coeffs[i,:] < 0)[0]
            flips_diff = np.where(cor_coeffs_diff[i,:] < 0)[0]

            if len(flips) > 1:
                # Multiple flips: choose the one with largest amplitude in differential data
                eod_chan[i] = flips[np.argmax(amps_diff[flips])]
                is_differential[i] = 1
                # print(f"Pulse {i}: Multiple flips, chosing highest amplitude difference")         
            elif len(flips) == 0:
                # No flip: use channel with largest single-ended amplitude
                eod_chan[i] = np.argmax(amps[i,:])
                is_differential[i] = 0
                # print(f"Pulse {i}: No flips, chosing largest single-ended amplitude")
            elif len(flips) == 1:
                # If there is no flip in differential channels, this is a noise pulse
                if len(flips_diff) == 0:
                    eod_chan[i] = -1
                    is_differential[i] = -1
                    # print(f"Pulse {i}: 1 flip, no diff flip, noise")

                else:
                    # Single flip: use it
                    eod_chan[i] = flips[0]
                    is_differential[i] = 1
                    # print(f"Pulse {i}: 1 flip, differential channel {eod_chan[i]} :)")

        else:
            # Single channel case
            eod_chan[i] = 0
            is_differential[i] = 0
        
    return eod_chan, is_differential, amps, cor_coeffs


def _estimate_differential_peak_location(data, peak_idx, trough_idx, channel_idx, n_channels):
    """
    Estimate interpolated peak location for differential extraction.
    
    Uses amplitude weighting between adjacent channels to provide sub-electrode
    spatial resolution similar to PCA extraction.
    
    Parameters
    ----------
    data : 2-D array
        Multi-channel data
    peak_idx : int
        Index of peak in original data
    trough_idx : int
        Index of trough in original data
    channel_idx : int
        Index of first channel in differential pair
    n_channels : int
        Total number of channels
    
    Returns
    -------
    weighted_location : float
        Continuous location estimate (channel_idx to channel_idx+1)
    """
    # For differential extraction, we use channel_idx and channel_idx+1
    # Extract the peak and trough values from both channels
    if channel_idx < 0 or channel_idx >= n_channels - 1:
        # Invalid channel or single-ended - use channel center
        return float(max(0, min(channel_idx, n_channels - 1)))
    
    try:
        # Get amplitudes at peak and trough for both channels
        ch1_peak = abs(data[peak_idx, channel_idx])
        ch1_trough = abs(data[trough_idx, channel_idx])
        ch2_peak = abs(data[peak_idx, channel_idx + 1])
        ch2_trough = abs(data[trough_idx, channel_idx + 1])
        
        # Weights = sqrt of amplitude for each channel
        ch1_weight = np.sqrt((ch1_peak + ch1_trough))
        ch2_weight = np.sqrt((ch2_peak + ch2_trough))
        
        # Weight the location between the two channels
        total_weight = ch1_weight + ch2_weight

        weighted_location = (channel_idx * ch1_weight + (channel_idx + 1) * ch2_weight) / total_weight if total_weight > 0 else channel_idx + 0.5
            
        return weighted_location
        
    except (IndexError, ValueError):
        # Fallback to midpoint if any error occurs
        return channel_idx + 0.5


def _extract_pca_waveform(snippet_multichannel, pca_component=0, interp_points=100, 
                         apply_common_average=True):
    """
    Extract EOD waveform using PCA-based noise reduction with 1D spatial interpolation.
    
    This method implements the approach from Zlenko et al. (2024) which uses PCA decomposition
    to separate signal from noise, followed by cubic spline interpolation for high spatial
    resolution along the linear electrode array.
    
    Parameters
    ----------
    snippet_multichannel : 2-D array
        Multi-channel snippet data (M samples × N channels)
    pca_component : int
        Which principal component to use (default 0 = first component with highest variance)
    interp_points : int
        Number of points for 1D spatial interpolation (default 100)
    apply_common_average : bool
        Whether to subtract common-mode signal before PCA (default True)
    
    Returns
    -------
    waveform_interpolated : 1-D array
        PCA-reconstructed waveform with spatial interpolation
    spatial_amplitudes : 1-D array
        Scaled amplitudes at each original electrode position
    explained_variance_ratio : float
        Proportion of variance explained by the selected component
    peak_electrode_channel : int
        Index of the electrode with the highest spatial amplitude (0 to N-1)
    peak_interpolated_location : float
        Interpolated continuous location of peak amplitude along electrode array (0.0 to N-1)
        Provides sub-electrode spatial resolution for peak localization
    """
    from scipy.interpolate import CubicSpline
    
    M, N = snippet_multichannel.shape  # M time samples, N channels
    
    if N < 2:
        raise ValueError("PCA extraction requires at least 2 channels")
    
    # Step 1: Common-mode suppression (optional but recommended)
    if apply_common_average:
        common_avg = np.mean(snippet_multichannel, axis=1, keepdims=True)
        D = snippet_multichannel - common_avg
    else:
        D = snippet_multichannel.copy()
    
    # Step 2: Calculate covariance matrix and perform PCA
    # Covariance of channels (N×N matrix)
    cov_matrix = np.cov(D.T)
    
    # Step 3: Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending order)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[pca_component] / np.sum(eigenvalues)
    
    # Step 4: Calculate projections P = D · C
    # Each column of P is a temporal signal (M samples)
    P = D @ eigenvectors
    
    # Step 5: Select target projection (typically first component)
    pk = P[:, pca_component]  # Shape: (M,)
    ck = eigenvectors[:, pca_component]  # Shape: (N,)
    
    # Step 6: Dimensional scaling - convert to physical units
    # φk = max(|pk|) · ck gives amplitudes at each electrode
    max_proj = np.max(np.abs(pk))
    spatial_amplitudes = max_proj * ck  # Shape: (N,)
    
    # Step 7: 1D Spatial Interpolation along electrode array
    # Physical positions of electrodes (assuming uniform spacing)
    electrode_positions = np.arange(N)
    
    # Create dense interpolation grid
    interp_positions = np.linspace(0, N-1, interp_points)
    
    # Cubic spline interpolation
    cs = CubicSpline(electrode_positions, spatial_amplitudes)
    amplitudes_interpolated = cs(interp_positions)
    
    # Reconstruct waveform: multiply interpolated spatial pattern by temporal signal
    # We need to find the peak position in the interpolated spatial profile
    peak_spatial_idx = np.argmax(np.abs(amplitudes_interpolated))
    
    # Return the temporal waveform scaled by the spatial amplitude at peak location
    # This gives us the "best" signal extracted from the multi-channel data
    waveform_interpolated = pk * (amplitudes_interpolated[peak_spatial_idx] / max_proj)
    
    # Determine which physical electrode has the highest amplitude
    peak_electrode_channel = np.argmax(np.abs(spatial_amplitudes))
    
    # Scale interpolated peak position to electrode coordinate range [0, N-1]
    # This gives the continuous spatial location of the peak signal
    peak_interpolated_location = peak_spatial_idx * (N - 1) / (interp_points - 1)
    
    return waveform_interpolated, spatial_amplitudes, explained_variance_ratio, peak_electrode_channel, peak_interpolated_location


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

def remove_proximity_duplicates(arrays_dict, proximity_threshold=3):
    """
    Remove duplicates based on proximity: same channel and midpoint within threshold samples.
    
    Args:
        arrays_dict: Dictionary containing all arrays to filter
        proximity_threshold: Maximum sample difference to consider as duplicate
    
    Returns:
        filtered_arrays_dict: Dictionary with filtered arrays
        n_removed: Number of duplicates removed
    """
    final_midpoint_idc = arrays_dict['final_midpoint_idc']
    eod_chan = arrays_dict['eod_chan']
    
    unique_mask = np.ones(len(final_midpoint_idc), dtype=bool)
    for i in range(len(final_midpoint_idc)):
        if unique_mask[i]:
            for j in range(i+1, len(final_midpoint_idc)):
                if (eod_chan[i] == eod_chan[j] and 
                    abs(final_midpoint_idc[i] - final_midpoint_idc[j]) <= proximity_threshold):
                    unique_mask[j] = False
    
    n_removed = np.sum(~unique_mask)
    
    # Filter all arrays by the unique mask
    filtered_arrays = {}
    for key, array in arrays_dict.items():
        if key == 'eod_waveforms':
            filtered_arrays[key] = [array[i] for i in range(len(array)) if unique_mask[i]]
        else:
            # Check if array is multidimensional before applying mask
            if np.ndim(array) > 0:
                filtered_arrays[key] = array[unique_mask]
            else:
                # Keep scalar values unchanged
                filtered_arrays[key] = array
    
    return filtered_arrays, n_removed


def remove_exact_duplicates(arrays_dict, parameters):
    """
    Remove pulses with identical peak/trough indices, keeping the best quality pulse.
    
    Args:
        arrays_dict: Dictionary containing all arrays to filter
        parameters: Parameters dictionary for amplitude ratio bounds
    
    Returns:
        filtered_arrays_dict: Dictionary with filtered arrays
        n_removed: Number of duplicates removed
    """
    final_peak_idc = arrays_dict['final_peak_idc']
    final_trough_idc = arrays_dict['final_trough_idc']
    waveform_lengths = arrays_dict['waveform_lengths']
    eod_amps = arrays_dict['eod_amps']
    amplitude_ratios = arrays_dict['amplitude_ratios']
    
    def get_param(param_name):
        return parameters.get(param_name, parameters[param_name])
    
    # Create unique identifier for each pulse based on peak and trough indices
    unique_pulses = {}
    duplicate_indices = []
    
    for i in range(len(final_peak_idc)):
        pulse_key = (final_peak_idc[i], final_trough_idc[i])
        
        if pulse_key in unique_pulses:
            # Found duplicate - keep the one with better characteristics
            existing_idx = unique_pulses[pulse_key]
            
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
                ratio_min = get_param('amplitude_ratio_min')
                ratio_max = get_param('amplitude_ratio_max')
                ratio_mid = (ratio_min + ratio_max) / 2
                
                current_ratio_distance = abs(current_ratio - ratio_mid)
                existing_ratio_distance = abs(existing_ratio - ratio_mid)
                keep_current = current_ratio_distance < existing_ratio_distance
            
            if keep_current:
                # Mark the existing one for removal and update the dictionary
                duplicate_indices.append(existing_idx)
                unique_pulses[pulse_key] = i
            else:
                # Mark the current one for removal
                duplicate_indices.append(i)
        else:
            # First occurrence of this peak/trough pair
            unique_pulses[pulse_key] = i
    
    n_removed = len(duplicate_indices)
    
    # Remove exact duplicates from all arrays and lists
    if duplicate_indices:
        # Create mask for non-duplicate indices
        non_duplicate_mask = np.ones(len(arrays_dict['eod_waveforms']), dtype=bool)
        non_duplicate_mask[duplicate_indices] = False
        
        # Filter all arrays and lists
        filtered_arrays = {}
                
        for key, array in arrays_dict.items():
            if key == 'eod_waveforms':
                filtered_arrays[key] = [array[i] for i in range(len(array)) if non_duplicate_mask[i]]
            else:
                # Check if array is multidimensional before applying mask
                if np.ndim(array) > 0:
                    filtered_arrays[key] = array[non_duplicate_mask]
                else:
                    # Keep scalar values unchanged
                    filtered_arrays[key] = array
                
    else:
        # No duplicates found, return original arrays
        filtered_arrays = arrays_dict.copy()
    
    return filtered_arrays, n_removed


def remove_duplicates(eod_snippets, eod_amps, eod_widths, eod_chan, is_differential,
                        snippet_peak_idc, snippet_trough_idc, raw_peak_idc, raw_trough_idc,
                        pulse_orientation, amp_ratios, fft_peak_freqs, peak_locations, parameters):
    """
    Complete duplicate removal pipeline for EOD detection results.
    
    Args:
        All the arrays returned from extract_pulse_snippets()
        parameters: Parameters dictionary
    
    Returns:
        All filtered arrays in the same order as input
    """
    print(f"    Starting duplicate removal from {len(eod_snippets)} pulses")
    
    # Create arrays dictionary for easier handling
    arrays_dict = {
        'eod_waveforms': eod_snippets,
        'eod_amps': eod_amps,
        'eod_widths': eod_widths,
        'ch_amps': np.zeros((len(eod_snippets), 1)),  # Placeholder - needs actual implementation
        'ch_cor_coeffs': np.zeros((len(eod_snippets), 1)),  # Placeholder - needs actual implementation
        'eod_chan': eod_chan,
        'is_differential': is_differential,
        'snippet_peak_idc': snippet_peak_idc,
        'snippet_trough_idc': snippet_trough_idc,
        'snippet_midpoint_idc': (snippet_peak_idc + snippet_trough_idc) // 2,  # Calculate if needed
        'final_peak_idc': raw_peak_idc,
        'final_trough_idc': raw_trough_idc,
        'final_midpoint_idc': (raw_peak_idc + raw_trough_idc) // 2,
        'original_pulse_orientation': pulse_orientation,
        'amplitude_ratios': amp_ratios,
        'waveform_lengths': np.array([len(wf) for wf in eod_snippets]),
        'fft_peak_freqs': fft_peak_freqs,
        'peak_locations': peak_locations
    }
    
    # Step 1: Remove proximity-based duplicates
    arrays_dict, n_proximity_removed = remove_proximity_duplicates(arrays_dict)
    print(f"    Removed {n_proximity_removed} proximity-based duplicate pulses")
    
    # Step 2: Remove exact duplicates
    arrays_dict, n_exact_removed = remove_exact_duplicates(arrays_dict, parameters)
    if n_exact_removed > 0:
        print(f"      Removing {n_exact_removed} exact duplicate pulses")
        print(f"      After exact duplicate removal: {len(arrays_dict['eod_waveforms'])} pulses")
    else:
        print(f"      No exact duplicates found")
    
    # Return filtered arrays in original order
    return (
        arrays_dict['eod_waveforms'],
        arrays_dict['eod_amps'], 
        arrays_dict['eod_widths'],
        arrays_dict['eod_chan'],
        arrays_dict['is_differential'],
        arrays_dict['snippet_peak_idc'],
        arrays_dict['snippet_trough_idc'],
        arrays_dict['final_peak_idc'],
        arrays_dict['final_trough_idc'],
        arrays_dict['original_pulse_orientation'],
        arrays_dict['amplitude_ratios'],
        arrays_dict['fft_peak_freqs'],
        arrays_dict['peak_locations']
    )


def filter_waveforms(eod_waveforms, eod_widths, amplitude_ratios, fft_peak_freqs, rate,
                     dur_min=20, dur_max=300,
                     pp_r_min=0.1, pp_r_max=5,
                     fft_freq_min=1000, fft_freq_max=10000,
                     interp_factor=1, return_features=False,
                     return_filteredout_features=False,
                     return_params=False):
    """
    Filter variable-length waveforms based on duration, amplitude ratio, and FFT frequency.
    Updated to work with extract_pulse_snippets output format.

    Parameters
    ----------
    eod_waveforms : list of 1-D arrays
        List of variable-length EOD waveforms from extract_pulse_snippets.
    eod_widths : 1-D array
        Width in microseconds between peak and trough from extract_pulse_snippets.
    amplitude_ratios : 1-D array
        Peak-to-trough amplitude ratios from extract_pulse_snippets.
    fft_peak_freqs : 1-D array
        Pre-calculated FFT peak frequencies from extract_pulse_snippets.
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
    return_filteredout_features : bool, optional
        Returns dataframe with filtered out features if True.
    return_params : bool, optional
        Returns dataframe with filter parameters if True.

    Returns
    -------
    keep_indices : 1-D array
        Indices of the kept waveforms.
    """
    n_snippets = len(eod_waveforms)
    wf_rate = rate * interp_factor

    # Use the pre-calculated eod_width (already in microseconds)
    wf_durs = eod_widths.copy()
    
    # Use the pre-calculated amplitude ratios
    wf_ratios = amplitude_ratios.copy()

    # Use the pre-calculated FFT peak frequencies
    fft_freqs = fft_peak_freqs.copy()

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

    if return_filteredout_features:
        all_indices = np.arange(len(eod_waveforms))
        filtered_out_indices = np.setdiff1d(all_indices, keep_indices)
        filteredout_features = pd.DataFrame({
            'pp_dur_us': wf_durs[filtered_out_indices],
            'pp_ratio': wf_ratios[filtered_out_indices],
            'fft_freq': fft_freqs[filtered_out_indices]
        })
        return_vars.append(filteredout_features)

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


def filter_waveforms_with_classifier(eod_waveforms, eod_widths, amplitude_ratios, fft_peak_freqs, rate,
                                    classifier=None, scaler=None,
                                    dur_min=20, dur_max=300,
                                    pp_r_min=0.1, pp_r_max=5,
                                    fft_freq_min=1000, fft_freq_max=10000,
                                    fish_probability_threshold=0.5,
                                    use_basic_filtering=True,
                                    interp_factor=1, return_features=False,
                                    return_filteredout_features=False,
                                    return_params=False):
    """
    Enhanced waveform filtering using trained Random Forest classifier for fish vs noise discrimination.
    
    This function combines traditional threshold-based filtering with machine learning classification
    to improve pulse quality assessment.

    Parameters
    ----------
    eod_waveforms : list of 1-D arrays
        List of variable-length EOD waveforms from extract_pulse_snippets.
    eod_widths : 1-D array
        Width in microseconds between peak and trough from extract_pulse_snippets.
    amplitude_ratios : 1-D array
        Peak-to-trough amplitude ratios from extract_pulse_snippets.
    fft_peak_freqs : 1-D array
        Pre-calculated FFT peak frequencies from extract_pulse_snippets.
    rate : int
        Sample rate of the original signal.
    classifier : sklearn estimator, optional
        Pre-loaded trained classifier object. If None, uses basic filtering only.
    scaler : sklearn StandardScaler, optional
        Pre-loaded feature scaler object. Required if classifier is provided.
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
    fish_probability_threshold : float, optional
        Minimum probability threshold for classifying as fish (default is 0.5).
    use_basic_filtering : bool, optional
        Whether to apply basic threshold filtering before ML classification (default is True).
    interp_factor : int, optional
        Factor by which the signal is interpolated (default is 1).
    return_features : bool, optional
        Returns dataframe with waveform features if True.
    return_filteredout_features : bool, optional
        Returns dataframe with filtered out features if True.
    return_params : bool, optional
        Returns dataframe with filter parameters if True.

    Returns
    -------
    keep_indices : 1-D array
        Indices of the kept waveforms.
    [features] : pandas.DataFrame, optional
        Features of kept waveforms if return_features=True.
    [filteredout_features] : pandas.DataFrame, optional
        Features of filtered out waveforms if return_filteredout_features=True.
    [params] : pandas.DataFrame, optional
        Filter parameters if return_params=True.
    """
    
    print(f"    Enhanced filtering: {len(eod_waveforms)} pulses to evaluate")
    
    n_snippets = len(eod_waveforms)
    wf_rate = rate * interp_factor

    # Use the pre-calculated features
    wf_durs = eod_widths.copy()
    wf_ratios = amplitude_ratios.copy()
    fft_freqs = fft_peak_freqs.copy()

    # Initialize keep mask - start with all True
    keep_mask = np.ones(n_snippets, dtype=bool)
    
    # Track filtering stages
    filtering_stages = {
        'initial': n_snippets,
        'after_basic': n_snippets,
        'after_ml': n_snippets
    }

    # Stage 1: Basic threshold filtering (if enabled)
    if use_basic_filtering:
        basic_mask = (
            (wf_durs >= dur_min) & (wf_durs <= dur_max) &
            (wf_ratios >= pp_r_min) & (wf_ratios <= pp_r_max) &
            (fft_freqs >= fft_freq_min) & (fft_freqs <= fft_freq_max)
        )
        keep_mask = keep_mask & basic_mask
        filtering_stages['after_basic'] = np.sum(keep_mask)
        print(f"    Basic filtering: {filtering_stages['initial'] - filtering_stages['after_basic']} pulses removed")

    # Stage 2: Machine Learning Classification (if classifier available)
    ml_predictions = None
    ml_probabilities = None
    
    if classifier is not None and scaler is not None:
        try:
            print(f"    Applying ML classification for fish vs noise discrimination")
            
            # Prepare features for classification
            # Note: feature order should be ['eod_width_us', 'fft_freq_max', 'eod_amplitude_ratio']
            classification_features = np.column_stack([
                wf_durs,      # eod_width_us
                fft_freqs,    # fft_freq_max  
                wf_ratios     # eod_amplitude_ratio
            ])
            
            # Remove any NaN values for classification
            valid_feature_mask = ~np.any(np.isnan(classification_features), axis=1)
            
            if np.sum(valid_feature_mask) > 0:
                # Scale features
                features_scaled = scaler.transform(classification_features[valid_feature_mask])
                
                # Predict probabilities
                probabilities = classifier.predict_proba(features_scaled)
                fish_probabilities = probabilities[:, 1]  # Probability of being fish (class 1)
                
                # Create full probability array
                ml_probabilities = np.full(n_snippets, np.nan)
                ml_probabilities[valid_feature_mask] = fish_probabilities
                
                # Apply probability threshold
                ml_mask = np.ones(n_snippets, dtype=bool)  # Default to keep
                ml_mask[valid_feature_mask] = fish_probabilities >= fish_probability_threshold
                
                # Combine with existing keep mask
                keep_mask = keep_mask & ml_mask
                
                filtering_stages['after_ml'] = np.sum(keep_mask)
                ml_removed = filtering_stages['after_basic'] - filtering_stages['after_ml']
                print(f"    ML classification: {ml_removed} additional pulses removed (noise)")
                print(f"    Fish probability threshold: {fish_probability_threshold}")
                
                # Store predictions for return
                ml_predictions = np.full(n_snippets, np.nan)
                ml_predictions[valid_feature_mask] = (fish_probabilities >= fish_probability_threshold).astype(int)
                
            else:
                print(f"    Warning: No valid features for ML classification")
                
        except Exception as e:
            print(f"    Warning: ML classification failed: {e}")
            print(f"    Continuing with basic filtering only")
    
    elif classifier is not None or scaler is not None:
        print(f"    Warning: Both classifier and scaler must be provided for ML classification")
        print(f"    Continuing with basic filtering only")
    
    # Get final keep indices
    keep_indices = np.where(keep_mask)[0]
    
    print(f"    Final result: {len(keep_indices)} / {n_snippets} pulses kept ({len(keep_indices)/n_snippets*100:.1f}%)")

    # Prepare return variables
    return_vars = [keep_indices]

    if return_features:
        features_dict = {
            'pp_dur_us': wf_durs[keep_indices],
            'pp_ratio': wf_ratios[keep_indices],
            'fft_freq': fft_freqs[keep_indices]
        }
        
        # Add ML predictions if available
        if ml_predictions is not None:
            features_dict['ml_fish_prediction'] = ml_predictions[keep_indices]
        if ml_probabilities is not None:
            features_dict['ml_fish_probability'] = ml_probabilities[keep_indices]
            
        features = pd.DataFrame(features_dict)
        return_vars.append(features)

    if return_filteredout_features:
        all_indices = np.arange(len(eod_waveforms))
        filtered_out_indices = np.setdiff1d(all_indices, keep_indices)
        
        filteredout_dict = {
            'pp_dur_us': wf_durs[filtered_out_indices],
            'pp_ratio': wf_ratios[filtered_out_indices],
            'fft_freq': fft_freqs[filtered_out_indices]
        }
        
        # Add ML predictions for filtered out pulses if available
        if ml_predictions is not None:
            filteredout_dict['ml_fish_prediction'] = ml_predictions[filtered_out_indices]
        if ml_probabilities is not None:
            filteredout_dict['ml_fish_probability'] = ml_probabilities[filtered_out_indices]
            
        filteredout_features = pd.DataFrame(filteredout_dict)
        return_vars.append(filteredout_features)

    if return_params:
        params_dict = {
            'dur_min': [dur_min],
            'dur_max': [dur_max],
            'pp_r_min': [pp_r_min],
            'pp_r_max': [pp_r_max],
            'fft_freq_min': [fft_freq_min],
            'fft_freq_max': [fft_freq_max],
            'fish_probability_threshold': [fish_probability_threshold],
            'use_basic_filtering': [use_basic_filtering],
            'classifier_used': [classifier is not None and scaler is not None]
        }
        
        params = pd.DataFrame(params_dict)
        return_vars.append(params)

    return return_vars


def save_fixed_length_waveforms(waveforms_list, output_path, format="csv"):
    """
    Save fixed-length waveforms in specified format.
    
    Parameters
    ----------
    waveforms_list : list of 1D arrays
        List of waveforms (all same length)
    output_path : str
        Base path for output files (without extension)
    format : str, optional
        Output format: 'csv' (default, human-readable) or 'npz' (compressed binary)
    
    Returns
    -------
    metadata : dict
        Basic metadata about the saved waveforms
    """
    if format not in ["csv", "npz"]:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'npz'.")
    
    if not waveforms_list:
        # Save empty file
        if format == "csv":
            pd.DataFrame().to_csv(f"{output_path}.csv", index=False)
        else:  # npz
            np.savez_compressed(f"{output_path}.npz", waveforms=np.array([]))
        
        metadata = {
            'n_waveforms': 0,
            'waveform_length': 0,
            'total_samples': 0,
            'shape': '(0, 0)' if format == "csv" else (0, 0),
            'dtype': 'float64'
        }
    else:
        # Stack waveforms into 2D array (n_waveforms, waveform_length)
        waveforms_array = np.stack(waveforms_list, axis=0)
        
        if format == "csv":
            # Save as CSV (each row is a waveform, each column is a time point)
            waveforms_df = pd.DataFrame(waveforms_array)
            waveforms_df.to_csv(f"{output_path}.csv", index=False)
        else:  # npz
            # Save as compressed npz
            np.savez_compressed(f"{output_path}.npz", waveforms=waveforms_array)
        
        metadata = {
            'n_waveforms': len(waveforms_list),
            'waveform_length': len(waveforms_list[0]),
            'total_samples': waveforms_array.size,
            'shape': str(waveforms_array.shape) if format == "csv" else waveforms_array.shape,
            'dtype': str(waveforms_array.dtype)
        }
    
    # Save minimal metadata
    if format == "csv":
        metadata_df = pd.DataFrame([metadata])
        metadata_df.to_csv(f"{output_path}_metadata.csv", index=False)
    else:  # npz -> json
        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, separators=(',', ':'))
    
    return metadata


def load_fixed_length_waveforms(base_path, format="csv"):
    """
    Load fixed-length waveforms from specified format.
    
    Parameters
    ----------
    base_path : str
        Base path (without extension)
    format : str, optional
        Input format: 'csv' (default, human-readable) or 'npz' (compressed binary)
    
    Returns
    -------
    waveforms_list : list of 1D arrays
        List of waveforms
    """
    if format not in ["csv", "npz"]:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'npz'.")
    
    try:
        if format == "csv":
            # Load from CSV file
            waveforms_df = pd.read_csv(f"{base_path}.csv")
            
            if waveforms_df.empty:
                return []
            
            # Convert to numpy array
            waveforms_array = waveforms_df.values
        else:  # npz
            # Load from npz file
            data = np.load(f"{base_path}.npz")
            waveforms_array = data['waveforms']
            
            if waveforms_array.size == 0:
                return []
        
        # Convert back to list of 1D arrays
        waveforms_list = [waveforms_array[i] for i in range(waveforms_array.shape[0])]
        
        return waveforms_list
        
    except FileNotFoundError:
        file_ext = ".csv" if format == "csv" else ".npz"
        print(f"Warning: File {base_path}{file_ext} not found")
        return []
    except Exception as e:
        print(f"Error loading waveforms: {e}")
        return []


############################### EVENT EXTRACTION ######################################
def create_channel_events(eod_table, max_ipi_seconds, verbose=False):
    """
    Create channel-wise events from EOD detections using temporal clustering.
    
    Parameters:
    -----------
    eod_table : pd.DataFrame
        DataFrame containing EOD detections with columns: 'timestamp', 'eod_channel', 'eod_amplitude'
    max_ipi_seconds : float
        Maximum inter-pulse interval for temporal clustering
    verbose : bool, default False
        Whether to print detailed processing information
        
    Returns:
    --------
    pd.DataFrame
        Channel events with added metadata columns
    """
    # Keep track of indices
    eod_table['original_index'] = eod_table.index

    # Sort by timestamp
    sorted_table = eod_table.sort_values('timestamp').copy()
    sorted_table['timestamp_dt'] = pd.to_datetime(sorted_table['timestamp'])

    # STAGE 1: Extract channel-wise events using temporal criteria only
    print("  Extracting events per channel ...")

    channel_events_list = []
    channel_event_counter = 0

    # Get unique channels
    channels = sorted(sorted_table['eod_channel'].unique())
    print(f"  Processing {len(channels)} channels: {channels}")

    for channel in channels:
        if verbose:
            print(f"    Processing channel {channel}...")
        
        # Get EODs for this channel
        channel_eods = sorted_table[sorted_table['eod_channel'] == channel].copy()
        
        if len(channel_eods) == 0:
            if verbose:
                print(f"      No EODs found for channel {channel}")
            continue

        if verbose:    
            print(f"      Found {len(channel_eods)} EODs")
        
        # Calculate inter-pulse intervals for this channel
        time_diffs = channel_eods['timestamp_dt'].diff().dt.total_seconds()
        
        # Identify event boundaries (gaps > max_ipi_seconds)
        event_boundaries = np.where(time_diffs > max_ipi_seconds)[0]
        
        # Create event segments for this channel
        event_starts = np.concatenate([[0], event_boundaries])
        event_ends = np.concatenate([event_boundaries, [len(channel_eods)]])
        
        if verbose:    
            print(f"      Found {len(event_starts)} temporal clusters")
        
        # Create channel events
        for start_idx, end_idx in zip(event_starts, event_ends):
            event_eods = channel_eods.iloc[start_idx:end_idx].copy()
            
            # Add channel event metadata
            event_eods['channel_event_id'] = channel_event_counter
            event_eods['channel'] = channel
            event_eods['channel_start_time'] = event_eods['timestamp_dt'].min()
            event_eods['channel_end_time'] = event_eods['timestamp_dt'].max()
            event_eods['channel_n_eods'] = len(event_eods)
            
            channel_events_list.append(event_eods)
            channel_event_counter += 1
            
            if verbose:    
                print(f"        Channel event {channel_event_counter-1}: {len(event_eods)} EODs")

    if not channel_events_list:
        print("  No channel events created!")
        return pd.DataFrame()
        
    channel_events = pd.concat(channel_events_list, ignore_index=True)
    print(f"  {len(channel_events)} total EODs in {channel_event_counter} channel events")
    
    return channel_events


def merge_channel_events(channel_events, max_merge_gap_seconds, verbose=False):
    """
    Merge channel events that are spatially and temporally connected.
    
    Parameters:
    -----------
    channel_events : pd.DataFrame
        DataFrame containing channel events with columns: channel_event_id, channel, 
        channel_start_time, channel_end_time, channel_n_eods
    max_merge_gap_seconds : float
        Maximum gap in seconds for merging neighboring channel events
    verbose : bool, default False
        Whether to print detailed processing information
        
    Returns:
    --------
    pd.DataFrame
        Merged events with additional columns: merged_event_id, merge_order, merge_iteration
    """
    # Get unique channel event summaries
    event_summaries = (channel_events
                    .groupby('channel_event_id')
                    .agg({
                        'channel': 'first',
                        'channel_start_time': 'first', 
                        'channel_end_time': 'first',
                        'channel_n_eods': 'first'
                    })
                    .reset_index())

    print(f"    Processing {len(event_summaries)} channel events for merging...")
    
    # Initialize merging data structures
    merged_events_list = []
    merged_event_counter = 0
    processed_channel_events = set()

    # Sort by start time for processing
    event_summaries = event_summaries.sort_values('channel_start_time')

    for idx, current_event in event_summaries.iterrows():
        current_id = current_event['channel_event_id']
        
        if current_id in processed_channel_events:
            continue
        
        if verbose:
            print(f"    Processing channel event {current_id} (channel {current_event['channel']})...")
        
        # Start a new merged event with iterative growing
        merged_group = [current_id]
        processed_channel_events.add(current_id)
        
        # Initialize merged event bounds
        current_start = current_event['channel_start_time']
        current_end = current_event['channel_end_time']
        current_channels = {current_event['channel']}
        
        # ITERATIVE MERGING: Keep expanding until no more events can be merged
        found_new_merge = True
        iteration = 0
        merge_iteration = {}
        
        while found_new_merge:
            found_new_merge = False
            iteration += 1
            # Convert datetime objects to pandas Timestamps for safe formatting
            current_start_pd = pd.Timestamp(current_start)
            current_end_pd = pd.Timestamp(current_end)
            if verbose:
                print(f"      Iteration {iteration}: bounds=({current_start_pd.strftime('%H:%M:%S')}-{current_end_pd.strftime('%H:%M:%S')}), channels={sorted(current_channels)}")

            # --- Spatially AND temporally connected merging ---
            # For each channel currently in the merged event, check its specific temporal bounds
            # Only merge neighbor events that are connected in BOTH space AND time
            events_to_merge_this_iteration = []
            new_channels_this_iteration = set()
            
            # For each channel in current merged event, get its specific temporal bounds
            channel_time_bounds = {}
            for ch in current_channels:
                # Find all events in merged_group for this channel and get their combined time bounds
                ch_event_ids = [eid for eid in merged_group if event_summaries.loc[event_summaries['channel_event_id'] == eid, 'channel'].values[0] == ch]
                if ch_event_ids:
                    ch_starts = [event_summaries.loc[event_summaries['channel_event_id'] == eid, 'channel_start_time'].values[0] for eid in ch_event_ids]
                    ch_ends = [event_summaries.loc[event_summaries['channel_event_id'] == eid, 'channel_end_time'].values[0] for eid in ch_event_ids]
                    channel_time_bounds[ch] = (min(ch_starts), max(ch_ends))
            
            # For each channel with events in the merged group, check its spatial neighbors
            for ch, (ch_start, ch_end) in channel_time_bounds.items():
                for neighbor_ch in [ch-1, ch+1]:
                    if neighbor_ch < 0 or neighbor_ch > 6:
                        continue
                        
                    # Find unprocessed events on neighbor channel
                    neighbor_events = event_summaries[
                        (event_summaries['channel'] == neighbor_ch) &
                        (~event_summaries['channel_event_id'].isin(processed_channel_events))
                    ]
                    
                    for _, neighbor_event in neighbor_events.iterrows():
                        neighbor_id = neighbor_event['channel_event_id']
                        if neighbor_id in processed_channel_events:
                            continue
                            
                        neighbor_start = neighbor_event['channel_start_time']
                        neighbor_end = neighbor_event['channel_end_time']
                        
                        # Direct temporal overlap between this channel and neighbor
                        direct_overlap = (neighbor_end >= ch_start) and (neighbor_start <= ch_end)
                        
                        # Sequential connection within gap tolerance
                        gap_after = (neighbor_start - ch_end).total_seconds() if neighbor_start > ch_end else float('inf')
                        gap_before = (ch_start - neighbor_end).total_seconds() if ch_start > neighbor_end else float('inf')
                        sequential_connection = min(gap_after, gap_before) <= max_merge_gap_seconds
                        
                        if direct_overlap:
                            gap = 0
                            gap_type = 'direct_overlap'
                            should_merge = True
                        elif sequential_connection:
                            gap = min(gap_after, gap_before)
                            gap_type = 'sequential_connection'
                            should_merge = True
                        else:
                            gap = min(gap_after, gap_before)
                            gap_type = 'too_distant'
                            should_merge = False
                        
                        if should_merge:
                            # Avoid duplicates - check if this neighbor event is already in the merge list
                            already_added = any(event_info['event_id'] == neighbor_id for event_info in events_to_merge_this_iteration)
                            if not already_added:
                                # Convert to pandas Timestamp for safe formatting
                                ch_start_pd = pd.Timestamp(ch_start)
                                ch_end_pd = pd.Timestamp(ch_end)
                                if verbose:
                                    print(f"          CANDIDATE for merging: event {neighbor_id} ch{neighbor_ch} ({gap_type}, gap: {gap:.2f}s) connected to ch{ch} ({ch_start_pd.strftime('%H:%M:%S')}-{ch_end_pd.strftime('%H:%M:%S')})")
                                events_to_merge_this_iteration.append({
                                    'event_id': neighbor_id,
                                    'channel': neighbor_ch,
                                    'start_time': neighbor_start,
                                    'end_time': neighbor_end,
                                    'gap': gap,
                                    'gap_type': gap_type,
                                    'connected_to_channel': ch
                                })
                                processed_channel_events.add(neighbor_id)

            # Merge all qualifying events simultaneously
            if events_to_merge_this_iteration:
                if verbose:
                    print(f"        MERGING {len(events_to_merge_this_iteration)} events simultaneously:")
                events_to_merge_this_iteration.sort(key=lambda x: x['gap'])
                for event_info in events_to_merge_this_iteration:
                    neighbor_id = event_info['event_id']
                    neighbor_channel = event_info['channel']
                    neighbor_start = event_info['start_time']
                    neighbor_end = event_info['end_time']
                    if verbose:
                        print(f"          - Event {neighbor_id} ch{neighbor_channel} ({event_info['gap_type']}, gap: {event_info['gap']:.2f}s)")
                    if neighbor_id not in merged_group:
                        merged_group.append(neighbor_id)
                        merge_iteration[neighbor_id] = iteration
                    if neighbor_id not in processed_channel_events:
                        processed_channel_events.add(neighbor_id)
                    if neighbor_channel not in new_channels_this_iteration:
                        new_channels_this_iteration.add(neighbor_channel)
                    if neighbor_channel not in current_channels:
                        current_channels.add(neighbor_channel)
                    current_start = min(current_start, neighbor_start)
                    current_end = max(current_end, neighbor_end)
                found_new_merge = True
                # Convert datetime objects to pandas Timestamps for safe formatting
                current_start_pd = pd.Timestamp(current_start)
                current_end_pd = pd.Timestamp(current_end)
                if verbose:
                    print(f"          UPDATED bounds after parallel merge: ({current_start_pd.strftime('%H:%M:%S')}-{current_end_pd.strftime('%H:%M:%S')})")
            else:
                if verbose:
                    print(f"        No qualifying events found for any temporal edge of channels {sorted(current_channels)}")
            if not found_new_merge:
                if verbose:
                    print(f"        No more mergeable neighbors found for any temporal edge of channels {sorted(current_channels)}")
            elif len(new_channels_this_iteration) == 0:
                if verbose:
                    print(f"        Temporal bounds changed but no new channels added - stopping to prevent infinite loop")
                found_new_merge = False
        
        # Assign merged event ID to all EODs in the final merged group
        if verbose:
            print(f"      Final merged group after {iteration} iterations: {merged_group} -> merged_event_id {merged_event_counter}")
        # Convert datetime objects to pandas Timestamps for safe formatting
        current_start_pd = pd.Timestamp(current_start)
        current_end_pd = pd.Timestamp(current_end)
        if verbose:
            print(f"      Final channels: {sorted(current_channels)}, final bounds: ({current_start_pd.strftime('%H:%M:%S')}-{current_end_pd.strftime('%H:%M:%S')})")

        for merge_order, channel_event_id in enumerate(merged_group):
            event_eods = channel_events[channel_events['channel_event_id'] == channel_event_id].copy()
            event_eods['merged_event_id'] = merged_event_counter
            event_eods['merge_order'] = merge_order  # Store the order in which this channel event was merged
            event_eods['merge_iteration'] = merge_iteration.get(channel_event_id, 1)
            event_eods['event_start_time'] = current_start_pd
            event_eods['event_end_time'] = current_end_pd
            merged_events_list.append(event_eods)

        merged_event_counter += 1

    # Clean up merging loop variables
    del processed_channel_events, merged_group, merge_iteration

    if merged_events_list:
        merged_events = pd.concat(merged_events_list, ignore_index=True)
        print(f"  {len(merged_events)} total EODs in {merged_event_counter} merged events")
        
        # Clean up merging variables to free memory
        del merged_events_list
        del event_summaries
        gc.collect()
        
        return merged_events
    else:
        print("  No merged events found!")
        return pd.DataFrame()
    

def filter_events(events_to_filter, min_eods_per_event, min_amplitude):
    """
    Filter events based on size and amplitude criteria.
    
    Parameters:
    -----------
    events_to_filter : pd.DataFrame
        Events that don't reach file end and need filtering
    min_eods_per_event : int
        Minimum number of EODs required per event
    min_amplitude : float
        Minimum amplitude threshold for events
    event_counter : int
        Current global event counter
        
    Returns:
    --------
    events_filtered : pd.DataFrame
        Filtered events with reassigned sequential IDs
    event_counter : int
        Updated global event counter
    """
    print(f"  Removing events with <{min_eods_per_event} pulses OR no pulse ≥{min_amplitude} amplitude...")

    if 'channel_event_id' in events_to_filter.columns and 'merged_event_id' not in events_to_filter.columns:
        id_col = 'channel_event_id'
    elif 'channel_event_id' in events_to_filter.columns and 'merged_event_id' in events_to_filter.columns:
        id_col = 'merged_event_id'

    # Criterion 1: Size filter (at least min_eods_per_event EODs)
    # Count EODs per merged event
    event_sizes = events_to_filter.groupby(id_col).size()

    print(f"  Event sizes: min={event_sizes.min()}, max={event_sizes.max()}, mean={event_sizes.mean():.1f}")
    valid_size_ids = event_sizes[event_sizes >= min_eods_per_event].index

    # Criterion 2: Amplitude filter (at least one pulse with amplitude ≥ min_amplitude)
    event_max_amplitudes = events_to_filter.groupby(id_col)['eod_amplitude'].max()
    valid_amplitude_ids = event_max_amplitudes[event_max_amplitudes >= min_amplitude].index

    # Combined filter: channel events must pass BOTH criteria
    valid_event_ids = valid_size_ids.intersection(valid_amplitude_ids)

    # Calculate removal statistics
    n_total = events_to_filter[id_col].nunique()
    n_removed_size = n_total - len(valid_size_ids)
    n_removed_amplitude = n_total - len(valid_amplitude_ids)
    n_removed_total = n_total - len(valid_event_ids)

    print(f"  Events passing size filter: {len(valid_event_ids)} / {len(event_sizes)}")
    print(f"    Original channel events: {n_total}")
    print(f"    Failed size criterion (<{min_eods_per_event} pulses): {n_removed_size}")
    print(f"    Failed amplitude criterion (max amplitude <{min_amplitude}): {n_removed_amplitude}")
    print(f"    Remaining after combined filter: {len(valid_event_ids)}")
    print(f"    Total removed: {n_removed_total}")

    if len(valid_event_ids) == 0:
        print("  No events passed size filtering!")
        # print("  Consider reducing min_eods_per_event parameter")
        events_filtered = pd.DataFrame(columns=events_to_filter.columns)
        return events_filtered
        # exit(1)

    # Filter to keep only valid events
    events_filtered = events_to_filter[events_to_filter[id_col].isin(valid_event_ids)].copy()

    return events_filtered


def create_event_plots(event_id, event_eods, event_data, event_start_time, sample_rate, output_path, max_plot_duration=300, extraction_method='Differential'):
    """
    Create detection plots for EOD events with memory-efficient handling.
    Adapts to PCA (single-ended) or Differential extraction methods.
    
    Parameters:
    -----------
    event_id : int
        Unique identifier for the event
    event_eods : pd.DataFrame
        DataFrame containing EOD detection data for this event
    event_data : np.ndarray
        Audio data for the event (samples x channels)
    event_start_time : datetime
        Start time of the event
    sample_rate : int
        Audio sampling rate
    output_path : str or Path
        Directory to save plots
    max_plot_duration : float
        Maximum duration (seconds) before applying downsampling
    extraction_method : str
        'PCA' for single-ended data or 'Differential' for differential data
    
    Returns:
    --------
    str : Path to saved plot file
    """
    print(f"        Creating event plot (method: {extraction_method})...")
    
    # Calculate event properties
    event_duration = len(event_data) / sample_rate
    event_size_mb = event_data.nbytes / (1024 * 1024)
    print(f"        Event duration: {event_duration:.1f}s, size: {event_size_mb:.1f}MB")
    
    if len(event_eods) == 0:
        print(f"        Warning: No EODs found for event {event_id}")
        return None
    
    # Get channel event information with fallback handling
    merged_channel_event_ids = event_eods['channel_event_id'].unique()
    
    # Handle merge_iteration data with robust fallback
    if 'merge_iteration' in event_eods.columns and not event_eods['merge_iteration'].isna().all():
        merge_event_data = (event_eods[['channel_event_id', 'merge_iteration']]
                            .drop_duplicates()
                            .sort_values('merge_iteration')
                            .reset_index(drop=True))
        
        if len(merge_event_data) > 0:
            max_iteration = int(merge_event_data['merge_iteration'].max())
            color_map = cm.get_cmap('tab20', max_iteration + 1)
            channel_event_colors = {
                int(iteration): color_map(i) 
                for i, iteration in enumerate(merge_event_data['merge_iteration'].unique())
            }
        else:
            # Fallback if merge data is empty
            merge_event_data = pd.DataFrame({
                'channel_event_id': merged_channel_event_ids, 
                'merge_iteration': [1] * len(merged_channel_event_ids)
            })
            channel_event_colors = {1: cm.get_cmap('tab20')(0)}
    else:
        print(f"        No merge iteration data available for event {event_id}, using default coloring")
        merge_event_data = pd.DataFrame({
            'channel_event_id': merged_channel_event_ids, 
            'merge_iteration': [1] * len(merged_channel_event_ids)
        })
        color_map = cm.get_cmap('tab20', len(merged_channel_event_ids))
        channel_event_colors = {1: color_map(0)}
    
    # Calculate plotting parameters
    offset_diff = np.max(event_eods['eod_amplitude']) * 1.5
    
    # Determine downsampling strategy
    if event_duration > max_plot_duration:
        plot_step = 3  # Fixed 3x downsampling for long events
        print(f"        Long event detected - applying 3x downsampling")
    else:
        plot_step = 1  # Original resolution for shorter events
        print(f"        Normal event - using original resolution")
    
    # Determine data format and plotting parameters based on extraction method
    n_channels = event_data.shape[1]
    if extraction_method == 'PCA':
        n_plot_channels = n_channels
        channel_label_prefix = 'Ch'
        plot_title_method = 'Single-Ended (PCA)'
    else:  # Differential
        n_plot_channels = n_channels - 1
        channel_label_prefix = 'Ch'
        plot_title_method = 'Differential'
    
    # Create the plot
    plt.figure(figsize=(20, 8))
    
    # Plot signals for each channel
    for i in range(n_plot_channels):
        # Calculate signal data based on extraction method
        if extraction_method == 'PCA':
            # Use single-ended channel data
            channel_data = event_data[::plot_step, i]
            ch_label = f'{channel_label_prefix}{i}'
        else:  # Differential
            # Calculate differential signal
            channel_data = np.diff(event_data[::plot_step, i:i+2], axis=1).flatten()
            ch_label = f'{channel_label_prefix}{i}-{i+1}'
        
        # Create time coordinates for downsampled data
        time_indices = np.arange(0, len(event_data), plot_step)[:len(channel_data)]
        time_offsets = pd.to_timedelta(time_indices / sample_rate, unit='s')
        x_coords = event_start_time + time_offsets
        
        plt.plot(x_coords, channel_data + i * offset_diff, linewidth=0.5, label=ch_label)
        
        # Plot detected pulses for this channel
        ch_eods = event_eods[event_eods['eod_channel'] == i]
        if len(ch_eods) > 0:
            # Plot peaks (red)
            if 'p1_idx' in ch_eods.columns:
                peak_sample_indices = ch_eods['p1_idx'].values.astype(np.int64)
                valid_peaks = ((peak_sample_indices >= 0) & 
                                (peak_sample_indices < len(event_data)) & 
                                (~np.isnan(peak_sample_indices)))
                
                if np.any(valid_peaks):
                    valid_peak_samples = peak_sample_indices[valid_peaks]
                    peak_timestamps = event_start_time + pd.to_timedelta(valid_peak_samples / sample_rate, unit='s')
                    
                    if extraction_method == 'PCA':
                        peak_values = event_data[valid_peak_samples, i]
                    else:
                        peak_values = np.diff(event_data[valid_peak_samples, i:i+2], axis=1).flatten()
                    
                    plt.plot(peak_timestamps, peak_values + i * offset_diff, 
                            'o', markersize=3, color='red', alpha=0.8, 
                            label='P1' if i == 0 else "")
            
            # Plot troughs (blue)
            if 'p2_idx' in ch_eods.columns:
                trough_sample_indices = ch_eods['p2_idx'].values.astype(np.int64)
                valid_troughs = ((trough_sample_indices >= 0) & 
                                (trough_sample_indices < len(event_data)) & 
                                (~np.isnan(trough_sample_indices)))
                
                if np.any(valid_troughs):
                    valid_trough_samples = trough_sample_indices[valid_troughs]
                    trough_timestamps = event_start_time + pd.to_timedelta(valid_trough_samples / sample_rate, unit='s')
                    
                    if extraction_method == 'PCA':
                        trough_values = event_data[valid_trough_samples, i]
                    else:
                        trough_values = np.diff(event_data[valid_trough_samples, i:i+2], axis=1).flatten()
                    
                    plt.plot(trough_timestamps, trough_values + i * offset_diff, 
                            'o', markersize=3, color='blue', alpha=0.8, 
                            label='P2' if i == 0 else "")
            
            # Plot pulse_location visualization for this channel
            if 'pulse_location' in ch_eods.columns and len(ch_eods) > 0:
                for _, eod_row in ch_eods.iterrows():
                    if 'p1_idx' in eod_row and not pd.isna(eod_row['p1_idx']):
                        p1_idx = int(eod_row['p1_idx'])
                        if p1_idx >= 0 and p1_idx < len(event_data) and 'pulse_location' in eod_row:
                            peak_loc = eod_row['pulse_location']
                            if not pd.isna(peak_loc):
                                p1_timestamp = event_start_time + pd.to_timedelta(p1_idx / sample_rate, unit='s')
                                # Draw thin line from channel offset to pulse_location offset
                                plt.plot([p1_timestamp, p1_timestamp], [i * offset_diff, peak_loc * offset_diff], 
                                        'k-', linewidth=0.5, alpha=0.6)
                                # Mark pulse_location with small black marker
                                plt.plot(p1_timestamp, peak_loc * offset_diff, 'ko', markersize=2, alpha=0.8)
        
        # Clean up arrays immediately to save memory
        del channel_data, x_coords
        gc.collect()
    
    # Overlay colored boxes for each original channel event
    midpoints_for_merge_lines = []
    for _, merge_event_row in merge_event_data.iterrows():
        cid = merge_event_row['channel_event_id']
        iteration = int(merge_event_row['merge_iteration'])
        ce_eods = event_eods[event_eods['channel_event_id'] == cid]
        
        if len(ce_eods) == 0:
            continue
            
        ch = ce_eods['eod_channel'].iloc[0]
        
        # Calculate time bounds for this channel event
        if 'midpoint_idx' in ce_eods.columns:
            min_sample = int(np.nanmin(ce_eods['midpoint_idx']))
            max_sample = int(np.nanmax(ce_eods['midpoint_idx']))
            min_timestamp = event_start_time + pd.to_timedelta(min_sample / sample_rate, unit='s')
            max_timestamp = event_start_time + pd.to_timedelta(max_sample / sample_rate, unit='s')
            mean_sample = int(np.nanmean(ce_eods['midpoint_idx']))
            mean_timestamp = event_start_time + pd.to_timedelta(mean_sample / sample_rate, unit='s')
        else:
            min_timestamp = event_start_time
            max_timestamp = event_start_time + pd.to_timedelta(len(event_data) / sample_rate, unit='s')
            mean_timestamp = event_start_time + pd.to_timedelta(len(event_data) / (2 * sample_rate), unit='s')
        
        width_seconds = (max_timestamp - min_timestamp).total_seconds()
        
        # Convert timestamps to matplotlib format
        min_timestamp_mpl = mdates.date2num(min_timestamp.to_pydatetime())
        width_days = width_seconds / 86400.0  # Convert to days for matplotlib
        mean_timestamp_mpl = mdates.date2num(mean_timestamp.to_pydatetime())
        
        # Store midpoint for merge lines
        midpoints_for_merge_lines.append((mean_timestamp_mpl, ch * offset_diff, iteration))
        
        # Draw colored rectangle for this channel event
        rect = mpatches.Rectangle(
            (min_timestamp_mpl, ch * offset_diff - 0.5 * offset_diff),
            width_days, offset_diff,
            linewidth=2, edgecolor=channel_event_colors[iteration], 
            facecolor='none', alpha=0.7, zorder=10
        )
        plt.gca().add_patch(rect)
        
        # Add iteration number annotation
        text_timestamp_mpl = min_timestamp_mpl + width_days * 0.05
        text_y = ch * offset_diff + 0.3 * offset_diff
        plt.text(text_timestamp_mpl, text_y, str(iteration), 
                fontsize=12, fontweight='bold', 
                color=channel_event_colors[iteration], 
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                            edgecolor=channel_event_colors[iteration], alpha=0.8),
                ha='center', va='center', zorder=15)
        
        # Draw mean midpoint marker
        plt.plot([mean_timestamp_mpl], [ch * offset_diff], 
                marker='s', color=channel_event_colors[iteration], 
                markersize=10, zorder=11)
    
    # Draw lines connecting merged channel events (merge path)
    if len(midpoints_for_merge_lines) > 1:
        midpoints_sorted = sorted(midpoints_for_merge_lines, key=lambda x: x[0])  # Sort by time
        plt.plot([m[0] for m in midpoints_sorted], [m[1] for m in midpoints_sorted], 
                color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=12)
    
    # Configure plot appearance
    # plt.ylim(bottom=None, top=(n_plot_channels - 0.5) * offset_diff)
    
    title_base = f'Event {event_id} - {plot_title_method} EOD Detections'
    if plot_step > 1:
        title_base += f' (downsampled {plot_step}x)'
    title_base += f'\nDuration: {event_duration:.1f}s - Colored boxes: merged channel events'
    plt.title(title_base)
    
    plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Voltage (stacked by channel)')
    
    # Format x-axis for better timestamp readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, int(event_duration/10))))
    plt.xticks(rotation=45)
    
    # Save plot
    plot_filename = f'event_{event_id:03d}_detection.png'
    plot_path = Path(output_path) / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')
    
    print(f"        Created plot: {plot_filename}")
    return str(plot_path)


############################### Clustering and Tracking ######################################

def normalize_waveforms(eod_snippets, snippet_p1_idc, snippet_p2_idc, 
                       method='p1_unity', baseline_correction=True,
                       min_amplitude_threshold=0.001, return_normalization_info=False,
                       crop_and_interpolate=False, crop_factor=7, target_length=150):
    """
    Normalize EOD waveforms for visualization, averaging, and classification.
    
    Parameters
    ----------
    eod_snippets : list of arrays
        List of EOD waveform snippets (variable or fixed length)
    snippet_p1_idc : array
        P1 (normalizing peak) indices within each snippet
    snippet_p2_idc : array  
        P2 (opposite polarity) indices within each snippet
    method : str, default 'p1_unity'
        Normalization method:
        - 'p1_unity': P1 amplitude = +1 (head-positive orientation)
        - 'peak_to_peak': Normalize by P1-P2 amplitude difference, P1 = +1
        - 'rms': Root mean square normalization
        - 'zscore': Z-score normalization (mean=0, std=1)
        - 'minmax': Min-max scaling to [0, 1] range
    baseline_correction : bool, default True
        Remove DC offset by centering around baseline
    min_amplitude_threshold : float, default 0.001
        Minimum P1-P2 amplitude difference for valid normalization
    return_normalization_info : bool, default False
        If True, return additional normalization metadata
    crop_and_interpolate : bool, default False
        If True, crop waveforms relative to P-P duration and interpolate to fixed length.
        Removes excess baseline and normalizes temporal scale across different pulse durations.
    crop_factor : float, default 7
        Multiplier for peak-peak duration to define crop window size (only used if crop_and_interpolate=True)
    target_length : int, default 150
        Target number of samples after interpolation (only used if crop_and_interpolate=True)
    
    Returns
    -------
    normalized_snippets : list of arrays
        Normalized waveform snippets
    normalization_info : dict (optional)
        Metadata about normalization process including:
        - original_p1_amplitudes, original_p2_amplitudes
        - scaling_factors, baseline_offsets
        - rejected_indices (waveforms that failed normalization)
    
    Notes
    -----
    - All waveforms are oriented so P1 is positive (head-positive)
    - Waveforms with insufficient P1-P2 amplitude difference are flagged # DEACTIVATED, UNNECESSARY
    - For machine learning applications, consider 'zscore' method
    - For visualization and averaging, 'p1_unity' or 'peak_to_peak' work well
    - Crop and interpolate option useful for waveform similarity comparisons with minimal baseline interference
    """
    
    if len(eod_snippets) == 0:
        if return_normalization_info:
            return [], {'rejected_indices': [], 'scaling_factors': [], 'baseline_offsets': []}
        return []
    
    # Validate inputs
    n_snippets = len(eod_snippets)
    if len(snippet_p1_idc) != n_snippets or len(snippet_p2_idc) != n_snippets:
        raise ValueError("Length mismatch between snippets and P1/P2 indices")
    
    # Initialize output
    normalized_snippets = []
    rejected_indices = []
    
    # Normalization metadata
    original_p1_amplitudes = []
    original_p2_amplitudes = []
    scaling_factors = []
    baseline_offsets = []
    
    for i, snippet in enumerate(eod_snippets):
        try:
            # Get P1 and P2 indices for this snippet
            p1_idx = int(snippet_p1_idc[i])
            p2_idx = int(snippet_p2_idc[i])
            
            # Validate indices
            if p1_idx < 0 or p1_idx >= len(snippet) or p2_idx < 0 or p2_idx >= len(snippet):
                print(f"Warning: Invalid P1/P2 indices for snippet {i}, skipping")
                rejected_indices.append(i)
                continue
            
            # Get original amplitudes
            # original_snippet = snippet.copy()
            p1_amp = snippet[p1_idx]
            p2_amp = snippet[p2_idx]
            original_p1_amplitudes.append(p1_amp)
            original_p2_amplitudes.append(p2_amp)
            
            # # Check for sufficient amplitude difference
            # amplitude_diff = abs(p1_amp - p2_amp)
            # if amplitude_diff < min_amplitude_threshold:
            #     print(f"Warning: Insufficient amplitude difference ({amplitude_diff:.6f}) for snippet {i}, skipping")
            #     rejected_indices.append(i)
            #     continue
            
            # Baseline correction
            baseline_offset = 0
            if baseline_correction:
                if method == 'zscore':
                    # For zscore, baseline correction is part of the method
                    baseline_offset = np.mean(snippet)
                else:
                    # Use pre-P1 baseline if available, otherwise whole waveform mean
                    if p1_idx >= 10:
                        baseline_offset = np.mean(snippet[:min(10, p1_idx)])
                    else:
                        baseline_offset = np.mean(snippet)
                snippet = snippet - baseline_offset
                # Update amplitudes after baseline correction
                p1_amp = snippet[p1_idx]
                p2_amp = snippet[p2_idx]
            
            baseline_offsets.append(baseline_offset)
            
            # Ensure P1 is positive (head-positive orientation)
            if p1_amp < 0:
                snippet = -snippet
                p1_amp = -p1_amp
                p2_amp = -p2_amp
            
            # Apply normalization method
            if method == 'p1_unity':
                # P1 amplitude = +1
                scale_factor = 1.0 / p1_amp if p1_amp != 0 else 1.0
                snippet = snippet * scale_factor
                
            elif method == 'peak_to_peak':
                # Normalize by P1-P2 difference, P1 = +1
                amplitude_diff = p1_amp - p2_amp
                scale_factor = 1.0 / amplitude_diff if amplitude_diff != 0 else 1.0
                snippet = snippet * scale_factor
                
            elif method == 'rms':
                # RMS normalization
                rms = np.sqrt(np.mean(snippet**2))
                scale_factor = 1.0 / rms if rms != 0 else 1.0
                snippet = snippet * scale_factor
                
            elif method == 'zscore':
                # Z-score normalization (mean=0, std=1)
                mean_val = np.mean(snippet)
                std_val = np.std(snippet)
                if std_val != 0:
                    snippet = (snippet - mean_val) / std_val
                    scale_factor = 1.0 / std_val
                else:
                    scale_factor = 1.0
                    
            elif method == 'minmax':
                # Min-max scaling to [0, 1]
                min_val = np.min(snippet)
                max_val = np.max(snippet)
                if max_val != min_val:
                    snippet = (snippet - min_val) / (max_val - min_val)
                    scale_factor = 1.0 / (max_val - min_val)
                else:
                    scale_factor = 1.0
                    
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            scaling_factors.append(scale_factor)
            
            # Apply cropping and interpolation if requested
            if crop_and_interpolate:
                # Calculate peak-peak duration and midpoint
                pp_duration = abs(p2_idx - p1_idx)
                midpoint = (p1_idx + p2_idx) // 2
                
                # Define crop window: crop_factor × P-P duration, centered on midpoint
                half_window = int(pp_duration * crop_factor / 2)
                start_idx = max(0, midpoint - half_window)
                end_idx = min(len(snippet), midpoint + half_window)
                
                # Crop waveform
                cropped = snippet[start_idx:end_idx]
                
                # Handle edge case: very short crop
                if len(cropped) < 2:
                    # Use wider window around midpoint
                    start_idx = max(0, midpoint - 25)
                    end_idx = min(len(snippet), midpoint + 25)
                    cropped = snippet[start_idx:end_idx]
                
                # Interpolate to fixed length
                if len(cropped) >= 2:
                    x_old = np.linspace(0, 1, len(cropped))
                    x_new = np.linspace(0, 1, target_length)
                    
                    # Use linear interpolation (fast and sufficient for EODs)
                    f = interp1d(x_old, cropped, kind='linear', fill_value='extrapolate')
                    snippet = f(x_new)
                else:
                    # Fallback: pad with zeros if something went wrong
                    snippet = np.zeros(target_length)
            
            normalized_snippets.append(snippet)
            
        except Exception as e:
            print(f"Warning: Failed to normalize snippet {i}: {e}")
            rejected_indices.append(i)
            # Add placeholder values for consistency
            original_p1_amplitudes.append(np.nan)
            original_p2_amplitudes.append(np.nan)
            scaling_factors.append(np.nan)
            baseline_offsets.append(np.nan)
    
    print(f"Normalized {len(normalized_snippets)}/{n_snippets} waveforms using '{method}' method")
    if len(rejected_indices) > 0:
        print(f"Rejected {len(rejected_indices)} waveforms due to normalization issues")
    
    if return_normalization_info:
        normalization_info = {
            'method': method,
            'baseline_correction': baseline_correction,
            'min_amplitude_threshold': min_amplitude_threshold,
            'original_p1_amplitudes': np.array(original_p1_amplitudes),
            'original_p2_amplitudes': np.array(original_p2_amplitudes),
            'scaling_factors': np.array(scaling_factors),
            'baseline_offsets': np.array(baseline_offsets),
            'rejected_indices': rejected_indices,
            'n_normalized': len(normalized_snippets),
            'n_rejected': len(rejected_indices)
        }
        return normalized_snippets, normalization_info
    
    return normalized_snippets


def plot_normalization_comparison(original_snippets, normalized_snippets, 
                                snippet_p1_idc, snippet_p2_idc,
                                normalization_info=None, n_examples=5, 
                                figsize=(15, 10), save_path=None):
    """
    Create comparison plots showing original vs normalized waveforms.
    
    Parameters
    ----------
    original_snippets : list of arrays
        Original EOD waveform snippets
    normalized_snippets : list of arrays
        Normalized EOD waveform snippets
    snippet_p1_idc : array
        P1 indices within each snippet
    snippet_p2_idc : array
        P2 indices within each snippet
    normalization_info : dict, optional
        Normalization metadata from normalize_waveforms
    n_examples : int, default 5
        Number of example waveforms to plot
    figsize : tuple, default (15, 10)
        Figure size
    save_path : str, optional
        Path to save the plot
    """
    
    if len(normalized_snippets) == 0:
        print("No normalized waveforms to plot")
        return
    
    n_examples = min(n_examples, len(normalized_snippets))
    indices = np.linspace(0, len(normalized_snippets)-1, n_examples, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Example original waveforms
    ax1 = axes[0, 0]
    for i, idx in enumerate(indices):
        if idx < len(original_snippets):
            snippet = original_snippets[idx]
            p1_idx = int(snippet_p1_idc[idx]) if idx < len(snippet_p1_idc) else len(snippet)//2
            p2_idx = int(snippet_p2_idc[idx]) if idx < len(snippet_p2_idc) else len(snippet)//2
            
            ax1.plot(snippet, alpha=0.7, label=f'EOD {idx}')
            ax1.plot(p1_idx, snippet[p1_idx], 'ro', markersize=4)
            ax1.plot(p2_idx, snippet[p2_idx], 'bo', markersize=4)
    
    ax1.set_title('Original Waveforms')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Example normalized waveforms
    ax2 = axes[0, 1]
    for i, idx in enumerate(indices):
        snippet = normalized_snippets[i]  # Note: normalized_snippets may be shorter
        # Find corresponding indices in the original arrays
        orig_idx = idx
        while orig_idx in (normalization_info['rejected_indices'] if normalization_info else []):
            orig_idx += 1
            if orig_idx >= len(original_snippets):
                break
        
        if orig_idx < len(snippet_p1_idc):
            p1_idx = int(snippet_p1_idc[orig_idx])
            p2_idx = int(snippet_p2_idc[orig_idx])
            
            ax2.plot(snippet, alpha=0.7, label=f'EOD {orig_idx}')
            if p1_idx < len(snippet):
                ax2.plot(p1_idx, snippet[p1_idx], 'ro', markersize=4)
            if p2_idx < len(snippet):
                ax2.plot(p2_idx, snippet[p2_idx], 'bo', markersize=4)
    
    method_title = normalization_info['method'] if normalization_info else 'normalized'
    ax2.set_title(f'Normalized Waveforms ({method_title})')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Normalized Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: All normalized waveforms overlay
    ax3 = axes[1, 0]
    for snippet in normalized_snippets:
        ax3.plot(snippet, alpha=0.3, color='blue', linewidth=0.5)
    
    # Plot mean waveform
    if len(normalized_snippets) > 0:
        # Align all waveforms to same length for averaging
        max_length = max(len(s) for s in normalized_snippets)
        aligned_waveforms = []
        for snippet in normalized_snippets:
            if len(snippet) == max_length:
                aligned_waveforms.append(snippet)
            else:
                # Simple padding/truncation
                if len(snippet) < max_length:
                    padded = np.pad(snippet, (0, max_length - len(snippet)), 'constant')
                    aligned_waveforms.append(padded)
                else:
                    aligned_waveforms.append(snippet[:max_length])
        
        if aligned_waveforms:
            mean_waveform = np.mean(aligned_waveforms, axis=0)
            ax3.plot(mean_waveform, 'r-', linewidth=2, label='Mean')
            ax3.legend()
    
    ax3.set_title(f'All Normalized Waveforms (n={len(normalized_snippets)})')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Normalized Amplitude')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Normalization statistics
    ax4 = axes[1, 1]
    if normalization_info:
        valid_indices = ~np.isnan(normalization_info['scaling_factors'])
        if np.any(valid_indices):
            scaling_factors = normalization_info['scaling_factors'][valid_indices]
            p1_amps = normalization_info['original_p1_amplitudes'][valid_indices]
            
            ax4.scatter(p1_amps, scaling_factors, alpha=0.6)
            ax4.set_xlabel('Original P1 Amplitude')
            ax4.set_ylabel('Scaling Factor')
            ax4.set_title(f'Normalization Scaling\n({normalization_info["n_normalized"]} normalized, {normalization_info["n_rejected"]} rejected)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No valid normalization data', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'No normalization info available', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Normalization comparison plot saved to: {save_path}")
    
    plt.show()
    
    return fig

def create_tracking_plot(event_id, event_eods, event_data, event_start_time, sample_rate, output_path, max_plot_duration=300, extraction_method='Differential'):
    """
    Create tracking plots for EOD events with memory-efficient handling.
    Adapts to PCA (single-ended) or Differential extraction methods.
    
    Parameters:
    -----------
    event_id : str
        Unique identifier for the event
    event_eods : pd.DataFrame
        DataFrame containing EOD detection data for this event
    event_data : np.ndarray
        Audio data for the event (samples x channels)
    event_start_time : datetime
        Start time of the event
    sample_rate : int
        Audio sampling rate
    output_path : str or Path
        Directory to save plots
    max_plot_duration : float
        Maximum duration (seconds) before applying downsampling
    extraction_method : str
        'PCA' for single-ended data or 'Differential' for differential data
    
    Returns:
    --------
    str : Path to saved plot file
    """
    print(f"        Creating event plot (method: {extraction_method})...")
    
    # Calculate event properties
    event_duration = len(event_data) / sample_rate
    event_size_mb = event_data.nbytes / (1024 * 1024)
    print(f"        Event duration: {event_duration:.1f}s, size: {event_size_mb:.1f}MB")
    
    if len(event_eods) == 0:
        print(f"        Warning: No EODs found for event {event_id}")
        return None
    
    # Determine fish IDs from tracking data if available
    fish_ids = event_eods['fish_id'].unique() if 'fish_id' in event_eods.columns else []
    if len(fish_ids) == 0:
        print(f"        No fish IDs associated with this event")
        return None
    
    # Create color map for fish IDs
    color_map = cm.get_cmap('tab10', len(fish_ids))
    
    if len(fish_ids) > 1:
        id_colors = {
            int(iteration): color_map(i) 
            for i, iteration in enumerate(event_eods['fish_id'].unique())
        }
    else:
        id_colors = {fish_ids[0]: color_map(0)}
    
    # Calculate plotting parameters
    offset_diff = np.max(event_eods['eod_amplitude']) * 1.5
    
    # Determine downsampling strategy
    if event_duration > max_plot_duration:
        plot_step = 3  # Fixed 3x downsampling for long events
        print(f"        Long event detected - applying 3x downsampling")
    else:
        plot_step = 1  # Original resolution for shorter events
        print(f"        Normal event - using original resolution")
    
    # Determine data format and plotting parameters based on extraction method
    n_channels = event_data.shape[1]
    if extraction_method == 'PCA':
        n_plot_channels = n_channels
        channel_label_prefix = 'Ch'
        plot_title_method = 'Single-Ended (PCA)'
    else:  # Differential
        n_plot_channels = n_channels - 1
        channel_label_prefix = 'Ch'
        plot_title_method = 'Differential'
    
    # Create the plot
    plt.figure(figsize=(20, 8))
    
    # Plot signals for each channel
    for i in range(n_plot_channels):
        # Calculate signal data based on extraction method
        if extraction_method == 'PCA':
            # Use single-ended channel data
            channel_data = event_data[::plot_step, i]
            ch_label = f'{channel_label_prefix}{i}'
        else:  # Differential
            # Calculate differential signal
            channel_data = np.diff(event_data[::plot_step, i:i+2], axis=1).flatten()
            ch_label = f'{channel_label_prefix}{i}-{i+1}'
        
        # Create time coordinates for downsampled data
        time_indices = np.arange(0, len(event_data), plot_step)[:len(channel_data)]
        time_offsets = pd.to_timedelta(time_indices / sample_rate, unit='s')
        x_coords = event_start_time + time_offsets
        
        plt.plot(x_coords, channel_data + i * offset_diff, linewidth=0.5, label=ch_label)
        
        # Plot detected pulses for this channel and fish id
        for fish_id in fish_ids:
        
            ch_eods = event_eods[(event_eods['eod_channel'] == i) & (event_eods['fish_id'] == fish_id)]
            if len(ch_eods) > 0:
                # Plot peaks (red)
                if 'p1_idx' in ch_eods.columns:
                    peak_sample_indices = ch_eods['p1_idx'].values.astype(np.int64)
                    valid_peaks = ((peak_sample_indices >= 0) & 
                                    (peak_sample_indices < len(event_data)) & 
                                    (~np.isnan(peak_sample_indices)))
                    
                    if np.any(valid_peaks):
                        valid_peak_samples = peak_sample_indices[valid_peaks]
                        peak_timestamps = event_start_time + pd.to_timedelta(valid_peak_samples / sample_rate, unit='s')
                        
                        if extraction_method == 'PCA':
                            peak_values = event_data[valid_peak_samples, i]
                        else:
                            peak_values = np.diff(event_data[valid_peak_samples, i:i+2], axis=1).flatten()
                        
                        plt.plot(peak_timestamps, peak_values + i * offset_diff, 
                                'o', markersize=3, color=id_colors[fish_id], alpha=0.8, 
                                label='P1' if i == 0 else "")
                
                # Plot troughs (blue)
                if 'p2_idx' in ch_eods.columns:
                    trough_sample_indices = ch_eods['p2_idx'].values.astype(np.int64)
                    valid_troughs = ((trough_sample_indices >= 0) & 
                                    (trough_sample_indices < len(event_data)) & 
                                    (~np.isnan(trough_sample_indices)))
                    
                    if np.any(valid_troughs):
                        valid_trough_samples = trough_sample_indices[valid_troughs]
                        trough_timestamps = event_start_time + pd.to_timedelta(valid_trough_samples / sample_rate, unit='s')
                        
                        if extraction_method == 'PCA':
                            trough_values = event_data[valid_trough_samples, i]
                        else:
                            trough_values = np.diff(event_data[valid_trough_samples, i:i+2], axis=1).flatten()
                        
                        plt.plot(trough_timestamps, trough_values + i * offset_diff, 
                                'o', markersize=3, color=id_colors[fish_id], alpha=0.8, 
                                label='P2' if i == 0 else "")
                
                # Plot pulse_location visualization for this channel
                if 'pulse_location' in ch_eods.columns and 'midpoint_idx' in ch_eods.columns:
                    midpoint_sample_indices = ch_eods['midpoint_idx'].values.astype(np.int64)
                    valid_midpoints = ((midpoint_sample_indices >= 0) &
                                    (midpoint_sample_indices < len(event_data)) & 
                                    (~np.isnan(midpoint_sample_indices)))
                    if np.any(valid_midpoints):
                        valid_midpoint_samples = midpoint_sample_indices[valid_midpoints]
                        midpoint_timestamps = event_start_time + pd.to_timedelta(valid_midpoint_samples / sample_rate, unit='s')
                    
                    plt.plot(midpoint_timestamps, ch_eods['pulse_location'].values[valid_midpoints] * offset_diff, 
                            '-', linewidth=0.5, color=id_colors[fish_id], alpha=0.8, 
                            label='Peak Loc' if i == 0 else "")

                    plt.plot(midpoint_timestamps, ch_eods['pulse_location'].values[valid_midpoints] * offset_diff, 
                            'x', markersize=3, color=id_colors[fish_id], alpha=0.8, 
                            label='Peak Loc' if i == 0 else "")
                    
        # Clean up arrays immediately to save memory
        del channel_data, x_coords
        gc.collect()
    
    
    # Configure plot appearance
    # plt.ylim(bottom=None, top=(n_plot_channels - 0.5) * offset_diff)
    
    title_base = f'Event {event_id} - {plot_title_method} EOD Detections'
    if plot_step > 1:
        title_base += f' (downsampled {plot_step}x)'
    title_base += f'\nDuration: {event_duration:.1f}s - Colored boxes: merged channel events'
    plt.title(title_base)
    
    # plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Voltage (stacked by channel)')
    
    # Format x-axis for better timestamp readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, int(event_duration/10))))
    plt.xticks(rotation=45)
    
    # Save plot
    plot_filename = f'{event_id}_detection.png'
    plot_path = Path(output_path) / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')
    
    print(f"        Created plot: {plot_filename}")
    return str(plot_path)