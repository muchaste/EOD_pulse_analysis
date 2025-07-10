# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:46:12 2024

@author: shuttlebox
"""
"""Load and cluster pulsefish recordings."""

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import gc

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

def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.

    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.

    Example of usage :

        # calculating frequencies in ranges [400, 500] and [1000, 1100]
        # of a windowed signal sampled at 44100 Hz

        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results

def analyze_r4_snippets(snippet_array, peak_indices, interp_factor):
    """
    Iterate through an array of short recording snippets from 8-channel
    EOD loggers (R4.0 and higher) that are centered around a peak.
    1) Extract amplitudes and correlation coefficients across channels to
    identify the head-to-tail channel pair.
    2) Calculate differential head-to-tail waveform
    3) Return amplitudes, correlation coefficients, head-to-tail channel
    (index of differential channel), head-to-tail waveforms, peak and trough
    indices, and indicator whether head-to-tail waveform could be extracted.

    Parameters
    ----------
    snippet_array: list of 2-D arrays of float
        The recording snippets to be analysed. Channels must be arranged in columns
        (1 column/chan)
    peak_indices: array of int
        Indices of peaks that contain information about the "position" of the
        snippets in the total recording

    Returns
    -------
    h2t_waveforms: 2-D array of float
        Head-to-tail waveform snippets from differential data
    amps: 2-D array of float
        Max. amplitudes across single-ended channels for each snippet
    h2t_amp: 1-D array of float
        Amplitude of head-to-tail waveform
    cor_coeffs: 2-D array of float
        Correlation coefficient between each single-ended channel
    h2t_chan: 1-D array of int
        Index of (differential) channel on which the head-to-tail waveform was identified
    h2t_found: 1-D array of int
        Indicator whether h2t channel pair could be identified
        1: yes
        0: no, the channel with the highest amplitude was chosen
        -1: no, the channel with the highest amp is not on the edge (no channel index is inserted in h2t_chan in this case)
    peak_idc: 1-D array of int
        Index of peaks (on original data)
    trough_idc: 1-D array of int
        Index of troughs (on original data)
    """

    n_channels = snippet_array[0].shape[1]
    n_samples = snippet_array[0].shape[0]
    n_snippets = len(snippet_array)
    interp_samples = interp_factor * n_samples

    # Preallocate arrays
    h2t_waveforms = np.zeros((n_snippets, interp_samples))
    amps = np.zeros((n_snippets, n_channels))
    amps_diff = np.zeros((n_snippets, n_channels - 1))
    h2t_amp = np.zeros(n_snippets)
    cor_coeffs = np.zeros((n_snippets, n_channels - 1))
    h2t_chan = np.zeros(n_snippets, dtype=int)
    h2t_found = np.ones(n_snippets, dtype=int)
    peak_idc = np.zeros(n_snippets, dtype=int)
    trough_idc = np.zeros(n_snippets, dtype=int)
    # h2t_orientation = np.ones(n_snippets, dtype=int)

    # Iterate through snippets, extract data
    for i, s in enumerate(snippet_array):
        # 1. Find peaks and amplitudes across channels
        for j in range(n_channels):
            p_idx = np.argmax(s[:, j])
            t_idx = np.argmin(s[:, j])
            amps[i, j] = abs(s[p_idx, j] - s[t_idx, j])

        # 2. Get correlation coefficient across channels (off-diagonal)
        cor_coeffs[i, :] = np.diagonal(np.corrcoef(s, rowvar=False), offset=1)

        # 3. Compute differential signal over channel pairs
        s_diff = np.diff(s, axis=1)
        # 3.1 Calculate peak/trough amp from differential data
        for j in range(n_channels - 1):
            p_idx = np.argmax(s_diff[:, j])
            t_idx = np.argmin(s_diff[:, j])
            amps_diff[i, j] = abs(s_diff[p_idx, j] - s_diff[t_idx, j])

        # 4. Extract head-to-tail waveform and channel
        # 4.1 Polarity flips happen where the corr. coeff is negative
        flips = np.where(cor_coeffs[i, :] < 0)[0]
        if len(flips) > 1:
            h2t_chan[i] = flips[np.argmax(amps_diff[i, flips])]
        elif len(flips) == 1:
            h2t_chan[i] = flips[0]
        else:
            h2t_chan[i] = np.argmax(amps_diff[i, :])
            h2t_found[i] = 0
            if h2t_chan[i] != 0 and h2t_chan[i] != n_channels - 2:
                h2t_found[i] = -1

        h2t_peak = s_diff[:, h2t_chan[i]]

        h2t_peak_idx = np.argmax(h2t_peak)
        h2t_trough_idx = np.argmin(h2t_peak)
        h2t_amp[i] = abs(h2t_peak[h2t_peak_idx] - h2t_peak[h2t_trough_idx])

        # 4.2 interpolate, rectify and normalize amplitude of h2t waveform
        if h2t_trough_idx < h2t_peak_idx:
            # h2t_orientation[i] = -1
            h2t_peak *= -1
            h2t_peak = np.roll(h2t_peak, (h2t_peak_idx-h2t_trough_idx))

        interper = interp1d(np.arange(len(h2t_peak)), h2t_peak, kind='quadratic')
        h2t_peak = interper(np.linspace(0, len(h2t_peak)-1, interp_samples))
        h2t_peak = np.roll(h2t_peak, interp_samples//2-np.argmax(h2t_peak))
        h2t_peak /= np.max(np.abs(h2t_peak))
        h2t_waveforms[i, :] = h2t_peak

        # Store again the indices of peak and trough
        peak_idc[i] = peak_indices[i] - n_samples // 2 + h2t_peak_idx
        trough_idc[i] = peak_indices[i] - n_samples // 2 + h2t_trough_idx

    return h2t_waveforms, amps, h2t_amp, cor_coeffs, h2t_chan, h2t_found, peak_idc, trough_idc

def filter_waveforms(h2t_waveforms, peak_idc, trough_idc, rate,
                     dur_min=20, dur_max=300,
                     pp_r_min=0.1, pp_r_max=5,
                     fft_freq_min=1000, fft_freq_max=10000,
                     interp_factor=1, return_features=False,
                     return_params=False):
    """
    Filter waveforms based on duration, amplitude ratio, and FFT frequency.

    Parameters
    ----------
    h2t_waveforms : 2-D array
        Array of head-to-tail waveforms.
    peak_idc : 1-D array
        Indices of peaks in the original signal.
    trough_idc : 1-D array
        Indices of troughs in the original signal.
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
    interpolation_factor : int, optional
        Factor by which the signal is interpolated (default is 1).
    return_params : bool, optional
        Returns dataframe with waveform parameters if True.

    Returns
    -------
    h2t_waveforms_keep : 2-D array
        Filtered array of head-to-tail waveforms.
    keep_indices : 1-D array
        Indices of the kept waveforms.
    """
    n_snippets = len(h2t_waveforms)
    h2t_rate = rate * interp_factor

    # Calculate durations in microseconds
    h2t_durs = np.abs(trough_idc - peak_idc) / rate * 1e6

    h2t_ratios = np.zeros(n_snippets)
    fft_freqs = np.zeros(n_snippets)
    h2t_ratios = np.abs(np.max(h2t_waveforms, axis = 1) / np.min(h2t_waveforms, axis = 1)) #<- vectorized, works outside of the loop but doesn't account for pulse orientation

    for i in range(n_snippets):
        # Calculate peak-to-peak amplitude ratios
        # if trough_idc[i] < peak_idc[i]:
        #     h2t_ratios[i] = np.abs(np.min(h2t_waveforms[i]) / np.max(h2t_waveforms[i]))
        # else:
        #     h2t_ratios[i] = np.abs(np.max(h2t_waveforms[i]) / np.min(h2t_waveforms[i]))

        # Calculate FFT peak frequencies
        # fft_freqs = np.array([calc_fft_peak(h2t_waveforms[i], h2t_rate) for i in range(n_snippets)])
        fft_freqs[i] = calc_fft_peak(h2t_waveforms[i], h2t_rate, zero_padding_factor=100)


    # Apply filters
    keep_mask = (
        (h2t_durs > dur_min) & (h2t_durs < dur_max) &
        (h2t_ratios > pp_r_min) & (h2t_ratios < pp_r_max) &
        (fft_freqs > fft_freq_min) & (fft_freqs < fft_freq_max)
    )

    keep_indices = np.where(keep_mask)[0]
    h2t_waveforms_keep = h2t_waveforms[keep_indices]

    return_vars = [h2t_waveforms_keep, keep_indices]

    if return_features:
        features = pd.DataFrame({
            'pp_dur': h2t_durs[keep_indices],
            'pp_ratio': h2t_ratios[keep_indices],
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



#%%
def analyze_channel(channel_index, filtered_h2t_waveforms, filtered_h2t_chan, filtered_peak_idc, filtered_trough_idc, data_diff, rate, offset_diff, time):
    """
    Analyzes a specific channel's head-to-tail waveforms by performing PCA and Gaussian Mixture Modeling.

    Parameters:
    channel_index (int): The index of the channel to analyze.
    filtered_h2t_waveforms (ndarray): The filtered head-to-tail waveforms.
    filtered_h2t_chan (ndarray): Array indicating which channel each waveform belongs to.
    filtered_peak_idc (ndarray): Indices of the peaks in the waveforms.
    filtered_trough_idc (ndarray): Indices of the troughs in the waveforms.
    data_diff (ndarray): The differential data of the recordings.
    rate (int): The sampling rate of the recordings.
    offset_diff (float): Offset value for plotting the data.
    time (ndarray): Time array corresponding to the recordings.

    Returns:
    None
    """
    h2t_idc = np.where(filtered_h2t_chan == channel_index)[0]  # Get relevant indices for channel
    waveforms_chan = filtered_h2t_waveforms[h2t_idc]
    peaks_chan = filtered_peak_idc[h2t_idc]
    # troughs_chan = filtered_trough_idc[h2t_idc]

    # Perform PCA
    pca = PCA(n_components=10)
    pcas = pca.fit_transform(waveforms_chan)

    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained variance ratio for channel {channel_index}: {explained_variance_ratio}")
    print(f"Sum of explained variance ratio for channel {channel_index}: {np.sum(explained_variance_ratio)}")

    # Fit Gaussian Mixture Model
    print("Fitting Gaussian Mixture Model")
    gmm = GaussianMixture(n_components=16)
    y = gmm.fit_predict(pcas)

    ulabels = np.unique(y)

    # Plot PCA results
    plot_pca_results(pcas, y, ulabels)

    # Plot waveform overlay for each cluster
    plot_waveform_clusters(waveforms_chan, y, ulabels)

    # Plot the original recording with peaks marked
    plot_peaks_on_recording(time, data_diff[:, channel_index], peaks_chan, y, ulabels, offset_diff, channel_index)

    gc.collect()

def plot_pca_results(pcas, labels, unique_labels):
    """
    Plots the results of PCA for the given data.

    Parameters:
    pcas (ndarray): PCA-transformed data.
    labels (ndarray): Cluster labels assigned to the data points.
    unique_labels (ndarray): Unique cluster labels.

    Returns:
    None
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for j, label in enumerate(unique_labels):
        ax[0].scatter(pcas[labels == label, 0], pcas[labels == label, 1], c=[colors[j]], label=f"Cluster {label}", s=5)
        ax[1].scatter(pcas[labels == label, 0], pcas[labels == label, 2], c=[colors[j]], label=f"Cluster {label}", s=5)

    ax[0].set_xlabel("PC 1")
    ax[0].set_ylabel("PC 2")
    ax[1].set_xlabel("PC 1")
    ax[1].set_ylabel("PC 3")
    fig.legend()
    plt.show()

def plot_waveform_clusters(waveforms, labels, unique_labels):
    """
    Plots the overlay of waveforms for each cluster.

    Parameters:
    waveforms (ndarray): Array of waveforms.
    labels (ndarray): Cluster labels assigned to the waveforms.
    unique_labels (ndarray): Unique cluster labels.

    Returns:
    None
    """
    rows = int(np.ceil(np.sqrt(len(unique_labels))))
    fig, axs = plt.subplots(rows, rows, sharex=True, sharey=True, figsize=(15, 15))
    axs = axs.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    n_samples = waveforms[0].shape[0]

    for j, label in enumerate(unique_labels):
        axs[j].plot(waveforms[labels == label].T, c=colors[j], lw=1, alpha=0.1)
        axs[j].axvline(n_samples / 2, c='k', lw=0.5, zorder=-1000)
        axs[j].axhline(0, c='k', lw=0.5, zorder=-1000)
        axs[j].set_title(f"Cluster {label}")

    plt.tight_layout()
    plt.show()

def plot_peaks_on_recording(time, data, peaks, labels, unique_labels, offset_diff, channel_index):
    """
    Plots the original recording with peaks marked, color-coded by cluster.

    Parameters:
    time (ndarray): Time array corresponding to the recordings.
    data (ndarray): Differential data of the recordings.
    peaks (ndarray): Indices of the peaks in the data.
    labels (ndarray): Cluster labels assigned to the peaks.
    unique_labels (ndarray): Unique cluster labels.
    offset_diff (float): Offset value for plotting the data.
    channel_index (int): The index of the channel being analyzed.

    Returns:
    None
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, data + channel_index * offset_diff)
    for j, label in enumerate(unique_labels):
        ax.plot(time[peaks[labels == label]], data[peaks[labels == label]] + channel_index * offset_diff, "o", c=colors[j], label=f"Cluster {label}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()