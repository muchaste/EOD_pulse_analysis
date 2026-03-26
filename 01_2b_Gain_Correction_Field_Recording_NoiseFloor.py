# -*- coding: utf-8 -*-
"""
Gain calibration using noise-floor RMS after masking fish pulses.
Scans a decreasing amplitude threshold per channel to find the knee where
peak count jumps sharply (= noise floor). Masks ±MASK_WIDTH_MS around
detected peaks, computes RMS on the clean baseline only.
"""
import matplotlib.pyplot as plt
import audioio as aio
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks
import gc

WINDOW_SEC = 60
MASK_WIDTH_MS = 2
N_THRESHOLDS = 50
MIN_PEAKS = 100
KNEE_REL_THRESHOLD = 0.05
plot_data = True

root = tk.Tk()
root.withdraw()
plt.ioff()

file_paths = filedialog.askopenfilenames(
    title="Select Audio Files",
    filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
)
file_list = list(file_paths)
output_path = filedialog.askdirectory(title="Select Folder to Store Analysis Results")

logger_id = file_list[0].split('/')[-1].split('-')[0]
print("Logger ID: %s" % logger_id)

cf_all_files = []

for fidx, fname in enumerate(file_list):
    print("Processing file %d of %d: %s" % (fidx + 1, len(file_list), fname))

    data, rate = aio.load_audio(fname)
    channels = data.shape[1]
    file_id = fname.split('/')[-1].split('.')[0]

    window_samples = int(min(rate * WINDOW_SEC, len(data)))
    data_window = data[:window_samples, :]

    W = int(rate * MASK_WIDTH_MS // 1000)
    min_peak_distance = int(rate // 500)  # 2 ms minimum spacing between peaks

    thresholds = np.linspace(0.95, 0.05, N_THRESHOLDS)
    rms = np.zeros(channels)

    knee_thresholds = np.full(channels, np.nan)
    knee_valid = np.zeros(channels, dtype=bool)
    peak_counts_all = np.zeros((channels, N_THRESHOLDS), dtype=int)

    for i in range(channels):
        channel_raw = data_window[:, i]
        channel_abs = np.abs(channel_raw)
        max_amp = np.max(channel_abs)

        if max_amp == 0:
            print("  Channel %d: zero amplitude, skipping (fallback)" % (i + 1))
            rms[i] = np.std(channel_raw) * np.sqrt(2)
            continue

        channel_norm = channel_abs / max_amp

        peak_counts = np.zeros(N_THRESHOLDS, dtype=int)
        for t_idx, thresh in enumerate(thresholds):
            peaks, _ = find_peaks(channel_norm, height=thresh, distance=min_peak_distance)
            peak_counts[t_idx] = len(peaks)
        peak_counts_all[i, :] = peak_counts

        diffs = np.diff(peak_counts)

        if peak_counts[-1] < MIN_PEAKS:
            print("  Channel %d: max peak count %d < %d — fallback to full window" % (
                i + 1, peak_counts[-1], MIN_PEAKS))
        elif max(diffs) / peak_counts[-1] < KNEE_REL_THRESHOLD:
            print("  Channel %d: no dominant knee (max_diff/max_count = %.3f) — fallback to full window" % (
                i + 1, max(diffs) / peak_counts[-1]))
        else:
            knee_idx = np.argmax(diffs)
            thresh_cut = thresholds[knee_idx]
            knee_thresholds[i] = thresh_cut
            knee_valid[i] = True

        if knee_valid[i]:
            thresh_raw = knee_thresholds[i] * max_amp
            peaks, _ = find_peaks(channel_abs, height=thresh_raw, distance=min_peak_distance)

            mask = np.ones(len(channel_raw), dtype=bool)
            for p in peaks:
                mask[max(0, p - W):p + W] = False

            retained_frac = mask.sum() / len(mask)
            print("  Channel %d: knee at %.2f (raw: %.4f), %d peaks masked, %.1f%% samples retained" % (
                i + 1, knee_thresholds[i], thresh_raw, len(peaks), retained_frac * 100))

            if retained_frac < 0.5:
                print("  WARNING: channel %d retained only %.1f%% of samples" % (i + 1, retained_frac * 100))

            rms[i] = np.std(channel_raw[mask]) * np.sqrt(2)
        else:
            rms[i] = np.std(channel_raw) * np.sqrt(2)

    rms_norm = rms / np.max(np.abs(rms))
    cor_factors = 1.0 / rms_norm
    cor_factors /= np.max(cor_factors)

    cf_all_files.append(cor_factors)

    if plot_data:
        n_cols = 4
        n_rows = int(np.ceil(channels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
        axes = axes.flatten()

        for i in range(channels):
            axes[i].plot(thresholds, peak_counts_all[i, :], color='steelblue')
            if knee_valid[i]:
                axes[i].axvline(knee_thresholds[i], color='red', linestyle='--', linewidth=1.5)
                status = "knee=%.2f" % knee_thresholds[i]
            else:
                status = "FALLBACK"
            axes[i].set_title("Ch %d — %s" % (i + 1, status), fontsize=9)
            axes[i].set_xlabel("Threshold (norm)", fontsize=8)
            axes[i].set_ylabel("Peak count", fontsize=8)
            axes[i].invert_xaxis()

        for j in range(channels, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(file_id, fontsize=10)
        plt.tight_layout()
        plt.savefig('%s\\%s_threshold_scan.png' % (output_path, file_id))
        plt.show(block=False)

    gc.collect()

cf_all_files = np.array(cf_all_files)
cf_df = pd.DataFrame(cf_all_files)

col_names = ['ch_%d' % (i + 1) for i in range(channels)]
cf_df.columns = col_names

file_ids = [fname.split('/')[-1].split('.')[0] for fname in file_list]
cf_df['file_id'] = file_ids

cf_df.to_csv('%s\\%s_correction_factors_all_files.csv' % (output_path, logger_id), index=False)

medians = np.median(cf_all_files, axis=0)
med_df = pd.DataFrame({'channel': np.arange(1, channels + 1), 'median_correction_factor': medians})
med_df.to_csv('%s\\%s_correction_factors_median_per_channel.csv' % (output_path, logger_id), index=False)

x_vals = np.arange(1, len(file_list) + 1)
plt.figure(figsize=(10, 6))
for i in range(channels):
    plt.plot(x_vals, cf_all_files[:, i], marker='o', linestyle='-', label='Channel %d' % (i + 1))
    plt.hlines(np.median(cf_all_files[:, i]), xmin=1, xmax=len(file_list),
               colors='k', linestyles='--', linewidth=2)
    plt.text(len(file_list) + 0.1, np.median(cf_all_files[:, i]),
             '%.2f' % np.median(cf_all_files[:, i]), fontsize=10, verticalalignment='center')

plt.xlabel('File index')
plt.ylabel('Correction Factor')
plt.title('Correction Factors per Channel and File')
plt.legend()
plt.tight_layout()
plt.savefig('%s\\%s_correction_factors_all_files.png' % (output_path, logger_id))
plt.show(block=False)
