import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

session_dir = r'G:\First Trip\Avanavero\Avanavero Location 1 - Entrance to lodge 27-28 feb 2024\EOD 2\recordings'
old_dir = os.path.join(session_dir, 'Envelope_extraction_0.001')
new_dir = os.path.join(session_dir, 'Envelope_extraction_0.001_win_8000')
output_dir = os.path.join(new_dir, 'comparison_with_old')
os.makedirs(output_dir, exist_ok=True)

time_columns = ['eod_start_time', 'eod_end_time', 'event_start_time', 'event_end_time']


def load_summaries(folder):
    summaries = pd.read_csv(os.path.join(folder, 'all_event_summaries.csv'))
    for column in time_columns:
        summaries[column] = pd.to_datetime(summaries[column])
    return summaries.sort_values('eod_start_time').reset_index(drop=True)


old = load_summaries(old_dir)
new = load_summaries(new_dir)
print(f'old events: {len(old)}, total EODs {old.n_eods.sum()}')
print(f'new events: {len(new)}, total EODs {new.n_eods.sum()}')

old_start = old.eod_start_time.values.astype('datetime64[ns]').astype('int64') / 1e9
old_end = old.eod_end_time.values.astype('datetime64[ns]').astype('int64') / 1e9
new_start = new.eod_start_time.values.astype('datetime64[ns]').astype('int64') / 1e9
new_end = new.eod_end_time.values.astype('datetime64[ns]').astype('int64') / 1e9

# Overlap of EOD spans in seconds for every old/new pair
overlap = (np.minimum(old_end[:, None], new_end[None, :])
           - np.maximum(old_start[:, None], new_start[None, :]))
overlap = np.clip(overlap, 0, None)

rows = []
for old_index in range(len(old)):
    matching_new = np.where(overlap[old_index] > 0)[0]
    if len(matching_new) == 0:
        rows.append({'old_event_id': old.event_id[old_index], 'new_event_ids': '',
                     'old_start': old.eod_start_time[old_index],
                     'old_n_eods': old.n_eods[old_index], 'new_n_eods': 0,
                     'old_duration_s': old.duration_seconds[old_index], 'new_duration_s': np.nan,
                     'old_median_ipi': old.median_ipi_seconds[old_index], 'new_median_ipi': np.nan,
                     'old_mean_amp': old.mean_amplitude[old_index], 'new_mean_amp': np.nan,
                     'old_mean_width_ms': old.mean_width_ms[old_index], 'new_mean_width_ms': np.nan,
                     'status': 'old_only'})
        continue
    new_rows = new.iloc[matching_new]
    rows.append({'old_event_id': old.event_id[old_index],
                 'new_event_ids': ','.join(str(i) for i in new_rows.event_id),
                 'old_start': old.eod_start_time[old_index],
                 'old_n_eods': old.n_eods[old_index], 'new_n_eods': new_rows.n_eods.sum(),
                 'old_duration_s': old.duration_seconds[old_index],
                 'new_duration_s': (new_rows.eod_end_time.max() - new_rows.eod_start_time.min()).total_seconds(),
                 'old_median_ipi': old.median_ipi_seconds[old_index],
                 'new_median_ipi': np.average(new_rows.median_ipi_seconds, weights=new_rows.n_eods),
                 'old_mean_amp': old.mean_amplitude[old_index],
                 'new_mean_amp': np.average(new_rows.mean_amplitude, weights=new_rows.n_eods),
                 'old_mean_width_ms': old.mean_width_ms[old_index],
                 'new_mean_width_ms': np.average(new_rows.mean_width_ms, weights=new_rows.n_eods),
                 'status': 'matched' if len(matching_new) == 1 else 'old_split_in_new'})

matched = pd.DataFrame(rows)

new_only_mask = overlap.sum(axis=0) == 0
new_only = new[new_only_mask][['event_id', 'eod_start_time', 'eod_end_time', 'duration_seconds', 'n_eods',
                               'median_ipi_seconds', 'ipi_cv', 'mean_amplitude', 'max_amplitude',
                               'mean_width_ms', 'file_names']]

matched.to_csv(os.path.join(output_dir, 'matched_events.csv'), index=False)
new_only.to_csv(os.path.join(output_dir, 'new_only_events.csv'), index=False)

# New events that span several old events (merges); their EODs are counted once per old event above
old_per_new = (overlap > 0).sum(axis=0)
new_merged = new[old_per_new > 1].copy()
new_merged['n_old_events'] = old_per_new[old_per_new > 1]
new_merged['old_event_ids'] = [','.join(str(i) for i in old.event_id[overlap[:, j] > 0])
                               for j in np.where(old_per_new > 1)[0]]
new_merged['old_n_eods'] = [old.n_eods[overlap[:, j] > 0].sum() for j in np.where(old_per_new > 1)[0]]
new_merged[['event_id', 'eod_start_time', 'duration_seconds', 'n_eods', 'n_old_events', 'old_event_ids',
            'old_n_eods']].to_csv(os.path.join(output_dir, 'new_events_merging_old.csv'), index=False)

unique_new_in_matched = new.n_eods[old_per_new > 0].sum()

old_only = matched[matched.status == 'old_only']
old_multi = matched[matched.status == 'old_split_in_new']
both = matched[matched.status != 'old_only']

print(f'\nold events matched in new: {len(both)} ({len(old_multi)} overlap more than one new event)')
print(f'old only: {len(old_only)}, EODs in them {old_only.old_n_eods.sum()}')
print(f'new only: {len(new_only)}, EODs in them {new_only.n_eods.sum()}')
print(f'\nEODs in matched events: old {both.old_n_eods.sum()}, new {unique_new_in_matched} (unique)')
print(f'new events overlapping more than one old event: {len(new_merged)}')
if len(new_merged) > 0:
    print(new_merged[['event_id', 'eod_start_time', 'duration_seconds', 'n_eods', 'n_old_events', 'old_event_ids',
                      'old_n_eods']].to_string(index=False))
print('ratio new/old n_eods per matched event:')
print((both.new_n_eods / both.old_n_eods).describe().round(2))
print('\nmedian IPI (s) old vs new:', both.old_median_ipi.median().round(3), both.new_median_ipi.median().round(3))
print('mean amplitude old vs new (median over events):', both.old_mean_amp.median(), both.new_mean_amp.median())
print('mean width ms old vs new (median over events):', both.old_mean_width_ms.median().round(3),
      both.new_mean_width_ms.median().round(3))

print('\nold-only events (check these for lost eels):')
print(old_only[['old_event_id', 'old_start', 'old_n_eods', 'old_duration_s', 'old_mean_amp']].to_string(index=False))
print('\nnew-only events (check these for false positives):')
print(new_only[['event_id', 'eod_start_time', 'n_eods', 'duration_seconds', 'mean_amplitude', 'max_amplitude',
                'mean_width_ms']].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].scatter(both.old_n_eods, both.new_n_eods, s=14)
limit = max(both.old_n_eods.max(), both.new_n_eods.max())
axes[0].plot([0, limit], [0, limit], 'k--', linewidth=0.8)
axes[0].set_xlabel('old n_eods')
axes[0].set_ylabel('new n_eods')
axes[0].set_title('EODs per matched event')

axes[1].scatter(both.old_mean_amp, both.new_mean_amp, s=14)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
low = min(both.old_mean_amp.min(), both.new_mean_amp.min())
high = max(both.old_mean_amp.max(), both.new_mean_amp.max())
axes[1].plot([low, high], [low, high], 'k--', linewidth=0.8)
axes[1].set_xlabel('old mean amplitude')
axes[1].set_ylabel('new mean amplitude')
axes[1].set_title('Mean amplitude')

axes[2].scatter(both.old_median_ipi, both.new_median_ipi, s=14)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
low = min(both.old_median_ipi.min(), both.new_median_ipi.min())
high = max(both.old_median_ipi.max(), both.new_median_ipi.max())
axes[2].plot([low, high], [low, high], 'k--', linewidth=0.8)
axes[2].set_xlabel('old median IPI (s)')
axes[2].set_ylabel('new median IPI (s)')
axes[2].set_title('Median IPI')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'old_vs_new_scatter.png'), dpi=150)

# ==============================================================================
# Pulse level: detection is identical in both runs (same envelope parameters), so the same pulse has
# the same (filename, original_p1_idx) in both tables. Compare amplitude, width and frequency per pulse.
# ==============================================================================
pulse_columns = ['filename', 'original_p1_idx', 'eod_amplitude', 'eod_width_us', 'fft_freq_max']
pulse_tables = {}
for run_name, run_dir in (('old', old_dir), ('new', new_dir)):
    frames = [pd.read_csv(table_file, usecols=pulse_columns)
              for table_file in glob.glob(os.path.join(run_dir, '*_eod_table.csv'))]
    pulse_tables[run_name] = pd.concat(frames, ignore_index=True).drop_duplicates(['filename', 'original_p1_idx'])

pulse_pairs = pulse_tables['old'].merge(pulse_tables['new'], on=['filename', 'original_p1_idx'],
                                        how='outer', suffixes=('_old', '_new'), indicator=True)
both_pulses = pulse_pairs[pulse_pairs._merge == 'both'].copy()
old_only_pulses = pulse_pairs[pulse_pairs._merge == 'left_only']
new_only_pulses = pulse_pairs[pulse_pairs._merge == 'right_only']

both_pulses['amp_ratio'] = both_pulses.eod_amplitude_new / both_pulses.eod_amplitude_old
both_pulses['width_ratio'] = both_pulses.eod_width_us_new / both_pulses.eod_width_us_old
both_pulses['fft_ratio'] = both_pulses.fft_freq_max_new / both_pulses.fft_freq_max_old

print(f'\npulses: old {len(pulse_tables["old"])}, new {len(pulse_tables["new"])}, matched {len(both_pulses)}, '
      f'old only {len(old_only_pulses)}, new only {len(new_only_pulses)}')
print('amplitude ratio new/old (matched pulses):')
print(both_pulses.amp_ratio.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3))

amp_bins = [0, 0.002, 0.005, 0.02, 0.05, np.inf]
both_pulses['old_amp_bin'] = pd.cut(both_pulses.eod_amplitude_old, amp_bins)
print('\nratios by old amplitude bin (median):')
print(both_pulses.groupby('old_amp_bin', observed=True)[['amp_ratio', 'width_ratio', 'fft_ratio']]
      .agg(['median', 'count']).round(3))

print('\nold-only pulses (median amplitude, width, frequency; n with width > 2000 us):',
      old_only_pulses.eod_amplitude_old.median().round(4), old_only_pulses.eod_width_us_old.median().round(0),
      old_only_pulses.fft_freq_max_old.median().round(0), (old_only_pulses.eod_width_us_old > 2000).sum())
print('new-only pulses (median amplitude, width, frequency; n with frequency > 1000 Hz):',
      new_only_pulses.eod_amplitude_new.median().round(4), new_only_pulses.eod_width_us_new.median().round(0),
      new_only_pulses.fft_freq_max_new.median().round(0), (new_only_pulses.fft_freq_max_new > 1000).sum())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(both_pulses.eod_amplitude_old, both_pulses.eod_amplitude_new, s=2, alpha=0.3)
limits = [both_pulses.eod_amplitude_old.min(), both_pulses.eod_amplitude_old.max()]
axes[0, 0].plot(limits, limits, 'k--', linewidth=0.8)
axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel('old amplitude')
axes[0, 0].set_ylabel('new amplitude')
axes[0, 0].set_title('Pulse amplitude')

axes[0, 1].scatter(both_pulses.eod_amplitude_old, both_pulses.amp_ratio, s=2, alpha=0.3)
axes[0, 1].axhline(1, color='k', linestyle='--', linewidth=0.8)
axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xlabel('old amplitude')
axes[0, 1].set_ylabel('amplitude ratio new / old')
axes[0, 1].set_title('Amplitude ratio vs old amplitude')

axes[1, 0].scatter(both_pulses.eod_width_us_old, both_pulses.eod_width_us_new, s=2, alpha=0.3)
limits = [both_pulses.eod_width_us_old.min(), both_pulses.eod_width_us_old.max()]
axes[1, 0].plot(limits, limits, 'k--', linewidth=0.8)
axes[1, 0].set_xlabel('old width (us)')
axes[1, 0].set_ylabel('new width (us)')
axes[1, 0].set_title('Pulse width (FWHM)')

axes[1, 1].scatter(both_pulses.fft_freq_max_old, both_pulses.fft_freq_max_new, s=2, alpha=0.3)
limits = [both_pulses.fft_freq_max_old.min(), both_pulses.fft_freq_max_old.max()]
axes[1, 1].plot(limits, limits, 'k--', linewidth=0.8)
axes[1, 1].set_xlabel('old frequency (Hz)')
axes[1, 1].set_ylabel('new frequency (Hz)')
axes[1, 1].set_title('Half-power frequency')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pulse_level_comparison.png'), dpi=150)
plt.show()
