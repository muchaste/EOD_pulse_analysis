# -*- coding: utf-8 -*-
"""
03_3_Select_annotation_sample.py

Draws the events to annotate for the fish-tracking ground truth (plan 2026-11-06, workstream A2).
Run once per annotation round.

Pool: every <site>/<session>/01_*/track/tracked_event_summary.csv under `root` (04_1 output of the
existing extractions). Each event gets a system (JR / LN / PL / SL from the site folder), a stratum
(tracker class x duration tercile, or 'hard') and the paths of its extraction and track folders.
Excluded: events drawn in earlier rounds (annotation_sample_round*.csv in `output_folder`), events
already annotated (annotations*.json next to the extraction folders or listed in `old_annotation_jsons`).
Within each system x stratum the quota is spread over the sessions in a round-robin over the season
(one event per session per pass, sessions in date order), so every session and both halves of the
season are represented; the RNG is seeded.

Yield report: for the previous rounds the annotation JSONs are matched to the sample CSVs and the
counts of verified (clear_fish with a fish count), unclear, noise and not yet annotated events are
printed per system x stratum, so the quotas / criteria of this round can be adjusted below.

Output: annotation_sample_round<k>.csv (system, site, session, date, extraction_folder, track_folder,
event_id, base_stem, stratum, tracker and event statistics). 03_1_Event_Annotation.py loads this file
("Load Sample CSV") and then shows only the sampled events of the opened folder.

Authors: Stefan Mucha with GitHub Copilot
"""

import glob
import json
import os
import re
import numpy as np
import pandas as pd

root = r'F:\\'
output_folder = r'F:\Tracking Results\annotation_rounds'
round_number = 1
seed = 20261106
old_annotation_jsons = []   # JSON files of the 41 old annotated events (excluded from the pool); fill in when known

# criteria for the clean strata
min_pulses = 100
min_assignment_rate = 0.9
duration_edges_s = [0, 5, 30, np.inf]   # terciles: brief / medium / long
duration_labels = ['brief', 'medium', 'long']

# quota per system and stratum for this round
quota = {
    '1fish_brief': 10, '1fish_medium': 10, '1fish_long': 10,
    '2fish_brief': 5, '2fish_medium': 5, '2fish_long': 5,
    'hard': 5,
}
system_of_site = {'Juma': 'JR', 'LN': 'LN', 'PL': 'PL', 'SL': 'SL'}

os.makedirs(output_folder, exist_ok=True)
rng = np.random.default_rng(seed + round_number)

summary_files = sorted(glob.glob(os.path.join(root, '*_L*', '*', '01_*', 'track', 'tracked_event_summary.csv')))
print(f"{len(summary_files)} track folders found under {root}")
pool_parts = []
for f in summary_files:
    track_folder = os.path.dirname(f)
    extraction_folder = os.path.dirname(track_folder)
    session_folder = os.path.dirname(extraction_folder)
    site = os.path.basename(os.path.dirname(session_folder))
    d = pd.read_csv(f)
    if len(d) == 0:
        continue
    d['site'] = site
    d['system'] = system_of_site[site.split('_')[0]]
    d['session'] = site + '_' + os.path.basename(session_folder)
    d['date'] = pd.to_datetime(re.match(r'\d{8}', os.path.basename(session_folder)).group(0), format='%Y%m%d')
    d['extraction_folder'] = extraction_folder
    d['track_folder'] = track_folder
    pool_parts.append(d)
pool = pd.concat(pool_parts, ignore_index=True)
pool['event_id'] = pool['event_id'].astype(int)
print(f"pool: {len(pool)} events, {pool['session'].nunique()} sessions, systems {pool['system'].value_counts().to_dict()}")

# base stem of each event (file name prefix of its eod table), needed by the annotation tool;
# one directory listing per extraction folder instead of one glob per event
stem_by_folder_event = {}
for folder in pool['extraction_folder'].unique():
    for name in os.listdir(folder):
        m = re.match(r'(.*_event_(\d+))_eod_table\.csv$', name)
        if m:
            stem_by_folder_event[(folder, int(m.group(2)))] = m.group(1)
pool['base_stem'] = [stem_by_folder_event.get((f, i), '') for f, i in zip(pool['extraction_folder'], pool['event_id'])]
missing = (pool['base_stem'] == '').sum()
if missing:
    print(f"  {missing} events without an eod table in the extraction folder: dropped from the pool")
pool = pool[pool['base_stem'] != ''].reset_index(drop=True)

# strata
pool['duration_class'] = pd.cut(pool['duration_seconds'], bins=duration_edges_s, labels=duration_labels, right=False).astype(str)
clean = (pool['n_eods'] >= min_pulses) & (pool['assignment_rate'] >= min_assignment_rate)
pool['stratum'] = 'hard'
pool.loc[clean & (pool['n_fish_total'] == 1), 'stratum'] = '1fish_' + pool['duration_class']
pool.loc[clean & (pool['n_fish_total'] == 2), 'stratum'] = '2fish_' + pool['duration_class']
print("pool per system x stratum:")
print(pd.crosstab(pool['system'], pool['stratum']).to_string())

# exclusions: earlier rounds, existing annotations
drawn_before = pd.DataFrame(columns=['extraction_folder', 'event_id', 'stratum', 'system', 'base_stem', 'round'])
for f in sorted(glob.glob(os.path.join(output_folder, 'annotation_sample_round*.csv'))):
    k = int(re.search(r'round(\d+)', os.path.basename(f)).group(1))
    if k >= round_number:
        continue
    d = pd.read_csv(f)
    d['round'] = k
    drawn_before = pd.concat([drawn_before, d[drawn_before.columns]], ignore_index=True)
drawn_keys = set(zip(drawn_before['extraction_folder'], drawn_before['event_id'].astype(int)))

annotated = {}   # base_stem -> (category, fish_count)
annotation_files = sorted(set(glob.glob(os.path.join(root, '*_L*', '*', 'annotations*.json')) +
                              glob.glob(os.path.join(root, '*_L*', '*', '01_*', 'annotations*.json')) +
                              list(old_annotation_jsons)))
for f in annotation_files:
    with open(f) as fh:
        data = json.load(fh)
    for stem, category in data.get('annotations', {}).items():
        if category == 'unannotated':
            continue
        annotated[stem] = (category, data.get('fish_counts', {}).get(stem, ''))
print(f"{len(annotation_files)} annotation files, {len(annotated)} annotated events (excluded from this round)")

# yield report of the previous rounds
if len(drawn_before):
    drawn_before['status'] = 'not annotated'
    for i, stem in enumerate(drawn_before['base_stem']):
        if stem in annotated:
            category, count = annotated[stem]
            if category == 'clear_fish' and str(count) not in ('', 'unclear', 'None'):
                drawn_before.loc[i, 'status'] = f'verified_{count}fish'
            elif category == 'clear_fish':
                drawn_before.loc[i, 'status'] = 'unclear count'
            else:
                drawn_before.loc[i, 'status'] = category
    print("\nyield of the previous rounds (system x stratum x status):")
    print(pd.crosstab([drawn_before['system'], drawn_before['stratum']], drawn_before['status']).to_string())
    verified = drawn_before[drawn_before['status'].str.startswith('verified')]
    print("verified fish counts per system:")
    print(pd.crosstab(verified['system'], verified['status']).to_string())

eligible = pool[[(e, i) not in drawn_keys for e, i in zip(pool['extraction_folder'], pool['event_id'])]]
eligible = eligible[~eligible['base_stem'].isin(annotated.keys())].reset_index(drop=True)
print(f"\neligible after exclusions: {len(eligible)}")

# draw: per system x stratum, round-robin over sessions in date order
drawn = []
for system in sorted(eligible['system'].unique()):
    for stratum, n_wanted in quota.items():
        cand = eligible[(eligible['system'] == system) & (eligible['stratum'] == stratum)]
        sessions = cand.groupby('session')['date'].first().sort_values().index.tolist()
        per_session = {s: cand[cand['session'] == s].sample(frac=1, random_state=rng.integers(1e9)) for s in sessions}
        pointers = {s: 0 for s in sessions}
        picked = []
        while len(picked) < n_wanted and any(pointers[s] < len(per_session[s]) for s in sessions):
            for s in sessions:
                if len(picked) >= n_wanted:
                    break
                if pointers[s] < len(per_session[s]):
                    picked.append(per_session[s].iloc[pointers[s]])
                    pointers[s] += 1
        if len(picked) < n_wanted:
            print(f"  {system} {stratum}: only {len(picked)} of {n_wanted} available")
        drawn.extend(picked)
sample = pd.DataFrame(drawn)
sample['round'] = round_number
columns = ['round', 'system', 'site', 'session', 'date', 'extraction_folder', 'track_folder', 'event_id', 'base_stem', 'stratum',
           'n_fish_total', 'assignment_rate', 'duration_seconds', 'n_eods', 'n_channels', 'mean_ipi_seconds', 'ipi_cv', 'max_amplitude']
sample = sample[columns].sort_values(['system', 'session', 'event_id']).reset_index(drop=True)
out_file = os.path.join(output_folder, f'annotation_sample_round{round_number}.csv')
sample.to_csv(out_file, index=False)
print(f"\nround {round_number}: {len(sample)} events written to {out_file}")
print(pd.crosstab(sample['system'], sample['stratum']).to_string())
print("sessions per system:", sample.groupby('system')['session'].nunique().to_dict())
