# -*- coding: utf-8 -*-
"""
04_5_Tuning_Analysis.py

This script analyzes the results of the tuning process, generating plots and summaries.

Authors: Stefan Mucha with Claude Sonnet 4.6
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import tkinter as tk
from tkinter import filedialog

# ============================================================
# CONFIGURATION
# ============================================================
root = tk.Tk()
root.withdraw()
CSV_PATH = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
OUTPUT_DIR = os.path.dirname(CSV_PATH)
ETA2_THRESHOLD = 0.01
TOP_QUANTILE   = 0.20
N_TOP_PAIRS    = 20

PARAM_COLS = [
    'waveform_target_length', 'crop_factor', 'min_ipi_s', 'max_track_gap_s',
    'max_location_jump_per_s', 'knn_percentile', 'min_shape_eps',
    'fft_artifact_threshold', 'location_tolerance', 'ipi_tolerance_fraction',
    'ipi_tolerance_min_s', 'pass1_new_frag_cost', 'pass2_max_gap_s',
    'pass2_cost_threshold', 'pass2_max_iterations', 'pass2_overlap_wf_threshold',
    'pass2_overlap_min_s', 'pass2_overlap_max_iterations', 'width_min_separation_us',
    'location_weight', 'ipi_weight', 'waveform_weight',
    'pass2_waveform_weight', 'pass2_spatial_weight',
]

WEIGHT_PARAMS = frozenset([
    'location_weight', 'ipi_weight', 'waveform_weight',
    'pass2_waveform_weight', 'pass2_spatial_weight',
])

PRIMARY   = 'score'
SECONDARY = 'acc_2fish'

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} rows  |  {len(PARAM_COLS)} parameter columns")
print(f"Score     — min: {df[PRIMARY].min():.4f}  median: {df[PRIMARY].median():.4f}  "
      f"max: {df[PRIMARY].max():.4f}  90th pct: {df[PRIMARY].quantile(0.9):.4f}")
print(f"acc_2fish — min: {df[SECONDARY].min():.4f}  median: {df[SECONDARY].median():.4f}  "
      f"max: {df[SECONDARY].max():.4f}  90th pct: {df[SECONDARY].quantile(0.9):.4f}")
print(f"\nNote: pass1 weights (loc+ipi+wf) sum=1; pass2 weights (wf+spatial) sum=1 — "
      f"within-group eta² values are not independent.\n")

# ============================================================
# PHASE 1 — SCORE DISTRIBUTION
# ============================================================
top_cutoff     = df[PRIMARY].quantile(1.0 - TOP_QUANTILE)
top_cutoff_sec = df[SECONDARY].quantile(1.0 - TOP_QUANTILE)

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

axes1[0].hist(df[PRIMARY], bins=60, color='steelblue', edgecolor='none', alpha=0.8)
axes1[0].axvline(top_cutoff, color='red', lw=1.5, linestyle='--',
                 label=f'top {int(TOP_QUANTILE * 100)}% cutoff ({top_cutoff:.3f})')
axes1[0].set_xlabel('Score (composite)')
axes1[0].set_ylabel('Count')
axes1[0].set_title(f'Score Distribution — {len(df):,} combos')
axes1[0].legend()

axes1[1].hist(df[SECONDARY], bins=60, color='darkorange', edgecolor='none', alpha=0.8)
axes1[1].axvline(top_cutoff_sec, color='red', lw=1.5, linestyle='--',
                 label=f'top {int(TOP_QUANTILE * 100)}% cutoff ({top_cutoff_sec:.3f})')
axes1[1].set_xlabel('acc_2fish')
axes1[1].set_ylabel('Count')
axes1[1].set_title('2-Fish Accuracy Distribution')
axes1[1].legend()

fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, '05_4_01_score_distribution.png'), dpi=150)
plt.close(fig1)
print("Saved: 05_4_01_score_distribution.png")

# ============================================================
# PHASE 2 — ETA² (ANOVA EFFECT SIZE)
# ============================================================
grand_mean_prim = df[PRIMARY].mean()
grand_mean_sec  = df[SECONDARY].mean()
ss_total_prim   = ((df[PRIMARY]   - grand_mean_prim) ** 2).sum()
ss_total_sec    = ((df[SECONDARY] - grand_mean_sec)  ** 2).sum()

eta2_primary   = {}
eta2_secondary = {}

for param in PARAM_COLS:
    gm_p = df.groupby(param)[PRIMARY].mean()
    gc_p = df.groupby(param)[PRIMARY].count()
    eta2_primary[param] = (gc_p * (gm_p - grand_mean_prim) ** 2).sum() / ss_total_prim

    gm_s = df.groupby(param)[SECONDARY].mean()
    gc_s = df.groupby(param)[SECONDARY].count()
    eta2_secondary[param] = (gc_s * (gm_s - grand_mean_sec) ** 2).sum() / ss_total_sec

eta2_df = pd.DataFrame({
    'eta2_score':    eta2_primary,
    'eta2_acc2fish': eta2_secondary,
}).sort_values('eta2_score', ascending=False)

print("\n=== eta² Effect Sizes ===")
print(eta2_df.to_string(float_format=lambda x: f'{x:.5f}'))
n_above = (eta2_df['eta2_score'] >= ETA2_THRESHOLD).sum()
print(f"\nAbove threshold (eta² >= {ETA2_THRESHOLD}): {n_above} / {len(PARAM_COLS)}\n")

params_sorted = eta2_df.index.tolist()
y_pos = np.arange(len(params_sorted))

fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.barh(y_pos - 0.2, [eta2_primary[p]   for p in params_sorted], height=0.35,
         label='score (composite)', color='steelblue', alpha=0.85)
ax2.barh(y_pos + 0.2, [eta2_secondary[p] for p in params_sorted], height=0.35,
         label='acc_2fish', color='darkorange', alpha=0.85)
ax2.axvline(ETA2_THRESHOLD, color='red', lw=1.2, linestyle='--',
            label=f'threshold {ETA2_THRESHOLD}')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(params_sorted, fontsize=9)
ax2.set_xlabel('eta²')
ax2.set_title('Parameter Effect Size (eta²) — composite score and 2-fish accuracy')
ax2.legend()
ax2.invert_yaxis()
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, '05_4_02_eta2_effect_sizes.png'), dpi=150)
plt.close(fig2)
print("Saved: 05_4_02_eta2_effect_sizes.png")

# ============================================================
# PHASE 3 — MARGINAL MEAN PLOTS
# ============================================================
n_params = len(PARAM_COLS)
n_cols3  = 4
n_rows3  = int(np.ceil(n_params / n_cols3))

fig3, axes3 = plt.subplots(n_rows3, n_cols3, figsize=(16, n_rows3 * 3.2))
axes3_flat  = axes3.flatten()

for i, param in enumerate(PARAM_COLS):
    ax    = axes3_flat[i]
    ax_r  = ax.twinx()

    means_p = df.groupby(param)[PRIMARY].mean().sort_index()
    stds_p  = df.groupby(param)[PRIMARY].std().sort_index()
    means_s = df.groupby(param)[SECONDARY].mean().sort_index()

    x_labels = [str(v) for v in means_p.index]
    x_pos    = np.arange(len(means_p))

    ax.bar(x_pos, means_p.values, yerr=stds_p.values, color='steelblue',
           alpha=0.65, capsize=3, error_kw={'lw': 1})
    best_idx = int(means_p.values.argmax())
    ax.bar(best_idx, means_p.values[best_idx], color='red', alpha=0.45)

    ax_r.plot(x_pos, means_s.values, 'o-', color='darkorange', lw=1.5, ms=5)
    ax_r.tick_params(axis='y', labelsize=6, labelcolor='darkorange')
    ax_r.set_ylabel('acc_2fish', fontsize=6, color='darkorange')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=30, ha='right')
    ax.set_title(f'{param}  η²={eta2_primary[param]:.3f}', fontsize=8)
    ax.set_ylabel('mean score', fontsize=7)
    ax.tick_params(axis='y', labelsize=7)

for j in range(i + 1, len(axes3_flat)):
    axes3_flat[j].set_visible(False)

fig3.suptitle(
    'Marginal Mean Score per Parameter Level  '
    '(red bar = best level,  orange line = acc_2fish)', fontsize=10)
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, '05_4_03_marginal_means.png'), dpi=150)
plt.close(fig3)
print("Saved: 05_4_03_marginal_means.png")

# ============================================================
# PHASE 4 — TOP-QUINTILE MODAL ANALYSIS
# ============================================================
df_top = df[df[PRIMARY] >= top_cutoff].copy()
print(f"Top {int(TOP_QUANTILE * 100)}% subset: {len(df_top):,} rows  (score >= {top_cutoff:.4f})\n")

fig4, axes4 = plt.subplots(n_rows3, n_cols3, figsize=(16, n_rows3 * 3.2))
axes4_flat  = axes4.flatten()

for i, param in enumerate(PARAM_COLS):
    ax = axes4_flat[i]

    full_freq  = df[param].value_counts(normalize=True).sort_index()
    top_freq   = df_top[param].value_counts(normalize=True).sort_index()
    all_levels = sorted(set(full_freq.index) | set(top_freq.index))
    x_pos      = np.arange(len(all_levels))

    full_vals = [full_freq.get(lv, 0.0) for lv in all_levels]
    top_vals  = [top_freq.get(lv, 0.0)  for lv in all_levels]

    ax.bar(x_pos - 0.18, full_vals, width=0.35, color='steelblue', alpha=0.7, label='all')
    ax.bar(x_pos + 0.18, top_vals,  width=0.35, color='red',       alpha=0.7, label='top 20%')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in all_levels], fontsize=7, rotation=30, ha='right')
    ax.set_title(param, fontsize=8)
    ax.set_ylabel('proportion', fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)

for j in range(i + 1, len(axes4_flat)):
    axes4_flat[j].set_visible(False)

fig4.suptitle(
    f'Value Frequency: All combos vs Top {int(TOP_QUANTILE * 100)}% by Score', fontsize=10)
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, '05_4_04_top_quintile_modal.png'), dpi=150)
plt.close(fig4)
print("Saved: 05_4_04_top_quintile_modal.png")

# ============================================================
# PHASE 5 — PAIRWISE INTERACTION EFFECTS (TOP N_TOP_PAIRS)
# ============================================================
all_pairs = list(combinations(PARAM_COLS, 2))
print(f"Computing interactions for {len(all_pairs)} pairs...")

pair_records = []
for pA, pB in all_pairs:
    joint_key = df[pA].astype(str) + '|' + df[pB].astype(str)
    gm_j = df[PRIMARY].groupby(joint_key).mean()
    gc_j = df[PRIMARY].groupby(joint_key).count()
    eta2_joint       = (gc_j * (gm_j - grand_mean_prim) ** 2).sum() / ss_total_prim
    eta2_interaction = max(0.0, eta2_joint - eta2_primary[pA] - eta2_primary[pB])
    pair_records.append({
        'param_A':          pA,
        'param_B':          pB,
        'eta2_A':           eta2_primary[pA],
        'eta2_B':           eta2_primary[pB],
        'eta2_joint':       eta2_joint,
        'eta2_interaction': eta2_interaction,
    })

pair_df = pd.DataFrame(pair_records).sort_values('eta2_interaction', ascending=False)
print(f"\n=== Top {N_TOP_PAIRS} Interaction Pairs (eta²_int = joint − main A − main B) ===")
print(pair_df.head(N_TOP_PAIRS).to_string(index=False, float_format=lambda x: f'{x:.5f}'))

top_pairs = pair_df.head(N_TOP_PAIRS).reset_index(drop=True)
n_rows5   = 4
n_cols5   = 5
vmin      = df[PRIMARY].min()
vmax      = df[PRIMARY].max()

fig5, axes5 = plt.subplots(n_rows5, n_cols5, figsize=(22, 14))
axes5_flat  = axes5.flatten()

for idx, row in top_pairs.iterrows():
    ax = axes5_flat[idx]
    pA = row['param_A']
    pB = row['param_B']
    heat = df.groupby([pA, pB])[PRIMARY].mean().unstack()
    im = ax.imshow(heat.values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(v) for v in heat.columns], fontsize=6, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([str(v) for v in heat.index], fontsize=6)
    ax.set_xlabel(pB, fontsize=7)
    ax.set_ylabel(pA, fontsize=7)
    ax.set_title(f'{pA[:12]} × {pB[:12]}\nη²_int={row["eta2_interaction"]:.4f}', fontsize=7)

for j in range(len(top_pairs), len(axes5_flat)):
    axes5_flat[j].set_visible(False)

fig5.suptitle(
    f'Top {N_TOP_PAIRS} Pairwise Interactions — Mean Score per Cell'
    f'  (score range {vmin:.3f}–{vmax:.3f})', fontsize=11)
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, '05_4_05_interactions.png'), dpi=150)
plt.close(fig5)
print("Saved: 05_4_05_interactions.png")

# ============================================================
# PHASE 6 — NARROWED GRID SUGGESTION
# ============================================================
non_weight_params = [p for p in PARAM_COLS if p not in WEIGHT_PARAMS]
suggested_grid    = {}
fixed_params      = {}

for param in non_weight_params:
    means_by_val = df.groupby(param)[PRIMARY].mean().sort_values(ascending=False)
    if eta2_primary[param] >= ETA2_THRESHOLD:
        top_vals = sorted(means_by_val.head(2).index.tolist())
        suggested_grid[param] = top_vals
    else:
        modal_val = df_top[param].mode().iloc[0]
        fixed_params[param]   = modal_val

w1_means = df.groupby(['location_weight', 'ipi_weight', 'waveform_weight'])[PRIMARY].mean()
w1_means = w1_means.sort_values(ascending=False)

w2_means = df.groupby(['pass2_waveform_weight', 'pass2_spatial_weight'])[PRIMARY].mean()
w2_means = w2_means.sort_values(ascending=False)

n_suggested = 1
for vals in suggested_grid.values():
    n_suggested *= len(vals)
n_w1 = min(3, len(w1_means))
n_w2 = min(3, len(w2_means))
total_narrowed = n_suggested * n_w1 * n_w2

print("\n=== Narrowed GRID Suggestion ===")
print(f"  eta² >= {ETA2_THRESHOLD} → top 2 values by mean score kept in grid")
print(f"  eta² <  {ETA2_THRESHOLD} → fixed at top-{int(TOP_QUANTILE*100)}%-quintile modal value")
print("  Weight params handled separately as triplets/pairs\n")
print("GRID = {")
for param, vals in suggested_grid.items():
    print(f"    {param!r}: {vals},")
print("}")
print(f"\n# Fixed (eta² < {ETA2_THRESHOLD}), set to top-quintile mode:")
for param, val in fixed_params.items():
    print(f"#   {param}: {val}")
print(f"\n# Pass 1 weight triplets — top 5:")
for (lw, iw, ww), s in w1_means.head(5).items():
    print(f"#   location={lw}  ipi={iw}  waveform={ww}  →  mean score {s:.4f}")
print(f"\n# Pass 2 weight pairs — top 5:")
for (ww, sw), s in w2_means.head(5).items():
    print(f"#   waveform={ww}  spatial={sw}  →  mean score {s:.4f}")
print(f"\nNarrowed grid: {n_suggested} param combos × {n_w1} weight triplets × {n_w2} weight pairs "
      f"= {total_narrowed:,} total combinations")

# ============================================================
# SAVE — ANALYSIS SUMMARY CSV
# ============================================================
# eta² per parameter (main effects)
summary_rows = []
for param in PARAM_COLS:
    means_by_val = df.groupby(param)[PRIMARY].mean().sort_values(ascending=False)
    best_val     = means_by_val.index[0]
    best_mean    = means_by_val.iloc[0]
    in_grid      = param in suggested_grid
    fixed_val    = fixed_params.get(param, None)
    summary_rows.append({
        'parameter':       param,
        'eta2_score':      eta2_primary[param],
        'eta2_acc2fish':   eta2_secondary[param],
        'above_threshold': eta2_primary[param] >= ETA2_THRESHOLD,
        'best_value':      best_val,
        'best_mean_score': round(best_mean, 5),
        'grid_values':     str(suggested_grid[param]) if in_grid else '',
        'fixed_value':     '' if in_grid else fixed_val,
    })

summary_df = pd.DataFrame(summary_rows).sort_values('eta2_score', ascending=False)

# pass 1 weight triplets
w1_rows = []
for (lw, iw, ww), s in w1_means.items():
    w1_rows.append({'location_weight': lw, 'ipi_weight': iw, 'waveform_weight': ww,
                    'mean_score': round(s, 5)})
w1_df = pd.DataFrame(w1_rows)

# pass 2 weight pairs
w2_rows = []
for (ww, sw), s in w2_means.items():
    w2_rows.append({'pass2_waveform_weight': ww, 'pass2_spatial_weight': sw,
                    'mean_score': round(s, 5)})
w2_df = pd.DataFrame(w2_rows)

# interaction effects (all pairs, sorted)
pair_df_out = pair_df[['param_A', 'param_B', 'eta2_A', 'eta2_B',
                        'eta2_joint', 'eta2_interaction']].copy()
pair_df_out = pair_df_out.round(6)

summary_path     = os.path.join(OUTPUT_DIR, '05_4_analysis_summary.csv')
w1_path          = os.path.join(OUTPUT_DIR, '05_4_w1_triplets.csv')
w2_path          = os.path.join(OUTPUT_DIR, '05_4_w2_pairs.csv')
interactions_path = os.path.join(OUTPUT_DIR, '05_4_interactions.csv')

summary_df.to_csv(summary_path,      index=False)
w1_df.to_csv(w1_path,               index=False)
w2_df.to_csv(w2_path,               index=False)
pair_df_out.to_csv(interactions_path, index=False)

print(f"\nSaved: {os.path.basename(summary_path)}")
print(f"Saved: {os.path.basename(w1_path)}")
print(f"Saved: {os.path.basename(w2_path)}")
print(f"Saved: {os.path.basename(interactions_path)}")

# ============================================================
# SAVE — NARROWED GRID CSV
# ============================================================
# One row per parameter. Grid params have comma-separated values; fixed params have a single value.
grid_rows = []
for param, vals in suggested_grid.items():
    grid_rows.append({
        'parameter': param,
        'type':      'grid',
        'values':    ','.join(str(v) for v in vals),
        'eta2_score': round(eta2_primary[param], 6),
    })
for param, val in fixed_params.items():
    grid_rows.append({
        'parameter':  param,
        'type':       'fixed',
        'values':     str(val),
        'eta2_score': round(eta2_primary[param], 6),
    })
# weight triplets (top 3)
for (lw, iw, ww), s in w1_means.head(3).items():
    grid_rows.append({
        'parameter':  f'weight_triplet__loc{lw}_ipi{iw}_wf{ww}',
        'type':       'weight_triplet',
        'values':     f'{lw},{iw},{ww}',
        'eta2_score': round(s, 6),
    })
# weight pairs (top 3)
for (ww, sw), s in w2_means.head(3).items():
    grid_rows.append({
        'parameter':  f'weight_pair__wf{ww}_sp{sw}',
        'type':       'weight_pair',
        'values':     f'{ww},{sw}',
        'eta2_score': round(s, 6),
    })

grid_out_path = os.path.join(OUTPUT_DIR, '05_4_narrowed_grid.csv')
pd.DataFrame(grid_rows).to_csv(grid_out_path, index=False)
print(f"Saved: {os.path.basename(grid_out_path)}")

print("\nDone.")
