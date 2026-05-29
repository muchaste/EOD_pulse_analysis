import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import filedialog

# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing all parameter_tuning_results*.csv files (accumulated across iterations).
# Also where proposals and plots are saved.
# Leave empty to open a folder dialog.
RESULTS_FOLDER   = ''

N_PROPOSALS      = 2000    # combos to hand to 05_2 next iteration
N_CANDIDATES     = 200000  # random candidates scored by surrogate for EI selection
RANDOM_SEED      = 42
RF_N_ESTIMATORS  = 500
RF_N_JOBS        = -1      # use all available CPUs for RF fit

# Exploration bonus: adds EXPLORATION_ALPHA * sigma to EI mu before EI calc.
# 0.0 = pure EI. Raise to ~0.3 if proposals cluster too tightly.
EXPLORATION_ALPHA = 0.0

PRIMARY   = 'score'
SECONDARY = 'acc_2fish'

# ============================================================
# PARAMETER BOUNDS  (min, max, type)
# Extended beyond original grid where best values were at edge.
# 'int' bounds are inclusive; 'float' bounds are continuous.
# Weight params handled separately via ratio encoding.
# ============================================================
PARAM_BOUNDS = {
    # --- high-impact, bounds extended ---
    'pass2_max_iterations':         (2,     30,    'int'),
    'pass2_max_gap_s':              (0.5,   10.0,  'float'),
    'ipi_tolerance_min_s':          (0.005, 0.3,   'float'),
    'pass1_new_frag_cost':          (0.5,   10.0,  'float'),
    # --- moderate impact ---
    'ipi_tolerance_fraction':       (0.1,   0.8,   'float'),
    'fft_artifact_threshold':       (0.4,   1.0,   'float'),
    'pass2_overlap_max_iterations': (1,     8,     'int'),
    # --- low impact, kept within sensible original range ---
    'min_ipi_s':                    (0.002, 0.02,  'float'),
    'max_track_gap_s':              (1,     15,    'float'),
    'max_location_jump_per_s':      (50,    600,   'float'),
    'knn_percentile':               (60,    95,    'int'),
    'min_shape_eps':                (0.1,   0.8,   'float'),
    'location_tolerance':           (3,     30,    'float'),
    'pass2_cost_threshold':         (0.5,   8.0,   'float'),
    'pass2_overlap_wf_threshold':   (0.1,   0.9,   'float'),
    'pass2_overlap_min_s':          (0.02,  0.3,   'float'),
    'width_min_separation_us':      (10,    60,    'float'),
}

# waveform_target_length and crop_factor require pre-normalization in 05_2 for each
# unique (wt, cf) combination. Constrain proposals to the discrete values already
# present in the GRID so the pre_normalized dict always has the needed key.
# Add new values here if you extend GRID['waveform_target_length'] / GRID['crop_factor'].
DISCRETE_ONLY = {
    'waveform_target_length': [150, 300],
    'crop_factor':            [4, 7],
}

# Expanded weight candidates  (triplets and pairs built from these)
_W_CANDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
WEIGHT_TRIPLETS_EXP = [
    (lw, iw, round(1.0 - lw - iw, 9))
    for lw in _W_CANDS for iw in _W_CANDS
    if 0.09 < round(1.0 - lw - iw, 9) < 0.91
]
WEIGHT_PAIRS_EXP = [(ww, round(1.0 - ww, 9)) for ww in _W_CANDS]

PARAM_COLS = (list(PARAM_BOUNDS.keys()) + list(DISCRETE_ONLY.keys()) + [
    'location_weight', 'ipi_weight', 'waveform_weight',
    'pass2_waveform_weight', 'pass2_spatial_weight',
])

# ============================================================
# FOLDER SELECTION
# ============================================================
if not RESULTS_FOLDER:
    RESULTS_FOLDER = os.environ.get('TUNING_RESULTS_FOLDER', '')
if not RESULTS_FOLDER:
    root = tk.Tk()
    root.withdraw()
    RESULTS_FOLDER = filedialog.askdirectory(title="Select results folder (contains parameter_tuning_results*.csv)")
    root.destroy()

print(f"Results folder: {RESULTS_FOLDER}\n")

# ============================================================
# LOAD ALL ACCUMULATED RESULTS
# ============================================================
csv_pattern = os.path.join(RESULTS_FOLDER, 'parameter_tuning_results*.csv')
all_csvs    = sorted(glob.glob(csv_pattern))
if not all_csvs:
    raise FileNotFoundError(f"No parameter_tuning_results*.csv found in {RESULTS_FOLDER}")

frames = []
for path in all_csvs:
    f = pd.read_csv(path)
    f['_source_file'] = os.path.basename(path)
    frames.append(f)

df_all = pd.concat(frames, ignore_index=True)
df_all = df_all.drop_duplicates(subset=PARAM_COLS)

n_nan = df_all[PRIMARY].isna().sum()
if n_nan > 0:
    print(f"Dropping {n_nan:,} rows with NaN score (all events failed — likely out-of-range params).")
    # Diagnose which params dominate NaN rows vs. valid rows
    nan_mask = df_all[PRIMARY].isna()
    shared   = [c for c in PARAM_BOUNDS if c in df_all.columns]
    if nan_mask.sum() > 0 and (~nan_mask).sum() > 0:
        print("  Mean param values — NaN rows vs valid rows:")
        for col in shared:
            m_nan   = df_all.loc[nan_mask,  col].mean()
            m_valid = df_all.loc[~nan_mask, col].mean()
            if abs(m_nan - m_valid) > 0.05 * (abs(m_valid) + 1e-9):
                print(f"    {col}: NaN={m_nan:.4g}  valid={m_valid:.4g}  Δ={m_nan-m_valid:+.4g}")
df_all = df_all.dropna(subset=[PRIMARY])

print(f"Loaded {len(df_all):,} unique evaluated combos from {len(all_csvs)} file(s)")
print(f"Score — min: {df_all[PRIMARY].min():.4f}  median: {df_all[PRIMARY].median():.4f}  "
      f"max: {df_all[PRIMARY].max():.4f}")

# Determine iteration number from existing proposal files
existing_proposals = sorted(glob.glob(os.path.join(RESULTS_FOLDER, 'proposals_iter*.csv')))
iteration = len(existing_proposals) + 1
print(f"Iteration: {iteration}\n")

# ============================================================
# FEATURE ENCODING
# Weight triplet: encode as (loc_weight, ipi_weight) — wf_weight = 1 - loc - ipi
# Weight pair:    encode as (pass2_waveform_weight)  — spatial = 1 - wf
# All other params: pass through as numeric
# ============================================================
feature_cols = (list(PARAM_BOUNDS.keys()) + list(DISCRETE_ONLY.keys()) +
                ['location_weight', 'ipi_weight', 'pass2_waveform_weight'])

X = df_all[feature_cols].values.astype(float)
y = df_all[PRIMARY].values

# ============================================================
# FIT RF SURROGATE
# ============================================================
print(f"Fitting RF surrogate ({RF_N_ESTIMATORS} trees)...")
rf = RandomForestRegressor(
    n_estimators=RF_N_ESTIMATORS,
    n_jobs=RF_N_JOBS,
    random_state=RANDOM_SEED,
    min_samples_leaf=1,
)

if len(X) >= 5:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    y_pred_val = rf.predict(X_val)
    r2_val     = r2_score(y_val, y_pred_val)
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  R²: {r2_val:.4f}")
    if r2_val < 0.3:
        print("  WARNING: R² < 0.3 — surrogate is weak. Proposals will be unreliable.")
    elif r2_val < 0.5:
        print("  NOTE: R² < 0.5 — surrogate is moderate. Proposals are directional.")
    else:
        print("  Surrogate quality: good.")
    print()
else:
    rf.fit(X, y)
    r2_val = float('nan')
    print(f"  Only {len(X)} sample(s) — skipping val split, training on all data.")
    print("  Proposals will be near-random with EI driven by uncertainty.\n")

# ============================================================
# GENERATE CANDIDATE POOL
# ============================================================
rng = np.random.default_rng(RANDOM_SEED + iteration)

n_weight_samples = N_CANDIDATES
candidates       = {}

for param, (lo, hi, ptype) in PARAM_BOUNDS.items():
    if ptype == 'int':
        candidates[param] = rng.integers(lo, hi + 1, size=n_weight_samples).astype(float)
    else:
        candidates[param] = rng.uniform(lo, hi, size=n_weight_samples)

# waveform_target_length and crop_factor: discrete only (GRID-valid values)
for param, allowed in DISCRETE_ONLY.items():
    idx = rng.integers(0, len(allowed), size=n_weight_samples)
    candidates[param] = np.array(allowed, dtype=float)[idx]

# Weight triplets: sample with replacement from expanded set
triplet_idx = rng.integers(0, len(WEIGHT_TRIPLETS_EXP), size=n_weight_samples)
triplet_arr = np.array(WEIGHT_TRIPLETS_EXP)[triplet_idx]
candidates['location_weight'] = triplet_arr[:, 0]
candidates['ipi_weight']      = triplet_arr[:, 1]
# waveform_weight = 1 - loc - ipi (stored in df but not in feature_cols, derived)

# Weight pairs
pair_idx = rng.integers(0, len(WEIGHT_PAIRS_EXP), size=n_weight_samples)
pair_arr  = np.array(WEIGHT_PAIRS_EXP)[pair_idx]
candidates['pass2_waveform_weight'] = pair_arr[:, 0]
candidates['pass2_spatial_weight']  = pair_arr[:, 1]

X_cand = np.column_stack([candidates[c] for c in feature_cols])
print(f"Scoring {N_CANDIDATES:,} candidates with surrogate...")

# Per-tree predictions for uncertainty estimate
tree_preds = np.array([tree.predict(X_cand) for tree in rf.estimators_])
mu_cand    = tree_preds.mean(axis=0)
sigma_cand = tree_preds.std(axis=0)

# ============================================================
# EXPECTED IMPROVEMENT ACQUISITION
# ============================================================
f_best  = y.max()
mu_adj  = mu_cand + EXPLORATION_ALPHA * sigma_cand
z       = np.where(sigma_cand > 1e-9, (mu_adj - f_best) / sigma_cand, 0.0)
ei      = np.where(sigma_cand > 1e-9,
                   (mu_adj - f_best) * norm.cdf(z) + sigma_cand * norm.pdf(z),
                   0.0)

top_idx    = np.argsort(ei)[::-1][:N_PROPOSALS]
top_mu     = mu_cand[top_idx]
top_sigma  = sigma_cand[top_idx]
top_ei     = ei[top_idx]

print(f"\nTop {N_PROPOSALS} proposals selected by EI")
print(f"  Predicted score — min: {top_mu.min():.4f}  "
      f"mean: {top_mu.mean():.4f}  max: {top_mu.max():.4f}")
print(f"  Predicted sigma — mean: {top_sigma.mean():.4f}  max: {top_sigma.max():.4f}")
print(f"  Current best observed score: {f_best:.4f}\n")

# ============================================================
# BUILD PROPOSALS DATAFRAME
# ============================================================
prop_rows = []
for i in top_idx:
    row = {}
    for param in PARAM_BOUNDS:
        lo, hi, ptype = PARAM_BOUNDS[param]
        val = candidates[param][i]
        row[param] = int(round(val)) if ptype == 'int' else float(val)
    for param in DISCRETE_ONLY:
        row[param] = int(candidates[param][i])
    lw   = candidates['location_weight'][i]
    iw   = candidates['ipi_weight'][i]
    ww   = round(1.0 - lw - iw, 9)
    p2ww = candidates['pass2_waveform_weight'][i]
    p2sw = candidates['pass2_spatial_weight'][i]
    row['location_weight']       = float(lw)
    row['ipi_weight']            = float(iw)
    row['waveform_weight']       = float(ww)
    row['pass2_waveform_weight'] = float(p2ww)
    row['pass2_spatial_weight']  = float(p2sw)
    prop_rows.append(row)

proposals_df = pd.DataFrame(prop_rows)
proposals_path = os.path.join(RESULTS_FOLDER, f'proposals_iter{iteration:02d}.csv')
proposals_df.to_csv(proposals_path, index=False)
print(f"Saved: {os.path.basename(proposals_path)}")

# ============================================================
# CONVERGENCE DIAGNOSTICS
# ============================================================
print("\n=== Convergence Diagnostics ===")
converged = False
if len(existing_proposals) == 0:
    print("  First optimization iteration — no prior iterations to compare.")
else:
    iter_scores = []
    for csv_path in all_csvs:
        chunk    = pd.read_csv(csv_path)
        best_in  = chunk[PRIMARY].dropna().max()
        iter_scores.append((os.path.basename(csv_path), best_in))
    print("  Best score per results file:")
    for fname, s in iter_scores:
        print(f"    {fname}: {s:.5f}")
    improvements = [iter_scores[i][1] - iter_scores[i - 1][1]
                    for i in range(1, len(iter_scores))]
    if improvements:
        last_imp = improvements[-1]
        print(f"  Last improvement: {last_imp:+.5f}")
        if len(improvements) >= 3 and all(abs(d) < 0.002 for d in improvements[-3:]):
            print("  CONVERGED: improvement < 0.002 for 3 consecutive iterations.")
            converged = True
        else:
            print("  Not yet converged.")

# Write convergence flag and proposals path for 05_6 to read
_flag_path = os.path.join(RESULTS_FOLDER, '_optimization_state.txt')
with open(_flag_path, 'w') as _f:
    _f.write(f'converged={int(converged)}\n')
    _f.write(f'proposals_path={proposals_path}\n')
    _f.write(f'iteration={iteration}\n')

# ============================================================
# PLOT 1 — CONVERGENCE CURVE (best score per results file)
# ============================================================
all_best = []
for csv_path in all_csvs:
    chunk = pd.read_csv(csv_path)
    all_best.append(chunk[PRIMARY].max())

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(range(1, len(all_best) + 1), all_best, 'o-', color='steelblue', lw=2)
ax1.axhline(max(all_best), color='red', lw=1, linestyle='--',
            label=f'best = {max(all_best):.4f}')
ax1.set_xlabel('Results file (iteration)')
ax1.set_ylabel('Best observed score')
ax1.set_title('Convergence Curve')
ax1.legend()
fig1.tight_layout()
fig1.savefig(os.path.join(RESULTS_FOLDER, f'05_5_iter{iteration:02d}_01_convergence.png'), dpi=150)
plt.close(fig1)
print(f"\nSaved: 05_5_iter{iteration:02d}_01_convergence.png")

# ============================================================
# PLOT 2 — RF FEATURE IMPORTANCES
# ============================================================
importances   = rf.feature_importances_
feat_order    = np.argsort(importances)
fig2, ax2     = plt.subplots(figsize=(9, 6))
ax2.barh(np.arange(len(feature_cols)), importances[feat_order],
         color='steelblue', alpha=0.85)
ax2.set_yticks(np.arange(len(feature_cols)))
ax2.set_yticklabels([feature_cols[i] for i in feat_order], fontsize=9)
ax2.set_xlabel('RF Feature Importance (impurity-based)')
ax2.set_title(f'Surrogate Feature Importances — iteration {iteration}  (R²={r2_val:.3f})')
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_FOLDER, f'05_5_iter{iteration:02d}_02_feature_importance.png'), dpi=150)
plt.close(fig2)
print(f"Saved: 05_5_iter{iteration:02d}_02_feature_importance.png")

# ============================================================
# PLOT 3 — PARTIAL DEPENDENCE for top 5 features
# ============================================================
top5_idx  = np.argsort(importances)[::-1][:5]
top5_cols = [feature_cols[i] for i in top5_idx]

fig3, axes3 = plt.subplots(1, 5, figsize=(20, 4))
X_pd = X_train if len(X) >= 5 else X
for ax, feat_idx, feat_name in zip(axes3, top5_idx, top5_cols):
    pd_result = partial_dependence(rf, X_pd, features=[feat_idx],
                                   kind='average', grid_resolution=40)
    ax.plot(pd_result['grid_values'][0], pd_result['average'][0],
            color='steelblue', lw=2)
    ax.set_xlabel(feat_name, fontsize=8)
    ax.set_ylabel('Partial dep. (score)', fontsize=8)
    ax.set_title(feat_name, fontsize=9)
    ax.tick_params(labelsize=7)

fig3.suptitle(f'Partial Dependence — top 5 features  (iteration {iteration})', fontsize=10)
fig3.tight_layout()
fig3.savefig(os.path.join(RESULTS_FOLDER, f'05_5_iter{iteration:02d}_03_partial_dependence.png'), dpi=150)
plt.close(fig3)
print(f"Saved: 05_5_iter{iteration:02d}_03_partial_dependence.png")

# ============================================================
# PLOT 4 — INTERACTION HEATMAPS for top 3 pairs (from surrogate)
# Using surrogate predictions on a 2D grid (not raw data bins)
# ============================================================
top3_pairs = []
pair_importances = []
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        pair_importances.append((importances[i] * importances[j], i, j))
pair_importances.sort(reverse=True)

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
for ax, (_, pi, pj) in zip(axes4, pair_importances[:3]):
    fn_i = feature_cols[pi]
    fn_j = feature_cols[pj]
    lo_i, hi_i, _ = PARAM_BOUNDS.get(fn_i, (X[:, pi].min(), X[:, pi].max(), 'float'))
    lo_j, hi_j, _ = PARAM_BOUNDS.get(fn_j, (X[:, pj].min(), X[:, pj].max(), 'float'))
    grid_i = np.linspace(lo_i, hi_i, 20)
    grid_j = np.linspace(lo_j, hi_j, 20)
    base_row = np.median(X_pd, axis=0)
    heat     = np.zeros((len(grid_i), len(grid_j)))
    for ri, vi in enumerate(grid_i):
        for rj, vj in enumerate(grid_j):
            r = base_row.copy()
            r[pi] = vi
            r[pj] = vj
            heat[ri, rj] = rf.predict(r.reshape(1, -1))[0]
    im = ax.imshow(heat, aspect='auto', origin='lower', cmap='viridis',
                   extent=[lo_j, hi_j, lo_i, hi_i])
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(fn_j, fontsize=8)
    ax.set_ylabel(fn_i, fontsize=8)
    ax.set_title(f'{fn_i[:14]} × {fn_j[:14]}', fontsize=9)

fig4.suptitle(f'Surrogate Interaction Heatmaps — top 3 feature pairs  (iteration {iteration})',
              fontsize=10)
fig4.tight_layout()
fig4.savefig(os.path.join(RESULTS_FOLDER, f'05_5_iter{iteration:02d}_04_interactions.png'), dpi=150)
plt.close(fig4)
print(f"Saved: 05_5_iter{iteration:02d}_04_interactions.png")

# ============================================================
# PRINT BEST OBSERVED COMBO SO FAR
# ============================================================
best_row = df_all.loc[df_all[PRIMARY].idxmax()]
print(f"\n=== Best observed combo so far (score={best_row[PRIMARY]:.5f}) ===")
for col in PARAM_COLS:
    if col in best_row:
        print(f"  {col}: {best_row[col]}")

print(f"\nNext step: set PROPOSALS_CSV = r'{proposals_path}' in 05_2 and run.\n")
print("Done.")
