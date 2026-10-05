# -*- coding: utf-8 -*-
"""
06_1_Waveform_Shape_Clustering.py

Fig 5 (RQ iii, breeding evidence) - Part 2: unsupervised clustering of every tracked fish's
mean EOD waveform SHAPE, across all sessions. Juvenile EOD pulses are described as a variable
mix of larval + adult discharge components (ratio changes with age) - not reliably captured by
width or amplitude alone (amplitude is heavily confounded by fish position relative to the
electrodes here, not just body size/condition). This script does not label juveniles itself -
it produces cluster visualizations for visual inspection, so the user can decide which
cluster(s) look juvenile-like before any downstream site/temporal analysis.

Note: fish flagged "Unknown" (species_uncertain) are DELIBERATELY included - a juvenile's mixed
waveform wouldn't match any adult reference individual well, which is exactly what triggers the
species classifier's out-of-distribution "uncertain" flag, so Unknown fish are strong juvenile
candidates and each cluster's Unknown fraction is reported to help spot this.

Authors: Stefan Mucha with Claude Sonnet 4.6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pulse_functions import load_waveforms

tracking_root = r"C:\Users\muchaste\Seafile\Uni\01 Research Projects\04 Uganda 2023 EOD loggers\01 Data\EOD Loggers\Tracking Results Reclassified\final_factor3.0_prob0.8"
output_root   = r"C:\Users\muchaste\Seafile\Uni\01 Research Projects\04 Uganda 2023 EOD loggers\03 Analysis Output\figures"

N_CLUSTERS = 12
N_PCA_COMPONENTS = 15
N_EXAMPLE_WAVEFORMS_PER_CLUSTER = 40

print("="*70)
print("WAVEFORM SHAPE CLUSTERING")
print("="*70)

session_folders = sorted(glob.glob(os.path.join(tracking_root, "*", "*")))
session_folders = [f for f in session_folders if os.path.isdir(f)
                   and os.path.exists(os.path.join(f, "tracked_fish_summary.csv"))]
print(f"✓ Found {len(session_folders)} tracked session folder(s)")

all_fish_meta = []
mean_wf_by_key = {}

for session_folder in session_folders:
    folder_parts = os.path.basename(session_folder).split("_")
    logger_id = folder_parts[0]
    site = folder_parts[2]
    session_date = folder_parts[1]
    system = os.path.basename(os.path.dirname(session_folder))

    fish_summary_path = os.path.join(session_folder, "tracked_fish_summary.csv")
    fish_df = pd.read_csv(fish_summary_path)
    fish_df["System"] = system
    fish_df["Site"] = site
    fish_df["Session_date"] = session_date
    fish_df["Logger_ID"] = logger_id
    all_fish_meta.append(fish_df[["fish_key", "species_assigned", "species_uncertain",
                                   "mean_width_us", "mean_location", "entry_time", "exit_time",
                                   "System", "Site", "Session_date", "Logger_ID"]])

    wf_npz_files = glob.glob(os.path.join(session_folder, "session_*_mean_waveforms.npz"))
    for wf_file in wf_npz_files:
        wf_base = wf_file[:-4]
        keys_file = wf_base + "_keys.csv"
        if not os.path.exists(keys_file):
            continue
        wf_list = load_waveforms(wf_base, format="npz", length="fixed")
        keys_df = pd.read_csv(keys_file)
        if len(wf_list) != len(keys_df):
            continue
        for fish_key, wf in zip(keys_df["fish_key"], wf_list):
            mean_wf_by_key[fish_key] = wf

fish_meta = pd.concat(all_fish_meta, ignore_index=True)
fish_meta = fish_meta[fish_meta["fish_key"].isin(mean_wf_by_key)].reset_index(drop=True)
print(f"✓ Loaded {len(fish_meta)} fish with matched mean waveforms "
      f"(from {len(mean_wf_by_key)} total stored waveforms)")

waveform_matrix = np.array([mean_wf_by_key[fk] for fk in fish_meta["fish_key"]])

print("\nFitting PCA + KMeans on waveform shapes...")
pca = PCA(n_components=N_PCA_COMPONENTS)
pca_scores = pca.fit_transform(waveform_matrix)
print(f"✓ PCA: {N_PCA_COMPONENTS} components explain "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}% of variance")

kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0)
fish_meta["cluster"] = kmeans.fit_predict(pca_scores)

# Per-cluster composition summary - helps spot candidate juvenile clusters (high Unknown
# fraction and/or unusual width) before visually inspecting the waveform shapes themselves
cluster_summary = fish_meta.groupby("cluster").agg(
    n_fish=("fish_key", "count"),
    pct_unknown=("species_uncertain", lambda x: 100 * (x == "True").mean()),
    mean_width_us=("mean_width_us", "mean"),
    std_width_us=("mean_width_us", "std"),
).reset_index()
species_by_cluster = (
    fish_meta.groupby(["cluster", "species_assigned"]).size().unstack(fill_value=0)
)
cluster_summary = cluster_summary.merge(species_by_cluster, on="cluster", how="left")
print("\n=== Cluster composition summary ===")
print(cluster_summary.to_string(index=False))

cluster_summary.to_csv(os.path.join(output_root, "waveform_cluster_summary.csv"), index=False)
fish_meta.to_csv(os.path.join(output_root, "waveform_cluster_assignment.csv"), index=False)
print(f"\n✓ Saved cluster_summary + per-fish cluster assignment to {output_root}")

# Visualization: one panel per cluster, thin sample waveforms + bold mean waveform, labeled
# with n_fish and %Unknown so juvenile-candidate clusters are easy to spot at a glance
n_cols = 4
n_rows = int(np.ceil(N_CLUSTERS / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, sharey=True)
axes = axes.flatten()

rng = np.random.default_rng(0)
for c in range(N_CLUSTERS):
    ax = axes[c]
    idx = np.where(fish_meta["cluster"].values == c)[0]
    sample_idx = rng.choice(idx, size=min(N_EXAMPLE_WAVEFORMS_PER_CLUSTER, len(idx)), replace=False)
    for i in sample_idx:
        ax.plot(waveform_matrix[i], color="gray", alpha=0.15, linewidth=0.5)
    ax.plot(waveform_matrix[idx].mean(axis=0), color="crimson", linewidth=1.8)
    row = cluster_summary.loc[cluster_summary["cluster"] == c].iloc[0]
    ax.set_title(f"Cluster {c}: n={row['n_fish']}, {row['pct_unknown']:.0f}% Unknown\n"
                 f"width={row['mean_width_us']:.0f}\u00b5s", fontsize=8)
    ax.set_xticks([])

for c in range(N_CLUSTERS, len(axes)):
    axes[c].axis("off")

plt.tight_layout()
fig_path = os.path.join(output_root, "waveform_cluster_shapes.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"✓ Saved cluster shape grid: {os.path.basename(fig_path)}")

print("\n" + "="*70)
print("DONE - inspect waveform_cluster_shapes.png and waveform_cluster_summary.csv,")
print("then flag which cluster ID(s) look juvenile-like for the R-side follow-up analysis.")
print("="*70)
