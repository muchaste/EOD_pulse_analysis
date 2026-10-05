# -*- coding: utf-8 -*-
"""
06_2_Species_Grouped_Waveform_Clustering.py

Fig 5 (RQ iii, breeding evidence) - Part 2 redesign. The pooled fixed-k KMeans approach
(06_1) likely spends most of its cluster budget separating SPECIES shape differences from
each other, diluting the within-species life-stage (juvenile/adult) signal actually of
interest. Fix: split by LDA-assigned species FIRST (including "Unknown" as its own group,
species_uncertain=="True" fish), then run DBSCAN (density-based, no fixed k, has a noise
label) SEPARATELY within each species - a better match for "mostly one adult blob + a rare
deviant juvenile subgroup" than forcing everything into one of a fixed number of clusters
shared across species.

Two outputs:
1. Overview: a single global PCA + DBSCAN pass across ALL fish, scatter plot colored by
   Species (not by DBSCAN cluster id) with global DBSCAN noise points marked separately, plus
   one representative mean waveform per species.
2. Per-species refinement: for each species group, its OWN PCA embedding (global PCA axes are
   dominated by between-species variance, not useful for within-species structure) and its OWN
   Gaussian Mixture Model (GMM) fit, with a cluster shape grid (same visual style as 06_1) for
   user inspection. GMM (not DBSCAN) is used here specifically: DBSCAN is density-based and can
   only recognize DENSE regions as clusters, so a diffuse/high-variance population (juvenile
   EODs are expected to be "plastic"/variable, not tightly clustered) gets dumped into noise
   regardless of eps/min_samples tuning - a GMM component can legitimately be broad and still
   register as a genuine cluster, which is the better match for this hypothesis.

Ground-truth anchor: two known small PN fish with a double-pulse waveform (per user) exist in
the data - ANCHOR_FISH_KEYS below identifies them by fish_key so the script can report which
GMM component they land in and its posterior probability, then list other high-posterior
members of that same component for visual review (semi-supervised validation).

Authors: Stefan Mucha with Claude Sonnet 4.6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from pulse_functions import load_waveforms

tracking_root = r"C:\Users\muchaste\Seafile\Uni\01 Research Projects\04 Uganda 2023 EOD loggers\01 Data\EOD Loggers\Tracking Results Reclassified\final_factor3.0_prob0.8"
output_root   = r"C:\Users\muchaste\Seafile\Uni\01 Research Projects\04 Uganda 2023 EOD loggers\03 Analysis Output\figures"

SPECIES_CODES = ["GL", "PN", "PD", "MV", "Unknown"]
SPECIES_COLORS = {"GL": "tab:blue", "PN": "tab:orange", "PD": "tab:green",
                   "MV": "tab:red", "Unknown": "black"}

# Known ground-truth double-pulse (juvenile-like) fish, identified by the user outside this
# pipeline - fill in with their exact fish_key values (format: e.g.
# "L5-20231011T091534_event_2_fish_3") to enable the anchor check below. Left empty skips it.
ANCHOR_FISH_KEYS = {"PN": []}

N_PCA_COMPONENTS_GLOBAL = 15
N_PCA_COMPONENTS_SPECIES = 15
MIN_SAMPLES = 15  # DBSCAN min_samples, global overview pass only
EPS_PERCENTILE_DEFAULT = 90  # eps chosen as this percentile of each point's MIN_SAMPLES-th
                             # nearest-neighbor distance (elbow heuristic) - global pass only
N_GMM_COMPONENTS_RANGE = range(2, 11)  # per-species pass: BIC-selected, not fixed - lets the
                                        # data pick how many sub-populations exist per species
N_EXAMPLE_WAVEFORMS_PER_CLUSTER = 40

print("="*70)
print("SPECIES-GROUPED WAVEFORM SHAPE CLUSTERING")
print("="*70)

# 1. LOAD DATA (same loading logic as 06_1) --------------------------------------------

session_folders = sorted(glob.glob(os.path.join(tracking_root, "*", "*")))
session_folders = [f for f in session_folders if os.path.isdir(f)
                   and os.path.exists(os.path.join(f, "tracked_fish_summary.csv"))]
print(f"Found {len(session_folders)} tracked session folder(s)")

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
# species_uncertain may be read as an actual bool (True/False) or as a string ("True"/"False")
# depending on the CSV - compare as lowercase string either way, and also catch any leftover
# "?" suffix in species_assigned directly (both should mean the same "uncertain" thing)
is_uncertain = (fish_meta["species_uncertain"].astype(str).str.strip().str.lower() == "true") \
               | fish_meta["species_assigned"].astype(str).str.contains(r"\?", regex=True)
fish_meta["Species"] = np.where(
    is_uncertain, "Unknown", fish_meta["species_assigned"].astype(str).str.replace("?", "", regex=False)
)
print(f"Loaded {len(fish_meta)} fish with matched mean waveforms "
      f"(from {len(mean_wf_by_key)} total stored waveforms)")
print(fish_meta["Species"].value_counts())

waveform_matrix = np.array([mean_wf_by_key[fk] for fk in fish_meta["fish_key"]])

# 2. GLOBAL OVERVIEW: PCA + DBSCAN across all fish, colored by Species -------------------

print("\nFitting global PCA + DBSCAN overview...")
pca_global = PCA(n_components=N_PCA_COMPONENTS_GLOBAL)
pca_scores_global = pca_global.fit_transform(waveform_matrix)
print(f"Global PCA: {N_PCA_COMPONENTS_GLOBAL} components explain "
      f"{pca_global.explained_variance_ratio_.sum()*100:.1f}% of variance")

nn = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(pca_scores_global)
kth_dist_global, _ = nn.kneighbors(pca_scores_global)
eps_global = np.percentile(kth_dist_global[:, -1], EPS_PERCENTILE_DEFAULT)
print(f"Global DBSCAN: eps={eps_global:.2f} (from {EPS_PERCENTILE_DEFAULT}th percentile of "
      f"{MIN_SAMPLES}-NN distance), min_samples={MIN_SAMPLES}")

dbscan_global = DBSCAN(eps=eps_global, min_samples=MIN_SAMPLES)
fish_meta["global_dbscan_cluster"] = dbscan_global.fit_predict(pca_scores_global)
n_noise_global = (fish_meta["global_dbscan_cluster"] == -1).sum()
print(f"Global DBSCAN: {fish_meta['global_dbscan_cluster'].nunique() - 1} cluster(s) found, "
      f"{n_noise_global} noise point(s) ({100*n_noise_global/len(fish_meta):.1f}%)")
print(pd.crosstab(fish_meta["Species"], fish_meta["global_dbscan_cluster"] == -1,
                   rownames=["Species"], colnames=["is_noise"]))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
for sp in SPECIES_CODES:
    idx = fish_meta["Species"].values == sp
    ax.scatter(pca_scores_global[idx, 0], pca_scores_global[idx, 1],
               s=6, alpha=0.4, color=SPECIES_COLORS[sp], label=sp)
# noise markers drawn small/faint and thin so they don't visually swamp the species coloring
# underneath - there can be thousands of noise points globally (~5-6% of a 150k-fish dataset)
noise_idx = fish_meta["global_dbscan_cluster"].values == -1
ax.scatter(pca_scores_global[noise_idx, 0], pca_scores_global[noise_idx, 1],
           s=10, facecolors="none", edgecolors="magenta", linewidths=0.4, alpha=0.25,
           label="DBSCAN noise")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Global PCA of all waveforms, colored by assigned Species\n"
             "(magenta rings = global DBSCAN noise/outlier points)")
ax.legend(fontsize=8, markerscale=2)

ax = axes[1]
for sp in SPECIES_CODES:
    idx = fish_meta["Species"].values == sp
    if idx.sum() == 0:
        continue
    ax.plot(waveform_matrix[idx].mean(axis=0), color=SPECIES_COLORS[sp],
             linewidth=1.8, label=f"{sp} (n={idx.sum()})")
ax.set_title("Representative (mean) waveform per Species")
ax.set_xticks([])
ax.legend(fontsize=8)

plt.tight_layout()
overview_path = os.path.join(output_root, "waveform_overview_by_species.png")
plt.savefig(overview_path, dpi=150)
plt.close()
print(f"Saved overview figure: {os.path.basename(overview_path)}")

# 3. PER-SPECIES REFINEMENT: own PCA + DBSCAN per species --------------------------------
# Global PCA axes are dominated by between-species shape variance - a separate embedding per
# species is needed to resolve within-species (life-stage) structure, which is the actual
# target here.

all_species_assignments = []

for sp in SPECIES_CODES:
    sp_idx = np.where(fish_meta["Species"].values == sp)[0]
    sp_waveforms = waveform_matrix[sp_idx]
    n_sp = len(sp_idx)
    print(f"\n--- {sp}: n={n_sp} ---")
    if n_sp < 3 * MIN_SAMPLES:
        print(f"Too few fish for a stable per-species DBSCAN (need >= {3*MIN_SAMPLES}), skipping.")
        continue

    n_comp_sp = min(N_PCA_COMPONENTS_SPECIES, n_sp - 1, sp_waveforms.shape[1])
    pca_sp = PCA(n_components=n_comp_sp)
    pca_scores_sp = pca_sp.fit_transform(sp_waveforms)
    print(f"{sp} PCA: {n_comp_sp} components explain "
          f"{pca_sp.explained_variance_ratio_.sum()*100:.1f}% of variance")

    bic_scores = []
    for k in N_GMM_COMPONENTS_RANGE:
        gmm_k = GaussianMixture(n_components=k, random_state=0, n_init=3)
        gmm_k.fit(pca_scores_sp)
        bic_scores.append(gmm_k.bic(pca_scores_sp))
    best_k = list(N_GMM_COMPONENTS_RANGE)[int(np.argmin(bic_scores))]
    print(f"{sp} GMM: BIC sweep over k={list(N_GMM_COMPONENTS_RANGE)} -> "
          f"{[f'{b:.0f}' for b in bic_scores]}, best k={best_k}")

    gmm_sp = GaussianMixture(n_components=best_k, random_state=0, n_init=5)
    cluster_labels_sp = gmm_sp.fit_predict(pca_scores_sp)
    posterior_sp = gmm_sp.predict_proba(pca_scores_sp)
    component_sizes = pd.Series(cluster_labels_sp).value_counts().sort_index()
    print(f"{sp}: {best_k} GMM component(s), sizes = {component_sizes.to_dict()}")

    sp_assignment = fish_meta.iloc[sp_idx][["fish_key", "Species", "mean_width_us",
                                             "mean_location", "System", "Site",
                                             "Session_date"]].copy()
    sp_assignment["cluster_within_species"] = cluster_labels_sp
    sp_assignment["posterior_max"] = posterior_sp.max(axis=1)
    all_species_assignments.append(sp_assignment)

    # DOUBLE-PULSE ANCHOR CHECK - semi-supervised validation using known ground-truth fish
    anchor_keys_sp = ANCHOR_FISH_KEYS.get(sp, [])
    if anchor_keys_sp:
        anchor_mask = sp_assignment["fish_key"].isin(anchor_keys_sp)
        if anchor_mask.any():
            print(f"\n{sp} double-pulse anchor fish:")
            print(sp_assignment.loc[anchor_mask, ["fish_key", "cluster_within_species", "posterior_max"]]
                  .to_string(index=False))
            anchor_components = sp_assignment.loc[anchor_mask, "cluster_within_species"].unique()
            for comp in anchor_components:
                comp_members = sp_assignment[sp_assignment["cluster_within_species"] == comp]
                top_members = comp_members.sort_values("posterior_max", ascending=False).head(20)
                print(f"  Top 20 highest-posterior members of component {comp} "
                      f"(n={len(comp_members)} total) - inspect these for juvenile-like shape:")
                print(top_members[["fish_key", "posterior_max"]].to_string(index=False))
        else:
            print(f"WARNING: anchor fish_key(s) for {sp} not found in this species' data - "
                  f"check ANCHOR_FISH_KEYS.")

    cluster_ids_sp = sorted(set(cluster_labels_sp))
    n_cols = 4
    n_rows = int(np.ceil(len(cluster_ids_sp) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False,
                              sharex=True, sharey=True)
    axes = axes.flatten()
    rng = np.random.default_rng(0)
    for panel_i, cluster_id in enumerate(cluster_ids_sp):
        ax = axes[panel_i]
        c_idx = np.where(cluster_labels_sp == cluster_id)[0]
        sample_idx = rng.choice(c_idx, size=min(N_EXAMPLE_WAVEFORMS_PER_CLUSTER, len(c_idx)),
                                 replace=False)
        for i in sample_idx:
            ax.plot(sp_waveforms[i], color="gray", alpha=0.15, linewidth=0.5)
        ax.plot(sp_waveforms[c_idx].mean(axis=0), color="crimson", linewidth=1.8)
        mean_width_c = fish_meta.iloc[sp_idx[c_idx]]["mean_width_us"].mean()
        ax.set_title(f"{sp} component {cluster_id}: n={len(c_idx)}, width={mean_width_c:.0f}\u00b5s",
                     fontsize=8)
        ax.set_xticks([])
    for panel_i in range(len(cluster_ids_sp), len(axes)):
        axes[panel_i].axis("off")

    plt.tight_layout()
    sp_fig_path = os.path.join(output_root, f"waveform_clusters_{sp}.png")
    plt.savefig(sp_fig_path, dpi=150)
    plt.close()
    print(f"Saved {sp} cluster shape grid: {os.path.basename(sp_fig_path)}")

species_grouped_assignment = pd.concat(all_species_assignments, ignore_index=True)
species_grouped_assignment.to_csv(
    os.path.join(output_root, "waveform_species_grouped_cluster_assignment.csv"), index=False)
print(f"\nSaved combined per-species cluster assignment to "
      f"waveform_species_grouped_cluster_assignment.csv")

print("\n" + "="*70)
print("DONE - inspect waveform_overview_by_species.png and waveform_clusters_<Species>.png,")
print("especially waveform_clusters_Unknown.png for juvenile-like sub-clusters.")
print("If eps looks off (too many/few clusters, most points -> noise), adjust EPS_PERCENTILE")
print("or MIN_SAMPLES and re-run.")
print("="*70)
