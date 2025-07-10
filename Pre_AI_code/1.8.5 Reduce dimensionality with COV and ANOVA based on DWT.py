import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import glob
import os

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Processed Files")

def plot_with_ellipsoids_subplots(data, species, wavelet):
    unique_species = np.unique(species)
    colors = sns.color_palette("hsv", len(unique_species))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    plot_titles = [
        f'Top 3 DWT Coefficients for {wavelet.capitalize()} Wavelet (c1 vs c2)',
        f'Top 3 DWT Coefficients for {wavelet.capitalize()} Wavelet (c1 vs c3)',
        f'Top 3 DWT Coefficients for {wavelet.capitalize()} Wavelet (c2 vs c3)'
    ]
    
    plot_indices = [(0, 1), (0, 2), (1, 2)]
    
    for ax, (title, (x_index, y_index)) in zip(axes, zip(plot_titles, plot_indices)):
        for i, specie in enumerate(unique_species):
            indices = species == specie
            x = data[indices, x_index]
            y = data[indices, y_index]
            ax.scatter(x, y, label=specie, color=colors[i], s=50, alpha=0.6)
            
            # Fit an ellipse to the species cluster
            if len(x) > 1 and len(y) > 1:  # Ensure there are enough points to fit an ellipse
                cov = np.cov(x, y)
                mean = [np.mean(x), np.mean(y)]
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = eigvals.argsort()[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
                width, height = 2 * np.sqrt(eigvals)
                ellip = Ellipse(xy=mean, width=width, height=height, angle=angle, color=colors[i], alpha=0.2)
                ax.add_patch(ellip)
        
        ax.set_title(title)
        ax.set_xlabel(f'Coefficient {x_index + 1}')
        ax.set_ylabel(f'Coefficient {y_index + 1}')
        ax.grid(True)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

# Load metadata
metadata = pd.read_csv(glob.glob(os.path.join(input_path, 'filtered_metadata.csv'))[0])
species = metadata['species'].values

wavelet_bases = ['haar', 'sym4', 'db4', 'db10']
results_cv = {}
results_anova = {}

for wavelet in wavelet_bases:
    # Load DWT coefficients
    dwt_coeffs_list = np.load(os.path.join(input_path, f'dwt_coefficients_{wavelet}.npy'), allow_pickle=True)
    
    # Convert list of arrays to a 2D array where each row is a flattened coefficient array
    max_length = max(len(np.hstack(coeffs)) for coeffs in dwt_coeffs_list)
    dwt_coeffs = np.array([np.pad(np.hstack(coeffs), (0, max_length - len(np.hstack(coeffs))), 'constant') for coeffs in dwt_coeffs_list])
    
    # Coefficient of Variation with transformation
    epsilon = 1  # Small constant to avoid division by zero
    mean_vals = np.mean(np.abs(dwt_coeffs), axis=0) + epsilon
    cv_scores = np.std(dwt_coeffs, axis=0) / mean_vals
    top_cv_indices = np.argsort(cv_scores)[-50:]
    results_cv[wavelet] = top_cv_indices

    # ANOVA
    p_values = [f_oneway(*[dwt_coeffs[species == s, i] for s in np.unique(species)]).pvalue for i in range(dwt_coeffs.shape[1])]
    top_anova_indices = np.argsort(p_values)[:50]
    results_anova[wavelet] = top_anova_indices
    
    # Plot the three most important coefficients based on ANOVA
    top_three_indices_aov = top_anova_indices[:3]
    top_three_coeffs_aov = dwt_coeffs[:, top_three_indices_aov]
    
    plot_with_ellipsoids_subplots(top_three_coeffs_aov, species, wavelet)
    
    # Plot the three most important coefficients based on cv
    top_three_indices_cv = top_cv_indices[:3]
    top_three_coeffs_cv = dwt_coeffs[:, top_three_indices_cv]
    
    plot_with_ellipsoids_subplots(top_three_coeffs_cv, species, wavelet)

# Save results
for wavelet in wavelet_bases:
    np.save(input_path + f'/top_cv_indices_{wavelet}.npy', results_cv[wavelet])
    np.save(input_path + f'/top_anova_indices_{wavelet}.npy', results_anova[wavelet])
