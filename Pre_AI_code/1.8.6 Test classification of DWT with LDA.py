import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import glob
import os
import pickle

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with Processed Files")
output_path = filedialog.askdirectory(title="Select Output Folder")

# Load metadata
metadata = pd.read_csv(glob.glob(os.path.join(input_path, 'filtered_metadata.csv'))[0])
species = metadata['species'].values

# Encode species labels
le = LabelEncoder()
species_encoded = le.fit_transform(species)

wavelet_bases = ['haar', 'sym4', 'db4', 'db10']

# Load the top 50 wavelet coefficients indices for each wavelet base (ANOVA and CV-based)
top_cv_indices = {wavelet: np.load(os.path.join(input_path, f'top_cv_indices_{wavelet}.npy'))[:50] for wavelet in wavelet_bases}
top_anova_indices = {wavelet: np.load(os.path.join(input_path, f'top_anova_indices_{wavelet}.npy'))[:50] for wavelet in wavelet_bases}

# Load DWT coefficients
dwt_coefficients = {wavelet: np.load(os.path.join(input_path, f'dwt_coefficients_{wavelet}.npy'), allow_pickle=True) for wavelet in wavelet_bases}

# Convert list of arrays to a 2D array where each row is a flattened coefficient array
dwt_coeffs_array = {}
for wavelet in wavelet_bases:
    max_length = max(len(np.hstack(coeffs)) for coeffs in dwt_coefficients[wavelet])
    dwt_coeffs_array[wavelet] = np.array([np.pad(np.hstack(coeffs), (0, max_length - len(np.hstack(coeffs))), 'constant') for coeffs in dwt_coefficients[wavelet]])

# Create train/test split once
X_train_indices, X_test_indices, y_train, y_test = train_test_split(np.arange(len(species_encoded)), species_encoded, test_size=0.2, stratify=species_encoded, random_state=42)

results = {}

for wavelet in wavelet_bases:
    results[wavelet] = {'cv': {}, 'anova': {}}
    
    for method, top_indices in [('cv', top_cv_indices[wavelet]), ('anova', top_anova_indices[wavelet])]:
        for num_coeffs in range(5, 51):
            indices = top_indices[:num_coeffs]
            X = dwt_coeffs_array[wavelet][:, indices]
            X_train, X_test = X[X_train_indices], X[X_test_indices]
            
            # LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train, y_train)
            y_pred_lda = lda.predict(X_test)
            lda_accuracy = accuracy_score(y_test, y_pred_lda)
            
            # MDA
            gmm_models = {}
            for spec in np.unique(y_train):
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(X_train[y_train == spec])
                gmm_models[spec] = gmm
                
            probs = np.array([gmm_models[spec].score_samples(X_test) for spec in gmm_models]).T
            y_pred_mda = np.argmax(probs, axis=1)
            mda_accuracy = accuracy_score(y_test, y_pred_mda)
            
            results[wavelet][method][num_coeffs] = {'lda': lda_accuracy, 'mda': mda_accuracy}

# Find the best combination
best_combination = {}
for wavelet in wavelet_bases:
    best_combination[wavelet] = {'method': None, 'analysis': None, 'num_coeffs': None, 'accuracy': 0, 'indices': []}
    for method in ['cv', 'anova']:
        for num_coeffs in range(5, 51):
            for analysis in ['lda', 'mda']:
                accuracy = results[wavelet][method][num_coeffs][analysis]
                if (accuracy > best_combination[wavelet]['accuracy']) or (accuracy == best_combination[wavelet]['accuracy'] and (best_combination[wavelet]['num_coeffs'] is None or num_coeffs < best_combination[wavelet]['num_coeffs'])):
                    best_combination[wavelet].update({'method': method, 'analysis': analysis, 'num_coeffs': num_coeffs, 'accuracy': accuracy, 'indices': top_anova_indices[wavelet][:num_coeffs]})

# Print the best combination
for wavelet in wavelet_bases:
    print(f"Best combination for {wavelet}: Method = {best_combination[wavelet]['method']}, Analysis = {best_combination[wavelet]['analysis']}, Number of Coefficients = {best_combination[wavelet]['num_coeffs']}, Accuracy = {best_combination[wavelet]['accuracy']:.4f}")

# Save the best combination as a .csv file
best_combination_df = pd.DataFrame(best_combination).transpose()
best_combination_df.to_csv(os.path.join(output_path, 'best_combination.csv'))

# Save the best LDA and MDA model for each wavelet
for wavelet in wavelet_bases:
    best_indices = best_combination[wavelet]['indices']
    X_best = dwt_coeffs_array[wavelet][:, best_indices]
    X_train_best, X_test_best = X_best[X_train_indices], X_best[X_test_indices]

    if best_combination[wavelet]['analysis'] == 'lda':
        model = LinearDiscriminantAnalysis()
        model.fit(X_train_best, y_train)
        with open(os.path.join(output_path, f'best_lda_model_{wavelet}.pkl'), 'wb') as f:
            pickle.dump(model, f)
    else:
        gmm_models = {}
        for spec in np.unique(y_train):
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X_train_best[y_train == spec])
            gmm_models[spec] = gmm
        with open(os.path.join(output_path, f'best_mda_model_{wavelet}.pkl'), 'wb') as f:
            pickle.dump(gmm_models, f)

# Extract top 50 classification combinations
combinations = []

for wavelet in wavelet_bases:
    for method in ['cv', 'anova']:
        for num_coeffs in range(5, 51):
            for analysis in ['lda', 'mda']:
                accuracy = results[wavelet][method][num_coeffs][analysis]
                combinations.append({
                    'wavelet': wavelet,
                    'method': method,
                    'analysis': analysis,
                    'num_coeffs': num_coeffs,
                    'accuracy': accuracy
                })

# Sort combinations by accuracy and take the top 50
top_50_combinations = sorted(combinations, key=lambda x: x['accuracy'], reverse=True)[:50]

# Convert to DataFrame and save as .csv
top_50_combinations_df = pd.DataFrame(top_50_combinations)
top_50_combinations_df.to_csv(os.path.join(output_path, 'top_50_combinations.csv'), index=False)

# Create a consistent color palette based on species labels
unique_species = np.unique(species)
palette = sns.color_palette("hsv", len(unique_species))
species_to_color = {species: palette[i] for i, species in enumerate(unique_species)}

# Plot classification results for the best combination
for wavelet in wavelet_bases:
    best_indices = best_combination[wavelet]['indices']
    X_best = dwt_coeffs_array[wavelet][:, best_indices]
    X_train_best, X_test_best = X_best[X_train_indices], X_best[X_test_indices]

    # LDA
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_best, species_encoded)

    y_pred_lda = lda.predict(X_best)

    # MDA
    gmm_models = {}
    for spec in np.unique(species_encoded):
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X_best[species_encoded == spec])
        gmm_models[spec] = gmm

    probs = np.array([gmm_models[spec].score_samples(X_best) for spec in gmm_models]).T
    y_pred_mda = np.argmax(probs, axis=1)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_best)

    # Plot LDA and MDA results
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=le.inverse_transform(y_pred_lda), palette=species_to_color, ax=ax[0], alpha=0.6)
    ax[0].set_title(f'LDA Classification Results: {wavelet.capitalize()} Wavelet with {best_combination[wavelet]["num_coeffs"]} Coefficients')

    sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=le.inverse_transform(y_pred_mda), palette=species_to_color, ax=ax[1], alpha=0.6)
    ax[1].set_title(f'MDA Classification Results: {wavelet.capitalize()} Wavelet with {best_combination[wavelet]["num_coeffs"]} Coefficients')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{wavelet}_PCA_classification_results.png'))
    plt.show()


# Plot classification results for the best combination
for wavelet in wavelet_bases:
    best_indices = top_anova_indices[wavelet][:best_combination[wavelet]['num_coeffs']]
    X_best = dwt_coeffs_array[wavelet][:, best_indices]

    # LDA
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_best, species_encoded)
    y_pred_lda = lda.predict(X_best)

    # Plot LDA results
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=le.inverse_transform(species_encoded), palette=species_to_color, ax=ax, alpha=0.6)
    ax.set_title(f'LDA Classification Results: {wavelet.capitalize()} Wavelet with {best_combination[wavelet]["num_coeffs"]} Coefficients')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{wavelet}_LDA_classification_results.png'))
    plt.show()
    
# Plot the results
fig, axes = plt.subplots(4, 2, figsize=(14, 20), sharex=True, sharey=True)
axes = axes.flatten()
for i, wavelet in enumerate(wavelet_bases):
    for j, method in enumerate(['cv', 'anova']):
        ax = axes[i*2 + j]
        for analysis in ['lda', 'mda']:
            x = list(results[wavelet][method].keys())
            y = [results[wavelet][method][num_coeffs][analysis] for num_coeffs in x]
            ax.plot(x, y, label=analysis.upper())
        ax.set_title(f'{wavelet.capitalize()} Wavelet ({method.upper()})')
        ax.set_xlabel('Number of Coefficients')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'classification_performance.png'))
plt.show()

# Save the species labels for the models
species_labels_df = pd.DataFrame({'species': le.classes_})
species_labels_df.to_csv(os.path.join(output_path, 'species_labels.csv'), index=False)

