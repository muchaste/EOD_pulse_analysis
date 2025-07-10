# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:32:07 2024

@author: shuttlebox
"""

"""Load and cluster pulsefish recordings."""

import matplotlib.pyplot as plt
# import audioio as aio
# from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
# import gc
import glob
# import datetime as dt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

#%%
plt.ioff()

# Set directories
root = tk.Tk()
root.withdraw()
fish_examples = filedialog.askdirectory(title = "Select Folder with Fish Features")

# List all .csv files
fish_wf_files = glob.glob(fish_examples + '/*waveforms.csv', recursive=True)
fish_feat_files = glob.glob(fish_examples + '/*eod_table.csv', recursive=True)

# Function to filter files based on exclusion criteria
def exclude_files(file_list, exclusion_patterns):
    filtered_files = []
    for fname in file_list:
        if not any(exclusion_pattern in fname for exclusion_pattern in exclusion_patterns):
            filtered_files.append(fname)
    return filtered_files

# Define exclusion patterns - field recordings with unknown fish that were used for fish/noise discrimination
exclusion_patterns = ['X', 'baby', 'clip', 'asym']

# Exclude files based on patterns
fish_wf_files = exclude_files(fish_wf_files, exclusion_patterns)
fish_feat_files = exclude_files(fish_feat_files, exclusion_patterns)

# Load data
fish_wf_list = [pd.read_csv(fname) for fname in fish_wf_files]
fish_feat_list = [pd.read_csv(fname) for fname in fish_feat_files]

fish_wfs = pd.concat(fish_wf_list, axis=0, ignore_index=True)
fish_features = pd.concat(fish_feat_list, axis=0, ignore_index=True)

# Count the occurrences of each species
species_counts = fish_features['species'].value_counts()
species_counts

# if there are too few cases, eliminate the species (e.g. PN_clip, PD_clip)
threshold = 10
# Identify species that meet the threshold
species_to_keep = species_counts[species_counts >= threshold].index
keep_idc = np.where(fish_features['species'].isin(species_to_keep))[0]

# Filter the DataFrame to keep only the rows with the specified species
fish_wfs = fish_wfs.iloc[keep_idc,:]
fish_features = fish_features.iloc[keep_idc,:]    


#%% Only if Features are used
# Define important feature variable names
features = ['pp_dur', 'pp_ratio', 'fft_freq']

# Plot features
fig, axs = plt.subplots(1,3, figsize=(30, 10))

# Get unique individuals for coloring
individuals = fish_features['fish_id'].unique()
colors = plt.cm.get_cmap('Set1', len(individuals))

# Plot each individual's data
for i, individual in enumerate(individuals):
    individual_data = fish_features[fish_features['fish_id'] == individual]
    axs[0].scatter(individual_data[features[0]], individual_data[features[1]], 
                color=colors(i), label=individual)
    axs[1].scatter(individual_data[features[0]], individual_data[features[2]], 
                color=colors(i), label=individual)
    axs[2].scatter(individual_data[features[1]], individual_data[features[2]], 
                color=colors(i), label=individual)

# Add labels and title
axs[0].set_xlabel(features[0])
axs[0].set_ylabel(features[1])
axs[1].set_xlabel(features[0])
axs[1].set_ylabel(features[2])
axs[2].set_xlabel(features[1])
axs[2].set_ylabel(features[2])
axs[2].legend(title='Individual')
plt.suptitle('Data Before Outlier Removal')
plt.show()

## Function to remove outliers using IQR method
def remove_outliers_iqr(df, variable_columns, thresh_factor):
    def filter_outliers(group):
        Q1 = group[variable_columns].quantile(0.25)
        Q3 = group[variable_columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - thresh_factor * IQR
        upper_bound = Q3 + thresh_factor * IQR
        return group[~((group[variable_columns] < lower_bound) | (group[variable_columns] > upper_bound)).any(axis=1)]
    
    return df.groupby('species').apply(filter_outliers).reset_index(drop=True)

# Remove outliers within each 'species' category
fish_features_cleaned = remove_outliers_iqr(fish_features, features, 10)

# Plot features
fig, axs = plt.subplots(1,3, figsize=(30, 10))

# Get unique individuals for coloring
individuals = fish_features_cleaned['fish_id'].unique()
colors = plt.cm.get_cmap('Set1', len(individuals))

# Plot each individual's data
for i, individual in enumerate(individuals):
    individual_data = fish_features_cleaned[fish_features_cleaned['fish_id'] == individual]
    axs[0].scatter(individual_data[features[0]], individual_data[features[1]], 
                color=colors(i), label=individual)
    axs[1].scatter(individual_data[features[0]], individual_data[features[2]], 
                color=colors(i), label=individual)
    axs[2].scatter(individual_data[features[1]], individual_data[features[2]], 
                color=colors(i), label=individual)

# Add labels and title
axs[0].set_xlabel(features[0])
axs[0].set_ylabel(features[1])
axs[1].set_xlabel(features[0])
axs[1].set_ylabel(features[2])
axs[2].set_xlabel(features[1])
axs[2].set_ylabel(features[2])
axs[2].legend(title='Individual')
# fig.grid(True)
plt.suptitle('Data After Outlier Removal')
plt.show()


# Only keep selected numeric features
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
fish_features_num = fish_features_cleaned.select_dtypes(include=[np.number])
fish_features_num = fish_features_num[features]

labs = np.array(fish_features_cleaned['species'])

#%% PCA for visualization
# decomp = PCA(n_components=10)
# pcas = decomp.fit_transform(fish_wfs)
# print(decomp.explained_variance_ratio_)
# print(np.sum(decomp.explained_variance_ratio_))

# # pcas = pcas[:10000, :]

# print(np.shape(pcas))

#%% Discriminant analysis
# Divide the dataset into features and target variable
# labs = np.array(fish_features_cleaned['species'])
# X = fish_features_num
output_path = filedialog.askdirectory(title = "Select Folder to Store Models")

labs = np.array(fish_features['species'])
X = fish_wfs
y = labs

# Save species labels
pd.DataFrame(np.unique(labs)).to_csv(f'{output_path}/species_labels.csv')
#%%
# Preprocess the dataset and divide into train and test
# sc = StandardScaler()
# X = sc.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%

# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Plot the scatterplot
plt.figure()
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('LDA Scatter Plot')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.show()

# Classify using Random Forest Classifier
classifier = RandomForestClassifier(max_depth=10, random_state=0)
classifier.fit(X_train_lda, y_train)
y_pred_lda = classifier.predict(X_test_lda)

# Print the accuracy and confusion matrix
print('LDA Accuracy: ' + str(accuracy_score(y_test, y_pred_lda)))
conf_m_lda = confusion_matrix(y_test, y_pred_lda)
print('LDA Confusion Matrix:\n', conf_m_lda)

#%% Store LDA
# Save the LDA model and Random Forest classifier
joblib.dump(lda, output_path+'/species_lda_transformer.pkl')
joblib.dump(classifier, output_path+'/species_random_forest_classifier.pkl')


#%% Mixture Discriminant Analysis
# Preprocess the dataset and divide into train and test
# sc = StandardScaler()
# X = sc.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Gaussian Mixture Models (GMM) to each class
gmm_models = {}
for species in np.unique(y_train):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_train[y_train == species])
    gmm_models[species] = gmm

# Calculate posterior probabilities for classification
probs = np.array([gmm_models[species].score_samples(X_test) for species in gmm_models]).T
y_pred_mda = np.argmax(probs, axis=1)

# Print the accuracy and confusion matrix
print('MDA Accuracy: ' + str(accuracy_score(y_test, y_pred_mda)))
conf_m_mda = confusion_matrix(y_test, y_pred_mda)
print('MDA Confusion Matrix:\n', conf_m_mda)

#%%
# Visualization
# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Scatter plot for MDA results
plt.figure(figsize=(10, 6))
for species in np.unique(y):
    plt.scatter(X_test_pca[y_pred_mda == species, 0], X_test_pca[y_pred_mda == species, 1], label=f'{np.unique(labs)[species]}')
plt.title('MDA Classification Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Scatter plot for LDA results
plt.figure(figsize=(10, 6))
for species in np.unique(y):
    plt.scatter(X_test_pca[y_pred_lda == species, 0], X_test_pca[y_pred_lda == species, 1], label=f'{np.unique(labs)[species]}')
plt.title('LDA Classification Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

#%%
# # Decision boundary plot for GMMs
# x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
# y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# # Stack the meshgrid points
# Z = np.c_[xx.ravel(), yy.ravel()]
# Z_std = sc.transform(pca.inverse_transform(Z))

# # Compute the probabilities for each point in the grid
# probs = np.array([gmm_models[species].score_samples(Z_std) for species in gmm_models]).T
# Z = np.argmax(probs, axis=1).reshape(xx.shape)

# plt.figure(figsize=(10, 6))
# plt.contourf(xx, yy, Z, alpha=0.3)
# for species in np.unique(y):
#     plt.scatter(X_test_pca[y_test == species, 0], X_test_pca[y_test == species, 1], label=f'Species {species}', edgecolor='k')
# plt.title('Decision Boundary of MDA')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()
# plt.show()

#%%
# Save the GMM models
for species, gmm in gmm_models.items():
    joblib.dump(gmm, f'{output_path}/gmm_{np.unique(labs)[species]}.pkl')


