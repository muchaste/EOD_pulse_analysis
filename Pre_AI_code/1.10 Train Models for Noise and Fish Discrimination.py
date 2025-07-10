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
noise_examples = filedialog.askdirectory(title = "Select Folder with Noise Features")

fish_wf_files = glob.glob(fish_examples + '/*waveforms.csv', recursive=True)
noise_wf_files = glob.glob(noise_examples + '/*waveforms.csv', recursive=True)

fish_wf_list = [pd.read_csv(fname) for fname in fish_wf_files]
noise_wf_list = [pd.read_csv(fname) for fname in noise_wf_files]

fish_wfs = pd.concat(fish_wf_list, axis=0, ignore_index=True)
noise_wfs = pd.concat(noise_wf_list, axis=0, ignore_index=True)

all_wfs = pd.concat([fish_wfs, noise_wfs], axis=0, ignore_index=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

labs = np.array(['fish'] * len(fish_wfs) + ['noise']*len(noise_wfs))


#%% Discriminant analysis
# Divide the dataset into features and target variable
X = all_wfs
y = labs

# Preprocess the dataset and divide into train and test
sc = StandardScaler()
X = sc.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Plot the scatterplot
plt.figure()
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 0], c=y_train, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('LDA Scatter Plot')
plt.show()

# Classify using Random Forest Classifier
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train_lda, y_train)
y_pred_lda = classifier.predict(X_test_lda)

# Print the accuracy and confusion matrix
print('LDA Accuracy: ' + str(accuracy_score(y_test, y_pred_lda)))
conf_m_lda = confusion_matrix(y_test, y_pred_lda)
print('LDA Confusion Matrix:\n', conf_m_lda)

#%% Mixture Discriminant Analysis
# Fit Gaussian Mixture Models (GMM) to each class
gmm_fish = GaussianMixture(n_components=2, random_state=42)
gmm_fish.fit(X_train[y_train == 0])

gmm_noise = GaussianMixture(n_components=2, random_state=42)
gmm_noise.fit(X_train[y_train == 1])

# Calculate posterior probabilities for classification
fish_probs = gmm_fish.score_samples(X_test)
noise_probs = gmm_noise.score_samples(X_test)

y_pred_mda = (fish_probs < noise_probs).astype(int)

# Print the accuracy and confusion matrix
print('MDA Accuracy: ' + str(accuracy_score(y_test, y_pred_mda)))
conf_m_mda = confusion_matrix(y_test, y_pred_mda)
print('MDA Confusion Matrix:\n', conf_m_mda)

# Compare LDA and MDA
print('\nComparison:')
print('LDA Accuracy: ', accuracy_score(y_test, y_pred_lda))
print('MDA Accuracy: ', accuracy_score(y_test, y_pred_mda))

#%%
# Visualization
# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Scatter plot for MDA results
plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[y_pred_mda == 0, 0], X_test_pca[y_pred_mda == 0, 1], c='red', label='Fish')
plt.scatter(X_test_pca[y_pred_mda == 1, 0], X_test_pca[y_pred_mda == 1, 1], c='blue', label='Noise')
plt.title('MDA Classification Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Scatter plot for LDA results
plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[y_pred_lda == 0, 0], X_test_pca[y_pred_lda == 0, 1], c='red', label='Fish')
plt.scatter(X_test_pca[y_pred_lda == 1, 0], X_test_pca[y_pred_lda == 1, 1], c='blue', label='Noise')
plt.title('LDA Classification Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Decision boundary plot for GMMs
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Stack the meshgrid points
Z = np.c_[xx.ravel(), yy.ravel()]
Z_std = sc.transform(pca.inverse_transform(Z))

fish_probs = gmm_fish.score_samples(Z_std)
noise_probs = gmm_noise.score_samples(Z_std)
Z = (fish_probs > noise_probs).astype(int).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, s=20, edgecolor='k')
plt.title('Decision Boundary of MDA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

#%%
# Save the trained Random Forest classifier
joblib.dump(classifier, fish_examples+'/random_forest_classifier.pkl')

# Save the trained LDA transformer
joblib.dump(lda, fish_examples+'/lda_transformer.pkl')

# Save the trained GMMs
joblib.dump(gmm_fish, fish_examples+'/gmm_fish.pkl')
joblib.dump(gmm_noise, fish_examples+'/gmm_noise.pkl')