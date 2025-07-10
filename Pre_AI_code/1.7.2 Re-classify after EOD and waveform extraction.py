# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:12:54 2024

@author: shuttlebox
"""

"""Load and cluster pulsefish recordings."""

# import matplotlib.pyplot as plt
# import audioio as aio
# from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import gc
import glob
# import datetime as dt
from sklearn.preprocessing import StandardScaler
# from R4_eodlogger_functions import analyze_r4_snippets, filter_waveforms
import joblib
# import matplotlib as mpl

def extract_and_sort_files(files):
    timecol = [pd.to_datetime(fname.split('-')[-1].split('_')[0], format='%Y%m%dT%H%M%S') for fname in files]
    files_set = pd.DataFrame({'timestamp': timecol, 'filename': files})
    return files_set.sort_values(by=['timestamp'], ignore_index=True)

# Set directories
root = tk.Tk()
root.withdraw()
input_path = filedialog.askdirectory(title="Select Folder with EOD tables ")
output_path = filedialog.askdirectory(title="Select Folder to Store Analysis Results")

# Pick directory with models for species classification
species_model_dir = filedialog.askdirectory(title="Select Folder with Species Models")
gmm_species_list = glob.glob(species_model_dir + '/gmm*.pkl')
lda_transformer_file = glob.glob(species_model_dir + '/*transformer.pkl')
lda_classifier_file = glob.glob(species_model_dir + '/*classifier.pkl')

#%%
print("Use waveforms (1) or features (0)?")
wf_input = int(input())
print("Use GMM (1) or LDA (0)?")
gmm_input = int(input())
#%%
# Load the models and species labels
species_labs = pd.read_csv(species_model_dir + '/species_labels.csv')

if gmm_input:
    gmm_species_models = {i: joblib.load(gmm) for i, gmm in enumerate(gmm_species_list)}
else:
    lda_transformer = joblib.load(lda_transformer_file[0])
    lda_classifier = joblib.load(lda_classifier_file[0])

# Load data
fish_feat_files = glob.glob(input_path + '/*eod_table.csv', recursive=True)
fish_feat_files_sorted = extract_and_sort_files(fish_feat_files)
fish_feat_list = [pd.read_csv(fname) for fname in fish_feat_files_sorted['filename']]
fish_features = pd.concat(fish_feat_list, axis=0, ignore_index=True)
del fish_feat_list

if wf_input:
    fish_wf_files = glob.glob(input_path + '/*waveforms.csv', recursive=True)
    fish_wf_files_sorted = extract_and_sort_files(fish_wf_files)
    fish_wf_list = [pd.read_csv(fname) for fname in fish_wf_files_sorted['filename']]
    fish_wfs = pd.concat(fish_wf_list, axis=0, ignore_index=True)
    del fish_wf_list
    X = fish_wfs
    del fish_wfs
else:
    fish_features_num = fish_features.select_dtypes(include=[np.number])
    # Define important feature variable names
    features = ['pp_dur', 'pp_ratio', 'fft_freq']
    print("Using the following features")
    print(features)
    fish_features_num = fish_features_num[features]
    X = fish_features_num

gc.collect()

#%% Only if features are used
# # Count the occurrences of each species
# species_counts = fish_features['species_pred'].value_counts()
# species_counts


# # Only keep numeric features
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# fish_features_num = fish_features.select_dtypes(include=[np.number])

# # Define important feature variable names
# features = ['pp_dur', 'pp_ratio', 'fft_freq']
# fish_features_num = fish_features_num[features]


#%%
if gmm_input:
    # Classify Species using the species GMMs
    if not wf_input:
        sc = StandardScaler()
        X = sc.fit_transform(X)
    print("Predicting species via GMM")
    species_preds = np.zeros(X.shape[0])
    species_probs = np.zeros((X.shape[0], len(species_labs['0'])))
    for i in range(len(species_labs['0'])):
        species_probs[:, i] = gmm_species_models[i].score_samples(X)
    species_preds = np.argmax(species_probs, axis=1)
    species_preds_labels = [species_labs['0'][pred] for pred in species_preds]
    fish_features['GMM_species_prob'] = np.max(species_probs, axis=1)
else:
    # Predict species using LDA
    print("Predicting species via LDA")
    X = lda_transformer.transform(X)
    species_preds = lda_classifier.predict(X)
    species_preds_labels = [species_labs['0'][pred] for pred in species_preds]

fish_features['species_pred'] = species_preds_labels

#%%
fname = fish_feat_files[0].split('\\')[-1].split('T')[0]
fish_features.to_csv('%s\\%s_eod_table.csv' % (output_path, fname), index=False)
# fish_wfs.to_csv('%s\\%s_h2t_waveforms.csv' % (output_path, fname), index=False)

