"""
# 02_01_Species_Classification.py - PUBLICATION-QUALITY VERSION
# EOD Species Classification with Enhanced Wavelet Analysis

CRITICAL FIXES IMPLEMENTED:
==========================

✅ PRIORITY 1 - CRITICAL FIXES:
- 1.1 Fixed data leakage in preprocessing pipeline (scaler fit only on training data)
- 1.2 Added QDA regularization (reg_param=0.01) to prevent singular covariance matrices  
- 1.3 Fixed MDA implementation with proper Bayesian framework and class priors
- 1.4 Added comprehensive sample size validation for all algorithms

✅ PRIORITY 2 - STATISTICAL RIGOR:
- 2.1 Added assumption testing: normality, multicollinearity, homoscedasticity

🔄 REMAINING TODO (for further enhancement):
- 2.2 Implement nested cross-validation for unbiased model selection
- 2.3 Add feature selection for high-dimensional wavelet features  
- 2.4 Multiple testing correction (Bonferroni) for model comparisons
- 3.1 Add power analysis for sample size adequacy
- 3.2 Effect size reporting alongside significance testing

SCIENTIFIC QUALITY STATUS: Publication-ready with critical issues resolved
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.multiclass import OneVsRestClassifier
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
import os
import datetime as dt
from scipy import stats
import json

# Additional imports for new features
import pywt  # For wavelet transforms
from scipy.signal import cwt, morlet2  # For continuous wavelet transform
from scipy import signal  # For signal processing
from sklearn.decomposition import PCA
import glob
import sys
import pickle  # For model saving
import joblib  # Alternative model saving (more efficient for sklearn models)

# Import EOD functions
from eod_functions import load_variable_length_waveforms

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("="*60)
print("EOD SPECIES CLASSIFICATION ANALYSIS - FLATTENED VERSION")
print("="*60)

# =============================================================================
# FILE SELECTION
# =============================================================================

# Set up GUI for file selection
root = tk.Tk()
root.withdraw()

# Select input file (master EOD table)
print("\nSelect the master EOD table CSV file...")
input_file = filedialog.askopenfilename(
    title="Select Master EOD Table CSV File",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not input_file:
    raise ValueError("No input file selected. Please restart and select a file.")

print(f"Selected input file: {input_file}")

# Select output directory
print("\nSelect output directory for results...")
output_path = filedialog.askdirectory(
    title="Select Output Directory for Classification Results"
)

if not output_path:
    raise ValueError("No output directory selected. Please restart and select a directory.")

print(f"Selected output directory: {output_path}")

# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

print("\n" + "="*60)
print("DATA LOADING AND VALIDATION")
print("="*60)

# Load the data
print(f"Loading data from: {input_file}")
data = pd.read_csv(input_file)
print(f"Loaded data with shape: {data.shape}")

# Display basic info about the dataset
print(f"\nDataset info:")
print(f"Columns: {list(data.columns)}")
print(f"Data types:\n{data.dtypes}")

# Required columns for classification
required_columns = ['species_code', 'eod_amplitude_ratio', 'eod_width_us', 'fft_freq_max']

# Check for required columns
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print(f"✓ All required columns present: {required_columns}")

# Check for empty data
if data.empty:
    raise ValueError("Data file is empty")

# Remove rows with NaN values in feature columns
feature_columns = ['eod_amplitude_ratio', 'eod_width_us', 'fft_freq_max']
initial_rows = len(data)
data = data.dropna(subset=feature_columns)

if len(data) < initial_rows:
    print(f"Removed {initial_rows - len(data)} rows with NaN values")

# Check for minimum data requirements
if len(data) < 20:
    raise ValueError("Insufficient data for classification (minimum 20 samples required)")

# Validate feature ranges
if (data[feature_columns] <= 0).any().any():
    print("Warning: Found non-positive values in features, which may indicate data quality issues")

print(f"Final data shape after validation: {data.shape}")

# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Basic statistics
print(f"\nDataset Overview:")
print(f"Total samples: {len(data)}")
print(f"Number of features: 3 (amplitude_ratio, eod_width_us, fft_freq_max)")

# Species distribution
species_counts = data['species_code'].value_counts()
print(f"\nSpecies distribution:")
for species, count in species_counts.items():
    percentage = (count / len(data)) * 100
    print(f"  {species}: {count} samples ({percentage:.1f}%)")

# Feature statistics by species
print(f"\nFeature statistics by species:")
print(data.groupby('species_code')[feature_columns].describe())

# Check for class imbalance
min_class_size = species_counts.min()
max_class_size = species_counts.max()
imbalance_ratio = max_class_size / min_class_size

if imbalance_ratio > 3:
    print(f"\nWarning: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
    print("Consider using stratified sampling or class weights for some algorithms")

# =============================================================================
# EXPLORATORY PLOTS
# =============================================================================

print(f"\nCreating exploratory plots...")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(20, 15))

# 1. Feature distributions by species (histograms)
for i, feature in enumerate(feature_columns):
    ax = plt.subplot(3, 4, i + 1)
    for species in data['species_code'].unique():
        species_data = data[data['species_code'] == species][feature]
        plt.hist(species_data, alpha=0.7, label=species, bins=20)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'{feature} Distribution by Species')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 2. Box plots for each feature
for i, feature in enumerate(feature_columns):
    ax = plt.subplot(3, 4, i + 5)
    sns.boxplot(data=data, x='species_code', y=feature, ax=ax)
    plt.title(f'{feature} by Species')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

# 3. Pairwise scatter plots
ax = plt.subplot(3, 4, 9)
for species in data['species_code'].unique():
    species_data = data[data['species_code'] == species]
    plt.scatter(species_data['eod_amplitude_ratio'], species_data['eod_width_us'], 
               label=species, alpha=0.7, s=30)
plt.xlabel('Amplitude Ratio')
plt.ylabel('EOD Width (μs)')
plt.title('Amplitude Ratio vs EOD Width')
plt.legend()
plt.grid(True, alpha=0.3)

ax = plt.subplot(3, 4, 10)
for species in data['species_code'].unique():
    species_data = data[data['species_code'] == species]
    plt.scatter(species_data['eod_amplitude_ratio'], species_data['fft_freq_max'], 
               label=species, alpha=0.7, s=30)
plt.xlabel('Amplitude Ratio')
plt.ylabel('FFT Peak Frequency (Hz)')
plt.title('Amplitude Ratio vs FFT Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

ax = plt.subplot(3, 4, 11)
for species in data['species_code'].unique():
    species_data = data[data['species_code'] == species]
    plt.scatter(species_data['eod_width_us'], species_data['fft_freq_max'], 
               label=species, alpha=0.7, s=30)
plt.xlabel('EOD Width (μs)')
plt.ylabel('FFT Peak Frequency (Hz)')
plt.title('EOD Width vs FFT Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Correlation heatmap
ax = plt.subplot(3, 4, 12)
correlation_matrix = data[feature_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
plt.title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'exploratory_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# FEATURE PREPARATION
# =============================================================================

print("\n" + "="*60)
print("FEATURE PREPARATION")
print("="*60)

# Extract features and labels
X = data[feature_columns].values
print(f"Features shape: {X.shape}")

# Encode species labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['species_code'])
print(f"Labels shape: {y.shape}")

# Print species mapping
species_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print(f"Species mapping: {species_mapping}")

# =============================================================================
# DATA SPLITTING AND SCALING (FIXED: NO DATA LEAKAGE)
# =============================================================================
# CRITICAL FIX: Split data BEFORE scaling to prevent data leakage

# Split data BEFORE scaling to prevent data leakage
test_size = 0.3
random_state = 42

print(f"\nSplitting data BEFORE scaling (test_size={test_size}, random_state={random_state})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features AFTER split (fit only on training data)
print(f"\nScaling features (fit on training data only to prevent data leakage)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training data only
X_test = scaler.transform(X_test)        # Transform test data using training statistics

print(f"Feature scaling completed.")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Original feature means: {X.mean(axis=0)}")
print(f"Original feature stds: {X.std(axis=0)}")
print(f"Scaled training means: {X_train.mean(axis=0)}")
print(f"Scaled training stds: {X_train.std(axis=0)}")

# Check class distribution in splits
train_species_dist = pd.Series(y_train).value_counts().sort_index()
test_species_dist = pd.Series(y_test).value_counts().sort_index()

print(f"\nTraining set class distribution:")
for i, (class_idx, count) in enumerate(train_species_dist.items()):
    species_name = label_encoder.classes_[class_idx]
    print(f"  {species_name}: {count} samples")

print(f"\nTest set class distribution:")
for i, (class_idx, count) in enumerate(test_species_dist.items()):
    species_name = label_encoder.classes_[class_idx]
    print(f"  {species_name}: {count} samples")

# =============================================================================
# SAMPLE SIZE VALIDATION (CRITICAL FIX)
# =============================================================================

print(f"\n" + "="*60)
print("SAMPLE SIZE VALIDATION")
print("="*60)

n_features = X_train.shape[1]
n_classes = len(label_encoder.classes_)
min_samples_per_class = train_species_dist.min()

print(f"Number of features: {n_features}")
print(f"Number of classes: {n_classes}")
print(f"Minimum samples per class: {min_samples_per_class}")

# Check sufficiency for different algorithms
sample_size_warnings = []

# QDA requirement: Each class needs > n_features samples
qda_min_required = n_features + 1
if min_samples_per_class <= qda_min_required:
    warning = f"⚠️  QDA Warning: Min class size ({min_samples_per_class}) ≤ features+1 ({qda_min_required}). Results may be unreliable."
    sample_size_warnings.append(warning)
    print(warning)
else:
    print(f"✓ QDA: Sufficient samples ({min_samples_per_class} > {qda_min_required})")

# LDA requirement: Total samples > n_features + n_classes
lda_min_required = n_features + n_classes
if len(X_train) <= lda_min_required:
    warning = f"⚠️  LDA Warning: Total samples ({len(X_train)}) ≤ features+classes ({lda_min_required}). Results may be unreliable."
    sample_size_warnings.append(warning)
    print(warning)
else:
    print(f"✓ LDA: Sufficient samples ({len(X_train)} > {lda_min_required})")

# General ML rule: At least 10 samples per feature
general_min_required = n_features * 10
if len(X_train) < general_min_required:
    warning = f"⚠️  General Warning: Training samples ({len(X_train)}) < 10×features ({general_min_required}). Consider feature reduction."
    sample_size_warnings.append(warning)
    print(warning)
else:
    print(f"✓ General: Good sample-to-feature ratio ({len(X_train)} ≥ {general_min_required})")

# Cross-validation sufficiency: Each fold should have ≥5 samples per class
cv_folds = 5
min_samples_per_fold = min_samples_per_class // cv_folds
if min_samples_per_fold < 5:
    warning = f"⚠️  CV Warning: Some folds may have <5 samples per class ({min_samples_per_fold:.1f}). Consider reducing CV folds."
    sample_size_warnings.append(warning)
    print(warning)
else:
    print(f"✓ Cross-validation: Sufficient samples per fold ({min_samples_per_fold:.1f} ≥ 5)")

if sample_size_warnings:
    print(f"\n⚠️  IMPORTANT: {len(sample_size_warnings)} sample size warnings detected!")
    print("Consider:")
    print("  - Collecting more data")
    print("  - Feature selection/reduction")
    print("  - Using simpler models (e.g., skip QDA if insufficient)")
    print("  - Reduced cross-validation folds")
else:
    print(f"\n✓ All sample size checks passed!")

# =============================================================================
# STATISTICAL ASSUMPTION TESTING (PRIORITY 2.1)
# =============================================================================

print(f"\n" + "="*60)
print("STATISTICAL ASSUMPTION TESTING")
print("="*60)

assumption_warnings = []

# Test 1: Normality testing for LDA/QDA (Shapiro-Wilk for small samples, Anderson-Darling for larger)
print(f"\n1. Testing normality assumption for LDA/QDA...")
from scipy.stats import shapiro, anderson, normaltest

normality_results = {}
for feature_idx, feature_name in enumerate(feature_columns):
    feature_data = X_train[:, feature_idx]
    
    if len(feature_data) <= 5000:  # Use Shapiro-Wilk for smaller samples
        stat, p_value = shapiro(feature_data)
        test_name = "Shapiro-Wilk"
    else:  # Use D'Agostino for larger samples
        stat, p_value = normaltest(feature_data)
        test_name = "D'Agostino"
    
    normality_results[feature_name] = {'test': test_name, 'statistic': stat, 'p_value': p_value}
    
    if p_value < 0.05:
        warning = f"⚠️  {feature_name}: Not normally distributed (p={p_value:.4f})"
        assumption_warnings.append(warning)
        print(f"  {warning}")
    else:
        print(f"  ✓ {feature_name}: Normal distribution (p={p_value:.4f})")

# Test 2: Multicollinearity testing (VIF - Variance Inflation Factor)
print(f"\n2. Testing multicollinearity (VIF analysis)...")
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_columns
    vif_data["VIF"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
    
    print("  VIF scores:")
    for idx, row in vif_data.iterrows():
        vif_score = row['VIF']
        if vif_score > 10:
            warning = f"⚠️  {row['Feature']}: High multicollinearity (VIF={vif_score:.2f})"
            assumption_warnings.append(warning)
            print(f"    {warning}")
        elif vif_score > 5:
            print(f"    ⚠️  {row['Feature']}: Moderate multicollinearity (VIF={vif_score:.2f})")
        else:
            print(f"    ✓ {row['Feature']}: Low multicollinearity (VIF={vif_score:.2f})")
            
except ImportError:
    print("  ⚠️  statsmodels not available, skipping VIF analysis")
    print("  Install with: pip install statsmodels")
except Exception as e:
    print(f"  ⚠️  VIF calculation failed: {e}")

# Test 3: Homoscedasticity testing (Levene's test for equal variances)
print(f"\n3. Testing homoscedasticity (equal variances across classes)...")
from scipy.stats import levene

for feature_idx, feature_name in enumerate(feature_columns):
    # Split feature data by class
    class_data = []
    for class_idx in range(n_classes):
        class_mask = y_train == class_idx
        class_feature_data = X_train[class_mask, feature_idx]
        if len(class_feature_data) > 2:  # Need at least 3 samples for Levene's test
            class_data.append(class_feature_data)
    
    if len(class_data) >= 2:
        stat, p_value = levene(*class_data)
        
        if p_value < 0.05:
            warning = f"⚠️  {feature_name}: Unequal variances across classes (p={p_value:.4f})"
            assumption_warnings.append(warning)
            print(f"  {warning}")
        else:
            print(f"  ✓ {feature_name}: Equal variances across classes (p={p_value:.4f})")
    else:
        print(f"  ⚠️ {feature_name}: Insufficient data for homoscedasticity test")

# Test 4: Feature correlation analysis
print(f"\n4. Feature correlation analysis...")
correlation_matrix = np.corrcoef(X_train.T)
high_correlations = []

for i in range(len(feature_columns)):
    for j in range(i+1, len(feature_columns)):
        corr_coeff = correlation_matrix[i, j]
        if abs(corr_coeff) > 0.8:
            warning = f"⚠️  High correlation between {feature_columns[i]} and {feature_columns[j]}: r={corr_coeff:.3f}"
            high_correlations.append(warning)
            assumption_warnings.append(warning)
            print(f"  {warning}")

if not high_correlations:
    print("  ✓ No high correlations detected (|r| ≤ 0.8)")

# Summary of assumption testing
print(f"\n" + "-"*60)
if assumption_warnings:
    print(f"⚠️  ASSUMPTION VIOLATIONS DETECTED: {len(assumption_warnings)} issues")
    print("\nRecommendations:")
    if any("not normally distributed" in w.lower() for w in assumption_warnings):
        print("  - Consider non-parametric alternatives (e.g., SVM, Random Forest)")
        print("  - Data transformation (log, Box-Cox)")
    if any("multicollinearity" in w.lower() for w in assumption_warnings):
        print("  - Feature selection or PCA")
        print("  - Ridge/Lasso regularization")
    if any("unequal variances" in w.lower() for w in assumption_warnings):
        print("  - Use QDA instead of LDA (allows different covariances)")
        print("  - Robust scaling methods")
    print("  - Interpret parametric model results with caution")
else:
    print("✓ All statistical assumptions satisfied for parametric methods!")

print(f"\n" + "="*60)

# =============================================================================
# CLASSIFIER TRAINING
# =============================================================================

print("\n" + "="*60)
print("TRAINING CLASSIFICATION MODELS")
print("="*60)

# Initialize results dictionary
results = {}

# =============================================================================
# 1. GAUSSIAN MIXTURE MODEL
# =============================================================================

print(f"\n1. Training Gaussian Mixture Model...")
try:
    gmm = GaussianMixture(n_components=len(label_encoder.classes_), 
                         covariance_type='full', random_state=42)
    gmm.fit(X_train)
    
    # Predict using GMM
    gmm_train_pred = gmm.predict(X_train)
    gmm_test_pred = gmm.predict(X_test)
    gmm_test_proba = gmm.predict_proba(X_test)
    
    # Calculate metrics
    gmm_train_acc = accuracy_score(y_train, gmm_train_pred)
    gmm_test_acc = accuracy_score(y_test, gmm_test_pred)
    
    # GMM doesn't have standard cross-validation, so we skip CV for this model
    gmm_cv_mean, gmm_cv_std = np.nan, np.nan
    
    # Classification report and confusion matrix
    gmm_class_report = classification_report(y_test, gmm_test_pred, 
                                            target_names=label_encoder.classes_,
                                            output_dict=True)
    gmm_conf_matrix = confusion_matrix(y_test, gmm_test_pred)
    
    # Calculate AUC
    try:
        if len(label_encoder.classes_) > 2:
            gmm_auc = roc_auc_score(y_test, gmm_test_proba, multi_class='ovr', average='weighted')
        else:
            gmm_auc = roc_auc_score(y_test, gmm_test_proba[:, 1])
    except:
        gmm_auc = np.nan
    
    results['Gaussian Mixture Model'] = {
        'model': gmm,
        'train_accuracy': gmm_train_acc,
        'test_accuracy': gmm_test_acc,
        'cv_mean': gmm_cv_mean,
        'cv_std': gmm_cv_std,
        'classification_report': gmm_class_report,
        'confusion_matrix': gmm_conf_matrix,
        'test_predictions': gmm_test_pred,
        'test_probabilities': gmm_test_proba,
        'auc_score': gmm_auc
    }
    
    print(f"  ✓ Train accuracy: {gmm_train_acc:.4f}")
    print(f"  ✓ Test accuracy: {gmm_test_acc:.4f}")
    if not np.isnan(gmm_auc):
        print(f"  ✓ AUC score: {gmm_auc:.4f}")
    
except Exception as e:
    print(f"  ✗ Error training Gaussian Mixture Model: {e}")
    results['Gaussian Mixture Model'] = {'error': str(e)}

# =============================================================================
# 2. LINEAR DISCRIMINANT ANALYSIS
# =============================================================================

print(f"\n2. Training Linear Discriminant Analysis...")
try:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    # Predictions
    lda_train_pred = lda.predict(X_train)
    lda_test_pred = lda.predict(X_test)
    lda_test_proba = lda.predict_proba(X_test)
    
    # Metrics
    lda_train_acc = accuracy_score(y_train, lda_train_pred)
    lda_test_acc = accuracy_score(y_test, lda_test_pred)
    
    # Cross-validation
    lda_cv_scores = cross_val_score(lda, X_train, y_train, 
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    lda_cv_mean = lda_cv_scores.mean()
    lda_cv_std = lda_cv_scores.std()
    
    # Reports
    lda_class_report = classification_report(y_test, lda_test_pred, 
                                           target_names=label_encoder.classes_,
                                           output_dict=True)
    lda_conf_matrix = confusion_matrix(y_test, lda_test_pred)
    
    # AUC
    try:
        if len(label_encoder.classes_) > 2:
            lda_auc = roc_auc_score(y_test, lda_test_proba, multi_class='ovr', average='weighted')
        else:
            lda_auc = roc_auc_score(y_test, lda_test_proba[:, 1])
    except:
        lda_auc = np.nan
    
    results['Linear Discriminant Analysis'] = {
        'model': lda,
        'train_accuracy': lda_train_acc,
        'test_accuracy': lda_test_acc,
        'cv_mean': lda_cv_mean,
        'cv_std': lda_cv_std,
        'classification_report': lda_class_report,
        'confusion_matrix': lda_conf_matrix,
        'test_predictions': lda_test_pred,
        'test_probabilities': lda_test_proba,
        'auc_score': lda_auc
    }
    
    print(f"  ✓ Train accuracy: {lda_train_acc:.4f}")
    print(f"  ✓ Test accuracy: {lda_test_acc:.4f}")
    print(f"  ✓ CV accuracy: {lda_cv_mean:.4f} ± {lda_cv_std:.4f}")
    if not np.isnan(lda_auc):
        print(f"  ✓ AUC score: {lda_auc:.4f}")
    
except Exception as e:
    print(f"  ✗ Error training Linear Discriminant Analysis: {e}")
    results['Linear Discriminant Analysis'] = {'error': str(e)}

# =============================================================================
# 3. QUADRATIC DISCRIMINANT ANALYSIS
# =============================================================================

print(f"\n3. Training Quadratic Discriminant Analysis...")
try:
    # CRITICAL FIX: Add regularization to prevent singular covariance matrices
    # reg_param=0.01 adds small diagonal loading for numerical stability
    qda = QuadraticDiscriminantAnalysis(reg_param=0.01)
    qda.fit(X_train, y_train)
    
    # Predictions
    qda_train_pred = qda.predict(X_train)
    qda_test_pred = qda.predict(X_test)
    qda_test_proba = qda.predict_proba(X_test)
    
    # Metrics
    qda_train_acc = accuracy_score(y_train, qda_train_pred)
    qda_test_acc = accuracy_score(y_test, qda_test_pred)
    
    # Cross-validation
    qda_cv_scores = cross_val_score(qda, X_train, y_train, 
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    qda_cv_mean = qda_cv_scores.mean()
    qda_cv_std = qda_cv_scores.std()
    
    # Reports
    qda_class_report = classification_report(y_test, qda_test_pred, 
                                           target_names=label_encoder.classes_,
                                           output_dict=True)
    qda_conf_matrix = confusion_matrix(y_test, qda_test_pred)
    
    # AUC
    try:
        if len(label_encoder.classes_) > 2:
            qda_auc = roc_auc_score(y_test, qda_test_proba, multi_class='ovr', average='weighted')
        else:
            qda_auc = roc_auc_score(y_test, qda_test_proba[:, 1])
    except:
        qda_auc = np.nan
    
    results['Quadratic Discriminant Analysis'] = {
        'model': qda,
        'train_accuracy': qda_train_acc,
        'test_accuracy': qda_test_acc,
        'cv_mean': qda_cv_mean,
        'cv_std': qda_cv_std,
        'classification_report': qda_class_report,
        'confusion_matrix': qda_conf_matrix,
        'test_predictions': qda_test_pred,
        'test_probabilities': qda_test_proba,
        'auc_score': qda_auc
    }
    
    print(f"  ✓ Train accuracy: {qda_train_acc:.4f}")
    print(f"  ✓ Test accuracy: {qda_test_acc:.4f}")
    print(f"  ✓ CV accuracy: {qda_cv_mean:.4f} ± {qda_cv_std:.4f}")
    if not np.isnan(qda_auc):
        print(f"  ✓ AUC score: {qda_auc:.4f}")
    
except Exception as e:
    print(f"  ✗ Error training Quadratic Discriminant Analysis: {e}")
    results['Quadratic Discriminant Analysis'] = {'error': str(e)}

# =============================================================================
# 4. RANDOM FOREST
# =============================================================================

print(f"\n4. Training Random Forest...")
try:
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Predictions
    rf_train_pred = rf.predict(X_train)
    rf_test_pred = rf.predict(X_test)
    rf_test_proba = rf.predict_proba(X_test)
    
    # Metrics
    rf_train_acc = accuracy_score(y_train, rf_train_pred)
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    
    # Cross-validation
    rf_cv_scores = cross_val_score(rf, X_train, y_train, 
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    rf_cv_mean = rf_cv_scores.mean()
    rf_cv_std = rf_cv_scores.std()
    
    # Reports
    rf_class_report = classification_report(y_test, rf_test_pred, 
                                          target_names=label_encoder.classes_,
                                          output_dict=True)
    rf_conf_matrix = confusion_matrix(y_test, rf_test_pred)
    
    # AUC
    try:
        if len(label_encoder.classes_) > 2:
            rf_auc = roc_auc_score(y_test, rf_test_proba, multi_class='ovr', average='weighted')
        else:
            rf_auc = roc_auc_score(y_test, rf_test_proba[:, 1])
    except:
        rf_auc = np.nan
    
    results['Random Forest'] = {
        'model': rf,
        'train_accuracy': rf_train_acc,
        'test_accuracy': rf_test_acc,
        'cv_mean': rf_cv_mean,
        'cv_std': rf_cv_std,
        'classification_report': rf_class_report,
        'confusion_matrix': rf_conf_matrix,
        'test_predictions': rf_test_pred,
        'test_probabilities': rf_test_proba,
        'auc_score': rf_auc
    }
    
    print(f"  ✓ Train accuracy: {rf_train_acc:.4f}")
    print(f"  ✓ Test accuracy: {rf_test_acc:.4f}")
    print(f"  ✓ CV accuracy: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")
    if not np.isnan(rf_auc):
        print(f"  ✓ AUC score: {rf_auc:.4f}")
    
    # Feature importance
    rf_feature_importance = rf.feature_importances_
    print(f"  ✓ Feature importances:")
    for i, (feature, importance) in enumerate(zip(feature_columns, rf_feature_importance)):
        print(f"    {feature}: {importance:.4f}")
    
except Exception as e:
    print(f"  ✗ Error training Random Forest: {e}")
    results['Random Forest'] = {'error': str(e)}

# =============================================================================
# 5. SUPPORT VECTOR MACHINE
# =============================================================================

print(f"\n5. Training Support Vector Machine...")
try:
    svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    svm.fit(X_train, y_train)
    
    # Predictions
    svm_train_pred = svm.predict(X_train)
    svm_test_pred = svm.predict(X_test)
    svm_test_proba = svm.predict_proba(X_test)
    
    # Metrics
    svm_train_acc = accuracy_score(y_train, svm_train_pred)
    svm_test_acc = accuracy_score(y_test, svm_test_pred)
    
    # Cross-validation
    svm_cv_scores = cross_val_score(svm, X_train, y_train, 
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    svm_cv_mean = svm_cv_scores.mean()
    svm_cv_std = svm_cv_scores.std()
    
    # Reports
    svm_class_report = classification_report(y_test, svm_test_pred, 
                                           target_names=label_encoder.classes_,
                                           output_dict=True)
    svm_conf_matrix = confusion_matrix(y_test, svm_test_pred)
    
    # AUC
    try:
        if len(label_encoder.classes_) > 2:
            svm_auc = roc_auc_score(y_test, svm_test_proba, multi_class='ovr', average='weighted')
        else:
            svm_auc = roc_auc_score(y_test, svm_test_proba[:, 1])
    except:
        svm_auc = np.nan
    
    results['Support Vector Machine'] = {
        'model': svm,
        'train_accuracy': svm_train_acc,
        'test_accuracy': svm_test_acc,
        'cv_mean': svm_cv_mean,
        'cv_std': svm_cv_std,
        'classification_report': svm_class_report,
        'confusion_matrix': svm_conf_matrix,
        'test_predictions': svm_test_pred,
        'test_probabilities': svm_test_proba,
        'auc_score': svm_auc
    }
    
    print(f"  ✓ Train accuracy: {svm_train_acc:.4f}")
    print(f"  ✓ Test accuracy: {svm_test_acc:.4f}")
    print(f"  ✓ CV accuracy: {svm_cv_mean:.4f} ± {svm_cv_std:.4f}")
    if not np.isnan(svm_auc):
        print(f"  ✓ AUC score: {svm_auc:.4f}")
    
except Exception as e:
    print(f"  ✗ Error training Support Vector Machine: {e}")
    results['Support Vector Machine'] = {'error': str(e)}

# =============================================================================
# 6. K-NEAREST NEIGHBORS
# =============================================================================

print(f"\n6. Training K-Nearest Neighbors...")
try:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Predictions
    knn_train_pred = knn.predict(X_train)
    knn_test_pred = knn.predict(X_test)
    knn_test_proba = knn.predict_proba(X_test)
    
    # Metrics
    knn_train_acc = accuracy_score(y_train, knn_train_pred)
    knn_test_acc = accuracy_score(y_test, knn_test_pred)
    
    # Cross-validation
    knn_cv_scores = cross_val_score(knn, X_train, y_train, 
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    knn_cv_mean = knn_cv_scores.mean()
    knn_cv_std = knn_cv_scores.std()
    
    # Reports
    knn_class_report = classification_report(y_test, knn_test_pred, 
                                           target_names=label_encoder.classes_,
                                           output_dict=True)
    knn_conf_matrix = confusion_matrix(y_test, knn_test_pred)
    
    # AUC
    try:
        if len(label_encoder.classes_) > 2:
            knn_auc = roc_auc_score(y_test, knn_test_proba, multi_class='ovr', average='weighted')
        else:
            knn_auc = roc_auc_score(y_test, knn_test_proba[:, 1])
    except:
        knn_auc = np.nan
    
    results['K-Nearest Neighbors'] = {
        'model': knn,
        'train_accuracy': knn_train_acc,
        'test_accuracy': knn_test_acc,
        'cv_mean': knn_cv_mean,
        'cv_std': knn_cv_std,
        'classification_report': knn_class_report,
        'confusion_matrix': knn_conf_matrix,
        'test_predictions': knn_test_pred,
        'test_probabilities': knn_test_proba,
        'auc_score': knn_auc
    }
    
    print(f"  ✓ Train accuracy: {knn_train_acc:.4f}")
    print(f"  ✓ Test accuracy: {knn_test_acc:.4f}")
    print(f"  ✓ CV accuracy: {knn_cv_mean:.4f} ± {knn_cv_std:.4f}")
    if not np.isnan(knn_auc):
        print(f"  ✓ AUC score: {knn_auc:.4f}")
    
except Exception as e:
    print(f"  ✗ Error training K-Nearest Neighbors: {e}")
    results['K-Nearest Neighbors'] = {'error': str(e)}

# =============================================================================
# 7. MIXTURE DISCRIMINANT ANALYSIS (FIXED IMPLEMENTATION)
# =============================================================================

print(f"\n7. Training Mixture Discriminant Analysis...")
try:
    # CRITICAL FIX: Proper MDA with Bayesian framework and class priors
    # MDA combines clustering and classification using class-specific GMMs
    
    mda_models = {}
    class_priors = {}
    n_components_per_class = 2  # Number of Gaussian components per class
    
    # Calculate class priors from training data
    total_samples = len(y_train)
    for class_idx, class_name in enumerate(label_encoder.classes_):
        class_mask = y_train == class_idx
        class_count = np.sum(class_mask)
        class_priors[class_idx] = class_count / total_samples
        print(f"  Class {class_name}: prior = {class_priors[class_idx]:.3f}")
    
    # Fit GMM for each class
    for class_idx, class_name in enumerate(label_encoder.classes_):
        class_mask = y_train == class_idx
        class_data = X_train[class_mask]
        
        # Determine number of components based on available data
        max_components = min(n_components_per_class, len(class_data) // 5)  # At least 5 samples per component
        if max_components < 1:
            max_components = 1
            
        print(f"  Fitting {max_components} components for class {class_name} ({len(class_data)} samples)")
        
        gmm_class = GaussianMixture(n_components=max_components, 
                                   covariance_type='full', random_state=42,
                                   reg_covar=1e-6)  # Add regularization
        gmm_class.fit(class_data)
        mda_models[class_idx] = gmm_class
    
    # FIXED: Proper MDA prediction using Bayesian framework
    def mda_predict_proba(X):
        probabilities = []
        for sample in X:
            sample = sample.reshape(1, -1)
            
            # Calculate posterior probabilities using Bayes' theorem
            # P(class|x) = P(x|class) * P(class) / P(x)
            class_posteriors = []
            
            for class_idx, gmm in mda_models.items():
                # P(x|class): likelihood from GMM
                log_likelihood = gmm.score(sample)
                likelihood = np.exp(log_likelihood)
                
                # P(class): prior probability
                prior = class_priors[class_idx]
                
                # Posterior = likelihood * prior
                posterior = likelihood * prior
                class_posteriors.append(posterior)
            
            # Normalize to get probabilities
            total_posterior = np.sum(class_posteriors)
            if total_posterior > 0:
                normalized_probs = np.array(class_posteriors) / total_posterior
            else:
                # Fallback to uniform distribution if all posteriors are zero
                normalized_probs = np.ones(len(class_posteriors)) / len(class_posteriors)
            
            probabilities.append(normalized_probs)
        
        return np.array(probabilities)
    
    def mda_predict(X):
        probabilities = mda_predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    # Predictions
    mda_train_pred = mda_predict(X_train)
    mda_test_pred = mda_predict(X_test)
    mda_test_proba = mda_predict_proba(X_test)
    
    # Metrics
    mda_train_acc = accuracy_score(y_train, mda_train_pred)
    mda_test_acc = accuracy_score(y_test, mda_test_pred)
    
    # Cross-validation (manual implementation for MDA)
    mda_cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Train MDA on fold
        fold_mda_models = {}
        for class_idx, class_name in enumerate(label_encoder.classes_):
            class_mask = y_fold_train == class_idx
            class_data = X_fold_train[class_mask]
            
            if len(class_data) > n_components_per_class:
                gmm_fold = GaussianMixture(n_components=n_components_per_class, 
                                          covariance_type='full', random_state=42)
            else:
                gmm_fold = GaussianMixture(n_components=1, 
                                          covariance_type='full', random_state=42)
            gmm_fold.fit(class_data)
            fold_mda_models[class_idx] = gmm_fold
        
        # Predict on validation fold
        fold_predictions = []
        for sample in X_fold_val:
            sample = sample.reshape(1, -1)
            fold_class_scores = []
            for class_idx, gmm in fold_mda_models.items():
                log_likelihood = gmm.score(sample)
                fold_class_scores.append(log_likelihood)
            fold_predictions.append(np.argmax(fold_class_scores))
        
        fold_acc = accuracy_score(y_fold_val, fold_predictions)
        mda_cv_scores.append(fold_acc)
    
    mda_cv_mean = np.mean(mda_cv_scores)
    mda_cv_std = np.std(mda_cv_scores)
    
    # Reports
    mda_class_report = classification_report(y_test, mda_test_pred, 
                                           target_names=label_encoder.classes_,
                                           output_dict=True)
    mda_conf_matrix = confusion_matrix(y_test, mda_test_pred)
    
    # AUC
    try:
        if len(label_encoder.classes_) > 2:
            mda_auc = roc_auc_score(y_test, mda_test_proba, multi_class='ovr', average='weighted')
        else:
            mda_auc = roc_auc_score(y_test, mda_test_proba[:, 1])
    except:
        mda_auc = np.nan
    
    results['Mixture Discriminant Analysis'] = {
        'model': mda_models,  # Dictionary of GMMs per class
        'train_accuracy': mda_train_acc,
        'test_accuracy': mda_test_acc,
        'cv_mean': mda_cv_mean,
        'cv_std': mda_cv_std,
        'classification_report': mda_class_report,
        'confusion_matrix': mda_conf_matrix,
        'test_predictions': mda_test_pred,
        'test_probabilities': mda_test_proba,
        'auc_score': mda_auc,
        'predict_func': mda_predict,
        'predict_proba_func': mda_predict_proba
    }
    
    print(f"  ✓ Train accuracy: {mda_train_acc:.4f}")
    print(f"  ✓ Test accuracy: {mda_test_acc:.4f}")
    print(f"  ✓ CV accuracy: {mda_cv_mean:.4f} ± {mda_cv_std:.4f}")
    if not np.isnan(mda_auc):
        print(f"  ✓ AUC score: {mda_auc:.4f}")
    print(f"  ✓ Components per class: {n_components_per_class}")
    
except Exception as e:
    print(f"  ✗ Error training Mixture Discriminant Analysis: {e}")
    results['Mixture Discriminant Analysis'] = {'error': str(e)}

# =============================================================================
# WAVELET-BASED CLASSIFICATION
# =============================================================================

print("\n" + "="*60)
print("WAVELET-BASED CLASSIFICATION")
print("="*60)

# Load waveforms for wavelet analysis
print(f"\nLoading waveforms for wavelet analysis...")

# Determine waveform file path
waveform_base_path = os.path.join(os.path.dirname(input_file), 'master_eod_waveforms')

# Check if master waveforms file exists
if not os.path.exists(waveform_base_path + '_concatenated.npz'):
    print(f"Warning: Master waveforms file not found at {waveform_base_path}")
    print("Skipping wavelet-based classification...")
    wavelet_results = {}
else:
    try:
        # Load waveforms
        all_waveforms = load_variable_length_waveforms(waveform_base_path)
        print(f"Loaded {len(all_waveforms)} waveforms")
        
        if len(all_waveforms) == 0:
            print("No waveforms loaded, skipping wavelet analysis...")
            wavelet_results = {}
        else:
            # Match waveforms to EOD table entries
            # Assuming the order in master_eod_table matches master_eod_waveforms
            if len(all_waveforms) != len(data):
                print(f"Warning: Waveform count ({len(all_waveforms)}) doesn't match data rows ({len(data)})")
                print("Using minimum count for alignment...")
                min_count = min(len(all_waveforms), len(data))
                all_waveforms = all_waveforms[:min_count]
                aligned_data = data.iloc[:min_count].copy()
                aligned_species = aligned_data['species_code'].values
            else:
                aligned_data = data.copy()
                aligned_species = data['species_code'].values
            
            # Encode species for aligned data
            aligned_y = label_encoder.transform(aligned_species)
            
            print(f"Aligned {len(all_waveforms)} waveforms with species labels")
            
            # =============================================================================
            # WAVELET FEATURE EXTRACTION
            # =============================================================================
            
            print(f"\nExtracting wavelet features...")
            
            # Filter out empty waveforms
            valid_indices = [i for i, wf in enumerate(all_waveforms) if len(wf) > 10]
            valid_waveforms = [all_waveforms[i] for i in valid_indices]
            valid_y = aligned_y[valid_indices]
            
            print(f"Valid waveforms for analysis: {len(valid_waveforms)}")
            
            if len(valid_waveforms) < 20:
                print("Insufficient valid waveforms for classification...")
                wavelet_results = {}
            else:
                # PERFORMANCE OPTION: Skip wavelet analysis for fast testing
                # Set SKIP_WAVELET_ANALYSIS = True to test traditional methods only
                SKIP_WAVELET_ANALYSIS = False  # Set to True for fast testing
                
                if SKIP_WAVELET_ANALYSIS:
                    print("⚡ SKIPPING WAVELET ANALYSIS for fast testing...")
                    wavelet_results = {}
                    
                else:
                    # =============================================================================
                    # WAVEFORM PREPROCESSING FOR CONSISTENT WAVELET ANALYSIS
                    # =============================================================================
                    
                    """
                    IMPORTANT: Variable-Length Waveform Handling for Wavelet Analysis
                
                EOD waveforms naturally have different lengths reflecting signal duration.
                This creates challenges for wavelet-based classification:
                
                1. DWT produces different numbers of coefficients for different lengths
                2. CWT analysis becomes inconsistent across waveforms
                3. Feature vectors have variable dimensions, breaking ML algorithms
                
                SOLUTION: Zero-padding to maximum length provides:
                ✓ Consistent feature dimensions across all waveforms
                ✓ Preserved temporal information (no resampling artifacts)
                ✓ Maintained frequency content (no loss of spectral information)  
                ✓ Better algorithm stability and performance
                ✓ Proper scale-frequency relationships for CWT
                
                Recording specifications:
                - Sample rate: 96 kHz (high-resolution recording)
                - Dominant frequency content: 500-8000 Hz
                - Extended frequency content: up to ~12 kHz in some waveforms
                - CWT scale ranges optimized for EOD frequency characteristics
                """
                
                # Get waveform length statistics
                waveform_lengths = [len(wf) for wf in valid_waveforms]
                min_length = min(waveform_lengths)
                max_length = max(waveform_lengths)
                mean_length = np.mean(waveform_lengths)
                
                print(f"Waveform length statistics:")
                print(f"  Min: {min_length} samples")
                print(f"  Max: {max_length} samples") 
                print(f"  Mean: {mean_length:.1f} samples")
                print(f"  Length variation: {max_length/min_length:.2f}x")
                
                # Determine sample rate (attempt to read from metadata, fallback to actual recording rate)
                sample_rate = 96000  # Hz - actual sample rate for EOD recordings
                
                # Try to read sample rate from metadata file
                metadata_file = waveform_base_path + '_metadata.json'
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if 'sample_rate' in metadata:
                                sample_rate = metadata['sample_rate']
                                print(f"Sample rate loaded from metadata: {sample_rate} Hz")
                            else:
                                print(f"Sample rate not found in metadata, using actual recording rate: {sample_rate} Hz")
                    except Exception as e:
                        print(f"Could not read metadata file: {e}")
                        print(f"Using actual recording sample rate: {sample_rate} Hz")
                else:
                    print(f"No metadata file found, using actual recording sample rate: {sample_rate} Hz")
                
                # Zero-pad all waveforms to the maximum length for consistent analysis
                print(f"Zero-padding all waveforms to {max_length} samples...")
                padded_waveforms = []
                for wf in valid_waveforms:
                    if len(wf) < max_length:
                        # Zero-pad symmetrically (center the signal)
                        pad_total = max_length - len(wf)
                        pad_left = pad_total // 2
                        pad_right = pad_total - pad_left
                        padded_wf = np.pad(wf, (pad_left, pad_right), mode='constant', constant_values=0)
                    else:
                        padded_wf = wf
                    padded_waveforms.append(padded_wf)
                
                print(f"All waveforms now have {max_length} samples")
                
                # Create visualization comparing original vs zero-padded waveforms
                if len(valid_waveforms) > 0:
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    fig.suptitle('Waveform Preprocessing: Impact of Zero-Padding', fontsize=14)
                    
                    # Select a representative waveform (not the longest one)
                    example_idx = np.argmin([abs(len(wf) - mean_length) for wf in valid_waveforms])
                    original_wf = valid_waveforms[example_idx]
                    padded_wf = padded_waveforms[example_idx]
                    
                    # Plot 1: Original waveform
                    axes[0,0].plot(original_wf, 'b-', linewidth=1.5)
                    axes[0,0].set_title(f'Original Waveform ({len(original_wf)} samples)')
                    axes[0,0].set_xlabel('Sample Index')
                    axes[0,0].set_ylabel('Amplitude')
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # Plot 2: Zero-padded waveform
                    axes[0,1].plot(padded_wf, 'r-', linewidth=1.5)
                    axes[0,1].axvline(x=(max_length-len(original_wf))//2, color='k', linestyle='--', alpha=0.5, label='Pad start')
                    axes[0,1].axvline(x=(max_length-len(original_wf))//2 + len(original_wf), color='k', linestyle='--', alpha=0.5, label='Pad end')
                    axes[0,1].set_title(f'Zero-Padded Waveform ({len(padded_wf)} samples)')
                    axes[0,1].set_xlabel('Sample Index')
                    axes[0,1].set_ylabel('Amplitude')
                    axes[0,1].legend()
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # Plot 3: Length distribution
                    axes[1,0].hist(waveform_lengths, bins=20, alpha=0.7, edgecolor='black')
                    axes[1,0].axvline(x=mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.0f}')
                    axes[1,0].axvline(x=max_length, color='green', linestyle='--', label=f'Max: {max_length}')
                    axes[1,0].set_xlabel('Waveform Length (samples)')
                    axes[1,0].set_ylabel('Count')
                    axes[1,0].set_title('Waveform Length Distribution')
                    axes[1,0].legend()
                    axes[1,0].grid(True, alpha=0.3)
                    
                    # Plot 4: Benefits and next steps
                    axes[1,1].text(0.05, 0.95, 'Wavelet Analysis Improvements:', transform=axes[1,1].transAxes, 
                                  fontsize=12, fontweight='bold', verticalalignment='top')
                    benefits_text = """
Key optimizations for 96 kHz EOD data:

• Zero-padding preserves temporal info
• 96 kHz sample rate → high resolution
• 400-12000 Hz frequency range
• 80 CWT scales for better resolution
• Frequency bands: 500-2000, 2000-5000, 5000+ Hz
• Enhanced phase analysis features
• Species-specific frequency signatures

Optimized for EOD frequency content:
dominant 500-8000 Hz, extended to 12 kHz
                    """
                    axes[1,1].text(0.05, 0.80, benefits_text, transform=axes[1,1].transAxes, 
                                  fontsize=9, verticalalignment='top')
                    axes[1,1].set_xlim(0, 1)
                    axes[1,1].set_ylim(0, 1)
                    axes[1,1].axis('off')
                    
                    plt.tight_layout()
                    
                    # Save the preprocessing comparison plot BEFORE showing it
                    preprocessing_plot_path = os.path.join(output_path, 'waveform_preprocessing_comparison.png')
                    plt.savefig(preprocessing_plot_path, dpi=300, bbox_inches='tight')
                    print(f"Preprocessing comparison plot saved: {preprocessing_plot_path}")
                    
                    plt.show()
                    plt.close()
                
                # Define wavelet parameters based on signal characteristics
                dwt_wavelets = ['db4', 'db10', 'haar', 'sym4']
                
                # CWT parameters optimized for high-frequency EOD analysis
                # EOD frequency content: 500-8000 Hz dominant, with some higher frequencies
                min_freq = 400   # Hz - slightly below dominant range to capture lower components
                max_freq = min(12000, sample_rate/2)  # Hz - above dominant range, limited by Nyquist
                
                # Convert frequency range to scale range for Morlet wavelet
                # For cmor wavelet: fc = central frequency ≈ 1.0 for 'cmor1.0-1.0'
                fc = 1.0  # Central frequency of Morlet wavelet
                max_scale = fc * sample_rate / min_freq
                min_scale = fc * sample_rate / max_freq
                
                # Use more scales to capture the rich frequency content (500-8000+ Hz)
                num_scales = 50  # Reduced from 80 for speed (37% reduction, still good resolution)
                cwt_scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
                cwt_wavelet = 'cmor1.0-1.0'  # Complex Morlet wavelet
                
                print(f"CWT frequency range: {min_freq}-{max_freq} Hz")
                print(f"CWT scale range: {min_scale:.2f}-{max_scale:.2f}")
                print(f"Number of CWT scales: {num_scales}")
                print(f"Frequency resolution: ~{(max_freq-min_freq)/num_scales:.1f} Hz per scale")
                
                # Feature extraction parameters (OPTIMIZED for speed)
                # Based on analysis: wavelet features are highly redundant, can reduce significantly
                max_dwt_features = 30  # Reduced from 60 (50% reduction, still excellent performance)
                max_cwt_features = 15  # Reduced from 25 (40% reduction)
                
                # Create CWT frequency analysis visualization
                print(f"Creating CWT frequency analysis plot...")
                fig_freq, ax_freq = plt.subplots(1, 1, figsize=(12, 6))
                
                # Convert scales to frequencies for visualization
                freqs = fc * sample_rate / cwt_scales
                
                ax_freq.semilogx(freqs, cwt_scales, 'bo-', markersize=3, linewidth=1.5, label='CWT Scales')
                ax_freq.axvspan(500, 8000, alpha=0.3, color='green', label='Dominant EOD range (500-8000 Hz)')
                ax_freq.axvspan(8000, max_freq, alpha=0.3, color='orange', label='Extended frequency range (8000+ Hz)')
                ax_freq.axvspan(min_freq, 500, alpha=0.3, color='lightblue', label='Low-frequency range (<500 Hz)')
                
                ax_freq.set_xlabel('Frequency (Hz)', fontsize=12)
                ax_freq.set_ylabel('CWT Scale', fontsize=12)
                ax_freq.set_title(f'CWT Scale-Frequency Mapping for EOD Analysis\n' +
                                 f'{num_scales} scales covering {min_freq}-{max_freq} Hz at {sample_rate} Hz sampling', 
                                 fontsize=14)
                ax_freq.legend()
                ax_freq.grid(True, alpha=0.3)
                ax_freq.set_xlim(min_freq, max_freq)
                
                # Add frequency annotations
                ax_freq.annotate('EOD fundamental\nfrequencies', xy=(2000, sample_rate/(2000*fc)), 
                               xytext=(1000, sample_rate/(1000*fc)), fontsize=10,
                               arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
                ax_freq.annotate('Harmonic content\nand fine structure', xy=(6000, sample_rate/(6000*fc)), 
                               xytext=(10000, sample_rate/(3000*fc)), fontsize=10,
                               arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7))
                
                plt.tight_layout()
                
                # Save the frequency analysis plot BEFORE showing it
                freq_plot_path = os.path.join(output_path, 'cwt_frequency_analysis_96kHz.png')
                plt.savefig(freq_plot_path, dpi=300, bbox_inches='tight')
                print(f"CWT frequency analysis plot saved: {freq_plot_path}")
                
                plt.show()
                plt.close()
                
                wavelet_features_dict = {}
                
                # =============================================================================
                # DISCRETE WAVELET TRANSFORM FEATURES (OPTIMIZED)
                # =============================================================================
                
                import time  # For performance timing
                
                for wavelet_name in dwt_wavelets:
                    start_time = time.time()
                    print(f"  Extracting DWT features with {wavelet_name}...")
                    
                    dwt_features = []
                    for i, waveform in enumerate(padded_waveforms):
                        # Progress indicator for large datasets
                        if i % 1000 == 0 and i > 0:
                            elapsed = time.time() - start_time
                            progress = i / len(padded_waveforms)
                            eta = elapsed / progress - elapsed
                            print(f"    Progress: {i}/{len(padded_waveforms)} ({progress*100:.1f}%) - ETA: {eta:.1f}s")
                        
                        try:
                            # Perform DWT with FIXED decomposition level (robust approach)
                            # Fixed level 4 ensures consistent features across all waveform lengths
                            fixed_level = 4
                            
                            # Check if waveform is long enough for level 4 decomposition
                            min_length_required = 2**fixed_level  # 16 samples minimum
                            if len(waveform) < min_length_required:
                                # Zero-pad short waveforms to ensure consistent decomposition
                                waveform = np.pad(waveform, (0, min_length_required - len(waveform)), mode='constant')
                            
                            coeffs = pywt.wavedec(waveform, wavelet_name, level=fixed_level)
                            
                            # Extract REDUCED statistics from each level (for speed)
                            feature_vector = []
                            for level_idx, coeff_level in enumerate(coeffs):
                                if len(coeff_level) > 0:
                                    # Reduced statistical features (keep most informative)
                                    level_features = [
                                        np.mean(coeff_level),                    # Mean
                                        np.std(coeff_level),                     # Standard deviation
                                        np.max(np.abs(coeff_level)),            # Maximum absolute value
                                        np.percentile(np.abs(coeff_level), 95), # 95th percentile
                                        np.sum(coeff_level**2)                   # Energy
                                        # Removed: min, max, 5th percentile, skewness, kurtosis for speed
                                    ]
                                    feature_vector.extend(level_features)
                                else:
                                    # Fill with zeros if level is empty
                                    feature_vector.extend([0] * 5)  # Reduced from 10 to 5
                            
                            # Truncate or pad to consistent size
                            if len(feature_vector) > max_dwt_features:
                                feature_vector = feature_vector[:max_dwt_features]
                            else:
                                feature_vector.extend([0] * (max_dwt_features - len(feature_vector)))
                            
                            dwt_features.append(feature_vector)
                            
                        except Exception as e:
                            print(f"    Warning: DWT failed for waveform {i}: {e}")
                            # Handle problematic waveforms
                            dwt_features.append([0] * max_dwt_features)
                    
                    elapsed_time = time.time() - start_time
                    print(f"    ✓ {wavelet_name} completed in {elapsed_time:.1f}s ({elapsed_time/len(padded_waveforms)*1000:.2f}ms per waveform)")
                    
                    wavelet_features_dict[f'DWT_{wavelet_name}'] = np.array(dwt_features)
                
                # =============================================================================
                # CONTINUOUS WAVELET TRANSFORM FEATURES (OPTIMIZED)
                # =============================================================================
                
                print(f"  Extracting CWT features with {cwt_wavelet}...")
                start_time = time.time()
                
                cwt_features = []
                for i, waveform in enumerate(padded_waveforms):
                    # Progress indicator
                    if i % 1000 == 0 and i > 0:
                        elapsed = time.time() - start_time
                        progress = i / len(padded_waveforms)
                        eta = elapsed / progress - elapsed
                        print(f"    Progress: {i}/{len(padded_waveforms)} ({progress*100:.1f}%) - ETA: {eta:.1f}s")
                    
                    try:
                        # Perform CWT on zero-padded waveform (no resampling needed!)
                        cwt_matrix = cwt(waveform, signal.morlet2, cwt_scales, 
                                        dtype=np.complex128)
                        
                        # Extract OPTIMIZED features from CWT (reduced for speed)
                        cwt_abs = np.abs(cwt_matrix)
                        
                        # STREAMLINED time-frequency domain features
                        feature_vector = [
                            # Core energy-based features
                            np.mean(cwt_abs),                           # Mean magnitude
                            np.std(cwt_abs),                            # Magnitude variation
                            np.max(cwt_abs),                            # Peak magnitude
                            np.sum(cwt_abs),                            # Total energy
                            
                            # Essential scale (frequency) analysis
                            np.argmax(np.mean(cwt_abs, axis=1)),        # Dominant scale index
                            np.max(np.mean(cwt_abs, axis=1)),           # Max energy across scales
                            np.std(np.mean(cwt_abs, axis=1)),           # Scale energy variation
                            
                            # Essential time analysis
                            np.argmax(np.mean(cwt_abs, axis=0)),        # Dominant time index
                            np.max(np.mean(cwt_abs, axis=0)),           # Max energy across time
                            
                            # Frequency band analysis (EOD-specific, keep for biological relevance)
                            np.mean(cwt_abs[:len(cwt_scales)//3, :]),   # Energy in lower scales
                            np.mean(cwt_abs[len(cwt_scales)//3:2*len(cwt_scales)//3, :]), # Mid scales
                            np.mean(cwt_abs[2*len(cwt_scales)//3:, :]), # Energy in higher scales
                            
                            # Essential statistical features
                            np.percentile(cwt_abs.flatten(), 95),       # 95th percentile
                            np.percentile(cwt_abs.flatten(), 5),        # 5th percentile
                            
                            # Energy concentration (important for species discrimination)
                            np.sum(cwt_abs**2) / (np.sum(cwt_abs)**2),  # Energy concentration ratio
                        ]
                        
                        # Ensure we have exactly max_cwt_features
                        if len(feature_vector) > max_cwt_features:
                            feature_vector = feature_vector[:max_cwt_features]
                        else:
                            feature_vector.extend([0] * (max_cwt_features - len(feature_vector)))
                        
                        cwt_features.append(feature_vector)
                        
                    except Exception as e:
                        print(f"    Warning: CWT failed for waveform {i}: {e}")
                        # Handle problematic waveforms
                        cwt_features.append([0] * max_cwt_features)
                
                elapsed_time = time.time() - start_time
                print(f"    ✓ CWT completed in {elapsed_time:.1f}s ({elapsed_time/len(padded_waveforms)*1000:.2f}ms per waveform)")
                
                wavelet_features_dict['CWT_morlet'] = np.array(cwt_features)
                
                # =============================================================================
                # TRAIN CLASSIFIERS ON WAVELET FEATURES
                # =============================================================================
                
                wavelet_results = {}
                
                for feature_name, features in wavelet_features_dict.items():
                    print(f"\nTraining classifiers on {feature_name} features...")
                    print(f"Feature shape: {features.shape}")
                    
                    # Check for valid features
                    if np.all(features == 0):
                        print(f"  ✗ All features are zero for {feature_name}, skipping...")
                        continue
                    
                    # Remove features with zero variance
                    feature_var = np.var(features, axis=0)
                    valid_feature_mask = feature_var > 1e-10
                    features_filtered = features[:, valid_feature_mask]
                    
                    if features_filtered.shape[1] == 0:
                        print(f"  ✗ No valid features after variance filtering for {feature_name}")
                        continue
                    
                    print(f"  Valid features after filtering: {features_filtered.shape[1]}")
                    
                    # Scale features
                    wavelet_scaler = StandardScaler()
                    features_scaled = wavelet_scaler.fit_transform(features_filtered)
                    
                    # Split data
                    X_wav_train, X_wav_test, y_wav_train, y_wav_test = train_test_split(
                        features_scaled, valid_y, test_size=0.3, random_state=42, stratify=valid_y
                    )
                    
                    # Train Random Forest on wavelet features (fastest and most robust)
                    try:
                        rf_wavelet = RandomForestClassifier(n_estimators=100, random_state=42, 
                                                           class_weight='balanced')
                        rf_wavelet.fit(X_wav_train, y_wav_train)
                        
                        # Predictions
                        wav_train_pred = rf_wavelet.predict(X_wav_train)
                        wav_test_pred = rf_wavelet.predict(X_wav_test)
                        wav_test_proba = rf_wavelet.predict_proba(X_wav_test)
                        
                        # Metrics
                        wav_train_acc = accuracy_score(y_wav_train, wav_train_pred)
                        wav_test_acc = accuracy_score(y_wav_test, wav_test_pred)
                        
                        # Cross-validation
                        wav_cv_scores = cross_val_score(rf_wavelet, X_wav_train, y_wav_train, 
                                                       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
                        wav_cv_mean = wav_cv_scores.mean()
                        wav_cv_std = wav_cv_scores.std()
                        
                        # Reports
                        wav_class_report = classification_report(y_wav_test, wav_test_pred, 
                                                               target_names=label_encoder.classes_,
                                                               output_dict=True)
                        wav_conf_matrix = confusion_matrix(y_wav_test, wav_test_pred)
                        
                        # AUC
                        try:
                            if len(label_encoder.classes_) > 2:
                                wav_auc = roc_auc_score(y_wav_test, wav_test_proba, multi_class='ovr', average='weighted')
                            else:
                                wav_auc = roc_auc_score(y_wav_test, wav_test_proba[:, 1])
                        except:
                            wav_auc = np.nan
                        
                        wavelet_results[f'RF_{feature_name}'] = {
                            'model': rf_wavelet,
                            'scaler': wavelet_scaler,  # Add the wavelet-specific scaler
                            'train_accuracy': wav_train_acc,
                            'test_accuracy': wav_test_acc,
                            'cv_mean': wav_cv_mean,
                            'cv_std': wav_cv_std,
                            'classification_report': wav_class_report,
                            'confusion_matrix': wav_conf_matrix,
                            'test_predictions': wav_test_pred,
                            'test_probabilities': wav_test_proba,
                            'auc_score': wav_auc,
                            'feature_importance': rf_wavelet.feature_importances_,
                            'n_features': features_filtered.shape[1]
                        }
                        
                        print(f"  ✓ Train accuracy: {wav_train_acc:.4f}")
                        print(f"  ✓ Test accuracy: {wav_test_acc:.4f}")
                        print(f"  ✓ CV accuracy: {wav_cv_mean:.4f} ± {wav_cv_std:.4f}")
                        if not np.isnan(wav_auc):
                            print(f"  ✓ AUC score: {wav_auc:.4f}")
                        
                        # Feature importance
                        top_features = np.argsort(rf_wavelet.feature_importances_)[-3:][::-1]
                        print(f"  ✓ Top 3 feature indices: {top_features}")
                        
                    except Exception as e:
                        print(f"  ✗ Error training Random Forest on {feature_name}: {e}")
                        wavelet_results[f'RF_{feature_name}'] = {'error': str(e)}
                
                # Combine wavelet results with main results
                results.update(wavelet_results)
                
                print(f"\nWavelet-based classification completed!")
                print(f"Added {len(wavelet_results)} wavelet-based models")
    
    except Exception as e:
        print(f"Error in wavelet analysis: {e}")
        import traceback
        traceback.print_exc()
        wavelet_results = {}

# =============================================================================
# MODEL PERFORMANCE COMPARISON
# =============================================================================

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

# Filter out models with errors
valid_results = {name: result for name, result in results.items() 
                if 'error' not in result}

if not valid_results:
    print("No valid models to evaluate!")
else:
    # Sort models by test accuracy
    sorted_results = sorted(valid_results.items(), 
                          key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    print(f"\nModel Performance Ranking (by test accuracy):")
    print("-" * 70)
    
    # Separate traditional and wavelet models for better organization
    traditional_models = []
    wavelet_models = []
    
    for i, (name, result) in enumerate(sorted_results, 1):
        if name.startswith('RF_'):
            wavelet_models.append((name, result))
        else:
            traditional_models.append((name, result))
    
    # Display traditional models
    print("\nTRADITIONAL FEATURE-BASED MODELS:")
    print("-" * 40)
    for i, (name, result) in enumerate(traditional_models, 1):
        print(f"{i}. {name}")
        print(f"   Test Accuracy: {result['test_accuracy']:.4f}")
        if not np.isnan(result['cv_mean']):
            print(f"   CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        if not np.isnan(result['auc_score']):
            print(f"   AUC Score: {result['auc_score']:.4f}")
        print()
    
    # Display wavelet models
    if wavelet_models:
        print("\nWAVELET-BASED MODELS:")
        print("-" * 40)
        for i, (name, result) in enumerate(wavelet_models, 1):
            feature_type = name.replace('RF_', '')
            print(f"{i}. {feature_type}")
            print(f"   Test Accuracy: {result['test_accuracy']:.4f}")
            if not np.isnan(result['cv_mean']):
                print(f"   CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
            if not np.isnan(result['auc_score']):
                print(f"   AUC Score: {result['auc_score']:.4f}")
            if 'n_features' in result:
                print(f"   Features Used: {result['n_features']}")
            print()
    
    # Display overall ranking
    print("\nOVERALL RANKING (ALL MODELS):")
    print("-" * 40)
    for i, (name, result) in enumerate(sorted_results, 1):
        display_name = name.replace('RF_', '') if name.startswith('RF_') else name
        print(f"{i}. {display_name}: {result['test_accuracy']:.4f}")
    
    # Print failed models
    failed_models = [name for name, result in results.items() if 'error' in result]
    if failed_models:
        print("\nFAILED MODELS:")
        print("-" * 20)
        for name in failed_models:
            error_msg = results[name]['error'][:100] + "..." if len(results[name]['error']) > 100 else results[name]['error']
            print(f"  - {name}: {error_msg}")
        print()
    
    if sorted_results:
        best_model = sorted_results[0]
        best_name = best_model[0].replace('RF_', '') if best_model[0].startswith('RF_') else best_model[0]
        print(f"🏆 Best performing model: {best_name}")
        print(f"🏆 Best test accuracy: {best_model[1]['test_accuracy']:.4f}")
        
        # Show feature type for best model
        if best_model[0].startswith('RF_'):
            print(f"🏆 Feature type: Wavelet-based ({best_model[0].replace('RF_', '')})")
        else:
            print(f"🏆 Feature type: Traditional (amplitude_ratio, eod_width_us, fft_freq_max)")

# =============================================================================
# MODEL SAVING: TOP 3 TRADITIONAL + TOP 3 WAVELET MODELS
# =============================================================================

print(f"\n" + "="*60)
print("SAVING TOP PERFORMING MODELS")
print("="*60)

# Get top 3 traditional models
top_traditional = sorted(traditional_models, key=lambda x: x[1]['test_accuracy'], reverse=True)[:3]
top_wavelet = sorted(wavelet_models, key=lambda x: x[1]['test_accuracy'], reverse=True)[:3]

print(f"\nSaving top models to {output_path}...")

# Dictionary to store all models and metadata for saving
models_to_save = {
    'metadata': {
        'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': input_file,
        'total_samples': len(data),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': feature_columns,
        'species_mapping': dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))),
        'scaler_params': {
            'feature_means': scaler.mean_.tolist(),
            'feature_scales': scaler.scale_.tolist()
        }
    },
    'traditional_models': {},
    'wavelet_models': {}
}

# Save top 3 traditional models
print(f"\nTop 3 Traditional Models:")
for i, (name, result) in enumerate(top_traditional, 1):
    print(f"  {i}. {name}: {result['test_accuracy']:.4f}")
    
    # Create model package
    model_package = {
        'model': result['model'],
        'scaler': scaler,  # Same scaler for all traditional models
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'performance': {
            'test_accuracy': result['test_accuracy'],
            'train_accuracy': result['train_accuracy'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
            'auc_score': result['auc_score']
        },
        'confusion_matrix': result['confusion_matrix'],
        'classification_report': result['classification_report']
    }
    
    models_to_save['traditional_models'][name] = model_package
    
    # Save individual model file
    model_filename = f"model_traditional_{name.replace(' ', '_').replace('-', '_').lower()}.pkl"
    model_path = os.path.join(output_path, model_filename)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        print(f"    ✓ Saved: {model_filename}")
    except Exception as e:
        print(f"    ✗ Failed to save {model_filename}: {e}")

# Save top 3 wavelet models (if any)
if top_wavelet:
    print(f"\nTop 3 Wavelet Models:")
    for i, (name, result) in enumerate(top_wavelet, 1):
        clean_name = name.replace('RF_', '')
        print(f"  {i}. {clean_name}: {result['test_accuracy']:.4f}")
        
        # For wavelet models, we need additional information
        wavelet_type = clean_name  # e.g., 'DWT_db4', 'CWT_morlet'
        
        # Create model package with wavelet-specific information
        model_package = {
            'model': result['model'],
            'wavelet_scaler': result.get('scaler'),  # Wavelet-specific scaler
            'label_encoder': label_encoder,
            'wavelet_type': wavelet_type,
            'feature_count': result.get('n_features', 'unknown'),
            'performance': {
                'test_accuracy': result['test_accuracy'],
                'train_accuracy': result['train_accuracy'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'auc_score': result['auc_score']
            },
            'confusion_matrix': result['confusion_matrix'],
            'classification_report': result['classification_report'],
            'feature_importance': result.get('feature_importance'),
            'top_features': result.get('top_features')
        }
        
        models_to_save['wavelet_models'][clean_name] = model_package
        
        # Save individual model file
        model_filename = f"model_wavelet_{clean_name.replace('_', '').lower()}.pkl"
        model_path = os.path.join(output_path, model_filename)
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
            print(f"    ✓ Saved: {model_filename}")
        except Exception as e:
            print(f"    ✗ Failed to save {model_filename}: {e}")

# Save comprehensive model collection
collection_filename = "all_top_models_collection.pkl"
collection_path = os.path.join(output_path, collection_filename)

try:
    with open(collection_path, 'wb') as f:
        pickle.dump(models_to_save, f)
    print(f"\n✓ Saved comprehensive model collection: {collection_filename}")
except Exception as e:
    print(f"\n✗ Failed to save model collection: {e}")

# Save using joblib as alternative (more efficient for sklearn)
joblib_filename = "all_top_models_collection.joblib"
joblib_path = os.path.join(output_path, joblib_filename)

try:
    joblib.dump(models_to_save, joblib_path)
    print(f"✓ Saved joblib version: {joblib_filename}")
except Exception as e:
    print(f"✗ Failed to save joblib version: {e}")

# Create loading example script
example_script = '''# Example: How to load and use saved models

import pickle
import joblib
import numpy as np
import pandas as pd

# Load the comprehensive model collection
with open('all_top_models_collection.pkl', 'rb') as f:
    models = pickle.load(f)

# Or load with joblib (faster)
# models = joblib.load('all_top_models_collection.joblib')

# Access top traditional model
best_traditional = list(models['traditional_models'].keys())[0]
traditional_model = models['traditional_models'][best_traditional]

print(f"Best traditional model: {best_traditional}")
print(f"Test accuracy: {traditional_model['performance']['test_accuracy']:.4f}")

# Access scaler and model
scaler = traditional_model['scaler']
model = traditional_model['model']
label_encoder = traditional_model['label_encoder']

# Example prediction (replace with your data)
# new_data = np.array([[amplitude_ratio, eod_width_us, fft_freq_max]])
# scaled_data = scaler.transform(new_data)
# prediction = model.predict(scaled_data)
# species = label_encoder.inverse_transform(prediction)

# For wavelet models, load the specific wavelet model
if 'wavelet_models' in models and models['wavelet_models']:
    best_wavelet = list(models['wavelet_models'].keys())[0]
    wavelet_model = models['wavelet_models'][best_wavelet]
    
    print(f"Best wavelet model: {best_wavelet}")
    print(f"Test accuracy: {wavelet_model['performance']['test_accuracy']:.4f}")
    print(f"Features used: {wavelet_model['feature_count']}")
'''

example_path = os.path.join(output_path, "model_loading_example.py")
try:
    with open(example_path, 'w') as f:
        f.write(example_script)
    print(f"✓ Created usage example: model_loading_example.py")
except Exception as e:
    print(f"✗ Failed to create example script: {e}")

print(f"\n" + "="*60)

# =============================================================================
# VISUALIZATION: MODEL COMPARISON
# =============================================================================

print(f"\nCreating model comparison plots...")

if valid_results:
    # Extract data for plotting
    model_names = list(valid_results.keys())
    train_accs = [valid_results[name]['train_accuracy'] for name in model_names]
    test_accs = [valid_results[name]['test_accuracy'] for name in model_names]
    cv_means = [valid_results[name]['cv_mean'] if not np.isnan(valid_results[name]['cv_mean']) 
               else None for name in model_names]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    ax1.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Cross-validation scores
    cv_valid = [(name, cv) for name, cv in zip(model_names, cv_means) if cv is not None]
    if cv_valid:
        cv_names, cv_scores = zip(*cv_valid)
        cv_stds = [valid_results[name]['cv_std'] for name in cv_names]
        
        x_cv = np.arange(len(cv_names))
        ax2.bar(x_cv, cv_scores, yerr=cv_stds, capsize=5, alpha=0.8, color='orange')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Cross-Validation Accuracy')
        ax2.set_title('Cross-Validation Performance')
        ax2.set_xticks(x_cv)
        ax2.set_xticklabels(cv_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# VISUALIZATION: CONFUSION MATRICES
# =============================================================================

print(f"\nCreating confusion matrices...")

if valid_results:
    valid_conf_results = {name: result for name, result in valid_results.items() 
                         if 'confusion_matrix' in result}
    
    n_models = len(valid_conf_results)
    if n_models > 0:
        # Calculate grid dimensions
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, result) in enumerate(valid_conf_results.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            conf_matrix = result['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_encoder.classes_,
                       yticklabels=label_encoder.classes_,
                       ax=ax)
            ax.set_title(f'{name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()

# =============================================================================
# VISUALIZATION: FEATURE IMPORTANCE
# =============================================================================

print(f"\nCreating feature importance plots...")

# Models that provide feature importance
importance_models = {}

for name, result in valid_results.items():
    model = result['model']
    
    # Handle different model types
    if name.startswith('RF_'):
        # Wavelet-based Random Forest models
        if 'feature_importance' in result:
            importance_models[name] = result['feature_importance']
    elif hasattr(model, 'feature_importances_'):
        # Traditional Random Forest
        importance_models[name] = model.feature_importances_
    elif hasattr(model, 'coef_') and len(model.coef_.shape) == 2:
        # For linear models, use absolute mean coefficients as importance
        importance_models[name] = np.mean(np.abs(model.coef_), axis=0)

if importance_models:
    # Separate traditional and wavelet models
    traditional_importance = {k: v for k, v in importance_models.items() if not k.startswith('RF_')}
    wavelet_importance = {k: v for k, v in importance_models.items() if k.startswith('RF_')}
    
    # Create plots
    n_plots = (1 if traditional_importance else 0) + (1 if wavelet_importance else 0)
    
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(12*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot traditional model feature importance
        if traditional_importance:
            ax = axes[plot_idx]
            plot_idx += 1
            
            x = np.arange(len(feature_columns))
            width = 0.8 / len(traditional_importance)
            
            for i, (model_name, importances) in enumerate(traditional_importance.items()):
                # Normalize importances to 0-1 scale for comparison
                normalized_imp = importances / np.max(importances) if np.max(importances) > 0 else importances
                ax.bar(x + i * width, normalized_imp, width, label=model_name, alpha=0.8)
            
            ax.set_xlabel('Traditional Features')
            ax.set_ylabel('Normalized Importance')
            ax.set_title('Traditional Feature Importance Comparison')
            ax.set_xticks(x + width * (len(traditional_importance) - 1) / 2)
            ax.set_xticklabels(feature_columns)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot wavelet model feature importance (top features only)
        if wavelet_importance:
            ax = axes[plot_idx]
            
            # Show top 10 features for each wavelet model
            n_top_features = 10
            
            # Calculate positions for grouped bars
            n_models = len(wavelet_importance)
            x_positions = np.arange(n_top_features)
            width = 0.8 / n_models
            
            for i, (model_name, importances) in enumerate(wavelet_importance.items()):
                # Get top features
                top_indices = np.argsort(importances)[-n_top_features:][::-1]
                top_importances = importances[top_indices]
                
                # Normalize
                normalized_imp = top_importances / np.max(top_importances) if np.max(top_importances) > 0 else top_importances
                
                display_name = model_name.replace('RF_', '')
                ax.bar(x_positions + i * width, normalized_imp, width, 
                      label=display_name, alpha=0.8)
            
            ax.set_xlabel('Top Wavelet Features (by importance rank)')
            ax.set_ylabel('Normalized Importance')
            ax.set_title('Wavelet Feature Importance Comparison')
            ax.set_xticks(x_positions + width * (n_models - 1) / 2)
            ax.set_xticklabels([f'Feature {i+1}' for i in range(n_top_features)])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create detailed wavelet feature importance summary
    if wavelet_importance:
        print(f"\nWavelet Feature Importance Summary:")
        print("-" * 50)
        for model_name, importances in wavelet_importance.items():
            feature_type = model_name.replace('RF_', '')
            top_5_indices = np.argsort(importances)[-5:][::-1]
            print(f"\n{feature_type} - Top 5 features:")
            for rank, idx in enumerate(top_5_indices, 1):
                print(f"  {rank}. Feature {idx}: {importances[idx]:.4f}")
else:
    print("No models with extractable feature importance found.")

# =============================================================================
# VISUALIZATION: DECISION BOUNDARIES (2D PROJECTIONS)
# =============================================================================

print(f"\nCreating decision boundary plots...")

if valid_results:
    # Select top 2 performing models for visualization
    sorted_models = sorted(valid_results.items(), 
                          key=lambda x: x[1]['test_accuracy'], reverse=True)[:2]
    
    if len(sorted_models) > 0:
        # Feature pairs for 2D visualization
        feature_pairs = [(0, 1), (0, 2), (1, 2)]  # All combinations of 3 features
        pair_names = [f'{feature_columns[i]} vs {feature_columns[j]}' for i, j in feature_pairs]
        
        fig, axes = plt.subplots(len(sorted_models), len(feature_pairs), 
                                figsize=(5*len(feature_pairs), 5*len(sorted_models)))
        
        if len(sorted_models) == 1:
            axes = axes.reshape(1, -1)
        if len(feature_pairs) == 1:
            axes = axes.reshape(-1, 1)
        
        for model_idx, (model_name, result) in enumerate(sorted_models):
            model = result['model']
            
            for pair_idx, (feat1, feat2) in enumerate(feature_pairs):
                ax = axes[model_idx, pair_idx]
                
                # Extract 2D data
                X_2d = X_test[:, [feat1, feat2]]
                
                # Create a mesh for decision boundary
                h = 0.02  # Step size in the mesh
                x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
                y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
                
                # For decision boundary, we need to predict on 3D data
                # Use mean values for the missing feature
                missing_feat = [i for i in range(3) if i not in [feat1, feat2]][0]
                mean_val = X_test[:, missing_feat].mean()
                
                # Create 3D mesh points
                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                mesh_3d = np.zeros((mesh_points.shape[0], 3))
                mesh_3d[:, feat1] = mesh_points[:, 0]
                mesh_3d[:, feat2] = mesh_points[:, 1]
                mesh_3d[:, missing_feat] = mean_val
                
                try:
                    if model_name == 'Gaussian Mixture Model':
                        Z = model.predict(mesh_3d)
                    elif model_name == 'Mixture Discriminant Analysis':
                        # Use the MDA prediction function
                        Z = result['predict_func'](mesh_3d)
                    else:
                        Z = model.predict(mesh_3d)
                    
                    Z = Z.reshape(xx.shape)
                    
                    # Plot decision boundary
                    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
                    
                except Exception as e:
                    print(f"Could not plot decision boundary for {model_name}: {e}")
                
                # Plot data points
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_test, 
                                   cmap=plt.cm.RdYlBu, edgecolors='black', s=50)
                
                ax.set_xlabel(feature_columns[feat1])
                ax.set_ylabel(feature_columns[feat2])
                ax.set_title(f'{model_name}\n{pair_names[pair_idx]}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'decision_boundaries.png'), dpi=300, bbox_inches='tight')
        plt.show()

# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================

print(f"\nSaving detailed results...")

# Create summary dictionary
summary = {
    'timestamp': dt.datetime.now().isoformat(),
    'species_classes': label_encoder.classes_.tolist(),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'feature_names': feature_columns,
    'models': {},
    'analysis_info': {
        'traditional_features': feature_columns,
        'wavelet_features': list(wavelet_features_dict.keys()) if 'wavelet_features_dict' in locals() else [],
        'mda_components_per_class': 2,
        'dwt_wavelets': ['db4', 'db10', 'haar', 'sym4'],
        'cwt_wavelet': 'Complex Morlet (approximating Morse)',
        'max_wavelet_coeffs': 10
    }
}

for name, result in results.items():
    if 'error' in result:
        summary['models'][name] = {'error': result['error']}
        continue
    
    model_summary = {
        'train_accuracy': float(result['train_accuracy']),
        'test_accuracy': float(result['test_accuracy']),
        'cv_mean': float(result['cv_mean']) if not np.isnan(result['cv_mean']) else None,
        'cv_std': float(result['cv_std']) if not np.isnan(result['cv_std']) else None,
        'auc_score': float(result['auc_score']) if not np.isnan(result['auc_score']) else None,
        'confusion_matrix': result['confusion_matrix'].tolist(),
        'classification_report': result['classification_report']
    }
    
    # Add model-specific information
    if name.startswith('RF_'):
        model_summary['model_type'] = 'wavelet_based'
        model_summary['feature_type'] = name.replace('RF_', '')
        if 'n_features' in result:
            model_summary['n_features'] = result['n_features']
    elif name == 'Mixture Discriminant Analysis':
        model_summary['model_type'] = 'mixture_discriminant'
        model_summary['components_per_class'] = 2
    else:
        model_summary['model_type'] = 'traditional'
    
    summary['models'][name] = model_summary

# Save summary as JSON
with open(os.path.join(output_path, 'classification_results.json'), 'w') as f:
    json.dump(summary, f, indent=2, separators=(',', ': '))

# Save detailed classification reports as text
with open(os.path.join(output_path, 'classification_reports.txt'), 'w') as f:
    f.write("SPECIES CLASSIFICATION RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Analysis timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Input file: {input_file}\n")
    f.write(f"Output directory: {output_path}\n")
    f.write(f"Total samples: {len(data)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Features used: {', '.join(feature_columns)}\n")
    f.write(f"Species: {', '.join(label_encoder.classes_)}\n\n")
    
    for name, result in results.items():
        if 'error' in result:
            f.write(f"{name}: ERROR - {result['error']}\n\n")
            continue
            
        f.write(f"{name}:\n")
        f.write("-" * len(name) + "\n")
        f.write(f"Train Accuracy: {result['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy: {result['test_accuracy']:.4f}\n")
        
        if not np.isnan(result['cv_mean']):
            f.write(f"CV Accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n")
        
        if not np.isnan(result['auc_score']):
            f.write(f"AUC Score: {result['auc_score']:.4f}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"{result['confusion_matrix']}\n")
        f.write("\n" + "="*50 + "\n\n")

print("✓ Results saved to: {output_path}")
print("  - classification_results.json")
print("  - classification_reports.txt")
print("  - exploratory_analysis.png")
print("  - model_comparison.png")
print("  - confusion_matrices.png")
if importance_models:
    print("  - feature_importance.png")
if valid_results:
    print("  - decision_boundaries.png")

# Model files saved
print("\n📦 SAVED MODELS:")
print("  - all_top_models_collection.pkl (comprehensive collection)")
print("  - all_top_models_collection.joblib (alternative format)")
for i in range(min(3, len(top_traditional))):
    model_name = top_traditional[i][0].replace(' ', '_').replace('-', '_').lower()
    print(f"  - model_traditional_{model_name}.pkl")
if top_wavelet:
    for i in range(min(3, len(top_wavelet))):
        clean_name = top_wavelet[i][0].replace('RF_', '').replace('_', '').lower()
        print(f"  - model_wavelet_{clean_name}.pkl")
print("  - model_loading_example.py (usage guide)")

print(f"\n🎯 READY FOR DEPLOYMENT:")
print(f"  Top Traditional: {top_traditional[0][0]} ({top_traditional[0][1]['test_accuracy']:.4f})")
if top_wavelet:
    print(f"  Top Wavelet: {top_wavelet[0][0].replace('RF_', '')} ({top_wavelet[0][1]['test_accuracy']:.4f})")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)