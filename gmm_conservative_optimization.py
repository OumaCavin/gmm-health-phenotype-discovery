#!/usr/bin/env python3
"""
===============================================================================
GMM CLUSTER OPTIMIZATION - CONSERVATIVE APPROACH
===============================================================================

Target: Recover and exceed previous best (0.0609)
Strategy: Conservative preprocessing, feature engineering, UMAP

This script implements a careful optimization that:
1. Preserves data signal (minimal removal)
2. Uses clinically relevant features
3. Applies UMAP for better cluster separation
4. Systematically tunes hyperparameters

Output: output_v2/
===============================================================================
"""

import os
import sys
import json
import time
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import (
    silhouette_score, silhouette_samples, 
    calinski_harabasz_score, davies_bouldin_score
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'output_v2'
DATA_PATH = 'data/raw/nhanes_health_data.csv'

# Create output directories
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/models', f'{OUTPUT_DIR}/figures', 
          f'{OUTPUT_DIR}/metrics', f'{OUTPUT_DIR}/predictions', 
          f'{OUTPUT_DIR}/reports', f'{OUTPUT_DIR}/logs']:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("GMM CLUSTER OPTIMIZATION - CONSERVATIVE APPROACH")
print("=" * 70)
print(f"Target: Recover and exceed 0.0609")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 70)

# =============================================================================
# STEP 1: DATA LOADING WITH CLINICAL FEATURE SELECTION
# =============================================================================

print("\n[STEP 1] Loading Data...")

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Original dataset: {df.shape}")

# Remove non-feature columns
exclude_cols = ['SEQN', 'respondent_id', 'cluster', 'cluster_label']
for col in exclude_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Convert categorical to numeric
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.Categorical(df[col]).codes

# Define clinically relevant feature groups (based on domain knowledge)
CLINICAL_FEATURES = {
    # Metabolic markers
    'metabolic': ['bmi', 'waist_circumference_cm', 'fasting_glucose_mg_dL', 
                  'insulin_uU_mL', 'total_cholesterol_mg_dL', 'hdl_cholesterol_mg_dL',
                  'ldl_cholesterol_mg_dL'],
    
    # Cardiovascular markers  
    'cardiovascular': ['systolic_bp_mmHg', 'diastolic_bp_mmHg', 'age'],
    
    # Behavioral factors
    'behavioral': ['smoked_100_cigarettes', 'alcohol_use_past_year', 
                   'drinks_per_week', 'vigorous_work_activity', 
                   'moderate_work_activity', 'vigorous_recreation_activity',
                   'moderate_recreation_activity'],
    
    # Mental health (PHQ-9)
    'mental_health': ['phq9_little_interest', 'phq9_feeling_down', 
                      'phq9_sleep_trouble', 'phq9_feeling_tired',
                      'phq9_total_score'],
    
    # Health conditions
    'conditions': ['arthritis', 'heart_failure', 'coronary_heart_disease',
                   'cancer_diagnosis', 'general_health_rating']
}

# Select all clinical features
all_clinical_features = []
for category, features in CLINICAL_FEATURES.items():
    for f in features:
        if f in df.columns:
            all_clinical_features.append(f)

# Remove duplicates and check which exist in data
all_clinical_features = list(dict.fromkeys([f for f in all_clinical_features if f in df.columns]))
print(f"Selected {len(all_clinical_features)} clinical features")

# Get numeric columns for clustering
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c in all_clinical_features]

# If clinical selection doesn't yield enough, use all numeric
if len(feature_cols) < 10:
    feature_cols = numeric_cols
    print(f"Using all {len(feature_cols)} numeric features")

X = df[feature_cols].values.astype(np.float64)
print(f"Feature matrix: {X.shape}")

# =============================================================================
# STEP 2: CONSERVATIVE PREPROCESSING
# =============================================================================

print("\n[STEP 2] Conservative Preprocessing...")

# A. Handle missing values (conservative - use median)
missing_before = np.isnan(X).sum()
print(f"Missing values: {missing_before}")

if missing_before > 0:
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    print("Applied median imputation")

# B. Remove constant features (minimal removal)
var_selector = VarianceThreshold(threshold=0.001)
X_var = var_selector.fit_transform(X)
var_features = var_selector.get_support()
print(f"Features after variance filter: {X_var.shape[1]}")

# C. Apply PowerTransformer for better GMM performance
# Yeo-Johnson works with positive and negative values
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_transformed = pt.fit_transform(X_var)
print("Applied Yeo-Johnson PowerTransformer")

# Skewness check
skewness = np.abs(stats.skew(X_transformed, axis=0)).mean()
print(f"Mean absolute skewness: {skewness:.4f}")

# D. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
print("Applied StandardScaler")

print(f"Final preprocessed shape: {X_scaled.shape}")
print(f"Data preservation: {100*X_scaled.shape[0]/5000:.1f}%")

# =============================================================================
# STEP 3: CONSERVATIVE OUTLIER DETECTION (minimal removal)
# =============================================================================

print("\n[STEP 3] Conservative Outlier Detection...")

original_size = X_scaled.shape[0]
outlier_masks = []

# Method 1: Very mild Isolation Forest (5% max)
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
labels = iso.fit_predict(X_scaled)
mask = labels != -1
outlier_masks.append(mask)
removed = (~mask).sum()
print(f"Isolation Forest (5%): Removed {removed} ({100*removed/original_size:.1f}%)")

# Method 2: Local Outlier Factor (3%)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
labels = lof.fit_predict(X_scaled)
mask = labels != -1
outlier_masks.append(mask)
removed = (~mask).sum()
print(f"LOF (3%): Removed {removed} ({100*removed/original_size:.1f}%)")

# Conservative: use only Isolation Forest results
# This preserves more signal
final_mask = outlier_masks[0]  # Only use the mildest method

X_clean = X_scaled[final_mask]
removed = original_size - X_clean.shape[0]
print(f"\nFinal conservative removal: {removed} ({100*removed/original_size:.1f}%)")
print(f"Preserved samples: {X_clean.shape[0]} ({100*X_clean.shape[0]/original_size:.1f}%)")

# Save clean data
np.save(f'{OUTPUT_DIR}/data/X_conservative_clean.npy', X_clean)

# =============================================================================
# STEP 4: DIMENSIONALITY REDUCTION (UMAP + PCA)
# =============================================================================

print("\n[STEP 4] Dimensionality Reduction...")

reductions = {}

# PCA for reference
pca_full = PCA(n_components=0.95, random_state=42)
X_pca_full = pca_full.fit_transform(X_clean)
reductions['pca_95'] = X_pca_full
print(f"PCA (95% variance): {X_pca_full.shape[1]} components")

# PCA with 2 components for visualization
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_clean)
reductions['pca_2d'] = X_pca_2d
print(f"PCA (2D): {X_pca_2d.shape[1]} components")

# UMAP for better cluster separation
print("Applying UMAP...")
try:
    import umap
    
    umap_configs = [
        {'n_neighbors': 15, 'min_dist': 0.1, 'name': 'umap_15_01'},
        {'n_neighbors': 30, 'min_dist': 0.0, 'name': 'umap_30_00'},
        {'n_neighbors': 50, 'min_dist': 0.5, 'name': 'umap_50_05'},
    ]
    
    for config in umap_configs:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=config['n_neighbors'],
            min_dist=config['min_dist'],
            random_state=42,
            metric='euclidean'
        )
        X_umap = reducer.fit_transform(X_clean)
        reductions[config['name']] = X_umap
        print(f"UMAP ({config['n_neighbors']}, {config['min_dist']}): {X_umap.shape}")
        
        # Save UMAP model
        import joblib
        joblib.dump(reducer, f'{OUTPUT_DIR}/models/{config['name']}.joblib')
        
except ImportError:
    print("UMAP not available, using PCA only")
    reductions['pca_2d'] = X_pca_2d

# Save PCA model
import joblib
joblib.dump(pca_2d, f'{OUTPUT_DIR}/models/pca_2d.joblib')

# =============================================================================
# STEP 5: COMPREHENSIVE GMM TUNING
# =============================================================================

print("\n[STEP 5] GMM Hyperparameter Tuning...")

def evaluate_gmm(X, n_components, covariance_type, reg_covar=1e-4):
    """Evaluate GMM configuration."""
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=10,
            max_iter=300,
            random_state=42,
            init_params='kmeans'
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        
        if len(np.unique(labels)) < 2:
            return None
            
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        
        return {
            'n_components': n_components,
            'covariance_type': covariance_type,
            'reg_covar': reg_covar,
            'silhouette_score': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'bic': gmm.bic(X),
            'aic': gmm.aic(X),
            'model': gmm,
            'labels': labels
        }
    except Exception as e:
        return None

# Test configurations
results = []
param_grid = {
    'n_components': [2, 3, 4, 5],
    'covariance_type': ['spherical', 'tied', 'diag', 'full'],
    'reg_covar': [1e-6, 1e-4, 1e-2]
}

print(f"Testing configurations on {len(reductions)} reduced datasets...")

for data_name, X_test in reductions.items():
    print(f"\n  Testing on {data_name}...")
    
    for n in param_grid['n_components']:
        for cov in param_grid['covariance_type']:
            for reg in param_grid['reg_covar']:
                result = evaluate_gmm(X_test, n, cov, reg)
                if result:
                    result['data_type'] = data_name
                    results.append(result)

# Sort by silhouette score
results.sort(key=lambda x: x['silhouette_score'], reverse=True)

print(f"\nTotal configurations tested: {len(results)}")
print(f"Best configuration:")
print(f"  - Silhouette Score: {results[0]['silhouette_score']:.4f}")
print(f"  - k: {results[0]['n_components']}")
print(f"  - Covariance: {results[0]['covariance_type']}")
print(f"  - Data type: {results[0]['data_type']}")

# =============================================================================
# STEP 6: ALTERNATIVE ALGORITHMS FOR COMPARISON
# =============================================================================

print("\n[STEP 6] Testing Alternative Algorithms...")

from sklearn.cluster import KMeans, AgglomerativeClustering

alt_results = []

# Use best UMAP configuration
X_best_umap = reductions.get('umap_30_00', reductions.get('umap_15_01', reductions['pca_2d']))

for k in [2, 3, 4, 5]:
    # K-Means
    try:
        kmeans = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(X_best_umap)
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X_best_umap, labels)
            alt_results.append({'algorithm': 'kmeans', 'k': k, 'silhouette': sil})
    except:
        pass
    
    # Agglomerative
    try:
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = agg.fit_predict(X_best_umap)
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X_best_umap, labels)
            alt_results.append({'algorithm': 'agglomerative', 'k': k, 'silhouette': sil})
    except:
        pass

alt_results.sort(key=lambda x: x['silhouette'], reverse=True)
print(f"Best alternative: {alt_results[0]['algorithm']} (k={alt_results[0]['k']}, sil={alt_results[0]['silhouette']:.4f})")

# =============================================================================
# STEP 7: SAVE ALL RESULTS
# =============================================================================

print("\n[STEP 7] Saving Results...")

# Combine all results
all_results = results + alt_results
all_results.sort(key=lambda x: x.get('silhouette_score', x.get('silhouette', 0)), reverse=True)
best = all_results[0]

# Save best model
if 'model' in best:
    import joblib
    joblib.dump(best['model'], f'{OUTPUT_DIR}/models/best_gmm_conservative.joblib')
    print(f"Saved best model")

# Save cluster assignments
if 'labels' in best:
    labels = best['labels']
else:
    labels = best['model'].predict(X_best_umap)

assignments = pd.DataFrame({
    'sample_id': range(len(labels)),
    'cluster': labels
})
assignments.to_csv(f'{OUTPUT_DIR}/predictions/conservative_cluster_assignments.csv', index=False)
print(f"Saved cluster assignments: {len(assignments)} samples")

# Save metrics
metrics = {
    'best_silhouette_score': float(best.get('silhouette_score', best.get('silhouette', 0))),
    'best_k': best.get('n_components', best.get('k', 'N/A')),
    'best_covariance_type': best.get('covariance_type', best.get('algorithm', 'N/A')),
    'best_data_type': best.get('data_type', 'umap'),
    'calinski_harabasz': float(best.get('calinski_harabasz', 0)),
    'davies_bouldin': float(best.get('davies_bouldin', 0)),
    'samples_preserved': X_clean.shape[0],
    'features_used': X_clean.shape[1],
    'data_preservation_rate': X_clean.shape[0] / original_size,
    'previous_best': 0.0609,
    'improvement_over_previous': float(best.get('silhouette_score', 0)) - 0.0609,
    'timestamp': datetime.datetime.now().isoformat()
}

with open(f'{OUTPUT_DIR}/metrics/conservative_optimization_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics")

# Save all GMM results
gmm_results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'labels']} 
                               for r in results])
gmm_results_df.to_csv(f'{OUTPUT_DIR}/metrics/all_gmm_configurations.csv', index=False)
print(f"Saved all configurations: {len(gmm_results_df)}")

# =============================================================================
# STEP 8: COMPREHENSIVE VISUALIZATION
# =============================================================================

print("\n[STEP 8] Generating Visualizations...")

# Use the best UMAP or PCA for visualization
viz_data = X_best_umap
viz_labels = labels

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Main Cluster Visualization (t-SNE or UMAP)
ax1 = fig.add_subplot(2, 3, 1)
scatter = ax1.scatter(viz_data[:, 0], viz_data[:, 1], c=viz_labels, 
                     cmap='viridis', alpha=0.6, s=15)
ax1.set_xlabel(f'{best["data_type"]} 1')
ax1.set_ylabel(f'{best["data_type"]} 2')
ax1.set_title(f'Best Clustering ({best["data_type"]})\nSilhouette: {best.get("silhouette_score", 0):.4f}')
plt.colorbar(scatter, ax=ax1, label='Cluster')

# 2. Silhouette Analysis
ax2 = fig.add_subplot(2, 3, 2)
sil_samples = silhouette_samples(X_clean, viz_labels)
y_lower = 10
n_clusters = len(np.unique(viz_labels))

for i in range(n_clusters):
    cluster_sil = sil_samples[viz_labels == i]
    cluster_sil.sort()
    size_i = cluster_sil.shape[0]
    y_upper = y_lower + size_i
    color = plt.cm.viridis(float(i) / n_clusters)
    ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax2.text(-0.05, y_lower + 0.5 * size_i, f'Cluster {i}')
    y_lower = y_upper + 10

best_sil = best.get('silhouette_score', best.get('silhouette', 0))
ax2.axvline(x=best_sil, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {best_sil:.4f}')
ax2.set_xlabel('Silhouette Coefficient')
ax2.set_ylabel('Cluster')
ax2.set_title('Silhouette Analysis')
ax2.legend()

# 3. Cluster Size Distribution
ax3 = fig.add_subplot(2, 3, 3)
unique, counts = np.unique(viz_labels, return_counts=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
bars = ax3.bar(unique, counts, color=colors)
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Sample Count')
ax3.set_title('Cluster Size Distribution')
for bar, count in zip(bars, counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(count), ha='center', va='bottom')

# 4. Top Configurations Comparison
ax4 = fig.add_subplot(2, 3, 4)
top_10 = results[:10]
config_labels = [f"k={r['n_components']}_{r['covariance_type'][:4]}" for r in top_10]
sil_scores = [r['silhouette_score'] for r in top_10]
bar_colors = ['green' if s >= 0.0609 else 'orange' if s >= 0.04 else 'red' for s in sil_scores]
ax4.barh(range(len(config_labels)), sil_scores, color=bar_colors)
ax4.set_yticks(range(len(config_labels)))
ax4.set_yticklabels(config_labels)
ax4.set_xlabel('Silhouette Score')
ax4.set_title('Top 10 GMM Configurations')
ax4.axvline(x=0.0609, color='green', linestyle='--', alpha=0.7, label='Previous Best (0.0609)')
ax4.axvline(x=0.87, color='blue', linestyle='--', alpha=0.7, label='Target (0.87)')
ax4.legend()

# 5. Score Summary
ax5 = fig.add_subplot(2, 3, 5)
ax5.axis('off')

prev_best = 0.0609
current_best = best.get('silhouette_score', 0)
improvement = (current_best - prev_best) / prev_best * 100

summary = f"""OPTIMIZATION RESULTS SUMMARY
{'='*45}

CURRENT BEST RESULTS
--------------------
Silhouette Score: {current_best:.4f}
Previous Best: {prev_best:.4f}
Improvement: {improvement:+.1f}%

BEST CONFIGURATION
------------------
Clusters (k): {best.get('n_components', best.get('k', 'N/A'))}
Covariance Type: {best.get('covariance_type', best.get('algorithm', 'N/A'))}
Data Type: {best.get('data_type', 'UMAP')}

DATA PRESERVATION
-----------------
Samples Preserved: {X_clean.shape[0]} ({100*X_clean.shape[0]/original_size:.1f}%)
Features Used: {X_clean.shape[1]}

STATUS: {'✓ EXCEEDED PREVIOUS BEST' if current_best > prev_best else '✗ BELOW PREVIOUS BEST'}

Config Tested: {len(results)}
"""

ax5.text(0.05, 0.95, summary, transform=ax5.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# 6. Score Distribution
ax6 = fig.add_subplot(2, 3, 6)
all_sil_scores = [r['silhouette_score'] for r in results]
ax6.hist(all_sil_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax6.axvline(x=current_best, color='red', linestyle='--', linewidth=2,
           label=f'Best: {current_best:.4f}')
ax6.axvline(x=prev_best, color='green', linestyle='--', linewidth=2,
           label=f'Previous: {prev_best:.4f}')
ax6.axvline(x=0.87, color='blue', linestyle='--', linewidth=2,
           label=f'Target: 0.87')
ax6.set_xlabel('Silhouette Score')
ax6.set_ylabel('Frequency')
ax6.set_title('Score Distribution Across All Configurations')
ax6.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/conservative_optimization_results.png', 
           dpi=150, bbox_inches='tight')
print(f"Saved visualization: conservative_optimization_results.png")

# =============================================================================
# STEP 9: FINAL REPORT
# =============================================================================

print("\n[STEP 9] Generating Report...")

prev_best = 0.0609
current_best = best.get('silhouette_score', 0)
target_achieved = current_best > prev_best

report = f"""# GMM Conservative Optimization Report

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best Silhouette Score** | {current_best:.4f} |
| **Previous Best** | {prev_best:.4f} |
| **Improvement** | {(current_best - prev_best) / prev_best * 100:+.1f}% |
| **Target (0.87-1.00)** | Not yet achieved (realistic for health data) |
| **Status** | {'✓ EXCEEDED PREVIOUS BEST' if target_achieved else '✗ BELOW PREVIOUS BEST'} |

## Configuration Details

| Parameter | Best Value |
|-----------|------------|
| Clusters (k) | {best.get('n_components', best.get('k', 'N/A'))} |
| Covariance Type | {best.get('covariance_type', best.get('algorithm', 'N/A'))} |
| Data Type | {best.get('data_type', 'UMAP')} |
| Regularization | {best.get('reg_covar', 'N/A')} |

## Methodology

### Conservative Preprocessing
- **Imputation**: Median imputation (robust to outliers)
- **Transformation**: Yeo-Johnson PowerTransformer
- **Scaling**: StandardScaler
- **Outlier Removal**: Conservative (5% via Isolation Forest)
- **Data Preservation**: {100*X_clean.shape[0]/original_size:.1f}%

### Dimensionality Reduction
- **Primary**: UMAP ({best.get('data_type', 'default')})
- **Alternative**: PCA (95% variance)

### Hyperparameter Search
- **k range**: 2-5
- **Covariance types**: spherical, tied, diag, full
- **Regularization**: 1e-6 to 1e-2
- **Total configurations tested**: {len(results)}

## Results Comparison

| Configuration | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|--------------|------------|----------------|-------------------|
| Best ({best.get('n_components', '?')}_{best.get('covariance_type', '?')[:4]}) | {current_best:.4f} | {best.get('davies_bouldin', 'N/A'):.4f} | {best.get('calinski_harabasz', 'N/A'):.2f} |
| Previous Best | 0.0609 | - | - |

## Top 5 Configurations

{results_df.head(5).to_markdown() if 'results_df' in dir() else 'See all_gmm_configurations.csv'}

## Why High Silhouette Scores Are Challenging

Achieving Silhouette scores of 0.87-1.00 requires:

1. **Perfect Cluster Separation**: Complete disjoint clusters with no overlap
2. **Compact Geometry**: All points tightly clustered around centroids
3. **No Noise**: Clean, artifact-free data
4. **Discrete Categories**: True categorical structure

### Reality of Health Phenotype Data

Health data (like NHANES) typically exhibits:

- **Continuous Phenotype Boundaries**: Health conditions exist on spectrums
- **Biological Variation**: Natural overlap between phenotype groups
- **Measurement Uncertainty**: Clinical measurement noise
- **Multi-morbidity**: Individuals with characteristics of multiple phenotypes

### Realistic Benchmarks

| Score Range | Interpretation | Achievable? |
|-------------|----------------|-------------|
| 0.87 - 1.00 | Excellent | Very rare in real health data |
| 0.51 - 0.70 | Good | Possible with curated features |
| 0.26 - 0.50 | Weak structure | Common for complex health data |
| < 0.25 | No structure | Typical for multi-dimensional data |

## Files Generated

- `models/best_gmm_conservative.joblib`: Best GMM model
- `models/pca_2d.joblib`: PCA transformer
- `models/umap_*.joblib`: UMAP transformers
- `metrics/conservative_optimization_results.json`: Complete metrics
- `metrics/all_gmm_configurations.csv`: All configurations tested
- `predictions/conservative_cluster_assignments.csv`: Cluster labels
- `figures/conservative_optimization_results.png`: Visualization

## Recommendations for Further Improvement

1. **Feature Engineering**: Create derived clinical scores
2. **Domain-Specific Features**: Incorporate metabolic syndrome indicators
3. **Semi-supervised Clustering**: Use partial label information
4. **Alternative Approaches**: Consider soft clustering or hierarchical methods

---
*Report generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

with open(f'{OUTPUT_DIR}/reports/conservative_optimization_report.md', 'w') as f:
    f.write(report)
print("Saved report: conservative_optimization_report.md")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)
print(f"Best Silhouette Score: {current_best:.4f}")
print(f"Previous Best: {prev_best:.4f}")
print(f"Improvement: {(current_best - prev_best) / prev_best * 100:+.1f}%")
print(f"Target Range: 0.87 - 1.00")
print(f"\nStatus: {'✓ EXCEEDED PREVIOUS BEST' if target_achieved else '✗ BELOW PREVIOUS BEST'}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("=" * 70)
