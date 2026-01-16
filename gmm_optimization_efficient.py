#!/usr/bin/env python3
"""
===============================================================================
GMM OPTIMIZATION PIPELINE - EFFICIENT VERSION
===============================================================================

Target: Achieve maximum Silhouette Score
Current Baseline: ~0.0609
Output Directory: output_v2/

This is an optimized version that focuses on the most promising configurations
while implementing all recommended strategies for maximizing Silhouette Score.
===============================================================================
"""

import os
import sys
import json
import time
import warnings
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    silhouette_score, silhouette_samples, calinski_harabasz_score,
    davies_bouldin_score
)
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'output_v2'
DATA_PATH = 'data/raw/nhanes_health_data.csv'

# Directory setup
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'cluster_profiles'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'reports'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)

print("=" * 70)
print("GMM OPTIMIZATION PIPELINE - EFFICIENT VERSION")
print("Target: Maximum Silhouette Score")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 70)

# =============================================================================
# STEP 1: DATA LOADING AND PREPARATION
# =============================================================================

print("\n[STEP 1] Loading data...")

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Remove excluded columns
exclude_cols = ['SEQN', 'cluster', 'cluster_label', 'Cluster', 'Cluster_Assignment', 'respondent_id']
for col in exclude_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Convert categorical to numeric
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = pd.Categorical(df[col]).codes

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric features: {len(numeric_cols)}")

X = df[numeric_cols].values.astype(np.float64)
print(f"Data matrix shape: {X.shape}")

# =============================================================================
# STEP 2: ADVANCED PREPROCESSING
# =============================================================================

print("\n[STEP 2] Advanced Preprocessing...")

# KNN Imputation
imputer = KNNImputer(n_neighbors=5, weights='distance')
X = imputer.fit_transform(X)
print(f"KNN Imputation completed: {X.shape}")

# Remove constant features
selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)
print(f"After variance threshold: {X.shape}")

# PowerTransformer (Yeo-Johnson)
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X = pt.fit_transform(X)
print(f"PowerTransformer applied")

# StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"StandardScaler applied")

# =============================================================================
# STEP 3: AGGRESSIVE OUTLIER REMOVAL
# =============================================================================

print("\n[STEP 3] Aggressive Outlier Removal...")

original_size = X.shape[0]
print(f"Original samples: {original_size}")

outlier_masks = []

# Isolation Forest with increasing contamination
for contamination in [0.10, 0.15, 0.20]:
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X)
    outlier_masks.append(labels != -1)
    removed = (~(labels != -1)).sum()
    print(f"Isolation Forest (contamination={contamination}): Removed {removed} ({100*removed/original_size:.1f}%)")

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.15)
labels = lof.fit_predict(X)
outlier_masks.append(labels != -1)
removed = (~(labels != -1)).sum()
print(f"LOF: Removed {removed} ({100*removed/original_size:.1f}%)")

# Mahalanobis Distance
mean = np.mean(X, axis=0)
cov = np.cov(X.T)
try:
    cov_inv = np.linalg.pinv(cov)
    diff = X - mean
    mahal_dist = np.sum(diff @ cov_inv * diff, axis=1)
    threshold = chi2.ppf(0.975, X.shape[1])
    mask = mahal_dist < threshold
    outlier_masks.append(mask)
    removed = (~mask).sum()
    print(f"Mahalanobis: Removed {removed} ({100*removed/original_size:.1f}%)")
except:
    print("Mahalanobis: Skipped (singular covariance)")

# Consensus approach
final_mask = np.all(outlier_masks, axis=0)
X_clean = X[final_mask]
removed = original_size - X_clean.shape[0]
print(f"Final: Removed {removed} ({100*removed/original_size:.1f}%)")
print(f"Clean samples: {X_clean.shape[0]}")

# Save cleaned data
np.save(os.path.join(OUTPUT_DIR, 'data', 'X_clean.npy'), X_clean)

# =============================================================================
# STEP 4: DIMENSIONALITY REDUCTION
# =============================================================================

print("\n[STEP 4] Dimensionality Reduction...")

reductions = {}

# PCA - different variance thresholds
for var_threshold in [0.90, 0.95, 0.99]:
    pca = PCA(n_components=var_threshold, random_state=42)
    X_pca = pca.fit_transform(X_clean)
    explained = pca.explained_variance_ratio_.sum()
    reductions[f'pca_{int(var_threshold*100)}'] = X_pca
    print(f"PCA ({int(var_threshold*100)}%): {X_pca.shape[1]} components")

# Save PCA model
pca_full = PCA(n_components=0.95, random_state=42)
pca_full.fit(X_clean)
joblib.dump(pca_full, os.path.join(OUTPUT_DIR, 'models', 'pca_95.joblib'))

# t-SNE for visualization
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(X_clean)
reductions['tsne_2d'] = X_tsne
print(f"t-SNE: {X_tsne.shape}")

# =============================================================================
# STEP 5: COMPREHENSIVE GMM TUNING
# =============================================================================

print("\n[STEP 5] GMM Tuning...")

def evaluate_gmm(X, k, cov_type, reg_covar=1e-4):
    """Evaluate GMM configuration."""
    try:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            reg_covar=reg_covar,
            n_init=10,
            max_iter=300,
            random_state=42
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        
        if len(np.unique(labels)) < 2:
            return None
            
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        
        return {
            'k': k,
            'cov_type': cov_type,
            'reg_covar': reg_covar,
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies,
            'bic': gmm.bic(X),
            'aic': gmm.aic(X),
            'model': gmm
        }
    except Exception as e:
        return None

# Test different configurations
k_values = [2, 3, 4]
cov_types = ['spherical', 'tied', 'diag']
reg_values = [1e-6, 1e-4, 1e-2]

all_results = []

# Test on cleaned data
print("Testing on cleaned data...")
for k in k_values:
    for cov in cov_types:
        for reg in reg_values:
            result = evaluate_gmm(X_clean, k, cov, reg)
            if result:
                result['data'] = 'cleaned'
                all_results.append(result)

# Test on PCA-reduced data
print("Testing on PCA-reduced data...")
X_pca = reductions['pca_95']
for k in k_values:
    for cov in cov_types:
        for reg in reg_values:
            result = evaluate_gmm(X_pca, k, cov, reg)
            if result:
                result['data'] = 'pca_95'
                all_results.append(result)

# Sort by silhouette score
all_results.sort(key=lambda x: x['silhouette'], reverse=True)

print(f"\nTotal configurations tested: {len(all_results)}")
if all_results:
    print(f"Best Silhouette Score: {all_results[0]['silhouette']:.4f}")
    print(f"Best Configuration: k={all_results[0]['k']}, covariance={all_results[0]['cov_type']}")

# =============================================================================
# STEP 6: ALTERNATIVE ALGORITHMS
# =============================================================================

print("\n[STEP 6] Testing Alternative Algorithms...")

alt_results = []

# K-Means
print("Testing K-Means...")
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(X_clean)
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X_clean, labels)
        alt_results.append({'algorithm': 'kmeans', 'k': k, 'silhouette': sil, 'model': kmeans})

# Agglomerative
print("Testing Agglomerative...")
for k in [2, 3, 4]:
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X_clean)
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X_clean, labels)
        alt_results.append({'algorithm': 'agglomerative', 'k': k, 'silhouette': sil})

# Sort alternative results
alt_results.sort(key=lambda x: x['silhouette'], reverse=True)
print(f"Best Alternative: {alt_results[0]['algorithm']} (k={alt_results[0]['k']}, silhouette={alt_results[0]['silhouette']:.4f})")

# =============================================================================
# STEP 7: ITERATIVE OPTIMIZATION - ULTRA AGGRESSIVE
# =============================================================================

print("\n[STEP 7] Iterative Optimization - Ultra Aggressive...")

# Try k=2 with spherical covariance (usually highest silhouette)
best_score = 0
best_result = None

for k in [2]:
    for cov in ['spherical']:
        for reg in [1e-6, 1e-5, 1e-4]:
            result = evaluate_gmm(X_clean, k, cov, reg)
            if result and result['silhouette'] > best_score:
                best_score = result['silhouette']
                best_result = result

print(f"Ultra-aggressive best: {best_score:.4f}")

# =============================================================================
# STEP 8: SAVE ALL RESULTS
# =============================================================================

print("\n[STEP 8] Saving Results...")

# Determine overall best
all_results.extend(alt_results)
if best_result:
    all_results.append(best_result)

all_results.sort(key=lambda x: x['silhouette'], reverse=True)
best = all_results[0]

# Save best model
if 'model' in best:
    joblib.dump(best['model'], os.path.join(OUTPUT_DIR, 'models', 'best_gmm_model.joblib'))

# Save cluster assignments
if 'labels' in best or 'model' in best:
    if 'labels' in best:
        labels = best['labels']
    else:
        labels = best['model'].predict(X_clean)
    
    assignments = pd.DataFrame({
        'sample_id': range(len(labels)),
        'cluster': labels
    })
    assignments.to_csv(os.path.join(OUTPUT_DIR, 'predictions', 'best_cluster_assignments.csv'), index=False)
    print(f"Saved cluster assignments: {len(assignments)} samples")

# Save metrics
metrics = {
    'best_silhouette': float(best['silhouette']),
    'best_k': best['k'],
    'best_covariance_type': best['cov_type'],
    'best_data': best.get('data', 'cleaned'),
    'calinski_harabasz': float(best.get('calinski', 0)),
    'davies_bouldin': float(best.get('davies', 0)),
    'total_configurations_tested': len(all_results),
    'samples_after_outlier_removal': X_clean.shape[0],
    'features_after_preprocessing': X_clean.shape[1],
    'timestamp': datetime.datetime.now().isoformat(),
    'target_achieved': best['silhouette'] >= 0.87
}

with open(os.path.join(OUTPUT_DIR, 'metrics', 'optimization_results.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics: {metrics}")

# Save all results
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in all_results])
results_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics', 'all_gmm_results.csv'), index=False)
print(f"Saved all results: {len(results_df)} configurations")

# =============================================================================
# STEP 9: GENERATE VISUALIZATIONS
# =============================================================================

print("\n[STEP 9] Generating Visualizations...")

# Get labels from best model
if 'labels' in best:
    labels = best['labels']
else:
    labels = best['model'].predict(X_clean)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# 1. Silhouette Plot
ax1 = fig.add_subplot(2, 3, 1)
sample_sil = silhouette_samples(X_clean, labels)
y_lower = 10
n_clusters = len(np.unique(labels))

for i in range(n_clusters):
    cluster_sil = sample_sil[labels == i]
    cluster_sil.sort()
    size_i = cluster_sil.shape[0]
    y_upper = y_lower + size_i
    color = plt.cm.viridis(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_i, f'Cluster {i}')
    y_lower = y_upper + 10

ax1.axvline(x=best['silhouette'], color='red', linestyle='--', linewidth=2,
           label=f'Mean: {best["silhouette"]:.4f}')
ax1.set_xlabel('Silhouette Coefficient')
ax1.set_ylabel('Cluster')
ax1.set_title('Silhouette Analysis')
ax1.legend()

# 2. PCA Projection
ax2 = fig.add_subplot(2, 3, 2)
scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, 
                     cmap='viridis', alpha=0.6, s=15)
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
ax2.set_title('t-SNE Cluster Visualization')
plt.colorbar(scatter, ax=ax2, label='Cluster')

# 3. Cluster Sizes
ax3 = fig.add_subplot(2, 3, 3)
unique, counts = np.unique(labels, return_counts=True)
bars = ax3.bar(unique, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique))))
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Count')
ax3.set_title('Cluster Size Distribution')
for bar, count in zip(bars, counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(count), ha='center', va='bottom')

# 4. Score Summary
ax4 = fig.add_subplot(2, 3, 4)
ax4.axis('off')
summary = f"""
CLUSTERING RESULTS SUMMARY
{'='*40}

Best Silhouette Score: {best['silhouette']:.4f}
Target Range: 0.87 - 1.00

Configuration:
  - Clusters (k): {best['k']}
  - Covariance: {best['cov_type']}
  - Data Type: {best.get('data', 'cleaned')}

Performance Metrics:
  - Silhouette: {best['silhouette']:.4f}
  - Calinski-Harabasz: {best.get('calinski', 0):.2f}
  - Davies-Bouldin: {best.get('davies', 0):.4f}

Data Info:
  - Samples: {X_clean.shape[0]}
  - Features: {X_clean.shape[1]}
  - Outliers Removed: {100*(1 - X_clean.shape[0]/original_size):.1f}%

{'✓ TARGET ACHIEVED' if best['silhouette'] >= 0.87 else '✗ BELOW TARGET (Expected for real health data)'}
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# 5. Top 10 Configurations
ax5 = fig.add_subplot(2, 3, 5)
top_10 = all_results[:10]
configs = [f"k={r['k']}_{r['cov_type']}" for r in top_10]
scores = [r['silhouette'] for r in top_10]
colors = ['green' if s >= 0.87 else 'orange' if s >= 0.5 else 'red' for s in scores]
ax5.barh(range(len(configs)), scores, color=colors)
ax5.set_yticks(range(len(configs)))
ax5.set_yticklabels(configs)
ax5.set_xlabel('Silhouette Score')
ax5.set_title('Top 10 Configurations')
ax5.axvline(x=0.87, color='green', linestyle='--', alpha=0.5, label='Target (0.87)')
ax5.legend()

# 6. Score Distribution
ax6 = fig.add_subplot(2, 3, 6)
all_scores = [r['silhouette'] for r in all_results]
ax6.hist(all_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax6.axvline(x=best['silhouette'], color='red', linestyle='--', linewidth=2,
           label=f'Best: {best["silhouette"]:.4f}')
ax6.axvline(x=0.87, color='green', linestyle='--', linewidth=2, label='Target: 0.87')
ax6.set_xlabel('Silhouette Score')
ax6.set_ylabel('Frequency')
ax6.set_title('Score Distribution')
ax6.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'optimization_results.png'), 
           dpi=150, bbox_inches='tight')
print("Saved visualization: optimization_results.png")

# =============================================================================
# STEP 10: GENERATE FINAL REPORT
# =============================================================================

print("\n[STEP 10] Generating Report...")

target_achieved = best['silhouette'] >= 0.87

report = f"""# GMM Optimization Report

## Executive Summary

**Target Silhouette Score**: 0.87 - 1.00
**Best Achieved Score**: {best['silhouette']:.4f}
**Status**: {'✓ TARGET ACHIEVED' if target_achieved else '✗ TARGET NOT REACHED'}

## Performance Analysis

### Why 0.87-1.00 Is Extremely Challenging

The theoretical maximum Silhouette score of 1.0 requires:

1. **Perfect Cluster Separation**: Clusters must be completely disjoint with no overlap
2. **Compact Clusters**: All points within a cluster must be tightly grouped around the centroid
3. **Spherical Geometry**: Clusters must be spherical and evenly distributed
4. **No Noise**: Data must be clean without measurement error or artifacts

### Realistic Expectations for Health Data (NHANES)

Health phenotype data typically exhibits:

- **Continuous Phenotype Boundaries**: Health conditions exist on spectrums, not discrete categories
- **Individual Variation**: Natural biological variation creates overlap between phenotypes
- **Measurement Noise**: Clinical measurements have inherent uncertainty
- **Multi-morbidity**: Individuals often have characteristics of multiple phenotypes

### Benchmark Silhouette Scores

| Score Range | Interpretation | Typical Achievable? |
|-------------|----------------|---------------------|
| 0.87 - 1.00 | Excellent separation | Very rare in real data |
| 0.51 - 0.70 | Good structure | Possible with clean synthetic data |
| 0.26 - 0.50 | Weak structure | Common for health phenotypes |
| < 0.25 | No structure | Typical for complex health data |

## Results

### Best Configuration

| Parameter | Value |
|-----------|-------|
| Clusters (k) | {best['k']} |
| Covariance Type | {best['cov_type']} |
| Data Source | {best.get('data', 'cleaned')} |
| Silhouette Score | {best['silhouette']:.4f} |
| Calinski-Harabasz | {best.get('calinski', 0):.2f} |
| Davies-Bouldin | {best.get('davies', 0):.4f} |

### Data Processing Summary

| Stage | Samples | Features |
|-------|---------|----------|
| Original | {original_size} | {X.shape[1]} |
| After Preprocessing | {X_clean.shape[0]} | {X_clean.shape[1]} |
| Outliers Removed | {100*(1 - X_clean.shape[0]/original_size):.1f}% | - |

### Configurations Tested

| Category | Count |
|----------|-------|
| GMM Configurations | {len([r for r in all_results if 'model' in r or 'cov_type' in r])} |
| Alternative Algorithms | {len(alt_results)} |
| Total | {len(all_results)} |

## Recommendations for Further Improvement

### 1. Feature Engineering
Create derived features that better capture phenotype differences:
- Metabolic syndrome indicators
- Cardiovascular risk scores
- Inflammatory markers

### 2. Semi-supervised Clustering
If partial phenotype labels exist, use semi-supervised approaches to guide clustering.

### 3. Different Problem Formulation
Consider:
- Soft clustering with probability thresholds
- Hierarchical clustering at multiple resolutions
- Consensus clustering from multiple algorithms

### 4. Alternative Evaluation Metrics
For health phenotypes, consider:
- Clinical meaningfulness over statistical separation
- Biological interpretability of clusters
- External validation with known phenotypes

## Files Generated

- `models/best_gmm_model.joblib`: Best performing model
- `metrics/optimization_results.json`: Complete metrics
- `metrics/all_gmm_results.csv`: All configurations tested
- `predictions/best_cluster_assignments.csv`: Cluster labels
- `figures/optimization_results.png`: Visualization

---

*Report generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

with open(os.path.join(OUTPUT_DIR, 'reports', 'optimization_report.md'), 'w') as f:
    f.write(report)
print("Saved report: optimization_report.md")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)
print(f"Best Silhouette Score: {best['silhouette']:.4f}")
print(f"Target Range: 0.87 - 1.00")
print(f"Status: {'✓ TARGET ACHIEVED' if target_achieved else '✗ BELOW TARGET'}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("=" * 70)
