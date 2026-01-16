#!/usr/bin/env python3
"""
===============================================================================
GMM OPTIMIZATION - MINIMAL EFFICIENT VERSION
===============================================================================
Target: Maximum Silhouette Score
All outputs to: output_v2/
"""

import os
import sys
import json
import time
import warnings
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

# Setup directories
OUTPUT_DIR = 'output_v2'
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/models', f'{OUTPUT_DIR}/figures', 
          f'{OUTPUT_DIR}/metrics', f'{OUTPUT_DIR}/predictions', 
          f'{OUTPUT_DIR}/reports']:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("GMM OPTIMIZATION - MINIMAL VERSION")
print("=" * 60)

# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

print("\n[1] Loading data...")
df = pd.read_csv('data/raw/nhanes_health_data.csv')
print(f"Original shape: {df.shape}")

# Remove ID columns
for col in ['SEQN', 'respondent_id', 'cluster', 'cluster_label']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Convert categorical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.Categorical(df[col]).codes

X = df.select_dtypes(include=[np.number]).values.astype(np.float64)
print(f"Data matrix: {X.shape}")

# =============================================================================
# STEP 2: PREPROCESSING
# =============================================================================

print("\n[2] Preprocessing...")

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X)

# Variance threshold
selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)

# PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
X = pt.fit_transform(X)

# Scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"After preprocessing: {X.shape}")

# =============================================================================
# STEP 3: OUTLIER REMOVAL
# =============================================================================

print("\n[3] Outlier removal...")
original_size = X.shape[0]

# Isolation Forest
iso = IsolationForest(n_estimators=100, contamination=0.15, random_state=42, n_jobs=-1)
mask = iso.fit_predict(X) != -1

# LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
mask &= (lof.fit_predict(X) != -1)

X = X[mask]
print(f"After outlier removal: {X.shape[0]} (removed {100*(1-X.shape[0]/original_size):.1f}%)")

# Save clean data
np.save(f'{OUTPUT_DIR}/data/X_clean.npy', X)

# =============================================================================
# STEP 4: DIMENSIONITY REDUCTION
# =============================================================================

print("\n[4] Dimensionality reduction...")

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X)
print(f"PCA: {X_pca.shape[1]} components ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

joblib.dump(pca, f'{OUTPUT_DIR}/models/pca.joblib')

# =============================================================================
# STEP 5: GMM TUNING (FOCUSED)
# =============================================================================

print("\n[5] GMM tuning...")

def test_gmm(X, k, cov_type):
    """Quick GMM test."""
    try:
        gmm = GaussianMixture(n_components=k, covariance_type=cov_type, 
                             n_init=5, max_iter=200, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        if len(np.unique(labels)) < 2:
            return None
        return {
            'k': k, 'cov': cov_type, 'silhouette': silhouette_score(X, labels),
            'bic': gmm.bic(X), 'model': gmm, 'labels': labels
        }
    except:
        return None

results = []

# Test focused configurations
print("Testing configurations...")
for k in [2, 3, 4]:
    for cov in ['spherical', 'tied']:
        # Test on cleaned data
        r = test_gmm(X, k, cov)
        if r: results.append({**r, 'data': 'cleaned'})
        
        # Test on PCA data
        r = test_gmm(X_pca, k, cov)
        if r: results.append({**r, 'data': 'pca'})

# Sort by silhouette
results.sort(key=lambda x: x['silhouette'], reverse=True)

print(f"\nTested {len(results)} configurations")
if results:
    print(f"Best Silhouette: {results[0]['silhouette']:.4f} (k={results[0]['k']}, {results[0]['cov']})")

# =============================================================================
# STEP 6: SAVE RESULTS
# =============================================================================

print("\n[6] Saving results...")

best = results[0] if results else None

if best:
    # Save model
    joblib.dump(best['model'], f'{OUTPUT_DIR}/models/best_gmm.joblib')
    
    # Save cluster assignments
    assignments = pd.DataFrame({
        'sample_id': range(len(best['labels'])),
        'cluster': best['labels']
    })
    assignments.to_csv(f'{OUTPUT_DIR}/predictions/cluster_assignments.csv', index=False)
    
    # Save metrics
    metrics = {
        'best_silhouette': float(best['silhouette']),
        'best_k': best['k'],
        'best_covariance': best['cov'],
        'best_data': best['data'],
        'bic': float(best['bic']),
        'samples': X.shape[0],
        'features': X.shape[1],
        'timestamp': datetime.datetime.now().isoformat()
    }
    with open(f'{OUTPUT_DIR}/metrics/results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save all results
    pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'labels']} 
                  for r in results]).to_csv(f'{OUTPUT_DIR}/metrics/all_results.csv', index=False)

# =============================================================================
# STEP 7: VISUALIZATION
# =============================================================================

print("\n[7] Generating visualization...")

if best:
    labels = best['labels']
    
    # Quick t-SNE for visualization only (on subset if needed)
    n_samples = X.shape[0]
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
               max_iter=500, learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(X)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Silhouette plot
    ax1 = axes[0, 0]
    sil_samples = silhouette_samples(X, labels)
    y_lower = 10
    for i in range(best['k']):
        cluster_sil = sil_samples[labels == i]
        cluster_sil.sort()
        size = cluster_sil.shape[0]
        ax1.fill_betweenx(np.arange(y_lower, y_lower + size), 0, cluster_sil,
                          alpha=0.7)
        ax1.text(-0.05, y_lower + size/2, f'Cluster {i}')
        y_lower += size + 10
    ax1.axvline(x=best['silhouette'], color='red', linestyle='--')
    ax1.set_title(f'Silhouette Plot (Score: {best["silhouette"]:.4f})')
    ax1.set_xlabel('Silhouette Coefficient')
    
    # t-SNE
    ax2 = axes[0, 1]
    scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    ax2.set_title('t-SNE Visualization')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    # Cluster sizes
    ax3 = axes[0, 2]
    unique, counts = np.unique(labels, return_counts=True)
    ax3.bar(unique, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique))))
    ax3.set_title('Cluster Sizes')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Count')
    
    # Score comparison
    ax4 = axes[1, 0]
    top_5 = results[:5]
    ax4.barh([f"k={r['k']}_{r['cov']}" for r in top_5], 
             [r['silhouette'] for r in top_5])
    ax4.axvline(x=0.87, color='green', linestyle='--', label='Target (0.87)')
    ax4.set_title('Top 5 Configurations')
    ax4.set_xlabel('Silhouette Score')
    ax4.legend()
    
    # Summary text
    ax5 = axes[1, 1]
    ax5.axis('off')
    summary = f"""OPTIMIZATION SUMMARY
{'='*40}

Best Silhouette Score: {best['silhouette']:.4f}
Target Range: 0.87 - 1.00

Configuration:
  k = {best['k']}
  Covariance = {best['cov']}
  Data = {best['data']}

Data Info:
  Samples: {X.shape[0]}
  Features: {X.shape[1]}
  
Status: {'✓ ACHIEVED' if best['silhouette'] >= 0.87 else '✗ BELOW TARGET'}
"""
    ax5.text(0.1, 0.9, summary, transform=ax5.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Score distribution
    ax6 = axes[1, 2]
    scores = [r['silhouette'] for r in results]
    ax6.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    ax6.axvline(x=best['silhouette'], color='red', linestyle='--', linewidth=2)
    ax6.set_title('Score Distribution')
    ax6.set_xlabel('Silhouette Score')
    ax6.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/optimization_results.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/figures/optimization_results.png")

# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n[8] Generating report...")

target = 0.87
achieved = best['silhouette'] if best else 0

report = f"""# GMM Optimization Results

## Summary

| Metric | Value |
|--------|-------|
| Best Silhouette Score | {achieved:.4f} |
| Target Range | 0.87 - 1.00 |
| Status | {'✓ TARGET ACHIEVED' if achieved >= target else '✗ BELOW TARGET'} |
| Clusters (k) | {best['k'] if best else 'N/A'} |
| Covariance Type | {best['cov'] if best else 'N/A'} |

## Why 0.87-1.00 Is Challenging

Achieving Silhouette scores of 0.87-1.00 requires:
- Perfectly separated, spherical clusters
- No overlap between clusters
- No noise or measurement error

Real-world health data (like NHANES) typically exhibits:
- Continuous phenotype boundaries
- Individual biological variation
- Measurement noise and artifacts
- Overlapping clinical characteristics

## Benchmark Expectations

| Score Range | Interpretation |
|-------------|----------------|
| 0.71 - 1.00 | Strong structure (rare in real data) |
| 0.51 - 0.70 | Good structure |
| 0.26 - 0.50 | Weak structure (typical for health data) |
| < 0.25 | No substantial structure |

## Files Generated

- `models/best_gmm.joblib`: Best GMM model
- `metrics/results.json`: Optimization metrics
- `metrics/all_results.csv`: All configurations tested
- `predictions/cluster_assignments.csv`: Cluster labels
- `figures/optimization_results.png`: Visualization

---
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

with open(f'{OUTPUT_DIR}/reports/optimization_report.md', 'w') as f:
    f.write(report)

print("\n" + "=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)
print(f"Best Silhouette Score: {achieved:.4f}")
print(f"Target Range: 0.87 - 1.00")
print(f"Status: {'✓ ACHIEVED' if achieved >= target else '✗ BELOW TARGET'}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("=" * 60)
