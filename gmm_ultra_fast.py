#!/usr/bin/env python3
"""
GMM OPTIMIZATION - ULTRA FAST VERSION
Target: Maximum Silhouette Score
Output: output_v2/
"""

import os
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

# Setup output directory
OUTPUT_DIR = 'output_v2'
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/models', f'{OUTPUT_DIR}/figures', 
          f'{OUTPUT_DIR}/metrics', f'{OUTPUT_DIR}/predictions', f'{OUTPUT_DIR}/reports']:
    os.makedirs(d, exist_ok=True)

print("GMM Optimization - Ultra Fast")
print("=" * 50)

# Load data
print("Loading data...")
df = pd.read_csv('data/raw/nhanes_health_data.csv')
for col in ['SEQN', 'respondent_id', 'cluster']:
    if col in df.columns:
        df = df.drop(columns=[col])
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.Categorical(df[col]).codes
X = df.select_dtypes(include=[np.number]).values.astype(np.float64)
print(f"Data: {X.shape}")

# Preprocessing
print("Preprocessing...")
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X)
pt = PowerTransformer(method='yeo-johnson')
X = pt.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"Preprocessed: {X.shape}")

# Aggressive outlier removal
print("Outlier removal...")
iso = IsolationForest(n_estimators=50, contamination=0.25, random_state=42, n_jobs=-1)
mask = iso.fit_predict(X) != -1
X = X[mask]
print(f"Clean: {X.shape[0]} samples (removed {100*(1-X.shape[0]/5000):.1f}%)")

# Save clean data
np.save(f'{OUTPUT_DIR}/data/X_clean.npy', X)

# PCA
print("PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X)
print(f"PCA: {X_pca.shape[1]} components")
joblib.dump(pca, f'{OUTPUT_DIR}/models/pca.joblib')

# Quick GMM test
print("GMM testing...")

def quick_gmm(X, k, cov):
    try:
        gmm = GaussianMixture(n_components=k, covariance_type=cov, 
                             n_init=2, max_iter=100, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        if len(np.unique(labels)) < 2:
            return None
        return {'k': k, 'cov': cov, 'sil': silhouette_score(X, labels), 
                'bic': gmm.bic(X), 'model': gmm, 'labels': labels}
    except:
        return None

results = []
# Test multiple configurations for best silhouette
for data_name, X_test in [('clean', X), ('pca', X_pca)]:
    for k in [2, 3, 4]:
        for cov in ['spherical', 'tied', 'diag']:
            r = quick_gmm(X_test, k, cov)
            if r:
                r['data'] = data_name
                results.append(r)

results.sort(key=lambda x: x['sil'], reverse=True)
print(f"Best: {results[0]['sil']:.4f} (k={results[0]['k']}, {results[0]['cov']}, {results[0]['data']})")

# Save results
best = results[0]
joblib.dump(best['model'], f'{OUTPUT_DIR}/models/best_gmm.joblib')

pd.DataFrame({'sample_id': range(len(best['labels'])), 
              'cluster': best['labels']}).to_csv(
    f'{OUTPUT_DIR}/predictions/assignments.csv', index=False)

json.dump({
    'best_silhouette': float(best['sil']),
    'best_k': best['k'],
    'best_covariance': best['cov'],
    'samples': X.shape[0],
    'features': X.shape[1]
}, open(f'{OUTPUT_DIR}/metrics/results.json', 'w'), indent=2)

# Visualization
print("Visualization...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Silhouette
ax1 = axes[0]
sil_samples = silhouette_samples(X, best['labels'])
y_lower = 10
for i in range(best['k']):
    cs = sil_samples[best['labels'] == i]
    cs.sort()
    ax1.fill_betweenx(np.arange(y_lower, y_lower + len(cs)), 0, cs, alpha=0.7)
    y_lower += len(cs) + 10
ax1.axvline(x=best['sil'], color='red', linestyle='--')
ax1.set_title(f'Silhouette: {best["sil"]:.4f}')
ax1.set_xlabel('Score')

# Clusters
ax2 = axes[1]
pca_vis = PCA(n_components=2, random_state=42).fit_transform(X)
ax2.scatter(pca_vis[:, 0], pca_vis[:, 1], c=best['labels'], cmap='viridis', s=5, alpha=0.5)
ax2.set_title('Clusters (PCA)')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

# Summary
ax3 = axes[2]
ax3.axis('off')
ax3.text(0.1, 0.9, f"""OPTIMIZATION RESULTS
{'='*40}

Best Silhouette: {best['sil']:.4f}
Target: 0.87 - 1.00

Configuration:
  k = {best['k']}
  Covariance = {best['cov']}
  Data = {best['data']}

Samples: {X.shape[0]}
Features: {X.shape[1]}

Status: {'✓ ACHIEVED' if best['sil'] >= 0.87 else '✗ BELOW TARGET'}
""", transform=ax3.transAxes, fontsize=11, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/results.png', dpi=150, bbox_inches='tight')

# Report
report = f"""# GMM Optimization Results

## Summary

| Metric | Value |
|--------|-------|
| Best Silhouette | {best['sil']:.4f} |
| Target | 0.87 - 1.00 |
| Status | {'✓ ACHIEVED' if best['sil'] >= 0.87 else '✗ BELOW TARGET'} |
| k | {best['k']} |
| Covariance | {best['cov']} |

## Analysis

Achieving 0.87-1.00 requires:
- Perfect cluster separation
- No overlap between clusters
- Clean, well-structured data

Health data typically exhibits:
- Continuous phenotype boundaries
- Individual variation
- Measurement noise
- Overlapping characteristics

## Realistic Expectations

| Score | Interpretation |
|-------|----------------|
| 0.71-1.00 | Excellent (rare in real data) |
| 0.51-0.70 | Good |
| 0.26-0.50 | Weak (typical for health) |
| < 0.25 | No structure |

---
Generated: {datetime.datetime.now()}
"""
open(f'{OUTPUT_DIR}/reports/report.md', 'w').write(report)

print("=" * 50)
print(f"Best Silhouette: {best['sil']:.4f}")
print(f"Target: 0.87 - 1.00")
print(f"Status: {'✓ ACHIEVED' if best['sil'] >= 0.87 else '✗ BELOW TARGET'}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 50)
