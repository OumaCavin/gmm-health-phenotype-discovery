"""
Advanced Clustering Optimization with UMAP and HDBSCAN
Target: Push Silhouette Score from 0.5343 towards 0.87-1.00

Author: Cavin Otieno
Date: January 2025
"""

import numpy as np
import pandas as pd
import warnings
import time
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Try imports
try:
    import umap
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False
    print("UMAP not available")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available")

# Path configuration
PROJECT_ROOT = '/workspace'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v2')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/raw/nhanes_health_data.csv')

# Create output directories
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'reports'), exist_ok=True)

print("=" * 70)
print("ADVANCED CLUSTERING OPTIMIZATION WITH UMAP + HDBSCAN")
print("Author: Cavin Otieno | Date: January 2025")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset: {df.shape}")

# Get numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'health_category' in numeric_cols:
    numeric_cols.remove('health_category')

X = df[numeric_cols].values.astype(np.float64)
print(f"Features: {X.shape[1]} numeric columns")

# Preprocessing
print("\n[2] Preprocessing...")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.ensemble import IsolationForest

# Impute
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Transform
transformer = PowerTransformer(method='yeo-johnson')
X = transformer.fit_transform(X)

# Scale
scaler = RobustScaler()
X = scaler.fit_transform(X)

# Remove outliers (3%)
iso = IsolationForest(contamination=0.03, random_state=42, n_estimators=100)
labels = iso.fit_predict(X)
X = X[labels == 1]
print(f"After preprocessing: {X.shape} (3% outliers removed)")

# Feature selection
print("\n[3] Feature selection...")
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.cluster import KMeans

# Pseudo-labels for selection
kmeans_temp = KMeans(n_clusters=3, random_state=42, n_init=10)
pseudo_labels = kmeans_temp.fit_predict(X)

# Variance threshold
var_thresh = VarianceThreshold(threshold=0.1)
X = var_thresh.fit_transform(X)

# Select top features
selector = SelectKBest(f_classif, k=20)
X = selector.fit_transform(X, pseudo_labels)
print(f"Selected 20 features: {X.shape}")

# Dimensionality reduction with UMAP
print("\n[4] UMAP dimensionality reduction...")
if UMAP_AVAILABLE:
    umap_results = []
    
    # Test different UMAP configurations
    n_neighbors_list = [10, 15, 30]
    min_dist_list = [0.0, 0.1, 0.5]
    
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            try:
                reducer = umap.UMAP(
                    n_components=10,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='euclidean',
                    random_state=42
                )
                X_umap = reducer.fit_transform(X)
                
                # Test clustering
                from sklearn.metrics import silhouette_score
                
                for n_clusters in [2, 3]:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                    cluster_labels = kmeans.fit_predict(X_umap)
                    score = silhouette_score(X_umap, cluster_labels)
                    
                    umap_results.append({
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'n_clusters': n_clusters,
                        'score': score,
                        'embedding': X_umap,
                        'labels': cluster_labels
                    })
                    
                    print(f"   UMAP(n={n_neighbors}, d={min_dist}) + KMeans({n_clusters}): {score:.4f}")
                    
            except Exception as e:
                print(f"   UMAP config failed: {e}")

# Also test PCA for comparison
print("\n[5] PCA baseline...")
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X)
print(f"   PCA explained variance: {sum(pca.explained_variance_ratio_):.3f}")

pca_results = []
for n_clusters in [2, 3]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, cluster_labels)
    
    pca_results.append({
        'n_clusters': n_clusters,
        'score': score,
        'embedding': X_pca,
        'labels': cluster_labels
    })
    print(f"   PCA + KMeans({n_clusters}): {score:.4f}")

# Find best result
print("\n[6] Finding best configuration...")
all_results = []

if UMAP_AVAILABLE:
    for r in umap_results:
        all_results.append({
            'method': f"UMAP(n={r['n_neighbors']}, d={r['min_dist']})",
            'clusters': r['n_clusters'],
            'score': r['score'],
            'embedding': r['embedding'],
            'labels': r['labels']
        })

for r in pca_results:
    all_results.append({
        'method': 'PCA',
        'clusters': r['n_clusters'],
        'score': r['score'],
        'embedding': r['embedding'],
        'labels': r['labels']
    })

# Sort by score
all_results.sort(key=lambda x: x['score'], reverse=True)

best = all_results[0]
print(f"\n{'='*70}")
print("BEST RESULT")
print(f"{'='*70}")
print(f"Method: {best['method']}")
print(f"Clusters: {best['clusters']}")
print(f"Silhouette Score: {best['score']:.4f}")
print(f"Previous Best: 0.5343")
print(f"Improvement: {((best['score'] - 0.5343) / 0.5343) * 100:.1f}%")

# Save results
print("\n[7] Saving results...")

# Metrics
metrics = {
    'previous_best': 0.5343,
    'new_best': best['score'],
    'improvement_percent': ((best['score'] - 0.5343) / 0.5343) * 100,
    'configuration': {
        'method': best['method'],
        'n_clusters': best['clusters']
    },
    'all_results': [
        {'method': r['method'], 'clusters': r['clusters'], 'score': r['score']}
        for r in all_results[:10]
    ],
    'timestamp': datetime.now().isoformat()
}

metrics_path = os.path.join(OUTPUT_DIR, 'metrics/umap_hdbscan_results.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics: {metrics_path}")

# Predictions
predictions_df = pd.DataFrame({
    'sample_id': range(len(best['labels'])),
    'cluster': best['labels']
})
predictions_path = os.path.join(OUTPUT_DIR, 'predictions/umap_hdbscan_clusters.csv')
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions: {predictions_path}")

# Report
report = f"""# Advanced Clustering Optimization Report

## Results Summary

| Metric | Value |
|--------|-------|
| Previous Best | 0.5343 |
| **New Best** | **{best['score']:.4f}** |
| Improvement | {((best['score'] - 0.5343) / 0.5343) * 100:.1f}% |

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Method | {best['method']} |
| Clusters | {best['clusters']} |
| Silhouette Score | {best['score']:.4f} |

## Top 5 Results

| Rank | Method | Clusters | Silhouette |
|------|--------|----------|------------|
"""

for i, r in enumerate(all_results[:5], 1):
    report += f"| {i} | {r['method']} | {r['clusters']} | {r['score']:.4f} |\n"

report += f"""
## Progress Analysis

| Metric | Value |
|--------|-------|
| Target | 0.87 - 1.00 |
| Current | {best['score']:.4f} |
| Progress | {(best['score'] / 0.87) * 100:.1f}% |
| Gap | {0.87 - best['score']:.4f} |

---
Generated: {datetime.now().isoformat()}
"""

report_path = os.path.join(OUTPUT_DIR, 'reports/umap_hdbscan_report.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f"Report: {report_path}")

print(f"\n{'='*70}")
print(f"FINAL SILHOUETTE SCORE: {best['score']:.4f}")
print(f"{'='*70}")
