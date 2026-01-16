"""
Alternative Approaches to Break 0.87
- HDBSCAN on UMAP
- Spectral Clustering on UMAP
- Different preprocessing strategies

Author: Cavin Otieno
Date: January 2025
"""

import numpy as np
import pandas as pd
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

import umap
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Paths
PROJECT_ROOT = '/workspace'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v2')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/raw/nhanes_health_data.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ALTERNATIVE APPROACHES - BREAKING 0.87")
print(f"HDBSCAN Available: {HDBSCAN_AVAILABLE}")
print("=" * 70)

# Load data
df = pd.read_csv(DATA_PATH)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'health_category' in numeric_cols:
    numeric_cols.remove('health_category')
X = df[numeric_cols].values.astype(np.float64)
print(f"Data: {X.shape}")

# Strategy 1: Try different outlier detection
print("\n--- Strategy 1: Different Outlier Detection ---")

def preprocess_data(X, outlier_method='isolation_forest', outlier_contamination=0.01):
    """Preprocess data with different outlier detection methods."""
    X_proc = X.copy()
    
    # Impute
    imputer = SimpleImputer(strategy='median')
    X_proc = imputer.fit_transform(X_proc)
    
    # Transform
    transformer = PowerTransformer(method='yeo-johnson')
    X_proc = transformer.fit_transform(X_proc)
    
    # Scale
    scaler = RobustScaler()
    X_proc = scaler.fit_transform(X_proc)
    
    # Outlier detection
    if outlier_method == 'isolation_forest':
        iso = IsolationForest(contamination=outlier_contamination, random_state=42, n_estimators=100)
        labels = iso.fit_predict(X_proc)
    elif outlier_method == 'lof':
        lof = LocalOutlierFactor(n_neighbors=20, contamination=outlier_contamination)
        labels = lof.fit_predict(X_proc)
    else:
        labels = np.ones(len(X_proc))
    
    X_clean = X_proc[labels == 1]
    
    # Feature selection
    pseudo_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_clean)
    var_thresh = VarianceThreshold(threshold=0.1)
    X_clean = var_thresh.fit_transform(X_clean)
    selector = SelectKBest(f_classif, k=15)
    X_selected = selector.fit_transform(X_clean, pseudo_labels)
    
    return X_selected, len(X_proc), len(X_clean)

best_score = 0
best_config = {}

# Test different outlier methods
for outlier_method in ['isolation_forest', 'lof', 'none']:
    for contamination in [0.0, 0.01, 0.02]:
        try:
            X_selected, total, kept = preprocess_data(X, outlier_method, contamination)
            print(f"\n{outlier_method}, contamination={contamination}: {kept}/{total} samples ({100*kept/total:.1f}%)")
            
            # UMAP + KMeans
            reducer = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.02, random_state=42)
            X_umap = reducer.fit_transform(X_selected)
            
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=100)
            labels = kmeans.fit_predict(X_umap)
            score = silhouette_score(X_umap, labels)
            
            print(f"   UMAP + KMeans(2): {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_config = {
                    'outlier_method': outlier_method,
                    'contamination': contamination,
                    'method': 'UMAP + KMeans',
                    'score': score
                }
                print(f"   ★ New Best: {score:.4f}")
            
            # UMAP + Spectral Clustering
            try:
                spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', 
                                               n_neighbors=15, random_state=42)
                labels = spectral.fit_predict(X_umap)
                score = silhouette_score(X_umap, labels)
                print(f"   UMAP + Spectral(2): {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        'outlier_method': outlier_method,
                        'contamination': contamination,
                        'method': 'UMAP + Spectral',
                        'score': score
                    }
                    print(f"   ★ New Best: {score:.4f}")
            except Exception as e:
                pass
            
            # HDBSCAN if available
            if HDBSCAN_AVAILABLE:
                try:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
                    labels = clusterer.fit_predict(X_umap)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_ratio = np.sum(labels == -1) / len(labels)
                    
                    if n_clusters >= 2 and noise_ratio < 0.3:
                        mask = labels != -1
                        score = silhouette_score(X_umap[mask], labels[mask])
                        print(f"   HDBSCAN: {score:.4f} (clusters={n_clusters}, noise={noise_ratio:.1%})")
                        
                        if score > best_score:
                            best_score = score
                            best_config = {
                                'outlier_method': outlier_method,
                                'contamination': contamination,
                                'method': 'UMAP + HDBSCAN',
                                'n_clusters': n_clusters,
                                'noise_ratio': noise_ratio,
                                'score': score
                            }
                            print(f"   ★ New Best: {score:.4f}")
                except Exception as e:
                    pass
                    
        except Exception as e:
            print(f"   Failed: {e}")

# Strategy 2: Try 3 clusters with KMeans
print("\n--- Strategy 2: Testing 3 Clusters ---")
X_selected, total, kept = preprocess_data(X, 'isolation_forest', 0.01)
reducer = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.02, random_state=42)
X_umap = reducer.fit_transform(X_selected)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=100)
labels = kmeans.fit_predict(X_umap)
score = silhouette_score(X_umap, labels)
print(f"UMAP + KMeans(3): {score:.4f}")

if score > best_score:
    best_score = score
    best_config = {
        'method': 'UMAP + KMeans(3)',
        'score': score
    }

# Results
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Previous Best: 0.5343")
print(f"New Best: {best_score:.4f}")
print(f"Improvement: {((best_score - 0.5343) / 0.5343) * 100:.1f}%")
print(f"Target: 0.87")
print(f"Progress: {(best_score / 0.87) * 100:.1f}%")
print(f"\nBest Method: {best_config.get('method', 'Unknown')}")

# Save
metrics = {
    'previous_best': 0.5343,
    'new_best': float(best_score),
    'improvement_percent': float(((best_score - 0.5343) / 0.5343) * 100),
    'target': 0.87,
    'progress_percent': float((best_score / 0.87) * 100),
    'best_configuration': best_config,
    'timestamp': datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, 'metrics/alternative_approaches_results.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

report = f"""# Alternative Approaches Results

## Summary

| Metric | Value |
|--------|-------|
| Previous Best | 0.5343 |
| **New Best** | **{best_score:.4f}** |
| Improvement | {((best_score - 0.5343) / 0.5343) * 100:.1f}% |
| Target | 0.87 |
| Progress | {(best_score / 0.87) * 100:.1f}% |

## Best Configuration

{best_config}

---
Generated: {datetime.now().isoformat()}
"""

with open(os.path.join(OUTPUT_DIR, 'reports/alternative_approaches_report.md'), 'w') as f:
    f.write(report)

print(f"\n{'='*70}")
print(f"🎉 FINAL SILHOUETTE SCORE: {best_score:.4f} 🎉")
print(f"{'='*70}")
