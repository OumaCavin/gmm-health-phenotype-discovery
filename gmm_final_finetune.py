"""
Final Fine-Tuning - Push from 0.8035 to 0.87+
Best config so far: 1% outliers, 15 features, UMAP(30, 0.02)

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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Paths
PROJECT_ROOT = '/workspace'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v2')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/raw/nhanes_health_data.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FINAL FINE-TUNING - PUSH TO 0.87+")
print("=" * 70)

# Load data
df = pd.read_csv(DATA_PATH)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'health_category' in numeric_cols:
    numeric_cols.remove('health_category')
X = df[numeric_cols].values.astype(np.float64)
print(f"Data: {X.shape}")

# Fixed preprocessing: 1% outliers, 15 features
print("\nUsing best preprocessing: 1% outliers, 15 features")

X_processed = X.copy()
imputer = SimpleImputer(strategy='median')
X_processed = imputer.fit_transform(X_processed)

transformer = PowerTransformer(method='yeo-johnson')
X_processed = transformer.fit_transform(X_processed)

scaler = RobustScaler()
X_processed = scaler.fit_transform(X_processed)

iso = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
labels = iso.fit_predict(X_processed)
X_clean = X_processed[labels == 1]

pseudo_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_clean)
var_thresh = VarianceThreshold(threshold=0.1)
X_clean = var_thresh.fit_transform(X_clean)
selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X_clean, pseudo_labels)

print(f"Prepared: {X_selected.shape}")

# Fine-tuning UMAP parameters
print("\nFine-tuning UMAP parameters...")

best_score = 0
best_config = {}

# Test around n_neighbors=30, min_dist=0.02
for n_neighbors in [28, 30, 32, 35]:
    for min_dist in [0.015, 0.02, 0.025, 0.03]:
        for n_comp in [8, 10, 12]:
            try:
                reducer = umap.UMAP(
                    n_components=n_comp,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='euclidean',
                    random_state=42
                )
                X_umap = reducer.fit_transform(X_selected)
                
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=100)
                labels = kmeans.fit_predict(X_umap)
                score = silhouette_score(X_umap, labels)
                
                if score > best_score:
                    best_score = score
                    best_config = {
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'n_components': n_comp,
                        'score': score
                    }
                    print(f"   ★ UMAP({n_neighbors}, {min_dist}, {n_comp}): {score:.4f}")
                    
            except Exception as e:
                pass

# Results
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Previous Best: 0.5343")
print(f"New Best: {best_score:.4f}")
print(f"Improvement: {((best_score - 0.5343) / 0.5343) * 100:.1f}%")
print(f"Target: 0.87")
print(f"Progress: {(best_score / 0.87) * 100:.1f}%")
print(f"Gap: {0.87 - best_score:.4f}")
print(f"\nBest UMAP Config:")
print(f"  n_neighbors: {best_config.get('n_neighbors', 30)}")
print(f"  min_dist: {best_config.get('min_dist', 0.02)}")
print(f"  n_components: {best_config.get('n_components', 10)}")

# Save
metrics = {
    'previous_best': 0.5343,
    'new_best': float(best_score),
    'improvement_percent': float(((best_score - 0.5343) / 0.5343) * 100),
    'target': 0.87,
    'progress_percent': float((best_score / 0.87) * 100),
    'configuration': best_config,
    'timestamp': datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, 'metrics/final_finetune_results.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

report = f"""# Final Fine-Tuning Results

## Summary

| Metric | Value |
|--------|-------|
| Previous Best | 0.5343 |
| **New Best** | **{best_score:.4f}** |
| Improvement | {((best_score - 0.5343) / 0.5343) * 100:.1f}% |
| Target | 0.87 |
| Progress | {(best_score / 0.87) * 100:.1f}% |

## Best Configuration

| Parameter | Value |
|-----------|-------|
| n_neighbors | {best_config.get('n_neighbors', 30)} |
| min_dist | {best_config.get('min_dist', 0.02)} |
| n_components | {best_config.get('n_components', 10)} |
| Clusters | 2 |

---
Generated: {datetime.now().isoformat()}
"""

with open(os.path.join(OUTPUT_DIR, 'reports/final_finetune_report.md'), 'w') as f:
    f.write(report)

print(f"\n{'='*70}")
print(f"🎉 FINAL SILHOUETTE SCORE: {best_score:.4f} 🎉")
print(f"{'='*70}")
