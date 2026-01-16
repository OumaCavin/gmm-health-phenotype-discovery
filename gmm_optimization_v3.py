"""
Streamlined High-Performance Clustering Optimization
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

# Path configuration
PROJECT_ROOT = '/workspace'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v2')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/raw/nhanes_health_data.csv')

# Create output directories
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'reports'), exist_ok=True)

def load_data():
    """Load NHANES data and prepare numeric features."""
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'health_category' in numeric_cols:
        numeric_cols.remove('health_category')
    
    X = df[numeric_cols].values.astype(np.float64)
    print(f"Feature matrix: {X.shape}")
    
    return X, numeric_cols

def preprocessing_pipeline(X):
    """Apply preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing")
    print("=" * 60)
    
    # Mean imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    print("1. Median imputation: OK")
    
    # Power transformation
    from sklearn.preprocessing import PowerTransformer
    transformer = PowerTransformer(method='yeo-johnson')
    X = transformer.fit_transform(X)
    print("2. Yeo-Johnson transform: OK")
    
    # Robust scaling
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    print("3. Robust scaling: OK")
    
    # Outlier removal (3%)
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.03, random_state=42, n_estimators=100)
    labels = iso.fit_predict(X)
    X = X[labels == 1]
    print(f"4. Outlier removal: {100 * (1 - X.shape[0] / 4850):.1f}% removed")
    print(f"   Final shape: {X.shape}")
    
    return X

def find_optimal_clusters(X, dim_method='PCA', n_components=10):
    """Find optimal cluster count and algorithm."""
    print(f"\n   Testing {dim_method} ({n_components} components)...")
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    
    # Apply dimensionality reduction
    if dim_method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
        X_red = reducer.fit_transform(X)
    elif dim_method == 'Isomap':
        reducer = Isomap(n_neighbors=15, n_components=n_components)
        X_red = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=n_components, random_state=42)
        X_red = reducer.fit_transform(X)
    
    best_score = 0
    best_config = {}
    
    # Test different cluster counts
    for n_clusters in [2, 3, 4]:
        # K-Means
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = model.fit_predict(X_red)
        score = silhouette_score(X_red, labels)
        if score > best_score:
            best_score = score
            best_config = {'method': 'KMeans', 'n_clusters': n_clusters, 'labels': labels, 'embedding': X_red}
        
        # GMM Full
        model = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, n_init=3)
        labels = model.fit_predict(X_red)
        score = silhouette_score(X_red, labels)
        if score > best_score:
            best_score = score
            best_config = {'method': 'GMM_Full', 'n_clusters': n_clusters, 'labels': labels, 'embedding': X_red}
        
        # GMM Diagonal
        model = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42, n_init=3)
        labels = model.fit_predict(X_red)
        score = silhouette_score(X_red, labels)
        if score > best_score:
            best_score = score
            best_config = {'method': 'GMM_Diag', 'n_clusters': n_clusters, 'labels': labels, 'embedding': X_red}
    
    return best_score, best_config

def run_optimization():
    """Run comprehensive but efficient optimization."""
    print("\n" + "=" * 60)
    print("CLUSTERING OPTIMIZATION v3")
    print("Target: Push Silhouette from 0.5343 towards 0.87-1.00")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load and preprocess
    X, feature_names = load_data()
    X = preprocessing_pipeline(X)
    
    # Test different configurations
    print("\n" + "=" * 60)
    print("STEP 3: Testing Configurations")
    print("=" * 60)
    
    configs = [
        ('PCA', 5),
        ('PCA', 8),
        ('PCA', 10),
        ('Isomap', 5),
        ('Isomap', 8),
    ]
    
    all_results = []
    best_overall_score = 0
    best_overall_config = None
    
    for dim_method, n_comp in configs:
        score, config = find_optimal_clusters(X, dim_method, n_comp)
        all_results.append({
            'dim_method': dim_method,
            'n_components': n_comp,
            'cluster_method': config.get('method', 'Unknown'),
            'n_clusters': config.get('n_clusters', 0),
            'score': score
        })
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_config = {
                'dimensionality_reduction': dim_method,
                'n_components': n_comp,
                'clustering_method': config.get('method', 'Unknown'),
                'n_clusters': config.get('n_clusters', 0),
                'silhouette_score': score,
                'labels': config.get('labels', None),
                'embedding': config.get('embedding', None)
            }
        
        print(f"   {dim_method}-{n_comp}: {score:.4f} ({config.get('method', 'Unknown')})")
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Previous Best: 0.5343")
    print(f"New Best: {best_overall_score:.4f}")
    print(f"Improvement: {((best_overall_score - 0.5343) / 0.5343) * 100:.1f}%")
    print(f"Best Config: {best_overall_config['dimensionality_reduction']} + {best_overall_config['clustering_method']}")
    print(f"Time: {elapsed:.1f}s")
    
    return best_overall_config, all_results

def save_results(best_config, all_results):
    """Save all results."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save metrics
    metrics = {
        'previous_best': 0.5343,
        'new_best': best_config['silhouette_score'],
        'improvement_percent': ((best_config['silhouette_score'] - 0.5343) / 0.5343) * 100,
        'configuration': {
            'dimensionality_reduction': best_config['dimensionality_reduction'],
            'n_components': best_config['n_components'],
            'clustering_method': best_config['clustering_method'],
            'n_clusters': best_config['n_clusters']
        },
        'all_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics/optimization_v3_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics_path}")
    
    # Save predictions
    if best_config['labels'] is not None:
        predictions_df = pd.DataFrame({
            'sample_id': range(len(best_config['labels'])),
            'cluster': best_config['labels']
        })
        predictions_path = os.path.join(OUTPUT_DIR, 'predictions/optimization_v3_clusters.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions: {predictions_path}")
    
    # Save report
    report = f"""# Clustering Optimization v3 Report

## Summary

**Previous Best:** 0.5343  
**New Best:** {best_config['silhouette_score']:.4f}  
**Improvement:** {((best_config['silhouette_score'] - 0.5343) / 0.5343) * 100:.1f}%

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Dimensionality Reduction | {best_config['dimensionality_reduction']} |
| Components | {best_config['n_components']} |
| Clustering Method | {best_config['clustering_method']} |
| Clusters | {best_config['n_clusters']} |
| Silhouette | {best_config['silhouette_score']:.4f} |

## Progress Analysis

| Metric | Value |
|--------|-------|
| Target | 0.87 - 1.00 |
| Current | {best_config['silhouette_score']:.4f} |
| Progress | {(best_config['silhouette_score'] / 0.87) * 100:.1f}% |
| Gap | {0.87 - best_config['silhouette_score']:.4f} |

---
Generated: {datetime.now().isoformat()}
"""
    
    report_path = os.path.join(OUTPUT_DIR, 'reports/optimization_v3_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report: {report_path}")

def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("HIGH-PERFORMANCE CLUSTERING OPTIMIZATION")
    print("Author: Cavin Otieno | January 2025")
    print("=" * 60)
    
    best_config, all_results = run_optimization()
    save_results(best_config, all_results)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)
    
    return best_config

if __name__ == "__main__":
    best_config = main()
    print(f"\nFinal Silhouette Score: {best_config['silhouette_score']:.4f}")
