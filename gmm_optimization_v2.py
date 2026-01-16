"""
High-Performance Clustering Optimization for Health Phenotype Discovery
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

def load_and_prepare_data():
    """Load NHANES data and prepare for clustering."""
    print("=" * 80)
    print("STEP 1: Loading and Preparing Data")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Check data types
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    
    # Separate features and target (remove non-numeric columns)
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column if present
    if 'health_category' in numeric_cols:
        numeric_cols.remove('health_category')
    
    X = df[numeric_cols].copy()
    
    # Convert to numpy array
    X = X.values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of numeric features: {len(numeric_cols)}")
    print(f"Missing values: {np.isnan(X).sum()}")
    
    return X, numeric_cols

def advanced_preprocessing(X, feature_names):
    """
    Apply advanced preprocessing pipeline.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Advanced Preprocessing")
    print("=" * 80)
    
    original_shape = X.shape
    X_processed = X.copy()
    
    # Step 1: Simple imputation (mean) - robust to missing values
    print("1. Applying Mean Imputation...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_processed)
    print(f"   Imputation complete. Shape: {X_imputed.shape}")
    
    # Step 2: Power Transformation (Yeo-Johnson)
    print("2. Applying Yeo-Johnson Power Transformation...")
    from sklearn.preprocessing import PowerTransformer
    transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    X_transformed = transformer.fit_transform(X_imputed)
    print(f"   Transformation complete. Shape: {X_transformed.shape}")
    
    # Step 3: Robust Scaling
    print("3. Applying Robust Scaling...")
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    print(f"   Scaling complete. Shape: {X_scaled.shape}")
    
    # Step 4: Isolation Forest Outlier Removal (conservative 3%)
    print("4. Applying Isolation Forest Outlier Removal (3%)...")
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(
        contamination=0.03,
        random_state=42,
        n_estimators=100
    )
    outlier_labels = iso_forest.fit_predict(X_scaled)
    X_clean = X_scaled[outlier_labels == 1]
    
    print(f"   Removed {np.sum(outlier_labels == -1)} outliers ({100 * np.sum(outlier_labels == -1) / len(outlier_labels):.1f}%)")
    print(f"   Final shape: {X_clean.shape}")
    print(f"   Data preserved: {100 * X_clean.shape[0] / original_shape[0]:.1f}%")
    
    return X_clean

def feature_engineering(X, n_features=20):
    """
    Apply feature selection to improve cluster separability.
    """
    print("\n" + "=" * 80)
    print("STEP 3: Feature Selection")
    print("=" * 80)
    
    print(f"Input features: {X.shape[1]}")
    
    # Create pseudo-labels using quick k-means
    print("1. Creating pseudo-labels for feature selection...")
    from sklearn.cluster import KMeans
    kmeans_temp = KMeans(n_clusters=3, random_state=42, n_init=10)
    pseudo_labels = kmeans_temp.fit_predict(X)
    
    # Variance threshold
    print("2. Applying Variance Threshold...")
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
    var_thresh = VarianceThreshold(threshold=0.1)
    X_var = var_thresh.fit_transform(X)
    print(f"   After variance threshold: {X_var.shape[1]} features")
    
    # SelectKBest
    print("3. Selecting Top Features using SelectKBest...")
    n_select = min(n_features, X_var.shape[1])
    selector = SelectKBest(f_classif, k=n_select)
    X_selected = selector.fit_transform(X_var, pseudo_labels)
    
    print(f"   Selected {X_selected.shape[1]} features")
    print(f"   Final feature matrix shape: {X_selected.shape}")
    
    return X_selected

def dimensionality_reduction_experiments(X):
    """
    Apply multiple dimensionality reduction methods.
    """
    print("\n" + "=" * 80)
    print("STEP 4: Dimensionality Reduction Experiments")
    print("=" * 80)
    
    results = {}
    n_components = min(10, X.shape[1] - 1)
    
    # PCA
    print("1. PCA...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    explained_var = np.sum(pca.explained_variance_ratio_)
    results['PCA'] = {
        'transform': X_pca,
        'explained_variance': explained_var,
        'n_components': X_pca.shape[1]
    }
    print(f"   Explained variance: {explained_var:.3f}")
    
    # Kernel PCA (RBF)
    print("2. Kernel PCA (RBF)...")
    from sklearn.decomposition import KernelPCA
    try:
        kpca = KernelPCA(n_components=n_components, 
                         kernel='rbf', 
                         gamma=0.01,
                         random_state=42)
        X_kpca = kpca.fit_transform(X)
        results['KernelPCA'] = {
            'transform': X_kpca,
            'n_components': X_kpca.shape[1]
        }
        print(f"   Shape: {X_kpca.shape}")
    except Exception as e:
        print(f"   Kernel PCA failed: {e}")
        results['KernelPCA'] = {'transform': X_pca, 'n_components': X_pca.shape[1]}
    
    # Isomap
    print("3. Isomap...")
    from sklearn.manifold import Isomap
    try:
        isomap = Isomap(n_neighbors=15, n_components=n_components)
        X_isomap = isomap.fit_transform(X)
        results['Isomap'] = {
            'transform': X_isomap,
            'n_components': X_isomap.shape[1]
        }
        print(f"   Shape: {X_isomap.shape}")
    except Exception as e:
        print(f"   Isomap failed: {e}")
        results['Isomap'] = {'transform': X_pca, 'n_components': X_pca.shape[1]}
    
    # LLE
    print("4. Locally Linear Embedding...")
    from sklearn.manifold import LocallyLinearEmbedding
    try:
        lle = LocallyLinearEmbedding(n_neighbors=15, 
                                     n_components=n_components,
                                     random_state=42)
        X_lle = lle.fit_transform(X)
        results['LLE'] = {
            'transform': X_lle,
            'n_components': X_lle.shape[1]
        }
        print(f"   Shape: {X_lle.shape}")
    except Exception as e:
        print(f"   LLE failed: {e}")
        results['LLE'] = {'transform': X_pca, 'n_components': X_pca.shape[1]}
    
    # t-SNE
    print("5. t-SNE...")
    from sklearn.manifold import TSNE
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate='auto',
            init='pca',
            random_state=42
        )
        X_tsne = tsne.fit_transform(X)
        results['tSNE'] = {
            'transform': X_tsne,
            'n_components': 2
        }
        print(f"   Shape: {X_tsne.shape}")
    except Exception as e:
        print(f"   t-SNE failed: {e}")
        results['tSNE'] = {'transform': X_pca[:, :2], 'n_components': 2}
    
    return results

def clustering_experiments(X_reduced, method_name, n_clusters=2):
    """
    Apply multiple clustering algorithms.
    """
    print(f"\n   Clustering experiments with {method_name} ({X_reduced.shape[1]} components)...")
    
    from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    
    results = {}
    
    # K-Means
    print("      - K-Means...", end=" ")
    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels_kmeans = kmeans.fit_predict(X_reduced)
    score_kmeans = silhouette_score(X_reduced, labels_kmeans)
    results['KMeans'] = {'labels': labels_kmeans, 'score': score_kmeans, 'time': time.time() - start}
    print(f"Silhouette: {score_kmeans:.4f}")
    
    # Spectral Clustering
    print("      - Spectral Clustering...", end=" ")
    start = time.time()
    spectral = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='nearest_neighbors',
                                   n_neighbors=15,
                                   random_state=42,
                                   assign_labels='kmeans')
    labels_spectral = spectral.fit_predict(X_reduced)
    score_spectral = silhouette_score(X_reduced, labels_spectral)
    results['Spectral'] = {'labels': labels_spectral, 'score': score_spectral, 'time': time.time() - start}
    print(f"Silhouette: {score_spectral:.4f}")
    
    # Agglomerative Clustering (Ward)
    print("      - Agglomerative (Ward)...", end=" ")
    start = time.time()
    agg_ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_agg_ward = agg_ward.fit_predict(X_reduced)
    score_agg_ward = silhouette_score(X_reduced, labels_agg_ward)
    results['Agglomerative_Ward'] = {'labels': labels_agg_ward, 'score': score_agg_ward, 'time': time.time() - start}
    print(f"Silhouette: {score_agg_ward:.4f}")
    
    # GMM Full
    print("      - GMM (Full)...", end=" ")
    start = time.time()
    gmm = GaussianMixture(n_components=n_clusters, 
                          covariance_type='full',
                          random_state=42,
                          n_init=5,
                          max_iter=200)
    gmm.fit(X_reduced)
    labels_gmm = gmm.predict(X_reduced)
    score_gmm = silhouette_score(X_reduced, labels_gmm)
    results['GMM_Full'] = {'labels': labels_gmm, 'score': score_gmm, 'time': time.time() - start, 'model': gmm}
    print(f"Silhouette: {score_gmm:.4f}")
    
    # GMM Diagonal
    print("      - GMM (Diagonal)...", end=" ")
    start = time.time()
    gmm_diag = GaussianMixture(n_components=n_clusters, 
                               covariance_type='diag',
                               random_state=42,
                               n_init=5,
                               max_iter=200)
    gmm_diag.fit(X_reduced)
    labels_gmm_diag = gmm_diag.predict(X_reduced)
    score_gmm_diag = silhouette_score(X_reduced, labels_gmm_diag)
    results['GMM_Diag'] = {'labels': labels_gmm_diag, 'score': score_gmm_diag, 'time': time.time() - start, 'model': gmm_diag}
    print(f"Silhouette: {score_gmm_diag:.4f}")
    
    return results

def optimize_clusters_grid(X_reduced, n_clusters_range=[2, 3, 4]):
    """
    Grid search over number of clusters.
    """
    print("\n" + "=" * 80)
    print("STEP 5: Cluster Count Optimization")
    print("=" * 80)
    
    from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score
    
    best_score = -1
    best_n_clusters = 2
    best_labels = None
    best_method = None
    
    results_summary = []
    
    for n_clusters in n_clusters_range:
        print(f"\n   Testing n_clusters = {n_clusters}...")
        
        methods = {
            'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=20),
            'Spectral': SpectralClustering(n_clusters=n_clusters, random_state=42),
            'GMM_Full': GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42),
            'GMM_Diag': GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42),
            'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        }
        
        for method_name, model in methods.items():
            try:
                if hasattr(model, 'predict'):
                    labels = model.fit_predict(X_reduced)
                else:
                    labels = model.fit(X_reduced)
                    if hasattr(model, 'predict'):
                        labels = model.predict(X_reduced)
                
                score = silhouette_score(X_reduced, labels)
                results_summary.append({
                    'n_clusters': n_clusters,
                    'method': method_name,
                    'silhouette': score
                })
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
                    best_method = method_name
                    
                print(f"      {method_name}: {score:.4f}")
                
            except Exception as e:
                print(f"      {method_name}: Failed - {e}")
    
    print(f"\n   Best Configuration: {best_method} with {best_n_clusters} clusters")
    print(f"   Best Silhouette Score: {best_score:.4f}")
    
    return {
        'best_n_clusters': best_n_clusters,
        'best_score': best_score,
        'best_labels': best_labels,
        'best_method': best_method,
        'all_results': results_summary
    }

def comprehensive_optimization(X, feature_names):
    """
    Run comprehensive optimization pipeline.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE CLUSTERING OPTIMIZATION")
    print(f"Target: Push Silhouette Score from 0.5343 towards 0.87-1.00")
    print("=" * 80)
    
    start_time = time.time()
    
    # Preprocessing
    X_clean = advanced_preprocessing(X, feature_names)
    
    # Feature selection
    X_engineered = feature_engineering(X_clean, n_features=25)
    
    # Dimensionality reduction experiments
    dim_reduction_results = dimensionality_reduction_experiments(X_engineered)
    
    # Optimize clustering for each embedding
    print("\n" + "=" * 80)
    print("STEP 5: Clustering Optimization Across All Embeddings")
    print("=" * 80)
    
    best_overall_score = 0
    best_overall_config = None
    
    for method_name, result in dim_reduction_results.items():
        X_reduced = result['transform']
        n_components = result['n_components']
        
        print(f"\n{'='*60}")
        print(f"Testing: {method_name} ({n_components} components)")
        print(f"{'='*60}")
        
        cluster_results = optimize_clusters_grid(X_reduced, n_clusters_range=[2, 3, 4])
        
        if cluster_results['best_score'] > best_overall_score:
            best_overall_score = cluster_results['best_score']
            best_overall_config = {
                'dimensionality_reduction': method_name,
                'n_components': n_components,
                'n_clusters': cluster_results['best_n_clusters'],
                'clustering_method': cluster_results['best_method'],
                'silhouette_score': cluster_results['best_score'],
                'labels': cluster_results['best_labels']
            }
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Previous Best Score: 0.5343")
    print(f"New Best Score: {best_overall_score:.4f}")
    print(f"Improvement: {((best_overall_score - 0.5343) / 0.5343) * 100:.1f}%")
    print(f"Best Configuration:")
    print(f"  - Dimensionality Reduction: {best_overall_config['dimensionality_reduction']}")
    print(f"  - Components: {best_overall_config['n_components']}")
    print(f"  - Clustering Method: {best_overall_config['clustering_method']}")
    print(f"  - Number of Clusters: {best_overall_config['n_clusters']}")
    print(f"Total Optimization Time: {total_time:.1f} seconds")
    
    return best_overall_config

def save_results(best_config, X_clean):
    """Save all results."""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
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
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics/comprehensive_v2_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'sample_id': range(len(best_config['labels'])),
        'cluster': best_config['labels']
    })
    predictions_path = os.path.join(OUTPUT_DIR, 'predictions/comprehensive_v2_clusters.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved: {predictions_path}")
    
    # Save report
    report = f"""# Comprehensive Clustering Optimization Report v2

## Executive Summary

**Objective:** Push Silhouette Score from 0.5343 towards 0.87-1.00

**Results Achieved:**
- **Previous Best:** 0.5343 (Spectral Clustering + PCA)
- **New Best:** {best_config['silhouette_score']:.4f}
- **Improvement:** {((best_config['silhouette_score'] - 0.5343) / 0.5343) * 100:.1f}%

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Dimensionality Reduction | {best_config['dimensionality_reduction']} |
| Number of Components | {best_config['n_components']} |
| Clustering Method | {best_config['clustering_method']} |
| Number of Clusters | {best_config['n_clusters']} |
| Silhouette Score | {best_config['silhouette_score']:.4f} |

## Progress Analysis

| Metric | Value |
|--------|-------|
| Target Score | 0.87 - 1.00 |
| Current Score | {best_config['silhouette_score']:.4f} |
| Progress | {(best_config['silhouette_score'] / 0.87) * 100:.1f}% |
| Gap to Target | {0.87 - best_config['silhouette_score']:.4f} |

## Next Steps for Further Improvement

1. **Fine-tune Parameters:** Continue optimizing hyperparameters with finer granularity
2. **Try HDBSCAN:** Install HDBSCAN for density-based clustering
3. **Feature Interactions:** Create polynomial feature interactions
4. **Ensemble Methods:** Implement consensus clustering

---
*Generated: {datetime.now().isoformat()}*
"""
    
    report_path = os.path.join(OUTPUT_DIR, 'reports/comprehensive_v2_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("HIGH-PERFORMANCE CLUSTERING OPTIMIZATION")
    print("Author: Cavin Otieno | Date: January 2025")
    print("=" * 80)
    
    # Load data
    X, feature_names = load_and_prepare_data()
    
    # Run comprehensive optimization
    best_config = comprehensive_optimization(X, feature_names)
    
    # Save results
    save_results(best_config, X)
    
    return best_config

if __name__ == "__main__":
    best_config = main()
    print(f"\nFinal Best Silhouette Score: {best_config['silhouette_score']:.4f}")
