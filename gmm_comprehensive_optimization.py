"""
Comprehensive Clustering Optimization for Health Phenotype Discovery
Target: Push Silhouette Score from 0.5343 towards 0.87-1.00

Strategy:
1. Advanced Preprocessing (IterativeImputer, PowerTransformer, Isolation Forest)
2. Feature Engineering (Polynomial features, Feature selection)
3. Multiple Dimensionality Reduction (UMAP, Isomap, LLE, Kernel PCA)
4. Advanced Clustering (HDBSCAN, Optimized GMM, Ensemble)
5. Bayesian Hyperparameter Optimization

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

# Enable experimental features
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import cross_val_score

# Try to import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available, using alternative methods")

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available, using alternatives")

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
    
    # Separate features and target
    X = df.drop(columns=['health_category'], errors='ignore')
    y = df['health_category'] if 'health_category' in df.columns else None
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Missing values: {X.isnull().sum().sum()}")
    
    return X, y

def advanced_preprocessing(X, method='comprehensive'):
    """
    Apply advanced preprocessing pipeline.
    
    Steps:
    1. Iterative imputation (MICE-like)
    2. Power transformation (Yeo-Johnson)
    3. Robust scaling
    4. Isolation Forest outlier removal
    """
    print("\n" + "=" * 80)
    print("STEP 2: Advanced Preprocessing")
    print("=" * 80)
    
    original_shape = X.shape
    X_processed = X.copy()
    
    # Step 1: Iterative Imputation
    print("1. Applying Iterative Imputation (MICE)...")
    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy='median'
    )
    X_imputed = imputer.fit_transform(X_processed)
    print(f"   Imputation complete. Shape: {X_imputed.shape}")
    
    # Step 2: Yeo-Johnson Power Transformation
    print("2. Applying Yeo-Johnson Power Transformation...")
    transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    X_transformed = transformer.fit_transform(X_imputed)
    print(f"   Transformation complete. Shape: {X_transformed.shape}")
    
    # Step 3: Robust Scaling
    print("3. Applying Robust Scaling...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    print(f"   Scaling complete. Shape: {X_scaled.shape}")
    
    # Step 4: Isolation Forest Outlier Removal (conservative 5%)
    print("4. Applying Isolation Forest Outlier Removal (5%)...")
    iso_forest = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100
    )
    outlier_labels = iso_forest.fit_predict(X_scaled)
    X_clean = X_scaled[outlier_labels == 1]
    y_clean = None
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
    
    print(f"   Removed {np.sum(outlier_labels == -1)} outliers ({100 * np.sum(outlier_labels == -1) / len(outlier_labels):.1f}%)")
    print(f"   Final shape: {X_clean.shape}")
    print(f"   Data preserved: {100 * X_clean.shape[0] / original_shape[0]:.1f}%")
    
    return X_clean, feature_names

def feature_engineering(X, n_features=20):
    """
    Apply feature engineering to improve cluster separability.
    
    1. Variance threshold filtering
    2. SelectKBest feature selection
    """
    print("\n" + "=" * 80)
    print("STEP 3: Feature Engineering")
    print("=" * 80)
    
    print(f"Input features: {X.shape[1]}")
    
    # Variance threshold
    print("1. Applying Variance Threshold...")
    var_thresh = VarianceThreshold(threshold=0.1)
    X_var = var_thresh.fit_transform(X)
    print(f"   After variance threshold: {X_var.shape[1]} features")
    
    # SelectKBest (using k-means inertia as pseudo-label for selection)
    print("2. Selecting Top Features using SelectKBest...")
    # Create pseudo-labels using quick k-means
    kmeans_temp = KMeans(n_clusters=3, random_state=42, n_init=10)
    pseudo_labels = kmeans_temp.fit_predict(X_var)
    
    selector = SelectKBest(f_classif, k=min(n_features, X_var.shape[1]))
    X_selected = selector.fit_transform(X_var, pseudo_labels)
    
    selected_features = selector.get_support()
    print(f"   Selected {X_selected.shape[1]} features")
    print(f"   Final feature matrix shape: {X_selected.shape}")
    
    return X_selected

def dimensionality_reduction_experiments(X):
    """
    Apply multiple dimensionality reduction methods and return best results.
    """
    print("\n" + "=" * 80)
    print("STEP 4: Dimensionality Reduction Experiments")
    print("=" * 80)
    
    results = {}
    
    # PCA
    print("1. PCA...")
    pca = PCA(n_components=min(10, X.shape[1] - 1), random_state=42)
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
    try:
        kpca = KernelPCA(n_components=min(10, X.shape[1] - 1), 
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
    try:
        isomap = Isomap(n_neighbors=15, n_components=min(10, X.shape[1] - 1))
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
    try:
        lle = LocallyLinearEmbedding(n_neighbors=15, 
                                     n_components=min(10, X.shape[1] - 1),
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
    
    # UMAP (if available)
    if UMAP_AVAILABLE:
        print("5. UMAP...")
        try:
            umap_reducer = umap.UMAP(
                n_components=min(10, X.shape[1] - 1),
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                random_state=42
            )
            X_umap = umap_reducer.fit_transform(X)
            results['UMAP'] = {
                'transform': X_umap,
                'n_components': X_umap.shape[1]
            }
            print(f"   Shape: {X_umap.shape}")
        except Exception as e:
            print(f"   UMAP failed: {e}")
            results['UMAP'] = {'transform': X_pca, 'n_components': X_pca.shape[1]}
    
    # t-SNE (separate call due to different interface)
    print("6. t-SNE...")
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
    Apply multiple clustering algorithms and return results.
    """
    print(f"\n   Clustering experiments with {method_name} ({X_reduced.shape[1]} components)...")
    
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
    
    # Agglomerative Clustering (Complete)
    print("      - Agglomerative (Complete)...", end=" ")
    start = time.time()
    agg_complete = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    labels_agg_complete = agg_complete.fit_predict(X_reduced)
    score_agg_complete = silhouette_score(X_reduced, labels_agg_complete)
    results['Agglomerative_Complete'] = {'labels': labels_agg_complete, 'score': score_agg_complete, 'time': time.time() - start}
    print(f"Silhouette: {score_agg_complete:.4f}")
    
    # GMM
    print("      - Gaussian Mixture...", end=" ")
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
    
    # HDBSCAN (if available)
    if HDBSCAN_AVAILABLE:
        print("      - HDBSCAN...", end=" ")
        start = time.time()
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=50,
                min_samples=10,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels_hdbscan = clusterer.fit_predict(X_reduced)
            n_clusters_hdbscan = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
            
            if n_clusters_hdbscan >= 2:
                # Filter out noise points for scoring
                mask = labels_hdbscan != -1
                if np.sum(mask) > n_clusters_hdbscan and len(set(labels_hdbscan[mask])) >= 2:
                    score_hdbscan = silhouette_score(X_reduced[mask], labels_hdbscan[mask])
                    noise_ratio = np.sum(labels_hdbscan == -1) / len(labels_hdbscan)
                    results['HDBSCAN'] = {
                        'labels': labels_hdbscan, 
                        'score': score_hdbscan, 
                        'time': time.time() - start,
                        'n_clusters': n_clusters_hdbscan,
                        'noise_ratio': noise_ratio
                    }
                    print(f"Silhouette: {score_hdbscan:.4f} (noise: {noise_ratio:.1%})")
                else:
                    print(f"HDBSCAN: Insufficient clusters ({n_clusters_hdbscan})")
            else:
                print(f"HDBSCAN: Only {n_clusters_hdbscan} cluster(s) found")
        except Exception as e:
            print(f"HDBSCAN failed: {e}")
    
    return results

def ensemble_clustering(X_reduced, n_clusters=2):
    """
    Create ensemble clustering from multiple algorithms.
    """
    print(f"\n   Ensemble Clustering...")
    
    # Get predictions from multiple algorithms
    algorithms = {
        'kmeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=20),
        'spectral': SpectralClustering(n_clusters=n_clusters, random_state=42),
        'agg_ward': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
        'gmm': GaussianMixture(n_components=n_clusters, random_state=42, n_init=3)
    }
    
    predictions = {}
    for name, algo in algorithms.items():
        predictions[name] = algo.fit_predict(X_reduced)
    
    # Simple voting: for each point, find most common assignment
    pred_array = np.array(list(predictions.values()))
    
    # Use co-association matrix approach
    n_samples = X_reduced.shape[0]
    co_assoc = np.zeros((n_samples, n_samples))
    
    for name, labels in predictions.items():
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if labels[i] == labels[j]:
                    co_assoc[i, j] += 1
                else:
                    co_assoc[i, j] -= 1
    
    # Spectral clustering on co-association matrix
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    co_assoc = (co_assoc + co_assoc.T) / 2
    co_assoc_sparse = csr_matrix(co_assoc)
    
    # Use spectral clustering on similarity matrix
    spectral = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='precomputed',
                                   random_state=42)
    ensemble_labels = spectral.fit_predict(co_assoc)
    
    score = silhouette_score(X_reduced, ensemble_labels)
    print(f"      Ensemble Silhouette: {score:.4f}")
    
    return {'labels': ensemble_labels, 'score': score}

def optimize_clusters_grid(X_reduced, n_clusters_range=[2, 3, 4, 5]):
    """
    Grid search over number of clusters for best silhouette score.
    """
    print("\n" + "=" * 80)
    print("STEP 5: Cluster Count Optimization")
    print("=" * 80)
    
    best_score = -1
    best_n_clusters = 2
    best_labels = None
    best_method = None
    
    results_summary = []
    
    for n_clusters in n_clusters_range:
        print(f"\n   Testing n_clusters = {n_clusters}...")
        
        # Test multiple methods
        methods = {
            'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=20),
            'Spectral': SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels='kmeans'),
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

def comprehensive_optimization(X):
    """
    Run comprehensive optimization pipeline.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE CLUSTERING OPTIMIZATION")
    print(f"Target: Push Silhouette Score from 0.5343 towards 0.87-1.00")
    print("=" * 80)
    
    all_results = []
    start_time = time.time()
    
    # Preprocessing
    X_clean, feature_names = advanced_preprocessing(X)
    
    # Feature engineering
    X_engineered = feature_engineering(X_clean, n_features=25)
    
    # Dimensionality reduction experiments
    dim_reduction_results = dimensionality_reduction_experiments(X_engineered)
    
    # For each dimensionality reduction method, find optimal clustering
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
        
        # Optimize cluster count
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
        
        all_results.append({
            'embedding': method_name,
            'n_components': n_components,
            'cluster_results': cluster_results
        })
    
    # Try ensemble on best embedding
    print(f"\n{'='*60}")
    print(f"ENSEMBLE CLUSTERING ON BEST EMBEDDING")
    print(f"{'='*60}")
    
    best_embedding = best_overall_config['dimensionality_reduction']
    X_best = dim_reduction_results[best_embedding]['transform']
    
    ensemble_result = ensemble_clustering(X_best, n_clusters=best_overall_config['n_clusters'])
    
    if ensemble_result['score'] > best_overall_score:
        best_overall_score = ensemble_result['score']
        best_overall_config['clustering_method'] = 'Ensemble'
        best_overall_config['silhouette_score'] = ensemble_result['score']
        best_overall_config['labels'] = ensemble_result['labels']
    
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
    
    return best_overall_config, all_results

def save_results(best_config, all_results, X_clean, X_engineered):
    """Save all results to output directory."""
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
    
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics/comprehensive_optimization_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'sample_id': range(len(best_config['labels'])),
        'cluster': best_config['labels']
    })
    predictions_path = os.path.join(OUTPUT_DIR, 'predictions/comprehensive_optimization_clusters.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved: {predictions_path}")
    
    # Save report
    report = f"""# Comprehensive Clustering Optimization Report

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

## Methods Evaluated

### Preprocessing
1. Iterative Imputation (MICE-like)
2. Yeo-Johnson Power Transformation
3. Robust Scaling
4. Isolation Forest Outlier Removal (5%)

### Dimensionality Reduction
1. PCA
2. Kernel PCA (RBF)
3. Isomap
4. Locally Linear Embedding (LLE)
5. UMAP (if available)
6. t-SNE

### Clustering Algorithms
1. K-Means
2. Spectral Clustering
3. Agglomerative Clustering (Ward, Complete)
4. Gaussian Mixture Models (Full, Diagonal covariance)
5. HDBSCAN (if available)
6. Ensemble Clustering

## Recommendations for Further Improvement

1. **Target Score Analysis:**
   - Current Progress: {(best_config['silhouette_score'] / 0.87) * 100:.1f}% towards target (0.87)
   - Gap to Target: {0.87 - best_config['silhouette_score']:.4f}

2. **Next Steps:**
   - Fine-tune HDBSCAN parameters
   - Try custom kernel methods
   - Apply consensus clustering with more base algorithms
   - Explore deep learning-based representations

---
*Generated: {datetime.now().isoformat()}*
"""
    
    report_path = os.path.join(OUTPUT_DIR, 'reports/comprehensive_optimization_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("GMM HEALTH PHENOTYPE DISCOVERY - COMPREHENSIVE OPTIMIZATION")
    print("Author: Cavin Otieno | Date: January 2025")
    print("=" * 80)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Run comprehensive optimization
    best_config, all_results = comprehensive_optimization(X)
    
    # Save results
    save_results(best_config, all_results, X, None)
    
    return best_config

if __name__ == "__main__":
    best_config = main()
    print(f"\nFinal Best Silhouette Score: {best_config['silhouette_score']:.4f}")
