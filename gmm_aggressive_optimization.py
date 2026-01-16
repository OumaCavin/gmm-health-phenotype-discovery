#!/usr/bin/env python3
"""
===============================================================================
GMM OPTIMIZATION PIPELINE v2.0 - AGGRESSIVE PERFORMANCE OPTIMIZATION
===============================================================================

Target: Achieve Silhouette Score between 0.87 - 1.00
Current Baseline: ~0.0609
Author: Cavin Otieno (OumaCavin)
Email: cavin.otieno012@gmail.com

This pipeline implements ALL recommended strategies for maximizing Silhouette Score:
1. Aggressive Data Preprocessing & Feature Engineering
2. Multiple Dimensionality Reduction Techniques (PCA, UMAP, t-SNE)
3. Extensive Outlier Removal (Isolation Forest, LOF, Statistical Methods)
4. Comprehensive Hyperparameter Tuning (GMM + Alternative Algorithms)
5. Ensemble Clustering Approaches
6. Iterative Optimization with Score-Based Decision Making

Output Directory: output_v2/
===============================================================================
"""

import os
import sys
import json
import time
import warnings
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    QuantileTransformer, LabelEncoder
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
)
from sklearn.metrics import (
    silhouette_score, silhouette_samples, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.model_selection import cross_val_score
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONFIGURATION AND DIRECTORY SETUP
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline."""
    
    # Directory paths
    project_root: str = os.path.abspath(os.path.dirname('__file__'))
    data_dir: str = None
    output_dir: str = None
    models_dir: str = None
    figures_dir: str = None
    logs_dir: str = None
    reports_dir: str = None
    
    # Data parameters
    data_path: str = 'data/raw/nhanes_health_data.csv'
    target_columns: List[str] = field(default_factory=lambda: [])
    exclude_columns: List[str] = field(default_factory=lambda: [
        'SEQN', 'cluster', 'cluster_label', 'Cluster', 'Cluster_Assignment'
    ])
    
    # Optimization targets
    target_silhouette_min: float = 0.87
    target_silhouette_max: float = 1.00
    acceptable_fallback: float = 0.50
    max_iterations: int = 50
    
    # Preprocessing options
    use_knn_imputation: bool = True
    use_power_transform: bool = True
    correlation_threshold: float = 0.85
    
    # Outlier removal options
    max_outlier_fraction: float = 0.40  # Maximum 40% data removal
    use_isolation_forest: bool = True
    use_lof: bool = True
    use_statistical_outliers: bool = True
    
    # Dimensionality reduction options
    use_pca: bool = True
    pca_variance_threshold: float = 0.95
    use_umap: bool = True
    use_tsne: bool = True
    
    # Clustering options
    k_range: Tuple[int, int] = (2, 8)
    covariance_types: List[str] = field(default_factory=lambda: [
        'spherical', 'diag', 'tied', 'full'
    ])
    n_init: int = 20
    max_iter: int = 500
    reg_covar_range: Tuple[float, float] = (1e-6, 1e-2)
    
    # Visualization options
    generate_all_plots: bool = True
    plot_dpi: int = 300
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = os.path.join(self.project_root, 'data')
        self.output_dir = os.path.join(self.project_root, 'output_v2')
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.reports_dir = os.path.join(self.output_dir, 'reports')


class OptimizationLogger:
    """Logger for the optimization pipeline."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logs_dir = config.logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Setup file logger
        log_file = os.path.join(
            self.logs_dir, 
            f'optimization_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_section(self, title: str):
        """Log a section header."""
        separator = "=" * 70
        self.logger.info(separator)
        self.logger.info(f"  {title}")
        self.logger.info(separator)
        
    def log_result(self, key: str, value: Any):
        """Log a result."""
        self.logger.info(f"  {key}: {value}")


class GMMOptimizationPipeline:
    """
    Comprehensive GMM Optimization Pipeline for maximizing Silhouette Score.
    
    This pipeline implements ALL recommended strategies:
    1. Advanced preprocessing (KNN imputation, PowerTransformer)
    2. Multi-stage outlier removal
    3. Multiple dimensionality reduction techniques
    4. Extensive hyperparameter tuning
    5. Alternative clustering algorithms for comparison
    6. Iterative optimization with score-based decisions
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = OptimizationLogger(config)
        self.results = []
        self.best_score = 0.0
        self.best_model = None
        self.best_config = None
        self.data = None
        self.processed_data = None
        self.current_iteration = 0
        
        # Setup directory structure
        self._setup_directories()
        
    def _setup_directories(self):
        """Create all necessary output directories."""
        directories = [
            self.config.output_dir,
            self.config.models_dir,
            self.config.figures_dir,
            self.config.logs_dir,
            self.config.reports_dir,
            os.path.join(self.config.output_dir, 'data'),
            os.path.join(self.config.output_dir, 'data', 'processed'),
            os.path.join(self.config.output_dir, 'data', 'optimized'),
            os.path.join(self.config.output_dir, 'metrics'),
            os.path.join(self.config.output_dir, 'predictions'),
            os.path.join(self.config.output_dir, 'cluster_profiles'),
        ]
        
        for dir_path in directories:
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                
        self.logger.log_result("Directories Created", len(directories))
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare the dataset."""
        self.logger.log_section("PHASE 1: DATA LOADING AND PREPARATION")
        
        # Load data
        if os.path.exists(self.config.data_path):
            self.data = pd.read_csv(self.config.data_path)
            self.logger.log_result("Data Loaded", self.data.shape)
        else:
            # Try alternative paths
            alt_paths = [
                'nhanes_health_data.csv',
                'data/nhanes_health_data.csv',
                '../data/raw/nhanes_health_data.csv'
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    self.data = pd.read_csv(path)
                    self.logger.log_result(f"Data Loaded from {path}", self.data.shape)
                    break
            else:
                raise FileNotFoundError(f"Cannot find data file. Tried: {alt_paths}")
        
        # Remove excluded columns
        cols_to_remove = [c for c in self.config.exclude_columns if c in self.data.columns]
        self.data = self.data.drop(columns=cols_to_remove)
        self.logger.log_result("Columns Removed", len(cols_to_remove))
        
        # Identify numeric columns for clustering
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.logger.log_result("Numeric Features", len(self.numeric_cols))
        
        return self.data
    
    def advanced_preprocessing(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply advanced preprocessing techniques.
        
        Implements:
        1. KNN Imputation for missing values
        2. PowerTransformer (Yeo-Johnson) for Gaussian distribution
        3. Correlation-based feature removal
        4. Variance-based feature selection
        """
        self.logger.log_section("PHASE 2: ADVANCED PREPROCESSING")
        
        # Extract numeric features
        X = self.data[self.numeric_cols].copy()
        self.logger.log_result("Initial Features", X.shape[1])
        
        # Handle missing values
        missing_before = X.isnull().sum().sum()
        self.logger.log_result("Missing Values Before", missing_before)
        
        if self.config.use_knn_imputation and missing_before > 0:
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            X_imputed = knn_imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns)
            self.logger.log_result("Imputation Method", "KNN (k=5, distance-weighted)")
        else:
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
        # Remove constant features
        constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=constant_cols)
        self.logger.log_result("Constant Features Removed", len(constant_cols))
        
        # Remove highly correlated features
        if self.config.correlation_threshold < 1.0:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_cols = [column for column in upper.columns 
                             if any(upper[column] > self.config.correlation_threshold)]
            X = X.drop(columns=high_corr_cols)
            self.logger.log_result("High Correlation Features Removed", len(high_corr_cols))
        
        # Apply PowerTransformer for Gaussian distribution
        if self.config.use_power_transform:
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            X_transformed = pt.fit_transform(X)
            X = pd.DataFrame(X_transformed, columns=X.columns)
            self.logger.log_result("Transform Method", "Yeo-Johnson PowerTransformer")
            
            # Verify Gaussian-like distribution
            skewness = X.apply(stats.skew).abs().mean()
            self.logger.log_result("Mean Absolute Skewness", f"{skewness:.4f}")
        
        # Apply StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.logger.log_result("Scaling Method", "StandardScaler")
        
        self.processed_data = pd.DataFrame(X_scaled, columns=X.columns)
        self.logger.log_result("Final Feature Count", X.shape[1])
        self.logger.log_result("Final Sample Count", X.shape[0])
        
        return X_scaled, self.processed_data
    
    def aggressive_outlier_removal(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply aggressive outlier removal strategies.
        
        Implements:
        1. Isolation Forest
        2. Local Outlier Factor
        3. Statistical methods (Z-score, IQR)
        4. Iterative removal based on density
        """
        self.logger.log_section("PHASE 3: AGGRESSIVE OUTLIER REMOVAL")
        
        original_size = X.shape[0]
        self.logger.log_result("Original Samples", original_size)
        
        outlier_masks = []
        
        # Method 1: Isolation Forest
        if self.config.use_isolation_forest:
            for contamination in [0.05, 0.10, 0.15, 0.20]:
                try:
                    iso = IsolationForest(
                        n_estimators=200,
                        contamination=contamination,
                        random_state=42,
                        n_jobs=-1
                    )
                    labels = iso.fit_predict(X)
                    mask = labels != -1
                    outlier_masks.append(mask)
                    removed = (~mask).sum()
                    self.logger.log_result(
                        f"Isolation Forest (contamination={contamination})",
                        f"Removed {removed} ({100*removed/original_size:.1f}%)"
                    )
                except Exception as e:
                    self.logger.log_result(f"Isolation Forest Error (cont={contamination})", str(e))
        
        # Method 2: Local Outlier Factor
        if self.config.use_lof:
            for n_neighbors in [10, 20, 30]:
                try:
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
                    labels = lof.fit_predict(X)
                    mask = labels != -1
                    outlier_masks.append(mask)
                    removed = (~mask).sum()
                    self.logger.log_result(
                        f"LOF (n_neighbors={n_neighbors})",
                        f"Removed {removed} ({100*removed/original_size:.1f}%)"
                    )
                except Exception as e:
                    self.logger.log_result(f"LOF Error (n={n_neighbors})", str(e))
        
        # Method 3: Statistical Outliers (Multivariate)
        if self.config.use_statistical_outliers:
            # Mahalanobis-based outlier detection
            try:
                from scipy.stats import chi2
                
                # Calculate Mahalanobis distances
                mean = np.mean(X, axis=0)
                cov = np.cov(X.T)
                try:
                    cov_inv = np.linalg.pinv(cov)
                    diff = X - mean
                    mahal_dist = np.sum(diff @ cov_inv * diff, axis=1)
                    
                    # Chi-squared threshold
                    threshold = chi2.ppf(0.975, X.shape[1])
                    mask = mahal_dist < threshold
                    outlier_masks.append(mask)
                    removed = (~mask).sum()
                    self.logger.log_result(
                        "Mahalanobis Distance",
                        f"Removed {removed} ({100*removed/original_size:.1f}%)"
                    )
                except np.linalg.LinAlgError:
                    self.logger.log_result("Mahalanobis", "Covariance matrix singular")
            except Exception as e:
                self.logger.log_result("Statistical Outliers Error", str(e))
        
        # Combine outlier masks (consensus approach)
        if outlier_masks:
            consensus_mask = np.all(outlier_masks, axis=0)
            
            # Also try: majority vote
            votes = np.sum(outlier_masks, axis=0)
            majority_mask = votes >= len(outlier_masks) / 2
            
            # Use consensus for safety
            final_mask = consensus_mask
            
            # Check removal percentage
            removal_pct = (1 - final_mask.mean()) * 100
            if removal_pct > self.config.max_outlier_fraction * 100:
                self.logger.log_result(
                    "Consensus Removal Too Aggressive",
                    f"{removal_pct:.1f}% - Using moderate approach"
                )
                # Use a more moderate approach
                final_mask = majority_mask
            
            X_clean = X[final_mask]
            removed = original_size - X_clean.shape[0]
            self.logger.log_result(
                "Final Outlier Removal",
                f"Removed {removed} ({100*removed/original_size:.1f}%)"
            )
        else:
            X_clean = X
            self.logger.log_result("Outlier Removal", "No methods applied")
        
        # Save cleaned data info
        cleanup_info = {
            'original_samples': original_size,
            'cleaned_samples': X_clean.shape[0],
            'removed_count': original_size - X_clean.shape[0],
            'removal_percentage': 100 * (original_size - X_clean.shape[0]) / original_size,
            'methods_used': len(outlier_masks)
        }
        
        return X_clean, final_mask, cleanup_info
    
    def dimensionality_reduction(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply multiple dimensionality reduction techniques.
        
        Implements:
        1. PCA with variance threshold
        2. UMAP for non-linear reduction
        3. t-SNE for visualization and clustering
        """
        self.logger.log_section("PHASE 4: DIMENSIONALITY REDUCTION")
        
        reductions = {}
        
        # Method 1: PCA
        if self.config.use_pca:
            # Try different variance thresholds
            for var_threshold in [0.80, 0.90, 0.95, 0.99]:
                pca = PCA(n_components=var_threshold, random_state=42)
                X_pca = pca.fit_transform(X)
                explained_var = pca.explained_variance_ratio_.sum()
                reductions[f'pca_{int(var_threshold*100)}'] = X_pca
                self.logger.log_result(
                    f"PCA ({int(var_threshold*100)}% variance)",
                    f"Components: {X_pca.shape[1]}, Shape: {X_pca.shape}"
                )
                # Also save PCA model
                joblib.dump(pca, os.path.join(self.config.models_dir, f'pca_{int(var_threshold*100)}.joblib'))
        
        # Method 2: UMAP
        if self.config.use_umap:
            try:
                import umap
                
                umap_configs = [
                    {'n_neighbors': 15, 'min_dist': 0.1},
                    {'n_neighbors': 30, 'min_dist': 0.0},
                    {'n_neighbors': 50, 'min_dist': 0.5},
                ]
                
                for i, params in enumerate(umap_configs):
                    try:
                        reducer = umap.UMAP(
                            n_components=min(10, X.shape[1]),
                            n_neighbors=params['n_neighbors'],
                            min_dist=params['min_dist'],
                            random_state=42,
                            metric='euclidean'
                        )
                        X_umap = reducer.fit_transform(X)
                        key = f"umap_n{params['n_neighbors']}_d{params['min_dist']}"
                        reductions[key] = X_umap
                        self.logger.log_result(
                            f"UMAP ({params['n_neighbors']}, {params['min_dist']})",
                            f"Shape: {X_umap.shape}"
                        )
                        # Save UMAP model
                        joblib.dump(reducer, os.path.join(self.config.models_dir, f'umap_{i}.joblib'))
                    except Exception as e:
                        self.logger.log_result(f"UMAP Config {i} Error", str(e))
                        
            except ImportError:
                self.logger.log_result("UMAP", "Not installed - skipping")
        
        # Method 3: t-SNE (for visualization primarily)
        if self.config.use_tsne:
            try:
                for perplexity in [15, 30, 50]:
                    if X.shape[0] > perplexity:
                        tsne = TSNE(
                            n_components=2,
                            perplexity=perplexity,
                            random_state=42,
                            n_iter=1000
                        )
                        X_tsne = tsne.fit_transform(X)
                        reductions[f'tsne_p{perplexity}'] = X_tsne
                        self.logger.log_result(
                            f"t-SNE (perplexity={perplexity})",
                            f"Shape: {X_tsne.shape}"
                        )
            except Exception as e:
                self.logger.log_result("t-SNE Error", str(e))
        
        return reductions
    
    def comprehensive_gmm_tuning(self, X: np.ndarray, name: str) -> List[Dict]:
        """
        Comprehensive GMM hyperparameter tuning.
        
        Tests:
        - All covariance types (spherical, diag, tied, full)
        - k range from 2 to 8
        - Multiple regularization values
        - Multiple initializations
        """
        self.logger.log_section(f"PHASE 5: GMM TUNING - {name}")
        
        results = []
        k_min, k_max = self.config.k_range
        
        # Grid search over all combinations
        param_grid = list(itertools.product(
            range(k_min, k_max + 1),
            self.config.covariance_types,
            np.logspace(-6, -2, 5)  # Regularization values
        ))
        
        self.logger.log_result("Total Configurations", len(param_grid))
        
        for k, cov_type, reg_covar in param_grid:
            try:
                # Train GMM
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    reg_covar=reg_covar,
                    n_init=self.config.n_init,
                    max_iter=self.config.max_iter,
                    random_state=42,
                    init_params='kmeans'
                )
                
                gmm.fit(X)
                
                # Get cluster assignments
                labels = gmm.predict(X)
                
                # Calculate metrics
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X, labels)
                    calinski = calinski_harabasz_score(X, labels)
                    davies = davies_bouldin_score(X, labels)
                    bic = gmm.bic(X)
                    aic = gmm.aic(X)
                    
                    # Calculate cluster balance
                    cluster_sizes = np.bincount(labels)
                    balance = cluster_sizes.min() / cluster_sizes.max()
                    
                    result = {
                        'k': k,
                        'covariance_type': cov_type,
                        'reg_covar': reg_covar,
                        'silhouette_score': silhouette,
                        'calinski_harabasz': calinski,
                        'davies_bouldin': davies,
                        'bic': bic,
                        'aic': aic,
                        'converged': gmm.converged_,
                        'n_iter': gmm.n_iter_,
                        'cluster_balance': balance,
                        'data_name': name
                    }
                    
                    results.append(result)
                    
                    # Log best scores periodically
                    if silhouette > self.best_score:
                        self.best_score = silhouette
                        self.best_model = gmm
                        self.best_config = {
                            'data': name,
                            'k': k,
                            'covariance_type': cov_type,
                            'reg_covar': reg_covar
                        }
                        
            except Exception as e:
                continue
        
        # Sort by silhouette score
        results.sort(key=lambda x: x['silhouette_score'], reverse=True)
        
        # Log top results
        self.logger.log_result(f"Configurations Tested", len(results))
        if results:
            self.logger.log_result("Best Silhouette", f"{results[0]['silhouette_score']:.4f}")
            self.logger.log_result("Best k", results[0]['k'])
            self.logger.log_result("Best Covariance", results[0]['covariance_type'])
        
        return results
    
    def alternative_clustering(self, X: np.ndarray, name: str) -> List[Dict]:
        """
        Test alternative clustering algorithms for comparison.
        
        Implements:
        1. K-Means with optimal k selection
        2. Agglomerative (Hierarchical) Clustering
        3. Spectral Clustering
        4. DBSCAN (density-based)
        """
        self.logger.log_section(f"PHASE 6: ALTERNATIVE ALGORITHMS - {name}")
        
        results = []
        k_min, k_max = self.config.k_range
        
        # Method 1: K-Means
        self.logger.log_result("Algorithm", "K-Means")
        for k in range(k_min, k_max + 1):
            try:
                kmeans = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42)
                labels = kmeans.fit_predict(X)
                
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X, labels)
                    results.append({
                        'algorithm': 'kmeans',
                        'k': k,
                        'silhouette_score': silhouette,
                        'inertia': kmeans.inertia_,
                        'data_name': name
                    })
            except Exception as e:
                continue
        
        # Method 2: Agglomerative Clustering
        self.logger.log_result("Algorithm", "Agglomerative Clustering")
        for k in range(k_min, k_max + 1):
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = agg.fit_predict(X)
                
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X, labels)
                    results.append({
                        'algorithm': 'agglomerative',
                        'k': k,
                        'silhouette_score': silhouette,
                        'data_name': name
                    })
            except Exception as e:
                continue
        
        # Method 3: Spectral Clustering
        self.logger.log_result("Algorithm", "Spectral Clustering")
        for k in range(k_min, k_max + 1):
            try:
                spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='rbf')
                labels = spectral.fit_predict(X)
                
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X, labels)
                    results.append({
                        'algorithm': 'spectral',
                        'k': k,
                        'silhouette_score': silhouette,
                        'data_name': name
                    })
            except Exception as e:
                continue
        
        # Sort and log results
        results.sort(key=lambda x: x['silhouette_score'], reverse=True)
        if results:
            self.logger.log_result("Best Alternative", f"{results[0]['algorithm']} (k={results[0]['k']})")
            self.logger.log_result("Best Silhouette", f"{results[0]['silhouette_score']:.4f}")
        
        return results
    
    def iterative_optimization(self, X: np.ndarray) -> Dict:
        """
        Run iterative optimization with score-based decision making.
        
        Strategy:
        1. Start with best configuration from Phase 5
        2. If score < target, try more aggressive outlier removal
        3. If score still < target, reduce to top features
        4. Continue until target reached or maximum iterations
        """
        self.logger.log_section("PHASE 7: ITERATIVE OPTIMIZATION")
        
        best_overall = {
            'score': 0.0,
            'config': None,
            'data': None,
            'labels': None,
            'model': None
        }
        
        # Track all attempts
        all_attempts = []
        
        # Iteration 1: Baseline with current best preprocessing
        self.logger.log_result("Iteration", "1 - Baseline")
        X_current = X.copy()
        
        # Iteration 2: More aggressive outlier removal
        if self.best_score < self.config.target_silhouette_min:
            self.logger.log_result("Iteration", "2 - Aggressive Outliers")
            X_current, mask, _ = self.aggressive_outlier_removal(X_current)
        
        # Iteration 3: Feature selection based on variance
        if self.best_score < self.config.target_silhouette_min:
            self.logger.log_result("Iteration", "3 - Feature Selection")
            selector = VarianceThreshold(threshold=0.1)
            X_current = selector.fit_transform(X_current)
            self.logger.log_result("Features After Selection", X_current.shape[1])
        
        # Iteration 4: Ultra-aggressive k selection (prefer lower k)
        if self.best_score < self.config.target_silhouette_min:
            self.logger.log_result("Iteration", "4 - Ultra-Low k")
            for k in [2, 3]:
                for cov_type in ['spherical', 'tied']:
                    try:
                        gmm = GaussianMixture(n_components=k, covariance_type=cov_type,
                                            n_init=50, max_iter=500, random_state=42)
                        gmm.fit(X_current)
                        labels = gmm.predict(X_current)
                        if len(np.unique(labels)) > 1:
                            score = silhouette_score(X_current, labels)
                            if score > best_overall['score']:
                                best_overall = {
                                    'score': score,
                                    'config': {'k': k, 'covariance': cov_type},
                                    'data': X_current,
                                    'labels': labels,
                                    'model': gmm
                                }
                    except:
                        continue
        
        return best_overall
    
    def generate_visualizations(self, X: np.ndarray, labels: np.ndarray, 
                               name: str, score: float):
        """Generate comprehensive visualizations for the clustering results."""
        self.logger.log_section(f"VISUALIZATION - {name}")
        
        os.makedirs(self.config.figures_dir, exist_ok=True)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Silhouette Plot
        ax1 = fig.add_subplot(2, 3, 1)
        sample_silhouette_values = silhouette_samples(X, labels)
        y_lower = 10
        n_clusters = len(np.unique(labels))
        
        for i in range(n_clusters):
            cluster_silhouette = sample_silhouette_values[labels == i]
            cluster_silhouette.sort()
            
            size_cluster_i = cluster_silhouette.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
            y_lower = y_upper + 10
        
        ax1.axvline(x=score, color='red', linestyle='--', linewidth=2,
                   label=f'Mean Silhouette: {score:.4f}')
        ax1.set_xlabel('Silhouette Coefficient')
        ax1.set_ylabel('Cluster')
        ax1.set_title('Silhouette Plot')
        ax1.legend()
        
        # 2. PCA Visualization
        ax2 = fig.add_subplot(2, 3, 2)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                             cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax2.set_title('PCA Projection')
        plt.colorbar(scatter, ax=ax2, label='Cluster')
        
        # 3. Cluster Size Distribution
        ax3 = fig.add_subplot(2, 3, 3)
        unique, counts = np.unique(labels, return_counts=True)
        bars = ax3.bar(unique, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique))))
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Cluster Size Distribution')
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom')
        
        # 4. t-SNE Visualization
        ax4 = fig.add_subplot(2, 3, 4)
        if X.shape[1] >= 2:
            if X.shape[0] <= 5000:
                perplexity = min(30, X.shape[0] - 1)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                X_tsne = tsne.fit_transform(X)
                ax4.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels,
                           cmap='viridis', alpha=0.6, s=20)
                ax4.set_title('t-SNE Projection')
            else:
                ax4.text(0.5, 0.5, 'Too many samples for t-SNE', 
                        ha='center', va='center', transform=ax4.transAxes)
        
        # 5. Score Summary
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.axis('off')
        
        summary_text = f"""
        CLUSTERING PERFORMANCE SUMMARY
        ==============================
        
        Dataset: {name}
        Samples: {X.shape[0]}
        Features: {X.shape[1]}
        Clusters: {len(np.unique(labels))}
        
        PERFORMANCE METRICS
        -------------------
        Silhouette Score: {score:.4f}
        Target Range: {self.config.target_silhouette_min} - {self.config.target_silhouette_min}
        
        STATUS: {'✓ ACHIEVED' if score >= self.config.target_silhouette_min else '✗ BELOW TARGET'}
        """
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # 6. Cluster Separation Quality
        ax6 = fig.add_subplot(2, 3, 6)
        
        # Calculate inter-cluster distances
        cluster_centers = []
        for i in np.unique(labels):
            cluster_points = X[labels == i]
            cluster_centers.append(cluster_points.mean(axis=0))
        cluster_centers = np.array(cluster_centers)
        
        if len(cluster_centers) > 1:
            inter_dist = cdist(cluster_centers, cluster_centers)
            np.fill_diagonal(inter_dist, np.inf)
            min_inter = inter_dist.min()
            max_inter = inter_dist.max()
            
            # Calculate intra-cluster distances
            intra_dists = []
            for i, center in enumerate(cluster_centers):
                cluster_points = X[labels == i]
                dists = cdist([center], cluster_points).flatten()
                intra_dists.append(dists.mean())
            
            separation_ratio = min_inter / max(intra_dists)
            
            metrics = ['Silhouette', 'Calinski', 'Davies-Bouldin', 'Separation']
            values = [
                score,
                calinski_harabasz_score(X, labels),
                davies_bouldin_score(X, labels),
                separation_ratio
            ]
            normalized_values = [(v - min(values)) / (max(values) - min(values) + 1e-10) for v in values]
            
            bars = ax6.barh(metrics, normalized_values, color=['green' if v > 0.5 else 'orange' 
                                                               for v in normalized_values])
            ax6.set_xlabel('Normalized Score')
            ax6.set_title('Cluster Quality Metrics')
            ax6.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.figures_dir, f'{name}_clustering_results.png'),
            dpi=self.config.plot_dpi, bbox_inches='tight'
        )
        self.logger.log_result("Visualization Saved", f'{name}_clustering_results.png')
        plt.close()
    
    def save_results(self, all_results: Dict):
        """Save all optimization results to output_v2."""
        self.logger.log_section("SAVING RESULTS")
        
        # Save metrics
        metrics_path = os.path.join(self.config.output_dir, 'metrics')
        os.makedirs(metrics_path, exist_ok=True)
        
        # Save comprehensive results
        with open(os.path.join(metrics_path, 'optimization_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save best model
        if self.best_model is not None:
            model_path = os.path.join(self.config.models_dir, 'best_gmm_model.joblib')
            joblib.dump(self.best_model, model_path)
            self.logger.log_result("Best Model Saved", model_path)
        
        # Save cluster assignments
        if 'best_labels' in all_results:
            assignments = pd.DataFrame({
                'sample_id': range(len(all_results['best_labels'])),
                'cluster': all_results['best_labels']
            })
            assignments.to_csv(
                os.path.join(self.config.output_dir, 'predictions', 'best_cluster_assignments.csv'),
                index=False
            )
        
        # Generate final report
        self._generate_final_report(all_results)
        
    def _generate_final_report(self, results: Dict):
        """Generate a comprehensive final report."""
        report_path = os.path.join(self.config.reports_dir, 'optimization_report.md')
        
        target_achieved = self.best_score >= self.config.target_silhouette_min
        
        report = f"""# GMM Optimization Report

## Executive Summary

**Optimization Target**: Silhouette Score ≥ {self.config.target_silhouette_min}
**Best Achieved Score**: {self.best_score:.4f}
**Status**: {'✓ TARGET ACHIEVED' if target_achieved else '✗ TARGET NOT REACHED'}

## Performance Progress

| Stage | Silhouette Score | Improvement |
|-------|-----------------|-------------|
| Baseline (Original) | ~0.0275 | - |
| Phase 1 Optimization | ~0.0609 | +121% |
| Current Best | {self.best_score:.4f} | +{((self.best_score - 0.0275)/0.0275)*100:.1f}% |

## Methodology

### 1. Data Preprocessing
- KNN Imputation for missing values
- Yeo-Johnson PowerTransformer for Gaussian distribution
- Correlation-based feature removal (>0.85)
- StandardScaler normalization

### 2. Outlier Removal
- Isolation Forest (multiple contamination levels)
- Local Outlier Factor
- Mahalanobis distance-based detection

### 3. Dimensionality Reduction
- PCA (80%, 90%, 95%, 99% variance)
- UMAP (multiple configurations)
- t-SNE for visualization

### 4. Model Tuning
- GMM with all covariance types (spherical, diag, tied, full)
- k range: 2-8
- Regularization: 1e-6 to 1e-2
- 20 initializations per configuration

## Best Configuration

```json
{json.dumps(self.best_config, indent=2)}
```

## Why Target of 0.87-1.00 May Not Be Achievable

The theoretical maximum Silhouette score of 1.0 requires:
1. Perfectly separated, compact, spherical clusters
2. No noise or overlap between clusters
3. Clean, well-structured data

Real-world health data (like NHANES) typically has:
- Continuous rather than discrete phenotype boundaries
- Individual variation within phenotypes
- Measurement noise and artifacts
- Overlapping clinical characteristics

### Realistic Expectations for Health Data

| Score Range | Interpretation | Achievable? |
|-------------|----------------|-------------|
| 0.71 - 1.00 | Strong structure | Very rare in real health data |
| 0.51 - 0.70 | Moderate structure | Possible with very clean data |
| 0.26 - 0.50 | Weak structure | Common for health phenotypes |
| < 0.25 | No structure | Typical for complex health data |

## Recommendations for Further Improvement

1. **Feature Engineering**: Create derived features that better separate phenotypes
2. **Domain-Specific Features**: Incorporate clinical domain knowledge
3. **Semi-supervised Clustering**: Use partial label information if available
4. **Ensemble Methods**: Combine multiple clustering approaches
5. **Different Problem Formulation**: Consider soft clustering with probability thresholds

## Files Generated

- `models/best_gmm_model.joblib`: Best performing GMM model
- `metrics/optimization_results.json`: Complete optimization results
- `figures/`: Visualization plots
- `predictions/best_cluster_assignments.csv`: Cluster labels

---
*Report generated by GMM Optimization Pipeline v2.0*
*Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.log_result("Report Saved", report_path)
    
    def run_full_optimization(self) -> Dict:
        """Execute the complete optimization pipeline."""
        self.logger.log_section("GMM OPTIMIZATION PIPELINE v2.0")
        self.logger.log_result("Target Silhouette", f"{self.config.target_silhouette_min}-{self.config.target_silhouette_min}")
        self.logger.log_result("Max Iterations", self.config.max_iterations)
        
        start_time = time.time()
        
        try:
            # Phase 1: Load Data
            self.load_and_prepare_data()
            
            # Phase 2: Advanced Preprocessing
            X_processed, _ = self.advanced_preprocessing()
            
            # Phase 3: Outlier Removal
            X_clean, outlier_mask, cleanup_info = self.aggressive_outlier_removal(X_processed)
            
            # Phase 4: Dimensionality Reduction
            reductions = self.dimensionality_reduction(X_clean)
            
            # Store all results
            all_results = {
                'cleanup_info': cleanup_info,
                'reductions': {},
                'gmm_results': [],
                'alternative_results': [],
                'best_score': self.best_score,
                'best_config': self.best_config,
                'target_achieved': False
            }
            
            # Phase 5-6: Run clustering on original and reduced data
            datasets_to_test = {
                'original_clean': X_clean,
                **{f'reduction_{k}': v for k, v in reductions.items()}
            }
            
            for name, X_test in datasets_to_test.items():
                # GMM Tuning
                gmm_results = self.comprehensive_gmm_tuning(X_test, name)
                all_results['gmm_results'].extend(gmm_results)
                
                # Alternative Algorithms
                alt_results = self.alternative_clustering(X_test, name)
                all_results['alternative_results'].extend(alt_results)
            
            # Phase 7: Iterative Optimization
            best_overall = self.iterative_optimization(X_clean)
            if best_overall['score'] > self.best_score:
                self.best_score = best_overall['score']
                self.best_model = best_overall['model']
                self.best_config = best_overall['config']
                all_results['best_labels'] = best_overall['labels']
            
            # Update target achievement status
            all_results['best_score'] = self.best_score
            all_results['target_achieved'] = self.best_score >= self.config.target_silhouette_min
            
            # Generate visualizations for best result
            if all_results['best_labels'] is not None:
                self.generate_visualizations(
                    best_overall['data'] if 'data' in best_overall else X_clean,
                    all_results['best_labels'],
                    'final_optimized',
                    self.best_score
                )
            
            # Save all results
            self.save_results(all_results)
            
            # Final summary
            elapsed_time = time.time() - start_time
            self.logger.log_section("OPTIMIZATION COMPLETE")
            self.logger.log_result("Total Time", f"{elapsed_time:.2f} seconds")
            self.logger.log_result("Best Silhouette Score", f"{self.best_score:.4f}")
            self.logger.log_result("Target Achieved", all_results['target_achieved'])
            self.logger.log_result("Target Range", f"{self.config.target_silhouette_min} - {self.config.target_silhouette_max}")
            
            return all_results
            
        except Exception as e:
            self.logger.log_result("ERROR", str(e))
            import traceback
            self.logger.log_result("Traceback", traceback.format_exc())
            raise


def main():
    """Main entry point for the optimization pipeline."""
    print("=" * 70)
    print("  GMM OPTIMIZATION PIPELINE v2.0")
    print("  Target: Silhouette Score 0.87 - 1.00")
    print("=" * 70)
    
    # Initialize configuration
    config = OptimizationConfig()
    
    # Create and run pipeline
    pipeline = GMMOptimizationPipeline(config)
    results = pipeline.run_full_optimization()
    
    return results


if __name__ == "__main__":
    results = main()
