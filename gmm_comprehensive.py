#!/usr/bin/env python3
"""
GMM Health Phenotype Discovery - COMPREHENSIVE OPTIMIZATION
=============================================================
Finds the optimal balance between cluster count and separation quality
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import joblib
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 80)
print("GMM COMPREHENSIVE OPTIMIZATION")
print("=" * 80)

# Load data
DATA_PATH = 'data/raw/nhanes_health_data.csv'
data = pd.read_csv(DATA_PATH)
print(f"\n[1] Data: {data.shape[0]} samples × {data.shape[1]} variables")

# =============================================================================
# OPTIMAL FEATURE SELECTION
# =============================================================================
print("\n[2] OPTIMAL FEATURE SELECTION")

# Key clinical features that discriminate health phenotypes
CLINICAL_FEATURES = [
    'bmi', 'waist_circumference_cm', 'systolic_bp_mmHg', 'diastolic_bp_mmHg',
    'hdl_cholesterol_mg_dL', 'fasting_glucose_mg_dL', 'total_cholesterol_mg_dL',
    'ldl_cholesterol_mg_dL', 'insulin_uU_mL', 'phq9_total_score',
    'general_health_rating', 'age'
]

FEATURE_COLS = [f for f in CLINICAL_FEATURES if f in data.columns]
print(f"[INFO] Using {len(FEATURE_COLS)} clinical features")

X = data[FEATURE_COLS].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# PCA FOR NOISE REDUCTION
# =============================================================================
print("\n[3] DIMENSIONALITY REDUCTION")

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"[OK] PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} dimensions")
print(f"[INFO] Variance retained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# =============================================================================
# COMPREHENSIVE MODEL SELECTION
# =============================================================================
print("\n[4] COMPREHENSIVE MODEL SELECTION")

k_range = range(2, 12)  # Test 2-11 clusters
cov_types = ['tied', 'spherical']  # Faster covariance types
n_init = 10

results = []

print(f"{'k':^4} | {'Cov':^8} | {'BIC':^12} | {'Silhouette':^10} | {'Davies':^8} | {'Score':^8}")
print("-" * 70)

for k in k_range:
    for cov in cov_types:
        gmm = GaussianMixture(
            n_components=k, 
            covariance_type=cov, 
            n_init=n_init, 
            random_state=42, 
            max_iter=200
        )
        gmm.fit(X_pca)
        
        labels = gmm.predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        bic = gmm.bic(X_pca)
        davies = davies_bouldin_score(X_pca, labels)
        calinski = calinski_harabasz_score(X_pca, labels)
        
        # Normalize for composite score
        sil_norm = (sil + 0.1) / 0.2  # Normalize to 0-1 range
        bic_norm = 1 - (bic - 150000) / 50000  # Approximate normalization
        davies_norm = 1 / (davies + 1)
        
        # Composite score emphasizing silhouette for good separation
        composite = 0.5 * sil_norm + 0.25 * bic_norm + 0.25 * davies_norm
        
        results.append({
            'k': k, 'cov': cov, 'sil': sil, 'bic': bic, 
            'davies': davies, 'calinski': calinski, 'composite': composite,
            'model': gmm
        })
        
        print(f"{k:^4} | {cov:^8} | {bic:^12.0f} | {sil:^10.4f} | {davies:^8.4f} | {composite:^8.4f}")

print("-" * 70)

# =============================================================================
# FIND OPTIMAL MODEL
# =============================================================================
print("\n[5] FINDING OPTIMAL MODEL")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Find best by composite score
best_composite = results_df.loc[results_df['composite'].idxmax()]
best_sil = results_df.loc[results_df['sil'].idxmax()]

print(f"\n[INFO] Best by Composite: k={int(best_composite['k'])}, cov={best_composite['cov']}")
print(f"       Silhouette: {best_composite['sil']:.4f}, BIC: {best_composite['bic']:.0f}")
print(f"\n[INFO] Best by Silhouette: k={int(best_sil['k'])}, cov={best_sil['cov']}")
print(f"       Silhouette: {best_sil['sil']:.4f}, BIC: {best_sil['bic']:.0f}")

# Select model - prioritize silhouette for better cluster separation
# But also consider having multiple phenotypes (k >= 4)
results_df_filtered = results_df[results_df['k'] >= 3]
best_balanced = results_df_filtered.loc[results_df_filtered['composite'].idxmax()]

print(f"\n[INFO] Best Balanced (k>=3): k={int(best_balanced['k'])}, cov={best_balanced['cov']}")
print(f"       Silhouette: {best_balanced['sil']:.4f}, BIC: {best_balanced['bic']:.0f}")

# Use the best balanced model for meaningful phenotypes
best_idx = best_balanced['model']
best_model = best_balanced['model']
best_k = int(best_balanced['k'])
best_cov = best_balanced['cov']

print("\n" + "=" * 60)
print(f"[OPTIMAL MODEL: k={best_k}, cov={best_cov}]")
print("=" * 60)

# =============================================================================
# FINAL EVALUATION
# =============================================================================
print("\n[6] FINAL EVALUATION")

labels = best_model.predict(X_pca)
probs = best_model.predict_proba(X_pca)
max_probs = probs.max(axis=1)
entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

final_sil = silhouette_score(X_pca, labels)
final_bic = best_model.bic(X_pca)
final_davies = davies_bouldin_score(X_pca, labels)
final_calinski = calinski_harabasz_score(X_pca, labels)
final_aic = best_model.aic(X_pca)

# Cluster distribution
unique, counts = np.unique(labels, return_counts=True)
print(f"\n[INFO] Cluster Distribution:")
for cluster, count in zip(unique, counts):
    pct = 100 * count / len(labels)
    print(f"  Cluster {cluster}: {count:,} samples ({pct:.1f}%)")

# Confidence analysis
high_conf = np.sum(max_probs >= 0.8)
mod_conf = np.sum((max_probs >= 0.5) & (max_probs < 0.8))
low_conf = np.sum(max_probs < 0.5)

print(f"\n[INFO] Assignment Confidence:")
print(f"  High (≥0.8):   {high_conf:,} ({100*high_conf/len(labels):.1f}%)")
print(f"  Moderate:      {mod_conf:,} ({100*mod_conf/len(labels):.1f}%)")
print(f"  Low (<0.5):    {low_conf:,} ({100*low_conf/len(labels):.1f}%)")
print(f"\n  Mean Prob:     {max_probs.mean():.4f}")
print(f"  Mean Entropy:  {entropy.mean():.4f}")

# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("PERFORMANCE IMPROVEMENT SUMMARY")
print("=" * 80)

sil_improvement = ((final_sil - 0.0275) / 0.0275) * 100
bic_change = ((final_bic - 149836.90) / 149836.90) * 100

print(f"\n{'Metric':<25} {'Before':<15} {'After':<15} {'Change':>12}")
print("-" * 70)
print(f"{'Silhouette Score':<25} {'0.0275':<15} {final_sil:<15.4f} {sil_improvement:>11.1f}%")
print(f"{'BIC Score':<25} {'149,836.90':<15} {final_bic:<15.0f} {bic_change:>11.1f}%")
print(f"{'Davies-Bouldin':<25} {'~2.0 (est)':<15} {final_davies:<15.4f} {((final_davies-2.0)/2.0)*100:>11.1f}%")
print("-" * 70)

print(f"\n[FINAL MODEL METRICS]")
print(f"  • Silhouette Score:      {final_sil:.4f} (higher=better)")
print(f"  • Calinski-Harabasz:     {final_calinski:.2f} (higher=better)")
print(f"  • Davies-Bouldin:        {final_davies:.4f} (lower=better)")
print(f"  • BIC Score:             {final_bic:.2f} (lower=better)")
print(f"  • AIC Score:             {final_aic:.2f} (lower=better)")
print(f"  • High Confidence:       {100*high_conf/len(labels):.1f}%")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n[7] GENERATING VISUALIZATIONS")

fig = plt.figure(figsize=(20, 16))
fig.suptitle(f'GMM Health Phenotype Discovery - OPTIMIZED (k={best_k}, Sil={final_sil:.4f})', 
             fontsize=18, fontweight='bold', y=0.98)

colors = plt.cm.Set2(np.linspace(0, 1, best_k))

# Plot 1: Model Selection Heatmap
ax1 = fig.add_subplot(2, 3, 1)
pivot_sil = results_df.pivot_table(values='sil', index='k', columns='cov')
sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, 
            vmin=0, vmax=max(0.1, pivot_sil.max().max()))
ax1.set_title('Silhouette Score by k and Covariance', fontsize=12, fontweight='bold')
ax1.set_xlabel('Covariance Type')
ax1.set_ylabel('Number of Clusters (k)')

# Plot 2: BIC Curves
ax2 = fig.add_subplot(2, 3, 2)
for cov in cov_types:
    subset = results_df[results_df['cov'] == cov].sort_values('k')
    ax2.plot(subset['k'], subset['bic'], '-o', label=cov, linewidth=2, markersize=6)
ax2.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={best_k}')
ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
ax2.set_ylabel('BIC Score', fontsize=11)
ax2.set_title('BIC Score by k and Covariance Type', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Silhouette Curves
ax3 = fig.add_subplot(2, 3, 3)
for cov in cov_types:
    subset = results_df[results_df['cov'] == cov].sort_values('k')
    ax3.plot(subset['k'], subset['sil'], '-o', label=cov, linewidth=2, markersize=6)
ax3.axhline(y=0.0275, color='red', linestyle=':', linewidth=2, label='Before (0.0275)')
ax3.axvline(x=best_k, color='green', linestyle='--', linewidth=2, label=f'Optimal k={best_k}')
ax3.set_xlabel('Number of Clusters (k)', fontsize=11)
ax3.set_ylabel('Silhouette Score', fontsize=11)
ax3.set_title('Silhouette Score by k and Covariance Type', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Cluster Distribution
ax4 = fig.add_subplot(2, 3, 4)
bars = ax4.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Cluster', fontsize=11)
ax4.set_ylabel('Number of Samples', fontsize=11)
ax4.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
ax4.set_xticks(unique)
for bar, count in zip(bars, counts):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{count:,}\n({100*count/len(labels):.1f}%)', ha='center', fontsize=10)

# Plot 5: PCA Visualization
ax5 = fig.add_subplot(2, 3, 5)
scatter = ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set2', alpha=0.6, s=15)
ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax5.set_title('Cluster Visualization (PCA)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax5, label='Cluster')

# Plot 6: Confidence Distribution
ax6 = fig.add_subplot(2, 3, 6)
ax6.hist(max_probs, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax6.axvline(x=0.8, color='green', linestyle='--', linewidth=2, label='High (0.8)')
ax6.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Moderate (0.5)')
ax6.set_xlabel('Maximum Cluster Probability', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Assignment Confidence Distribution', fontsize=12, fontweight='bold')
ax6.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('optimized_gmm_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Visualization saved: optimized_gmm_results.png")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n[8] SAVING RESULTS")

# Results summary
results_data = {
    'optimal_k': best_k,
    'covariance_type': best_cov,
    'silhouette_score': float(final_sil),
    'calinski_harabasz_score': float(final_calinski),
    'davies_bouldin_score': float(final_davies),
    'bic_score': float(final_bic),
    'aic_score': float(final_aic),
    'n_features_used': len(FEATURE_COLS),
    'features_used': FEATURE_COLS,
    'n_samples': len(data),
    'pca_components': int(X_pca.shape[1]),
    'high_confidence_pct': float(100 * high_conf / len(labels)),
    'mean_entropy': float(entropy.mean()),
    'cluster_sizes': {int(k): int(v) for k, v in zip(unique, counts)},
    'confidence_distribution': {
        'high': int(high_conf),
        'moderate': int(mod_conf),
        'low': int(low_conf)
    }
}

with open('optimized_gmm_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)
print("[OK] Results saved: optimized_gmm_results.json")

# Save model and transformers
joblib.dump(best_model, 'optimized_gmm_model.joblib')
joblib.dump(scaler, 'optimized_scaler.joblib')
joblib.dump(pca, 'optimized_pca.joblib')
print("[OK] Model saved: optimized_gmm_model.joblib")
print("[OK] Scaler saved: optimized_scaler.joblib")
print("[OK] PCA saved: optimized_pca.joblib")

# Save cluster assignments
data['cluster'] = labels
data['max_probability'] = max_probs
data['entropy'] = entropy
data.to_csv('optimized_cluster_assignments.csv', index=False)
print("[OK] Cluster assignments saved: optimized_cluster_assignments.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"""
KEY IMPROVEMENTS IMPLEMENTED:
1. Feature Selection:
   - Selected {len(FEATURE_COLS)} clinically relevant features
   - Focused on cardiovascular, metabolic, and mental health indicators
   - Removed noisy/redundant features

2. Dimensionality Reduction:
   - PCA retained {sum(pca.explained_variance_ratio_)*100:.1f}% of variance
   - Reduced from {X_scaled.shape[1]} to {X_pca.shape[1]} dimensions

3. Optimal Model Selection:
   - Tested k=2 to 11 clusters
   - Tested tied and spherical covariance types
   - Used 10 random initializations per configuration
   - Selected k={best_k} using composite scoring

PERFORMANCE RESULTS:
- Silhouette: 0.0275 → {final_sil:.4f} ({sil_improvement:.1f}% improvement)
- BIC: 149,837 → {final_bic:.0f}
- High-confidence assignments: {100*high_conf/len(labels):.1f}%

IDENTIFIED PHENOTYPES: {best_k} distinct health clusters
""")
