#!/usr/bin/env python3
"""
GMM Health Phenotype Discovery - HIGHLY OPTIMIZED VERSION
==========================================================
Fast execution with key performance improvements
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
print("GMM OPTIMIZATION - FAST VERSION")
print("=" * 80)

# Load data
DATA_PATH = 'data/raw/nhanes_health_data.csv'
data = pd.read_csv(DATA_PATH)
print(f"\n[1] Data: {data.shape[0]} samples × {data.shape[1]} variables")

# Feature engineering - select key clinical features
CLINICAL_FEATURES = [
    'bmi', 'waist_circumference_cm', 'systolic_bp_mmHg', 'diastolic_bp_mmHg',
    'hdl_cholesterol_mg_dL', 'fasting_glucose_mg_dL', 'total_cholesterol_mg_dL',
    'ldl_cholesterol_mg_dL', 'insulin_uU_mL', 'phq9_total_score',
    'general_health_rating', 'age'
]

# Use only features that exist
FEATURE_COLS = [f for f in CLINICAL_FEATURES if f in data.columns]
print(f"[2] Features: {len(FEATURE_COLS)}")

X = data[FEATURE_COLS].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"[3] PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} dims ({sum(pca.explained_variance_ratio_)*100:.1f}% variance)")

# Model selection - fast grid search
print("\n[4] Model Selection...")
k_range = range(2, 8)
cov_types = ['tied', 'spherical']  # Faster covariance types
n_init = 5  # Fewer initializations

best_score = -np.inf
best_k = 2
best_cov = 'tied'
best_model = None

results = []

for k in k_range:
    for cov in cov_types:
        gmm = GaussianMixture(n_components=k, covariance_type=cov, 
                             n_init=n_init, random_state=42, max_iter=100)
        gmm.fit(X_pca)
        labels = gmm.predict(X_pca)
        
        sil = silhouette_score(X_pca, labels)
        bic = gmm.bic(X_pca)
        davies = davies_bouldin_score(X_pca, labels)
        
        # Composite score (higher is better)
        score = 0.5 * sil + 0.3 * (1/(davies+1)) + 0.2 * (1/(bic/100000))
        
        results.append({'k': k, 'cov': cov, 'sil': sil, 'bic': bic, 'davies': davies, 'score': score})
        
        if score > best_score:
            best_score = score
            best_k = k
            best_cov = cov
            best_model = gmm
            
        print(f"  k={k}, {cov}: sil={sil:.4f}, bic={bic:.0f}")

print(f"\n[5] Best: k={best_k}, cov={best_cov}, sil={results[-1]['sil']:.4f}")

# Final evaluation
labels = best_model.predict(X_pca)
probs = best_model.predict_proba(X_pca)
max_probs = probs.max(axis=1)

final_sil = silhouette_score(X_pca, labels)
final_bic = best_model.bic(X_pca)
final_davies = davies_bouldin_score(X_pca, labels)
final_calinski = calinski_harabasz_score(X_pca, labels)

# Cluster distribution
unique, counts = np.unique(labels, return_counts=True)
print(f"\n[6] Clusters: {dict(zip(unique, counts))}")

# Confidence
high_conf = np.sum(max_probs >= 0.8)
print(f"\n[7] High confidence: {100*high_conf/len(labels):.1f}%")

# Results
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"\n{'Metric':<25} {'Before':<12} {'After':<12} {'Change':>10}")
print("-" * 60)
print(f"{'Silhouette Score':<25} {'0.0275':<12} {final_sil:<12.4f} {((final_sil-0.0275)/0.0275)*100:>9.1f}%")
print(f"{'BIC Score':<25} {'149837':<12} {final_bic:<12.0f} {((final_bic-149837)/149837)*100:>9.1f}%")
print("-" * 60)
print(f"\n[Final Metrics]")
print(f"  Silhouette: {final_sil:.4f}")
print(f"  BIC: {final_bic:.0f}")
print(f"  Davies-Bouldin: {final_davies:.4f}")
print(f"  High-confidence: {100*high_conf/len(labels):.1f}%")

# Visualization
print("\n[8] Creating visualization...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle('GMM Optimization Results', fontsize=16, fontweight='bold')

colors = plt.cm.Set2(np.linspace(0, 1, best_k))

# Plot 1: Results table
ax1 = fig.add_subplot(2, 3, 1)
ax1.axis('off')
results_text = "Model Selection Results\n" + "="*40 + "\n\n"
for r in results:
    results_text += f"k={r['k']}, {r['cov']}: sil={r['sil']:.4f}\n"
ax1.text(0.1, 0.9, results_text, transform=ax1.transAxes, fontsize=10, 
         verticalalignment='top', fontfamily='monospace')

# Plot 2: Silhouette comparison
ax2 = fig.add_subplot(2, 3, 2)
df_results = pd.DataFrame(results)
for cov in cov_types:
    subset = df_results[df_results['cov'] == cov]
    ax2.plot(subset['k'], subset['sil'], '-o', label=cov, linewidth=2)
ax2.axhline(y=0.0275, color='red', linestyle='--', label='Before (0.0275)')
ax2.set_xlabel('k (clusters)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette by k and Covariance')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cluster sizes
ax3 = fig.add_subplot(2, 3, 3)
bars = ax3.bar(unique, counts, color=colors, edgecolor='black')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Count')
ax3.set_title(f'Cluster Distribution (k={best_k})')
for bar, count in zip(bars, counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{count}', ha='center', fontsize=10)

# Plot 4: PCA
ax4 = fig.add_subplot(2, 3, 4)
scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set2', alpha=0.5, s=10)
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_title('Clusters in PCA Space')
plt.colorbar(scatter, ax=ax4, label='Cluster')

# Plot 5: Confidence
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(max_probs, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax5.axvline(x=0.8, color='green', linestyle='--', label='High (0.8)', linewidth=2)
ax5.set_xlabel('Max Probability')
ax5.set_ylabel('Frequency')
ax5.set_title('Assignment Confidence')
ax5.legend()

# Plot 6: Performance comparison
ax6 = fig.add_subplot(2, 3, 6)
metrics = ['Before', 'After']
silhouettes = [0.0275, final_sil]
bars = ax6.bar(metrics, silhouettes, color=['lightcoral', 'lightgreen'], edgecolor='black')
ax6.set_ylabel('Silhouette Score')
ax6.set_title('Performance Improvement')
for bar, val in zip(bars, silhouettes):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('optimized_gmm_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: optimized_gmm_results.png")

# Save outputs
results_data = {
    'optimal_k': best_k,
    'covariance_type': best_cov,
    'silhouette_score': float(final_sil),
    'bic_score': float(final_bic),
    'davies_bouldin_score': float(final_davies),
    'calinski_harabasz_score': float(final_calinski),
    'high_confidence_pct': float(100 * high_conf / len(labels)),
    'cluster_sizes': {int(k): int(v) for k, v in zip(unique, counts)},
    'features_used': FEATURE_COLS,
    'pca_components': X_pca.shape[1]
}

with open('optimized_gmm_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)
print("[OK] Saved: optimized_gmm_results.json")

# Save model
joblib.dump(best_model, 'optimized_gmm_model.joblib')
joblib.dump(scaler, 'optimized_scaler.joblib')
joblib.dump(pca, 'optimized_pca.joblib')
print("[OK] Saved: optimized_gmm_model.joblib")

# Cluster assignments
data['cluster'] = labels
data['max_probability'] = max_probs
data.to_csv('optimized_cluster_assignments.csv', index=False)
print("[OK] Saved: optimized_cluster_assignments.csv")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"""
IMPROVEMENTS:
- Silhouette: 0.0275 → {final_sil:.4f} ({((final_sil-0.0275)/0.0275)*100:.1f}% improvement)
- BIC: 149,837 → {final_bic:.0f}
- High-confidence: {100*high_conf/len(labels):.1f}%
""")
