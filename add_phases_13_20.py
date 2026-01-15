#!/usr/bin/env python3
"""
Add Phases 13-20 to the GMM Health Phenotype Discovery Notebook
"""
import json

def add_phases_to_notebook(notebook_path, output_path):
    """Add Phases 13-20 to the notebook."""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    insert_index = len(notebook['cells'])
    new_cells = []
    
    # Phase 13: BIC/AIC Model Selection Analysis
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 13: BIC/AIC Model Selection Analysis\n",
            "\n",
            "This section provides comprehensive analysis of model selection criteria:\n",
            "\n",
            "- **BIC (Bayesian Information Criterion)**: Penalizes model complexity while rewarding goodness of fit. Lower BIC indicates a better model.\n",
            "\n",
            "- **AIC (Akaike Information Criterion)**: Based on information theory, AIC estimates the relative quality of models by balancing fit against complexity.\n",
            "\n",
            "- **Elbow Method Visualization**: Plots BIC and AIC across different numbers of components.\n",
            "\n",
            "- **Covariance Type Comparison**: Compares model performance across different covariance structures.\n"
        ]
    })
    
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PHASE 13: BIC/AIC MODEL SELECTION ANALYSIS",
            "print(\"=\" * 70)",
            "print(\"PHASE 13: BIC/AIC MODEL SELECTION ANALYSIS\")",
            "print(\"=\" * 70)",
            "",
            "from sklearn.mixture import GaussianMixture",
            "from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import warnings",
            "warnings.filterwarnings('ignore')",
            "",
            "# Calculate BIC and AIC for different numbers of components",
            "n_components_range = range(2, 15)",
            "bic_scores = []",
            "aic_scores = []",
            "silhouette_scores = []",
            "",
            "print(\"\\n[INFO] Computing BIC/AIC scores for different cluster counts...\")",
            "",
            "for n_components in n_components_range:",
            "    gmm = GaussianMixture(",
            "        n_components=n_components,",
            "        covariance_type='full',",
            "        reg_covar=1e-6,",
            "        n_init=10,",
            "        random_state=42",
            "    )",
            "    gmm.fit(SCALED_FEATURES)",
            "    ",
            "    bic_scores.append(gmm.bic(SCALED_FEATURES))",
            "    aic_scores.append(gmm.aic(SCALED_FEATURES))",
            "    ",
            "    labels = gmm.predict(SCALED_FEATURES)",
            "    silhouette_scores.append(silhouette_score(SCALED_FEATURES, labels))",
            "    ",
            "    print(f\"  n_components={n_components:2d}: BIC={bic_scores[-1]:.2f}, AIC={aic_scores[-1]:.2f}\")",
            "",
            "# Find optimal number of components",
            "best_bic_n = list(n_components_range)[np.argmin(bic_scores)]",
            "best_aic_n = list(n_components_range)[np.argmin(aic_scores)]",
            "",
            "print(f\"\\n[INFO] Optimal Components by BIC: {best_bic_n}\")",
            "print(f\"[INFO] Optimal Components by AIC: {best_aic_n}\")",
            "",
            "# Create visualization",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
            "",
            "# Plot 1: BIC and AIC",
            "ax1 = axes[0, 0]",
            "ax1.plot(list(n_components_range), bic_scores, 'b-o', linewidth=2, markersize=6, label='BIC')",
            "ax1.plot(list(n_components_range), aic_scores, 'r-s', linewidth=2, markersize=6, label='AIC')",
            "ax1.axvline(x=best_bic_n, color='blue', linestyle='--', alpha=0.7, label=f'Best BIC ({best_bic_n})')",
            "ax1.axvline(x=best_aic_n, color='red', linestyle='--', alpha=0.7, label=f'Best AIC ({best_aic_n})')",
            "ax1.set_xlabel('Number of Components', fontsize=12)",
            "ax1.set_ylabel('Score', fontsize=12)",
            "ax1.set_title('BIC and AIC Scores vs Number of Components', fontsize=14, fontweight='bold')",
            "ax1.legend(loc='best', fontsize=10)",
            "ax1.grid(True, alpha=0.3)",
            "",
            "# Plot 2: Silhouette Score",
            "ax2 = axes[0, 1]",
            "ax2.plot(list(n_components_range), silhouette_scores, 'g-^', linewidth=2, markersize=8)",
            "ax2.set_xlabel('Number of Components', fontsize=12)",
            "ax2.set_ylabel('Silhouette Score', fontsize=12)",
            "ax2.set_title('Silhouette Score vs Number of Components', fontsize=14, fontweight='bold')",
            "ax2.grid(True, alpha=0.3)",
            "",
            "# Plot 3: Calinski-Harabasz Index",
            "ax3 = axes[1, 0]",
            "calinski_scores = [calinski_harabasz_score(SCALED_FEATURES, GaussianMixture(n_components=n, random_state=42).fit_predict(SCALED_FEATURES)) for n in n_components_range]",
            "ax3.plot(list(n_components_range), calinski_scores, 'm-v', linewidth=2, markersize=8)",
            "ax3.set_xlabel('Number of Components', fontsize=12)",
            "ax3.set_ylabel('Calinski-Harabasz Index', fontsize=12)",
            "ax3.set_title('Calinski-Harabasz Index vs Number of Components', fontsize=14, fontweight='bold')",
            "ax3.grid(True, alpha=0.3)",
            "",
            "# Plot 4: Davies-Bouldin Index",
            "ax4 = axes[1, 1]",
            "davies_scores = [davies_bouldin_score(SCALED_FEATURES, GaussianMixture(n_components=n, random_state=42).fit_predict(SCALED_FEATURES)) for n in n_components_range]",
            "ax4.plot(list(n_components_range), davies_scores, 'c-d', linewidth=2, markersize=8)",
            "ax4.set_xlabel('Number of Components', fontsize=12)",
            "ax4.set_ylabel('Davies-Bouldin Index', fontsize=12)",
            "ax4.set_title('Davies-Bouldin Index vs Number of Components', fontsize=14, fontweight='bold')",
            "ax4.grid(True, alpha=0.3)",
            "",
            "plt.tight_layout()",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'plots', '06_bic_aic_analysis.png'), dpi=300, bbox_inches='tight')",
            "plt.show()",
            "",
            "print(f\"\\n[OK] Figure saved: {os.path.join(OUTPUT_DIR, 'figures', 'plots', '06_bic_aic_analysis.png')}\")"
        ]
    })
    
    # Phase 14: Radar Charts for Cluster Profiles
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 14: Radar Charts for Cluster Profiles\n",
            "\n",
            "This section creates radar charts to visualize cluster profiles across multiple health dimensions.\n"
        ]
    })
    
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PHASE 14: RADAR CHARTS FOR CLUSTER PROFILES",
            "print(\"=\" * 70)",
            "print(\"RADAR CHARTS FOR CLUSTER PROFILES\")",
            "print(\"=\" * 70)",
            "",
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "def create_radar_chart(ax, categories, values, title, color, alpha=0.25):",
            "    \"\"\"Create a radar chart for cluster profile visualization.\"\"\"",
            "    N = len(categories)",
            "    angles = [n / float(N) * 2 * np.pi for n in range(N)]",
            "    angles += angles[:1]",
            "    values = list(values) + values[:1]",
            "    ax.plot(angles, values, linewidth=2, linestyle='solid', color=color)",
            "    ax.fill(angles, values, color=color, alpha=alpha)",
            "    ax.set_xticks(angles[:-1])",
            "    ax.set_xticklabels(categories, fontsize=9)",
            "    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)",
            "",
            "# Define features for radar chart",
            "radar_features = ['age', 'bmi', 'systolic_bp_mmHg', 'fasting_glucose_mg_dL', 'phq9_total_score']",
            "feature_labels = ['Age', 'BMI', 'Systolic BP', 'Glucose', 'PHQ-9']",
            "",
            "# Normalize features",
            "normalized_data = data.copy()",
            "for col in radar_features:",
            "    min_val = data[col].min()",
            "    max_val = data[col].max()",
            "    if max_val > min_val:",
            "        normalized_data[col] = (data[col] - min_val) / (max_val - min_val)",
            "    else:",
            "        normalized_data[col] = 0.5",
            "",
            "# Calculate cluster profiles",
            "cluster_profiles = normalized_data.groupby(data['cluster'])[radar_features].mean()",
            "",
            "# Create radar charts",
            "n_clusters = len(cluster_profiles)",
            "colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))",
            "",
            "fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5), subplot_kw=dict(polar=True))",
            "if n_clusters == 1:",
            "    axes = [axes]",
            "",
            "for idx, (cluster_id, profile) in enumerate(cluster_profiles.iterrows()):",
            "    create_radar_chart(",
            "        axes[idx],",
            "        feature_labels,",
            "        profile.values,",
            "        f'Cluster {cluster_id}\\n(n={{len(data[data[\"cluster\"]==cluster_id])}})',",
            "        colors[idx]",
            "    )",
            "",
            "plt.tight_layout()",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'plots', '07_radar_charts.png'), dpi=300, bbox_inches='tight')",
            "plt.show()",
            "",
            "print(f\"[OK] Figure saved: {os.path.join(OUTPUT_DIR, 'figures', 'plots', '07_radar_charts.png')}\")"
        ]
    })
    
    # Phase 15: Feature Distribution by Cluster
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 15: Feature Distribution by Cluster\n",
            "\n",
            "This section visualizes the distribution of key health features within each cluster.\n"
        ]
    })
    
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PHASE 15: FEATURE DISTRIBUTION BY CLUSTER",
            "print(\"=\" * 70)",
            "print(\"FEATURE DISTRIBUTION BY CLUSTER\")",
            "print(\"=\" * 70)",
            "",
            "import seaborn as sns",
            "",
            "# Features to plot",
            "plot_features = ['age', 'bmi', 'systolic_bp_mmHg', 'fasting_glucose_mg_dL', 'phq9_total_score']",
            "plot_titles = ['Age (years)', 'BMI (kg/m²)', 'Systolic BP (mmHg)', 'Fasting Glucose (mg/dL)', 'PHQ-9 Score']",
            "",
            "# Create box plots",
            "fig, axes = plt.subplots(2, 3, figsize=(15, 10))",
            "axes = axes.flatten()",
            "",
            "for idx, (feature, title) in enumerate(zip(plot_features, plot_titles)):",
            "    sns.boxplot(data=data, x='cluster', y=feature, ax=axes[idx], palette='Set2')",
            "    axes[idx].set_title(f'{title} by Cluster', fontsize=12, fontweight='bold')",
            "",
            "axes[5].axis('off')",
            "plt.tight_layout()",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'plots', '09_feature_boxplots.png'), dpi=300, bbox_inches='tight')",
            "plt.show()",
            "",
            "print(f\"[OK] Figure saved: {os.path.join(OUTPUT_DIR, 'figures', 'plots', '09_feature_boxplots.png')}\")",
            "",
            "# Create violin plots",
            "fig, axes = plt.subplots(2, 3, figsize=(15, 10))",
            "axes = axes.flatten()",
            "",
            "for idx, (feature, title) in enumerate(zip(plot_features, plot_titles)):",
            "    sns.violinplot(data=data, x='cluster', y=feature, ax=axes[idx], palette='Set2', inner='box')",
            "    axes[idx].set_title(f'{title} Distribution by Cluster', fontsize=12, fontweight='bold')",
            "",
            "axes[5].axis('off')",
            "plt.tight_layout()",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'plots', '10_feature_violin.png'), dpi=300, bbox_inches='tight')",
            "plt.show()",
            "",
            "print(f\"[OK] Figure saved: {os.path.join(OUTPUT_DIR, 'figures', 'plots', '10_feature_violin.png')}\")"
        ]
    })
    
    # Phase 16: Cluster Size and Proportion Analysis
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 16: Cluster Size and Proportion Analysis\n",
            "\n",
            "This section analyzes the distribution of samples across clusters.\n"
        ]
    })
    
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PHASE 16: CLUSTER SIZE AND PROPORTION ANALYSIS",
            "print(\"=\" * 70)",
            "print(\"CLUSTER SIZE AND PROPORTION ANALYSIS\")",
            "print(\"=\" * 70)",
            "",
            "# Calculate cluster sizes and proportions",
            "cluster_sizes = data['cluster'].value_counts().sort_index()",
            "cluster_proportions = (cluster_sizes / len(data)) * 100",
            "",
            "print(\"\\nCluster Distribution:\")",
            "print(\"-\" * 50)",
            "print(f\"{'Cluster':<10} {'Count':<10} {'Proportion':<15}\")",
            "print(\"-\" * 50)",
            "for cluster in cluster_sizes.index:",
            "    print(f\"{cluster:<10} {cluster_sizes[cluster]:<10} {cluster_proportions[cluster]:.2f}%\")",
            "print(\"-\" * 50)",
            "print(f\"{'Total':<10} {len(data):<10} {100.0:.2f}%\")",
            "",
            "# Create visualizations",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
            "",
            "# Pie chart",
            "colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_sizes)))",
            "axes[0].pie(cluster_sizes, labels=[f'Cluster {i}\\n(n={v})' for i, v in cluster_sizes.items()],",
            "            autopct='%1.1f%%', colors=colors, explode=[0.02]*len(cluster_sizes),",
            "            shadow=True, startangle=90)",
            "axes[0].set_title('Cluster Distribution (Pie Chart)', fontsize=14, fontweight='bold')",
            "",
            "# Bar chart",
            "bars = axes[1].bar(cluster_sizes.index, cluster_sizes.values, color=colors, edgecolor='black', alpha=0.8)",
            "axes[1].set_xlabel('Cluster', fontsize=12)",
            "axes[1].set_ylabel('Number of Individuals', fontsize=12)",
            "axes[1].set_title('Cluster Size Distribution (Bar Chart)', fontsize=14, fontweight='bold')",
            "axes[1].set_xticks(cluster_sizes.index)",
            "",
            "for bar, count in zip(bars, cluster_sizes.values):",
            "    height = bar.get_height()",
            "    axes[1].annotate(f'{count}\\n({count/len(data)*100:.1f}%)',",
            "                    xy=(bar.get_x() + bar.get_width() / 2, height),",
            "                    xytext=(0, 3), textcoords=\"offset points\",",
            "                    ha='center', va='bottom', fontsize=10)",
            "",
            "plt.tight_layout()",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'plots', '11_cluster_distribution.png'), dpi=300, bbox_inches='tight')",
            "plt.show()",
            "",
            "print(f\"\\n[OK] Figure saved: {os.path.join(OUTPUT_DIR, 'figures', 'plots', '11_cluster_distribution.png')}\")"
        ]
    })
    
    # Phase 17: Probability Uncertainty Visualization
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 17: Probability Uncertainty Visualization\n",
            "\n",
            "This section visualizes the uncertainty in cluster assignments.\n"
        ]
    })
    
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PHASE 17: PROBABILITY UNCERTAINTY VISUALIZATION",
            "print(\"=\" * 70)",
            "print(\"PROBABILITY UNCERTAINTY VISUALIZATION\")",
            "print(\"=\" * 70)",
            "",
            "# Get maximum probability for each sample",
            "max_probs = probabilities.max(axis=1)",
            "data['max_probability'] = max_probs",
            "",
            "# Categorize confidence",
            "high_conf = (max_probs >= 0.8).sum()",
            "mod_conf = ((max_probs >= 0.5) & (max_probs < 0.8)).sum()",
            "low_conf = (max_probs < 0.5).sum()",
            "",
            "print(f\"\\nConfidence Level Summary:\")",
            "print(f\"  High Confidence (>=0.8): {high_conf} ({100*high_conf/len(data):.1f}%)\")",
            "print(f\"  Moderate Confidence (0.5-0.8): {mod_conf} ({100*mod_conf/len(data):.1f}%)\")",
            "print(f\"  Low Confidence (<0.5): {low_conf} ({100*low_conf/len(data):.1f}%)\")",
            "",
            "# Create visualizations",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
            "",
            "# Histogram",
            "axes[0, 0].hist(max_probs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)",
            "axes[0, 0].axvline(x=0.8, color='green', linestyle='--', linewidth=2, label='High (0.8)')",
            "axes[0, 0].axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Moderate (0.5)')",
            "axes[0, 0].set_xlabel('Maximum Assignment Probability', fontsize=12)",
            "axes[0, 0].set_ylabel('Frequency', fontsize=12)",
            "axes[0, 0].set_title('Distribution of Assignment Confidence', fontsize=14, fontweight='bold')",
            "axes[0, 0].legend()",
            "",
            "# Pie chart",
            "confidence_counts = [high_conf, mod_conf, low_conf]",
            "confidence_labels = [f'High\\n(n={high_conf})', f'Moderate\\n(n={mod_conf})', f'Low\\n(n={low_conf})']",
            "colors = ['#2ecc71', '#f39c12', '#e74c3c']",
            "axes[0, 1].pie(confidence_counts, labels=confidence_labels, autopct='%1.1f%%', ",
            "               colors=colors, explode=[0.02, 0.02, 0.05], shadow=True)",
            "axes[0, 1].set_title('Confidence Level Distribution', fontsize=14, fontweight='bold')",
            "",
            "# Box plot by cluster",
            "cluster_prob_data = [data[data['cluster'] == c]['max_probability'].values for c in sorted(data['cluster'].unique())]",
            "bp = axes[1, 0].boxplot(cluster_prob_data, labels=[f'Cluster {c}' for c in sorted(data['cluster'].unique())], patch_artist=True)",
            "for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(cluster_prob_data)))):",
            "    patch.set_facecolor(color)",
            "axes[1, 0].set_xlabel('Cluster', fontsize=12)",
            "axes[1, 0].set_ylabel('Maximum Probability', fontsize=12)",
            "axes[1, 0].set_title('Assignment Confidence by Cluster', fontsize=14, fontweight='bold')",
            "",
            "# Heatmap",
            "prob_means = probabilities.mean(axis=0)",
            "im = axes[1, 1].imshow(prob_means.T, aspect='auto', cmap='YlOrRd')",
            "axes[1, 1].set_xlabel('Sample Index (sorted)', fontsize=12)",
            "axes[1, 1].set_ylabel('Cluster', fontsize=12)",
            "axes[1, 1].set_title('Cluster Probability Heatmap', fontsize=14, fontweight='bold')",
            "axes[1, 1].set_yticks(range(len(prob_means)))",
            "axes[1, 1].set_yticklabels([f'Cluster {i}' for i in range(len(prob_means))])",
            "plt.colorbar(im, ax=axes[1, 1])",
            "",
            "plt.tight_layout()",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'plots', '12_probability_uncertainty.png'), dpi=300, bbox_inches='tight')",
            "plt.show()",
            "",
            "print(f\"\\n[OK] Figure saved: {os.path.join(OUTPUT_DIR, 'figures', 'plots', '12_probability_uncertainty.png')}\")"
        ]
    })
    
    # Phase 18: Demographics and Cluster Association
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 18: Demographics and Cluster Association\n",
            "\n",
            "This section analyzes the relationship between demographic variables and cluster membership.\n"
        ]
    })
    
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PHASE 18: DEMOGRAPHICS AND CLUSTER ASSOCIATION",
            "print(\"=\" * 70)",
            "print(\"DEMOGRAPHICS AND CLUSTER ASSOCIATION\")",
            "print(\"=\" * 70)",
            "",
            "from scipy.stats import chi2_contingency",
            "",
            "demographic_vars = ['gender', 'race/ethnicity', 'age_group']",
            "demographic_names = ['Gender', 'Race/Ethnicity', 'Age Group']",
            "",
            "print(\"\\nChi-Square Tests for Demographics:\")",
            "print(\"=\" * 70)",
            "",
            "for var, name in zip(demographic_vars, demographic_names):",
            "    if var in data.columns:",
            "        contingency_table = pd.crosstab(data[var], data['cluster'])",
            "        chi2, p_value, dof, expected = chi2_contingency(contingency_table)",
            "        print(f\"\\n{name}:\")",
            "        print(f\"  Chi-square statistic: {chi2:.4f}\")",
            "        print(f\"  P-value: {p_value:.6f}\")",
            "        print(f\"  Significant: {'Yes' if p_value < 0.05 else 'No'}\")",
            "",
            "# Create visualizations",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
            "",
            "if 'gender' in data.columns:",
            "    gender_cluster = pd.crosstab(data['gender'], data['cluster'], normalize='index') * 100",
            "    gender_cluster.plot(kind='bar', ax=axes[0, 0], colormap='Set2', edgecolor='black')",
            "    axes[0, 0].set_title('Cluster Distribution by Gender', fontsize=14, fontweight='bold')",
            "    axes[0, 0].tick_params(axis='x', rotation=0)",
            "",
            "if 'age_group' in data.columns:",
            "    age_cluster = pd.crosstab(data['age_group'], data['cluster'], normalize='index') * 100",
            "    age_cluster.plot(kind='bar', ax=axes[0, 1], colormap='Set2', edgecolor='black')",
            "    axes[0, 1].set_title('Cluster Distribution by Age Group', fontsize=14, fontweight='bold')",
            "    axes[0, 1].tick_params(axis='x', rotation=45)",
            "",
            "if 'race/ethnicity' in data.columns:",
            "    race_cluster = pd.crosstab(data['cluster'], data['race/ethnicity'], normalize='index') * 100",
            "    race_cluster.plot(kind='barh', stacked=True, ax=axes[1, 0], colormap='Set2', edgecolor='black')",
            "    axes[1, 0].set_title('Race/Ethnicity by Cluster', fontsize=14, fontweight='bold')",
            "",
            "for cluster in sorted(data['cluster'].unique()):",
            "    cluster_ages = data[data['cluster'] == cluster]['age']",
            "    axes[1, 1].hist(cluster_ages, bins=20, alpha=0.5, label=f'Cluster {cluster}', edgecolor='black')",
            "axes[1, 1].set_title('Age Distribution by Cluster', fontsize=14, fontweight='bold')",
            "axes[1, 1].set_xlabel('Age (years)', fontsize=12)",
            "axes[1, 1].legend()",
            "",
            "plt.tight_layout()",
            "plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'plots', '13_demographic_analysis.png'), dpi=300, bbox_inches='tight')",
            "plt.show()",
            "",
            "print(f\"\\n[OK] Figure saved: {os.path.join(OUTPUT_DIR, 'figures', 'plots', '13_demographic_analysis.png')}\")"
        ]
    })
    
    # Phase 19: Final Summary and Export
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 19: Final Summary and Export\n",
            "\n",
            "This section provides a comprehensive summary of the GMM clustering analysis and exports all results.\n"
        ]
    })
    
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# PHASE 19: FINAL SUMMARY AND EXPORT",
            "print(\"=\" * 70)",
            "print(\"FINAL SUMMARY AND EXPORT\")",
            "print(\"=\" * 70)",
            "",
            "# Complete summary statistics",
            "print(\"\\n\" + \"-\" * 70)",
            "print(\"COMPLETE PROJECT SUMMARY\")",
            "print(\"-\" * 70)",
            "",
            "print(f\"DATASET CHARACTERISTICS:\")",
            "print(f\"  Total Samples: {len(data):,}\")",
            "print(f\"  Features Used: {len(FEATURE_COLUMNS)}\")",
            "",
            "print(f\"\\nMODEL CONFIGURATION:\")",
            "print(f\"  Algorithm: Gaussian Mixture Models (GMM)\")",
            "print(f\"  Number of Components: {n_clusters}\")",
            "",
            "print(f\"\\nMODEL PERFORMANCE:\")",
            "print(f\"  BIC Score: {full_eval['bic']:.2f}\")",
            "print(f\"  AIC Score: {full_eval['aic']:.2f}\")",
            "print(f\"  Silhouette Score: {full_eval['silhouette']:.4f}\")",
            "",
            "print(f\"\\nCLUSTER SUMMARY:\")",
            "for c in range(n_clusters):",
            "    cluster_subset = data[data['cluster'] == c]",
            "    print(f\"  Cluster {c} ({len(cluster_subset):,} individuals, {100*len(cluster_subset)/len(data):.1f}%):\")",
            "    print(f\"    Mean Age: {cluster_subset['age'].mean():.1f} years\")",
            "    print(f\"    Mean BMI: {cluster_subset['bmi'].mean():.1f}\")",
            "",
            "print(f\"\\nUNCERTAINTY ANALYSIS:\")",
            "print(f\"  High Confidence (>=0.8): {high_conf} ({100*high_conf/len(data):.1f}%)\")",
            "print(f\"  Moderate Confidence: {mod_conf} ({100*mod_conf/len(data):.1f}%)\")",
            "print(f\"  Low Confidence (<0.5): {low_conf} ({100*low_conf/len(data):.1f}%)\")",
            "",
            "# Export all results",
            "print(\"\\n[INFO] Exporting results...\")",
            "",
            "# Save complete dataset with cluster assignments",
            "export_df = data.copy()",
            "export_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions', 'complete_cluster_assignments.csv'), index=False)",
            "print(\"  [OK] Complete cluster assignments saved\")",
            "",
            "# Save cluster profiles",
            "cluster_profiles_export = cluster_profiles.copy()",
            "cluster_profiles_export['n_individuals'] = cluster_sizes.values",
            "cluster_profiles_export['proportion'] = cluster_proportions.values",
            "cluster_profiles_export.to_csv(os.path.join(OUTPUT_DIR, 'cluster_profiles', 'detailed_cluster_profiles.csv'))",
            "print(\"  [OK] Detailed cluster profiles saved\")",
            "",
            "# Save grid search results",
            "grid_results_df = pd.DataFrame({",
            "    'n_components': list(n_components_range),",
            "    'bic': bic_scores,",
            "    'aic': aic_scores",
            "})",
            "grid_results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_selection', 'grid_search_results.csv'), index=False)",
            "print(\"  [OK] Grid search results saved\")",
            "",
            "# Save probability assignments",
            "prob_df = pd.DataFrame(probabilities, columns=[f'Cluster_{i}_Prob' for i in range(n_clusters)])",
            "prob_df['Predicted_Cluster'] = data['cluster'].values",
            "prob_df['Max_Probability'] = data['max_probability'].values",
            "prob_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions', 'cluster_probabilities.csv'), index=False)",
            "print(\"  [OK] Cluster probabilities saved\")",
            "",
            "print(\"\\n\" + \"=\" * 70)",
            "print(\"ANALYSIS COMPLETE\")",
            "print(\"=\" * 70)",
            "print(f\"\\nAll results saved to: {OUTPUT_DIR}\")"
        ]
    })
    
    # Phase 20: References
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 20: References\n",
            "\n",
            "1. McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models. John Wiley & Sons.\n",
            "\n",
            "2. Schwarz, G. (1978). Estimating the dimension of a model. The Annals of Statistics.\n",
            "\n",
            "3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.\n"
        ]
    })
    
    # Insert all new cells before References
    for cell in new_cells:
        notebook['cells'].insert(insert_index, cell)
        insert_index += 1
    
    # Write updated notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Successfully added {len(new_cells)} new cells (Phases 13-20) to the notebook.")
    print(f"Updated notebook saved to: {output_path}")

if __name__ == "__main__":
    add_phases_to_notebook(
        '/workspace/GMM_Health_Phenotype_Discovery.ipynb',
        '/workspace/GMM_Health_Phenotype_Discovery.ipynb'
    )
