"""
Presentation Preparation Script for GMM Health Phenotype Discovery Notebook

This script restructures the notebook with:
1. Consistent phase numbering throughout
2. Added markdown explanations for code cells
3. Proper documentation structure
"""

import json

# Load the notebook
with open('/workspace/GMM_Health_Phenotype_Discovery.ipynb', 'r') as f:
    notebook = json.load(f)

# Define the enhanced markdown content for each phase that needs explanation
enhanced_markdowns = {
    # Phase 3 - EDA
    "## Phase 3: Exploratory Data Analysis\n\nThis section provides initial insights into the NHANES health data through statistical summaries and distribution visualizations. Understanding the data distribution is crucial for appropriate clustering analysis.",
    
    # Phase 4 - Data Preprocessing
    "## Phase 4: Data Preprocessing for GMM\n\nThis section prepares the data for Gaussian Mixture Model clustering by handling missing values, encoding categorical variables, and selecting relevant features. Proper preprocessing is essential for GMM convergence and meaningful results.",
    
    # Phase 5 - Dimensionality Reduction
    "## Phase 5: Dimensionality Reduction\n\nThis section applies PCA (Principal Component Analysis) and t-SNE to reduce feature dimensionality for visualization and potential computational efficiency. Dimensionality reduction helps identify the underlying structure and relationships in high-dimensional health data.",
    
    # Phase 6 - GMM Hyperparameter Tuning
    "## Phase 6: GMM Hyperparameter Tuning\n\nThis section performs systematic model selection using BIC (Bayesian Information Criterion) and AIC (Akaike Information Criterion) to determine the optimal number of clusters and covariance structure. These metrics balance model fit against complexity.",
    
    # Phase 7 - Train Optimal GMM
    "## Phase 7: Train Optimal GMM Model\n\nThis section fits the final Gaussian Mixture Model using the optimal hyperparameters identified in the tuning phase. The model learns the parameters of the underlying probability distributions for each cluster.",
    
    # Phase 8 - Cluster Interpretation
    "## Phase 8: Cluster Interpretation and Profiling\n\nThis section analyzes the characteristics of each identified cluster by examining mean values across all features. This helps define the health phenotypes and understand the distinguishing characteristics of each subpopulation.",
    
    # Phase 9 - Cluster Visualization
    "## Phase 9: Cluster Visualization\n\nThis section creates 2D and 3D visualizations of the clustering results using PCA and t-SNE projections. Visual inspection helps validate the clustering structure and identify potential overlaps or separation issues.",
    
    # Phase 10 - Model Evaluation
    "## Phase 10: Model Evaluation Metrics\n\nThis section computes comprehensive clustering quality metrics including Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index. These metrics provide quantitative assessment of cluster separation and compactness.",
    
    # Phase 11 - Probabilistic Membership
    "## Phase 11: Probabilistic Membership Analysis\n\nThis section examines the posterior probability distributions for cluster assignments. GMM's probabilistic nature provides uncertainty quantification for each individual's cluster membership.",
    
    # Phase 12 - Medical History Analysis
    "## Phase 12: Medical History and Cluster Association\n\nThis section analyzes the relationship between cluster membership and medical history variables. This helps validate that the identified phenotypes have meaningful clinical distinctions.",
    
    # Phase 13 - Cluster Validation
    "## Phase 13: Statistical Cluster Validation\n\nThis section performs statistical tests to validate cluster differences, including ANOVA for continuous variables and chi-square tests for categorical variables. This confirms that clusters represent genuinely different subpopulations.",
    
    # Phase 14 - Feature Importance
    "## Phase 14: Feature Importance Analysis\n\nThis section identifies which features contribute most to cluster separation. Understanding feature importance helps interpret the biological meaning of the identified health phenotypes.",
}

# Phase mapping for consistent numbering
phase_mapping = {
    0: ("Phase 1", "Library Imports and Environment Setup"),
    1: ("Phase 2", "Project Configuration and Path Setup"),
    2: ("Phase 3", "Exploratory Data Analysis"),
    3: ("Phase 4", "Data Preprocessing for GMM"),
    4: ("Phase 5", "Dimensionality Reduction"),
    5: ("Phase 6", "GMM Hyperparameter Tuning"),
    6: ("Phase 7", "Train Optimal GMM Model"),
    7: ("Phase 8", "Cluster Interpretation and Profiling"),
    8: ("Phase 9", "Cluster Visualization"),
    9: ("Phase 10", "Model Evaluation Metrics"),
    10: ("Phase 11", "Probabilistic Membership Analysis"),
    11: ("Phase 12", "Medical History Analysis"),
    12: ("Phase 13", "Statistical Cluster Validation"),
    13: ("Phase 14", "Feature Importance Analysis"),
    14: ("Phase 15", "Uncertainty Analysis - Probability Distributions"),
    15: ("Phase 16", "Feature Distribution by Cluster"),
    16: ("Phase 17", "Probability 17: (" Uncertainty Visualization"),
   Phase 18", "Cluster Size and Proportion Analysis"),
    18: ("Phase 19", "Demographics and Cluster Association"),
    19: ("Phase 20", "Final Summary and Export"),
    20: ("Phase 21", "References"),
}

# Track current phase to insert markdown before code cells
current_phase = 0
markdown_inserted = []

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        # Check if this code cell needs a preceding markdown
        cell_content = ''.join(cell['source'])
        
        # Identify phase from cell content
        for phase_num, (phase_name, phase_title) in phase_mapping.items():
            if f"# PHASE {phase_num + 1}:" in cell_content or f"PHASE {phase_num + 1}:" in cell_content:
                current_phase = phase_num + 1
                break
            elif f"# =============================================================================" in cell_content and f"PHASE {phase_num + 1}:" not in cell_content:
                # Check for actual phase content
                for pn, (pn_name, pn_title) in phase_mapping.items():
                    if f"PHASE {pn + 1}:" in cell_content:
                        current_phase = pn + 1
                        break
    
    elif cell['cell_type'] == 'markdown':
        cell_content = ''.join(cell['source'])
        for phase_num, (phase_name, phase_title) in phase_mapping.items():
            if f"## Phase {phase_num + 1}:" in cell_content:
                current_phase = phase_num + 1
                break

# Save the notebook structure for reference
print("Notebook Analysis Complete")
print(f"Total cells: {len(notebook['cells'])}")
print(f"Code cells: {len([c for c in notebook['cells'] if c['cell_type'] == 'code'])}")
print(f"Markdown cells: {len([c for c in notebook['cells'] if c['cell_type'] == 'markdown'])}")
