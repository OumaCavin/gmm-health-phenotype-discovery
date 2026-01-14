# GMM Health Phenotype Discovery - Project Documentation

<div align="center">

**MSc Public Health Data Science - SDS6217 Advanced Machine Learning**  
**Student ID:** SDS6/46982/2025  
**Author:** Cavin Otieno  
**Date:** January 2025  
**Institution:** University of Nairobi

</div>

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Methodology](#methodology)
4. [Project Structure](#project-structure)
5. [Installation and Setup](#installation-and-setup)
6. [Usage Guide](#usage-guide)
7. [Path Management System](#path-management-system)
8. [Output Files](#output-files)
9. [Key Findings](#key-findings)
10. [References](#references)

---

## Project Overview

This project applies Gaussian Mixture Models (GMM) to identify latent subpopulations in the National Health and Nutrition Examination Survey (NHANES) health data. The analysis demonstrates how probabilistic clustering can capture population heterogeneity that traditional hard-clustering methods may miss.

### Research Questions

1. Can GMM identify distinct health phenotypes in the NHANES population?
2. What are the characteristic profiles of each identified cluster?
3. How does hyperparameter tuning affect model performance?

### Objectives

- Implement a comprehensive GMM clustering pipeline
- Perform systematic hyperparameter optimization using BIC/AIC criteria
- Identify and interpret meaningful health subpopulations
- Provide reproducible methodology for public health research

---

## Dataset Description

**Source:** National Health and Nutrition Examination Survey (NHANES)  
**Location:** `data/raw/nhanes_health_data.csv`  
**Samples:** 5,000 adult respondents  
**Variables:** 47 health indicators

### Variable Categories

| Category | Variables | Description |
|----------|-----------|-------------|
| **Demographics** | 5 | sex, age, race/ethnicity, education, income |
| **Body Measures** | 4 | weight, height, BMI, waist circumference |
| **Blood Pressure** | 2 | systolic BP, diastolic BP |
| **Laboratory** | 5 | total cholesterol, HDL, LDL, glucose, insulin |
| **Behavioral** | 6 | smoking, alcohol, physical activity levels |
| **Medical Conditions** | 8 | arthritis, heart disease, stroke, etc. |
| **Mental Health** | 10 | PHQ-9 depression screening items |
| **Derived Features** | 4 | clinical category assignments |

### Data Quality

- Missing values handled through mean imputation for continuous variables
- Categorical variables encoded as numerical indices
- Continuous variables standardized for GMM implementation

---

## Methodology

### Gaussian Mixture Models (GMM)

GMM is a probabilistic clustering algorithm that models data as a mixture of multiple Gaussian distributions. Unlike K-Means which assigns each point to a single cluster, GMM provides soft assignments based on posterior probabilities.

#### Key Advantages for Public Health

1. **Probabilistic Cluster Assignment**: Each individual receives a probability of belonging to each cluster
2. **Population Heterogeneity**: Captures continuous distributions of risk factors
3. **Flexible Covariance Structures**: Four types (full, tied, diag, spherical) for different cluster shapes
4. **Uncertainty Quantification**: Confidence in cluster assignments for clinical decision-making

### Hyperparameter Tuning

The following hyperparameters are optimized through grid search:

| Parameter | Description | Search Space |
|-----------|-------------|--------------|
| n_components | Number of clusters | 2-10 |
| covariance_type | Covariance structure | full, tied, diag, spherical |
| reg_covar | Regularization term | 1e-6, 1e-5, 1e-4 |
| max_iter | Maximum iterations | 100, 200, 500 |
| n_init | Initializations | 5, 10, 20 |

### Model Selection Criteria

- **BIC (Bayesian Information Criterion)**: Primary selection criterion
- **AIC (Akaike Information Criterion)**: Secondary validation
- **Silhouette Score**: Cluster cohesion and separation quality

---

## Project Structure

```
gmm-health-phenotype-discovery/
├── GMM_Health_Phenotype_Discovery.ipynb    # Main analysis notebook
├── GMM_Health_Phenotype_Discovery.py       # Standalone Python script
├── README.md                                # Project overview
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore rules
├── LICENSE                                  # MIT License
├── docs/
│   └── project_documentation.md            # This documentation
├── data/
│   └── raw/
│       └── nhanes_health_data.csv          # NHANES dataset (5000 samples, 47 vars)
├── output_v2/
│   ├── reports/                            # Analysis reports
│   ├── metrics/                            # Model evaluation metrics
│   ├── predictions/                        # Cluster assignments
│   ├── cluster_profiles/                   # Cluster characteristic profiles
│   └── logs/                               # Execution logs
├── models/
│   ├── gmm_clustering/                     # Trained GMM models
│   └── scalers/                            # Data scalers
└── figures/
    └── plots/                              # Visualization outputs
```

---

## Installation and Setup

### Prerequisites

- Python 3.12.3 or higher
- pip package manager
- Git (for version control)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/OumaCavin/gmm-health-phenotype-discovery.git
   cd gmm-health-phenotype-discovery
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook GMM_Health_Phenotype_Discovery.ipynb
   ```

---

## Usage Guide

### Running the Analysis

1. Open `GMM_Health_Phenotype_Discovery.ipynb` in Jupyter
2. Execute cells sequentially from top to bottom
3. Each phase builds upon previous results

### Phase Breakdown

| Phase | Title | Description |
|-------|-------|-------------|
| 1 | Library Imports | Import required packages and configure environment |
| 2 | Data Loading | Load NHANES dataset from persistent path |
| 3 | Exploratory Data Analysis | Explore dataset characteristics and distributions |
| 4 | Data Preprocessing | Handle missing values and standardize features |
| 5 | Dimensionality Reduction | PCA and t-SNE for visualization |
| 6 | Hyperparameter Tuning | Grid search with BIC optimization |
| 7 | Train Optimal Model | Fit best GMM configuration |
| 8 | Cluster Analysis | Profile and interpret identified clusters |
| 9 | Cluster Visualization | Visualize clusters in reduced dimensions |
| 10 | Model Evaluation | Comprehensive model assessment |
| 11 | Probabilistic Membership | Analyze cluster membership probabilities |
| 12 | Conclusions | Summary and future research directions |

---

## Path Management System

This project implements a comprehensive path management system to ensure all outputs are saved to persistent file paths rather than in-memory storage.

### Path Configuration

All paths are centralized in the PROJECT CONFIGURATION cell:

```python
# Main directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v2')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

# Phase-specific subdirectories
PHASE_DIRS = {
    'data': os.path.join(DATA_DIR, 'raw'),
    'processed': os.path.join(DATA_DIR, 'processed'),
    'reports': os.path.join(OUTPUT_DIR, 'reports'),
    'logs': os.path.join(OUTPUT_DIR, 'logs'),
    'plots': os.path.join(FIGURES_DIR, 'plots')
}
```

### Saving Utilities

The project includes standardized saving functions:

| Function | Purpose | Location |
|----------|---------|----------|
| `save_fig()` | Save figures in multiple formats | `figures/plots/` |
| `save_model()` | Persist trained models | `models/gmm_clustering/` |
| `save_data()` | Export data to CSV | `output_v2/subdirectories/` |

### Figure Display and Saving

All visualization cells follow this pattern:

```python
# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...
plt.tight_layout()

# Display figure
plt.show()

# Save to persistent path
save_fig(fig, 'figure_name', subdir='plots')
```

This ensures:
1. Figures are displayed in the notebook
2. Figures are saved to persistent storage
3. Outputs survive kernel restarts

---

## Output Files

### Generated Outputs

After running the full notebook, the following files are generated:

#### Figures (`figures/plots/`)

| File | Description |
|------|-------------|
| `01_health_indicator_distributions.png` | Distribution of health variables |
| `02_correlation_heatmap.png` | Feature correlation matrix |
| `03_dimensionality_reduction.png` | PCA/t-SNE visualization |
| `04_cluster_profiles_heatmap.png` | Cluster characteristic profiles |
| `05_gmm_clustering_results.png` | Final cluster visualization |

#### Models (`models/gmm_clustering/`)

| File | Description |
|------|-------------|
| `standard_scaler.joblib` | Fitted StandardScaler |
| `gmm_optimal_model.joblib` | Best GMM model |

#### Data (`output_v2/`)

| File | Description |
|------|-------------|
| `cluster_profiles/gmm_cluster_profiles.csv` | Cluster statistics |
| `predictions/gmm_cluster_predictions.csv` | Sample cluster assignments |
| `metrics/project_config.json` | Configuration summary |

---

## Key Findings

### Model Performance

- **Optimal Number of Clusters**: [To be filled after execution]
- **Best Covariance Type**: [To be filled after execution]
- **BIC Score**: [To be filled after execution]
- **AIC Score**: [To be filled after execution]

### Cluster Interpretations

| Cluster | Profile | Key Characteristics |
|---------|---------|---------------------|
| Cluster 1 | [Name] | [Description] |
| Cluster 2 | [Name] | [Description] |
| ... | ... | ... |

### Implications for Public Health

1. **Risk Stratification**: Identified clusters can inform targeted interventions
2. **Resource Allocation**: Cluster sizes guide healthcare planning
3. **Prevention Strategies**: Behavioral patterns inform prevention programs

---

## References

1. McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models. Wiley.
2. CDC National Health and Nutrition Examination Survey. https://www.cdc.gov/nchs/nhanes/
3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, 2011.
4. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**Cavin Otieno**  
MSc Public Health Data Science  
Advanced Machine Learning (SDS6217)  
University of Nairobi  

GitHub: [@OumaCavin](https://github.com/OumaCavin)
