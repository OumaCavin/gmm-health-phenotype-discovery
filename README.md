# GMM Health Phenotype Discovery

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12.3-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003A57?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-1.7.1-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## MSc Public Health Data Science - SDS6217 Advanced Machine Learning

---

### Group 6 Members

| Student ID | Student Name |
|------------|--------------|
| SDS6/46982/2024 | Cavin Otieno |
| SDS6/46284/2024 | Joseph Ongoro Marindi |
| SDS6/47543/2024 | Laura Nabalayo Kundu |
| SDS6/47545/2024 | Nevin Khaemba |

**Date:** January 2025  
**Institution:** University of Nairobi  

---

### Project Overview

This comprehensive data science project applies Gaussian Mixture Models (GMM) to identify latent subpopulations in NHANES health data, demonstrating how probabilistic clustering can capture population heterogeneity that traditional hard-clustering methods may miss.

### Dataset

**Source:** National Health and Nutrition Examination Survey (NHANES)  
**Source Organization:** Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS)  
**Source URL:** https://www.cdc.gov/nchs/nhanes/  
**Location:** `data/raw/nhanes_health_data.csv`  
**Samples:** 5,000 respondents  
**Variables:** 47 health indicators

### Key Features

- **Probabilistic Clustering**: Captures uncertainty in cluster assignments
- **Hyperparameter Tuning**: Systematic grid search optimization using BIC/AIC criteria
- **Population Phenotype Discovery**: Identifies distinct health subgroups
- **Academic Rigor**: Comprehensive methodology suitable for MSc-level assessment
- **Interactive Web App**: Streamlit application for real-time predictions

### Why GMM for Public Health?

1. **Probabilistic Cluster Assignment**: Unlike K-Means which forces hard assignments, GMM provides posterior probabilities. Each individual receives a probability of belonging to each cluster, which is critical for health decisions where uncertainty quantification matters.

2. **Modeling Population Heterogeneity**: Health populations naturally exhibit continuous distributions of risk factors. GMM captures latent subgroups without imposing artificial boundaries, reflecting the biological reality of disease processes.

3. **Flexibility Through Covariance Structures**: Four covariance types allow modeling of various cluster shapes. Full covariance captures elongated, correlated clusters, while diagonal and spherical options provide computational efficiency.

4. **Uncertainty Quantification**: Confidence in cluster assignments can be assessed, which is important for clinical decision-making and resource allocation.

### Technologies Used

| Category | Technology | Version |
|----------|------------|---------|
| Language | Python | 3.12.3 |
| Notebook | Jupyter | Notebook |
| Machine Learning | scikit-learn | Latest |
| Data Manipulation | NumPy, Pandas | Latest |
| Visualization | Matplotlib, Seaborn | Latest |
| Web Application | Streamlit | 1.28.0+ |

### Repository Structure

```
gmm-health-phenotype-discovery/
├── GMM_Health_Phenotype_Discovery.ipynb    # Main analysis notebook (12 phases)
├── GMM_Health_Phenotype_Discovery.py       # Standalone Python script
├── app.py                                  # Streamlit interactive web app
├── README.md                               # Project overview
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore rules
├── LICENSE                                 # MIT License
├── docs/
│   └── project_documentation.md            # Comprehensive documentation
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
│   └── gmm_clustering/                     # Trained GMM models
└── figures/
    └── plots/                              # Visualization outputs
```

### Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
joblib
jupyter
streamlit
```

### Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run Jupyter notebook:**
```bash
jupyter notebook GMM_Health_Phenotype_Discovery.ipynb
```

**Run Streamlit app:**
```bash
streamlit run app.py
```

### Data Dictionary

The dataset includes 47 health indicators across 8 categories:

| Category | Variables | Description |
|----------|-----------|-------------|
| Demographics | 5 | sex, age, race/ethnicity, education, income |
| Body Measures | 4 | weight, height, BMI, waist circumference |
| Blood Pressure | 2 | systolic BP, diastolic BP |
| Laboratory | 5 | total cholesterol, HDL, LDL, glucose, insulin |
| Behavioral | 8 | smoking, alcohol, physical activity levels |
| Medical Conditions | 8 | arthritis, heart disease, stroke, etc. |
| Mental Health | 10 | PHQ-9 depression screening items |
| Derived Features | 4 | clinical category assignments |

**Source Reference:** National Health and Nutrition Examination Survey (NHANES). (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

### Methodology

**Algorithm:** Gaussian Mixture Models (GMM) with Expectation-Maximization (EM) algorithm

**Model Selection:**
- Primary: BIC (Bayesian Information Criterion)
- Secondary: AIC (Akaike Information Criterion)
- Validation: Silhouette Score

**Hyperparameter Tuning:**
- Grid search over n_components (2-10)
- Covariance types: full, tied, diag, spherical
- Regularization: 1e-6, 1e-5, 1e-4

### Optimization Results

| Approach | Silhouette Score | Improvement | Data Preserved |
|----------|-----------------|-------------|----------------|
| Original | 0.0275 | Baseline | 100% |
| Previous Best | 0.0609 | +121% | ~95% |
| Conservative | 0.4465 | +633% | 97.0% |
| Aggressive | 0.3936 | +546% | 84.6% |
| **Spectral Clustering (New Best)** | **0.5343** | **+1843%** | **95.0%** |

**Best Configuration:**
- **k = 2** clusters
- **Diagonal covariance** structure
- **PCA dimensionality reduction** for better cluster separation
- **10 clinical features** (BMI, Glucose, BP, etc.)
- **95% data preservation** with conservative outlier removal (5%)
- **Spectral Clustering** algorithm for improved cluster geometry handling

**Key Findings:**
1. Conservative preprocessing outperforms aggressive outlier removal
2. Spectral Clustering captures complex cluster geometries that GMM cannot handle
3. PCA provides reliable dimensionality reduction when UMAP is unavailable
4. Feature selection using clinical domain knowledge improves results
5. The Silhouette score of 0.5343 represents a significant improvement for health phenotype data

**Target Analysis:**
- Target: 0.87 - 1.00
- Achieved: 0.5343
- Progress: 57.0% toward target
- Note: Scores of 0.87+ require perfect cluster separation, which is unrealistic for continuous health phenotype data

### References

1. McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models. Wiley.
2. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
3. Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). The PHQ-9: Validity of a brief depression severity measure. J Gen Intern Med.
4. NHANES. (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/
5. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.

### License

MIT License

---

### Author

**Group 6 - MSc Public Health Data Science**

| Student ID | Name | Role |
|------------|------|------|
| SDS6/46982/2024 | Cavin Otieno | Lead Developer |
| SDS6/46284/2024 | Joseph Ongoro Marindi | Data Analyst |
| SDS6/47543/2024 | Laura Nabalayo Kundu | Research Lead |
| SDS6/47545/2024 | Nevin Khaemba | Visualization Lead |

**Course:** Advanced Machine Learning (SDS6217)  
**Institution:** University of Nairobi
