# GMM Health Phenotype Discovery

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12.3-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-003A57?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-1.7.1-blue?style=for-the-badge)

</div>

---

## MSc Public Health Data Science - SDS6217 Advanced Machine Learning

---

**Student ID:** SDS6/46982/2025  
**Date:** January 2025  
**Institution:** University of Nairobi  

---

### Project Overview

This comprehensive data science project applies Gaussian Mixture Models (GMM) to identify latent subpopulations in NHANES health data, demonstrating how probabilistic clustering can capture population heterogeneity that traditional hard-clustering methods may miss.

### Dataset

**Source:** National Health and Nutrition Examination Survey (NHANES)  
**Location:** `data/raw/nhanes_health_data.csv`  
**Samples:** 5,000 respondents  
**Variables:** 47 health indicators

### Key Features

- **Probabilistic Clustering**: Captures uncertainty in cluster assignments
- **Hyperparameter Tuning**: Systematic grid search optimization
- **Population Phenotype Discovery**: Identifies distinct health subgroups
- **Academic Rigor**: Comprehensive methodology suitable for MSc-level assessment

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

### Repository Structure

```
gmm-health-phenotype-discovery/
├── GMM_Health_Phenotype_Discovery.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── nhanes_health_data.csv
├── output_v2/
│   ├── metrics/
│   ├── predictions/
│   └── cluster_profiles/
├── models/
│   └── gmm_clustering/
└── figures/
    └── plots/
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
```

### Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Open Jupyter notebook: `jupyter notebook GMM_Health_Phenotype_Discovery.ipynb`
3. Run cells sequentially

### Author

**Cavin Otieno**  
MSc Public Health Data Science  
Advanced Machine Learning (SDS6217)  
University of Nairobi

### License

MIT License
