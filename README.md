# GMM Health Phenotype Discovery

## MSc Public Health Data Science - SDS6217 Advanced Machine Learning

---

**Student ID:** SDS6/46982/2025  
**Date:** January 2025  
**Institution:** University of Nairobi  

---

### Project Overview

This comprehensive data science project applies Gaussian Mixture Models (GMM) to identify latent subpopulations in public health data, demonstrating how probabilistic clustering can capture population heterogeneity that traditional hard-clustering methods may miss.

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

### Repository Structure

```
gmm-health-phenotype-discovery/
├── GMM_Health_Phenotype_Discovery.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── raw/
│       └── brfss_health_data.csv
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

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

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
