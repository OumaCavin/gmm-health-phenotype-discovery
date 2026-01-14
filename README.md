# GMM Health Phenotype Discovery

## MSc Public Health Data Science - SDS6217 Advanced Machine Learning

A comprehensive data science project applying Gaussian Mixture Models (GMM) to identify latent subpopulations in public health data.

### Project Overview

This project demonstrates the application of Gaussian Mixture Models for population health stratification, addressing the fundamental challenge of heterogeneity in public health datasets.

### Key Features

- **Probabilistic Clustering**: Captures uncertainty in cluster assignments
- **Hyperparameter Tuning**: Systematic grid search optimization
- **Population Phenotype Discovery**: Identifies distinct health subgroups
- **Academic Rigor**: Comprehensive methodology suitable for MSc-level assessment

### Repository Structure

```
gmm-health-phenotype-discovery/
├── README.md
├── requirements.txt
├── GMM_Health_Phenotype_Discovery.ipynb
├── figures/
│   ├── 01_distributions.png
│   ├── 02_correlation_heatmap.png
│   └── ...
├── models/
│   ├── gmm_optimal_model.joblib
│   └── standard_scaler.joblib
└── results/
    └── cluster_profiles.csv
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

### License

MIT License
