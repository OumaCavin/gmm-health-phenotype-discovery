# GMM Health Phenotype Discovery - Project Documentation

<div align="center">

**MSc Public Health Data Science - SDS6217 Advanced Machine Learning**  

**Group 6 Members:**
| Student ID | Student Name |
|------------|--------------|
| SDS6/46982/2024 | Cavin Otieno |
| SDS6/46284/2024 | Joseph Ongoro Marindi |
| SDS6/47543/2024 | Laura Nabalayo Kundu |
| SDS6/47545/2024 | Nevin Khaemba |

**Date:** January 2025  
**Institution:** University of Nairobi

</div>

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Data Dictionary](#data-dictionary)
4. [Methodology](#methodology)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Project Structure](#project-structure)
7. [Installation and Setup](#installation-and-setup)
8. [Usage Guide](#usage-guide)
9. [Path Management System](#path-management-system)
10. [Output Files](#output-files)
11. [Interactive Streamlit App](#interactive-streamlit-app)
12. [Key Findings](#key-findings)
13. [References](#references)

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
**Source Organization:** Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS)  
**Source URL:** https://www.cdc.gov/nchs/nhanes/  
**Location:** `data/raw/nhanes_health_data.csv`  
**Samples:** 5,000 adult respondents  
**Variables:** 47 health indicators

### Data Collection Methodology

NHANES combines interviews and physical examinations to assess the health and nutritional status of adults and children in the United States. The survey is conducted by the National Center for Health Statistics (NCHS), part of the CDC.

### Data Quality

- Missing values handled through mean imputation for continuous variables
- Categorical variables encoded as numerical indices
- Continuous variables standardized for GMM implementation

---

## Data Dictionary

### Complete Variable Documentation with Source References

#### 1. Demographics (5 Variables)

| Variable Name | Description | Unit/Range | Source |
|---------------|-------------|------------|--------|
| `sex` | Biological sex of respondent | 0=Female, 1=Male | NHANES Demographics Questionnaire |
| `age` | Age at screening | Years (20-80) | NHANES Demographics Questionnaire |
| `race_ethnicity` | Race/ethnicity category | 1-5 (5-level classification) | NHANES Demographics Questionnaire |
| `education_level` | Education level completed | 1-5 (less than 9th grade to graduate) | NHANES Demographics Questionnaire |
| `income_category` | Annual household income | 1-20 (20 income brackets) | NHANES Demographics Questionnaire |

**Source Reference:** NHANES Demographics Module. (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

#### 2. Body Measures (4 Variables)

| Variable Name | Description | Unit/Range | Source |
|---------------|-------------|------------|--------|
| `weight_kg` | Body weight | Kilograms | NHANES Physical Examination (Exam Component) |
| `height_cm` | Standing height | Centimeters | NHANES Physical Examination |
| `bmi` | Body Mass Index | kg/m² (10-60) | Calculated from weight and height |
| `waist_circumference_cm` | Waist circumference | Centimeters | NHANES Physical Examination |

**Source Reference:** NHANES Anthropometry Manual. (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

#### 3. Blood Pressure (2 Variables)

| Variable Name | Description | Unit/Range | Source |
|---------------|-------------|------------|--------|
| `systolic_bp_mmHg` | Systolic blood pressure | mmHg (70-200) | NHANES Physical Examination |
| `diastolic_bp_mmHg` | Diastolic blood pressure | mmHg (40-120) | NHANES Physical Examination |

**Source Reference:** NHANES Blood Pressure Procedures Manual. (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

#### 4. Laboratory Measures (5 Variables)

| Variable Name | Description | Unit/Range | Source |
|---------------|-------------|------------|--------|
| `total_cholesterol_mg_dL` | Total cholesterol | mg/dL (100-400) | NHANES Laboratory Component |
| `hdl_cholesterol_mg_dL` | High-density lipoprotein cholesterol | mg/dL (20-100) | NHANES Laboratory Component |
| `ldl_cholesterol_mg_dL` | Low-density lipoprotein cholesterol | mg/dL (30-250) | NHANES Laboratory Component |
| `fasting_glucose_mg_dL` | Fasting plasma glucose | mg/dL (50-300) | NHANES Laboratory Component |
| `insulin_uU_mL` | Fasting insulin | μU/mL | NHANES Laboratory Component |

**Source Reference:** NHANES Laboratory Procedures Manual. (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

#### 5. Behavioral Factors (8 Variables)

| Variable Name | Description | Unit/Range | Source |
|---------------|-------------|------------|--------|
| `smoked_100_cigarettes` | Ever smoked 100 cigarettes | 0=No, 1=Yes | NHANES Smoking Questionnaire |
| `cigarettes_per_day` | Cigarettes smoked per day | 0-99 | NHANES Smoking Questionnaire |
| `alcohol_use_past_year` | Alcohol use in past year | 0=No, 1=Yes | NHANES Alcohol Questionnaire |
| `drinks_per_week` | Average drinks per week | 0-50 | NHANES Alcohol Questionnaire |
| `vigorous_work_activity` | Vigorous work activity (days/week) | 0-7 | NHANES Physical Activity Questionnaire |
| `moderate_work_activity` | Moderate work activity (days/week) | 0-7 | NHANES Physical Activity Questionnaire |
| `vigorous_recreation_activity` | Vigorous recreational activity (days/week) | 0-7 | NHANES Physical Activity Questionnaire |
| `moderate_recreation_activity` | Moderate recreational activity (days/week) | 0-7 | NHANES Physical Activity Questionnaire |

**Source Reference:** NHANES Smoking Questionnaire, Alcohol Questionnaire, and Physical Activity Questionnaire. (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

#### 6. Medical Conditions (8 Variables)

| Variable Name | Description | Unit/Range | Source |
|---------------|-------------|------------|--------|
| `general_health_rating` | General health self-rating | 1-5 (Excellent to Poor) | NHANES Health Status Questionnaire |
| `arthritis` | Ever told had arthritis | 0=No, 1=Yes | NHANES Medical Conditions Questionnaire |
| `heart_failure` | Ever told had heart failure | 0=No, 1=Yes | NHANES Medical Conditions Questionnaire |
| `coronary_heart_disease` | Ever told had coronary heart disease | 0=No, 1=Yes | NHANES Medical Conditions Questionnaire |
| `angina_pectoris` | Ever told had angina/angina pectoris | 0=No, 1=Yes | NHANES Medical Conditions Questionnaire |
| `heart_attack` | Ever told had heart attack (myocardial infarction) | 0=No, 1=Yes | NHANES Medical Conditions Questionnaire |
| `stroke` | Ever told had stroke | 0=No, 1=Yes | NHANES Medical Conditions Questionnaire |
| `cancer_diagnosis` | Ever told had cancer or malignancy | 0=No, 1=Yes | NHANES Medical Conditions Questionnaire |

**Source Reference:** NHANES Medical Conditions Questionnaire. (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

#### 7. Mental Health - PHQ-9 (10 Variables)

The PHQ-9 is a validated depression screening instrument based on the DSM-IV criteria.

| Variable Name | Description | Unit/Range | Source |
|---------------|-------------|------------|--------|
| `phq9_little_interest` | Little interest in doing things (past 2 weeks) | 0-3 | PHQ-9 Depression Screen |
| `phq9_feeling_down` | Feeling down, depressed, or hopeless (past 2 weeks) | 0-3 | PHQ-9 Depression Screen |
| `phq9_sleep_trouble` | Trouble falling or staying asleep, or sleeping too much | 0-3 | PHQ-9 Depression Screen |
| `phq9_feeling_tired` | Feeling tired or having little energy | 0-3 | PHQ-9 Depression Screen |
| `phq9_poor_appetite` | Poor appetite or overeating | 0-3 | PHQ-9 Depression Screen |
| `phq9_feeling_bad_about_self` | Feeling bad about yourself or that you are a failure | 0-3 | PHQ-9 Depression Screen |
| `phq9_trouble_concentrating` | Trouble concentrating on things | 0-3 | PHQ-9 Depression Screen |
| `phq9_moving_speaking` | Moving or speaking slowly, or being fidgety/restless | 0-3 | PHQ-9 Depression Screen |
| `phq9_suicidal_thoughts` | Thoughts that you would be better off dead | 0-3 | PHQ-9 Depression Screen |
| `phq9_total_score` | Total PHQ-9 score (sum of 9 items) | 0-27 | Calculated |

**Source Reference:** Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). The PHQ-9: Validity of a brief depression severity measure. Journal of General Internal Medicine, 16(9), 606-613. https://www.cdc.gov/nchs/nhanes/

#### 8. Derived Features (4 Variables)

| Variable Name | Description | Unit/Range | Source |
|---------------|-------------|------------|--------|
| `cardiovascular_risk_score` | Calculated cardiovascular risk | Derived | Derived from pooled cohort equations |
| `metabolic_syndrome_indicator` | Metabolic syndrome status | 0=No, 1=Yes | ATP III criteria |
| `health_category` | Overall health category | 1-4 | Derived from multiple indicators |
| `risk_level` | Risk stratification level | 1-3 (Low, Medium, High) | Derived composite score |

**Source Reference:** Grundy, S. M., et al. (2005). Diagnosis and management of the metabolic syndrome. Circulation, 112(17), 2735-2752.

---

## Methodology

### Gaussian Mixture Models (GMM)

GMM is a probabilistic clustering algorithm that models data as a mixture of multiple Gaussian distributions. Unlike K-Means which assigns each point to a single cluster, GMM provides soft assignments based on posterior probabilities.

**Theoretical Foundation:** GMM assumes that the data is generated from a mixture of K Gaussian distributions, where each distribution represents a cluster. The model learns the parameters (mean, covariance, and mixing weight) of each distribution using the Expectation-Maximization (EM) algorithm.

**Why GMM for Public Health:**

1. **Probabilistic Cluster Assignment**: Each individual receives a probability of belonging to each cluster
2. **Population Heterogeneity**: Captures continuous distributions of risk factors
3. **Flexible Covariance Structures**: Four types (full, tied, diag, spherical) for different cluster shapes
4. **Uncertainty Quantification**: Confidence in cluster assignments for clinical decision-making

### Model Selection Criteria

| Criterion | Description | Selection Rule |
|-----------|-------------|----------------|
| **BIC** | Bayesian Information Criterion | Lower is better - primary criterion |
| **AIC** | Akaike Information Criterion | Lower is better - secondary validation |
| **Silhouette Score** | Cluster cohesion and separation | Higher is better (-1 to 1) |

---

## Hyperparameter Tuning

The following hyperparameters are optimized through exhaustive grid search:

| Parameter | Description | Search Space | Justification |
|-----------|-------------|--------------|---------------|
| `n_components` | Number of clusters | 2-10 | Evaluates different population strata |
| `covariance_type` | Covariance structure | full, tied, diag, spherical | Tests different cluster shapes |
| `reg_covar` | Regularization term | 1e-6, 1e-5, 1e-4 | Prevents singular covariance matrices |
| `max_iter` | Maximum iterations | 100, 200, 500 | Ensures algorithm convergence |
| `n_init` | Initializations | 5, 10, 20 | Avoids local optima |

### Grid Search Configuration

- **Total Combinations:** 4 × 4 × 3 × 3 × 3 = 432 configurations
- **Evaluation Metric:** BIC (primary), AIC (secondary)
- **Validation:** 5-fold cross-validation

---

## Project Structure

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

---

## Installation and Setup

### Prerequisites

- Python 3.12.3 or higher
- pip package manager
- Git (for version control)

### Installation Steps

```bash
# Clone the Repository
git clone https://github.com/OumaCavin/gmm-health-phenotype-discovery.git
cd gmm-health-phenotype-discovery

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install Dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook GMM_Health_Phenotype_Discovery.ipynb

# Launch Streamlit App (in separate terminal)
streamlit run app.py
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

| Function | Purpose | Location |
|----------|---------|----------|
| `save_fig()` | Save figures in multiple formats | `figures/plots/` |
| `save_model()` | Persist trained models | `models/gmm_clustering/` |
| `save_data()` | Export data to CSV | `output_v2/subdirectories/` |

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

## Interactive Streamlit App

The project includes an interactive Streamlit web application (`app.py`) for real-time predictions.

### App Features

| Tab | Feature | Description |
|-----|---------|-------------|
| Tab 1 | 🔮 Predict Cluster | Interactive health parameter input and prediction |
| Tab 2 | 📈 Cluster Profiles | Visualize and compare cluster characteristics |
| Tab 3 | 📊 Model Performance | Display BIC, AIC, and Silhouette scores |
| Tab 4 | ℹ️ About | Project and team information |
| Tab 5 | 📥 Data Download | Download all model outputs |

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Key Findings

### Model Performance

| Metric | Value |
|--------|-------|
| **Optimal Number of Clusters** | 2 |
| **Best Covariance Type** | Diagonal |
| **Best Silhouette Score** | 0.4465 |
| **BIC Score** | Model selection criterion |
| **Primary Selection Criterion** | BIC (Bayesian Information Criterion) |
| **Validation Metric** | Silhouette Score for cluster separation |

### Optimization Progress

| Approach | Silhouette Score | Improvement | Data Preserved |
|----------|-----------------|-------------|----------------|
| Original | 0.0275 | Baseline | 100% |
| Previous Best | 0.0609 | +121% | ~95% |
| **Conservative (Best)** | **0.4465** | **+633%** | **97.0%** |
| Aggressive | 0.3936 | +546% | 84.6% |

### Best Configuration Details

| Parameter | Value |
|-----------|-------|
| Number of Clusters (k) | 2 |
| Covariance Type | Diagonal |
| Dimensionality Reduction | UMAP |
| Features Used | 10 clinical features (BMI, Glucose, BP, etc.) |
| Data Preservation | 97% (conservative 5% outlier removal) |

### Key Optimization Insights

1. **Conservative Preprocessing Works Best**: Removing only 5% of outliers preserved signal while improving cluster separation. Aggressive removal (25-35%) actually degraded performance.

2. **UMAP Superior to PCA**: UMAP dimensionality reduction achieved better cluster separation than raw features or PCA projections.

3. **Feature Selection Matters**: Using clinically relevant features (BMI, Glucose, Blood Pressure, HDL Cholesterol, etc.) improved results over using all 46 features.

4. **Realistic Expectations**: The Silhouette score of 0.4465 represents the realistic maximum for health phenotype data, as health data naturally exhibits continuous rather than discrete cluster structures.

### Target Analysis

| Score Range | Interpretation | Achievable? |
|-------------|----------------|-------------|
| 0.87 - 1.00 | Excellent | Very rare in real health data |
| 0.51 - 0.70 | Good | Possible with curated features |
| **0.26 - 0.50** | **Weak structure** | **Achieved - Typical for health data** |
| < 0.25 | No structure | Typical for multi-dimensional data |

The target of 0.87-1.00 was not achieved because:
- Health phenotypes exist on continuous spectrums, not discrete categories
- Individual biological variation creates natural overlap between groups
- Clinical measurements have inherent uncertainty
- Multi-morbidity means individuals often have characteristics of multiple phenotypes

### Cluster Interpretations

| Cluster | Profile | Key Characteristics |
|---------|---------|---------------------|
| Cluster 1 | Lower Risk | Lower BMI, normal glucose, better cardiovascular markers |
| Cluster 2 | Higher Risk | Elevated BMI, higher glucose, increased cardiovascular risk |

### Implications for Public Health

1. **Risk Stratification:** Identified clusters can inform targeted interventions for high-risk individuals
2. **Resource Allocation:** Cluster sizes guide healthcare planning and resource distribution
3. **Prevention Strategies:** Behavioral patterns inform prevention programs for specific phenotypes
4. **Personalized Medicine:** Cluster membership can guide individualized healthcare recommendations

---

## References

### Academic References

1. McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley.

2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

3. Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). The PHQ-9: Validity of a brief depression severity measure. *Journal of General Internal Medicine*, 16(9), 606-613.

4. Grundy, S. M., et al. (2005). Diagnosis and management of the metabolic syndrome. *Circulation*, 112(17), 2735-2752.

5. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

### NHANES Data References

6. National Health and Nutrition Examination Survey. (2023). Centers for Disease Control and Prevention, National Center for Health Statistics. https://www.cdc.gov/nchs/nhanes/

7. NHANES Analytic Guidelines. (2023). Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/analytic_guidelines.htm

### Software and Tools

8. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362.

9. pandas development team. (2020). pandas: powerful Python data analysis toolkit. https://pandas.pydata.org/

10. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95.

11. Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.

12. Streamlit Documentation. (2024). Streamlit Inc. https://docs.streamlit.io/

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**Group 6 - MSc Public Health Data Science**

| Student ID | Name | Role |
|------------|------|------|
| SDS6/46982/2024 | Cavin Otieno | Lead Developer |
| SDS6/46284/2024 | Joseph Ongoro Marindi | Data Analyst |
| SDS6/47543/2024 | Laura Nabalayo Kundu | Research Lead |
| SDS6/47545/2024 | Nevin Khaemba | Visualization Lead |

**Course:** Advanced Machine Learning (SDS6217)  
**Institution:** University of Nairobi  
**Date:** January 2025

---

*Last Updated: January 2025*
*Version: 1.0*
