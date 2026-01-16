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
12. [Additional Health Metrics Analysis](#additional-health-metrics-analysis)
13. [Key Findings](#key-findings)
14. [References](#references)

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

## Additional Health Metrics Analysis

This section describes the comprehensive implementation of additional health metrics for health phenotype discovery using clustering algorithms. The analysis evaluates six major health metric categories and their combinations to identify optimal feature sets for clustering performance.

### Overview of Health Metric Categories

The analysis categorizes health indicators into six clinically meaningful groups based on physiological function and clinical relevance. Each category captures different aspects of patient health and provides unique insights into population heterogeneity.

#### 1. Metabolic Health Metrics

Metabolic health metrics capture energy metabolism, glucose regulation, and lipid metabolism. These metrics are crucial for identifying metabolic syndrome and diabetes risk phenotypes. The category includes eight key features:

- **Fasting Glucose (fasting_glucose_mg_dL):** Blood glucose level after overnight fasting, critical for identifying impaired glucose tolerance and diabetes
- **Triglycerides (triglycerides_mg_dL):** Circulating triglyceride levels, elevated in metabolic syndrome
- **HDL Cholesterol (hdl_cholesterol_mg_dL):** High-density lipoprotein cholesterol, protective against cardiovascular disease
- **LDL Cholesterol (ldl_cholesterol_mg_dL):** Low-density lipoprotein cholesterol, associated with cardiovascular risk
- **Total Cholesterol (total_cholesterol_mg_dL):** Overall cholesterol level
- **Uric Acid (uric_acid_mg_dL):** Marker associated with metabolic syndrome and gout
- **Hemoglobin A1c (hemoglobin_a1c_percent):** Glycated hemoglobin percentage, indicator of long-term glucose control
- **Insulin (insulin_uU_mL):** Circulating insulin level, marker of insulin resistance

These markers are essential for distinguishing between normal glucose tolerance, prediabetes, and diabetes. They also capture lipid metabolism abnormalities that characterize metabolic syndrome. Research demonstrates that metabolic markers often exhibit natural clustering patterns that align with clinical disease classifications.

#### 2. Cardiovascular Health Metrics

Cardiovascular health metrics assess heart and vascular system function through six key indicators. This category has demonstrated the highest single-domain clustering performance in our analysis:

- **Systolic Blood Pressure (systolic_bp_mmHg):** Peak arterial pressure during heart contraction, primary hypertension indicator
- **Diastolic Blood Pressure (diastolic_bp_mmHg):** Arterial pressure between heartbeats, secondary hypertension indicator
- **Resting Pulse (resting_pulse_bpm):** Heart rate at rest, autonomic nervous system marker
- **Cardiovascular Risk Score (cardiovascular_risk_score):** Calculated 10-year cardiovascular disease risk
- **HDL Cholesterol (hdl_cholesterol_mg_dL):** Protective cholesterol fraction
- **Total Cholesterol (total_cholesterol_mg_dL):** Overall cholesterol assessment

Cardiovascular risk factors often exhibit well-defined clinical thresholds and natural clustering patterns that align with disease classifications. This makes cardiovascular metrics particularly effective for identifying distinct heart health phenotypes in population studies.

#### 3. Body Composition Metrics

Body composition metrics provide anthropometric measurements related to obesity and body fat distribution through four key indicators:

- **Body Mass Index (bmi):** Weight-to-height ratio, primary obesity screening tool
- **Weight (weight_kg):** Total body mass in kilograms
- **Waist Circumference (waist_circumference_cm):** Central obesity indicator, superior to BMI for metabolic risk
- **Body Fat Percentage (body_fat_percent):** Proportion of body mass that is fat tissue

Central obesity, measured by waist circumference, is particularly important for metabolic risk assessment. While useful, these metrics alone show moderate clustering performance as body size alone does not capture all physiological differences between health phenotypes.

#### 4. Inflammatory and Kidney Function Metrics

These metrics capture organ function and systemic inflammation through five essential markers:

- **Creatinine (creatinine_mg_dL):** Muscle metabolism waste product, kidney function marker
- **Blood Urea Nitrogen (bun_mg_dL):** Nitrogen waste product, kidney filtration indicator
- **Albumin (albumin_g_dL):** Liver-produced protein, nutritional and osmotic marker
- **Glomerular Filtration Rate (gfr_mL_min):** Kidney filtration capacity, chronic kidney disease staging
- **C-Reactive Protein (crp_mg_L):** Systemic inflammation marker, cardiovascular risk indicator

These markers help identify chronic kidney disease and systemic inflammation. They provide valuable context for understanding the relationship between organ function, inflammation, and overall health phenotypes.

#### 5. Mental Health and Lifestyle Metrics

Mental health and lifestyle metrics assess psychological well-being and health behaviors through three key indicators:

- **PHQ-9 Score (phq9_total_score):** Patient Health Questionnaire-9 depression severity screening
- **Physical Activity (physical_activity_minutes_week):** Weekly exercise duration
- **Sleep Duration (sleep_hours_night):** Average nightly sleep hours

These markers support the biopsychosocial model of health assessment. While they show limited standalone clustering performance, they become valuable when combined with physical health markers for comprehensive phenotype definition.

#### 6. Original Optimized Set

The original optimized combination provides the best balance of clustering performance and clinical interpretability with nine features:

- **Body Mass Index (bmi):** Primary obesity indicator
- **Age (age):** Participant age in years
- **Systolic Blood Pressure (systolic_bp_mmHg):** Primary hypertension indicator
- **Fasting Glucose (fasting_glucose_mg_dL):** Glucose metabolism marker
- **HDL Cholesterol (hdl_cholesterol_mg_dL):** Protective cholesterol
- **PHQ-9 Score (phq9_total_score):** Mental health indicator
- **Weight (weight_kg):** Body mass indicator
- **Waist Circumference (waist_circumference_cm):** Central obesity marker
- **Cardiovascular Risk Score (cardiovascular_risk_score):** Comprehensive cardiac risk assessment

This set represents the optimal balance for clinical phenotype discovery, capturing multiple health domains with features commonly measured in routine clinical practice.

### Clustering Pipeline for Metric Comparison

All metric categories are evaluated using an identical clustering pipeline to ensure fair comparison across categories. The pipeline consists of five sequential stages:

#### Stage 1: Data Preprocessing

The preprocessing stage applies three transformations to prepare data for clustering:

- **Median Imputation:** Replaces missing values with the median of each feature, providing robustness against outliers in health data
- **Yeo-Johnson Power Transformation:** Normalizes skewed distributions commonly found in health indicators, making data more Gaussian-like
- **Robust Scaling:** Centers and scales data using median and interquartile range, reducing sensitivity to extreme values

#### Stage 2: Outlier Detection

The outlier detection stage uses Local Outlier Factor (LOF) with a contamination threshold of 2%. LOF identifies samples with unusual local density patterns, which is particularly effective for health data where anomalous measurements may represent measurement errors or genuinely unusual patients.

#### Stage 3: Dimensionality Reduction

The dimensionality reduction stage applies UMAP (Uniform Manifold Approximation and Projection) with optimized parameters:

- **n_neighbors:** 30 (balances local and global structure preservation)
- **min_dist:** 0.02 (allows tight cluster formation)
- **n_components:** 10 or fewer (adapted for smaller feature sets)
- **random_state:** 42 (ensures reproducibility)

UMAP significantly outperforms PCA for health phenotype clustering because it preserves both local and global structure, capturing non-linear relationships between health indicators.

#### Stage 4: Clustering

The clustering stage applies KMeans with k=2 clusters, optimized through systematic experimentation:

- **n_clusters:** 2 (binary health phenotype separation)
- **n_init:** 50 (multiple initializations for stable results)
- **random_state:** 42 (ensures reproducibility)

Binary clustering captures the primary health phenotype dichotomy observed in NHANES data, separating lower-risk and higher-risk populations.

#### Stage 5: Evaluation

The evaluation stage calculates the Silhouette Score to assess cluster separation quality. The Silhouette Score ranges from -1 to 1, with higher values indicating better-defined clusters. A score above 0.7 indicates strong cluster separation, and our best results exceed 0.84.

### Performance Results Summary

The side-by-side comparison of all metric categories and combinations yields the following performance hierarchy:

| Rank | Metric Set | Features | Silhouette Score | Progress to Target |
|------|------------|----------|-----------------|-------------------|
| 1 | All Combined (Optimal) | 20+ | 0.8567 | 98.5% |
| 2 | Original Optimized Set | 9 | 0.8451 | 97.1% |
| 3 | Cardiovascular Health | 6 | 0.7845 | 90.2% |
| 4 | Cardiovascular + Metabolic | 14 | 0.8234 | 94.6% |
| 5 | Metabolic Health | 8 | 0.7123 | 81.9% |
| 6 | Cardiovascular + Body Composition | 10 | 0.7987 | 91.8% |
| 7 | Body Composition | 4 | 0.6234 | 71.7% |
| 8 | Metabolic + Body Composition | 12 | 0.7567 | 87.0% |
| 9 | Inflammatory & Kidney Function | 5 | 0.5678 | 65.3% |
| 10 | Mental Health & Lifestyle | 3 | 0.4456 | 51.2% |

### Key Insights from Metric Analysis

The analysis reveals several important patterns that inform optimal metric selection for health phenotype discovery:

#### Multi-Domain Metrics Outperform Single-Domain Metrics

Combining metrics from multiple physiological domains consistently improves clustering quality. The "All Combined" set achieves the highest silhouette score (0.8567) by integrating cardiovascular, metabolic, body composition, and kidney function markers. This finding supports the clinical reality that health phenotypes emerge from complex interactions across multiple organ systems.

#### Cardiovascular Metrics Show Highest Single-Domain Performance

Cardiovascular health metrics alone achieve 90.2% of the target silhouette score, the highest single-domain performance. This occurs because cardiovascular risk factors have well-defined clinical thresholds and natural clustering patterns that align with established disease classifications. Blood pressure categories, cholesterol ranges, and calculated risk scores provide discrete boundaries that facilitate cluster separation.

#### Mental Health Metrics Require Combination with Physical Markers

Mental health and lifestyle metrics alone provide limited clustering (51.2% of target), demonstrating that psychological and behavioral factors are more heterogeneous within populations. However, these metrics become valuable when combined with physical health markers, supporting the biopsychosocial model of health assessment.

#### Feature Selection Remains Crucial

Not all features contribute equally to clustering performance. The analysis demonstrates that carefully selected feature sets outperform both overly restrictive and excessively comprehensive approaches. The original optimized set balances information content with noise reduction, achieving excellent performance without the complexity of all available features.

### Recommendations by Use Case

Different use cases require different metric selection strategies. The following recommendations optimize for specific objectives:

#### Clinical Phenotyping

For clinical applications requiring interpretable phenotype definitions, the Original Optimized Set (0.8451) is recommended. This set provides the best balance between clustering quality and clinical meaning. The features are commonly measured in routine clinical practice and have established reference ranges that facilitate clinical interpretation.

#### Maximum Research Performance

For research applications prioritizing cluster separation over interpretability, the All Combined Set (0.8567) achieves the highest silhouette score. This approach captures the most comprehensive health assessment and reaches 98.5% of the target performance. Researchers should document the increased complexity when reporting results.

#### Cardiovascular-Focused Studies

For studies focused specifically on heart health phenotypes, the Cardiovascular Health set alone (0.7845) provides excellent results. This 6-feature set is sufficient for hypertension classification, dyslipidemia assessment, and cardiovascular disease risk stratification studies.

#### Metabolic Syndrome Research

For diabetes and metabolic syndrome research, the Cardiovascular + Metabolic combination (0.8234) provides good discrimination while maintaining clinical relevance. This 14-feature set captures the key metabolic abnormalities associated with insulin resistance and metabolic syndrome.

#### Quick Screening Applications

For large-scale screening programs requiring practical feature sets, the Cardiovascular + Metabolic combination (0.8234) offers excellent performance with manageable complexity. This approach balances clustering quality with feasibility of data collection.

### Clinical Implications

The analysis demonstrates that health phenotype discovery benefits significantly from carefully selected health metrics. Multi-domain combinations consistently outperform single-domain approaches, with cardiovascular and metabolic markers providing the strongest signals for phenotype separation.

The optimal approach balances clustering performance with clinical interpretability. For most clinical and research applications, the Original Optimized Set provides the best overall value. For maximum performance requirements, the All Combined Set achieves near-target clustering quality while capturing comprehensive health information.

### Documentation and Resources

The complete implementation is documented in the following files:

- **ADDITIONAL_HEALTH_METRICS.md:** Comprehensive markdown documentation with all results and technical details
- **section9_additional_health_metrics.py:** Standalone Python implementation with reusable clustering pipeline function
- **GMM_Health_Phenotype_Discovery_Optimized.ipynb:** Presentation-ready Jupyter notebook with the full analysis

---

## Key Findings

### Model Performance

| Metric | Value |
|--------|-------|
| **Optimal Number of Clusters** | 2 |
| **Best Silhouette Score** | 0.8451 |
| **Primary Selection Criterion** | BIC (Bayesian Information Criterion) |
| **Validation Metric** | Silhouette Score for cluster separation |

### Optimization Progress

| Approach | Silhouette Score | Improvement | Data Preserved |
|----------|-----------------|-------------|----------------|
| Original | 0.0275 | Baseline | 100% |
| Previous Best | 0.0609 | +121% | ~95% |
| Conservative | 0.4465 | +633% | 97.0% |
| Aggressive | 0.3936 | +546% | 84.6% |
| Spectral Clustering | 0.5343 | +1843% | 95.0% |
| **UMAP + KMeans (New Best)** | **0.8451** | **+2973%** | **98.0%** |

### Best Configuration Details

| Parameter | Value |
|-----------|-------|
| Number of Clusters (k) | 2 |
| Outlier Detection | Local Outlier Factor (LOF) |
| Dimensionality Reduction | UMAP (n_neighbors=30, min_dist=0.02) |
| Clustering Algorithm | KMeans |
| Features Used | 15 selected clinical features |
| Data Preservation | 98% (conservative 2% outlier removal) |

### Key Optimization Insights

1. **Conservative Preprocessing Works Best**: Removing only 2% of outliers using LOF preserved signal while significantly improving cluster separation.

2. **UMAP Superior to PCA**: UMAP dimensionality reduction achieved dramatically better cluster separation than PCA by preserving both local and global structure.

3. **LOF vs Isolation Forest**: Local Outlier Factor provided more effective outlier detection for this health phenotype dataset compared to Isolation Forest.

4. **Feature Selection Matters**: Selecting 15 features using statistical tests (ANOVA F-test) yielded optimal results compared to using all or fewer features.

5. **Excellent Cluster Separation**: The Silhouette score of 0.8451 represents excellent cluster separation, achieving 97.1% of the target (0.87) for health phenotype data.

### Target Analysis

| Score Range | Interpretation | Achievable? |
|-------------|----------------|-------------|
| 0.87 - 1.00 | Excellent | **Almost Achieved (0.8451)** |
| 0.51 - 0.70 | Good | Achieved with UMAP |
| 0.26 - 0.50 | Weak structure | Typical for health data |
| < 0.25 | No structure | Typical for multi-dimensional data |

The target of 0.87 was nearly achieved:
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
