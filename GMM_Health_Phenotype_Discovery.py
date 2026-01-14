"""
================================================================================
MSc Public Health Data Science - SDS6217 Advanced Machine Learning
Project: Gaussian Mixture Models for Population Health Stratification

This notebook demonstrates a complete data science workflow using GMM clustering
to identify latent subpopulations in public health data.

Author: Cavin Otieno
Student ID: [Your ID]
Date: January 2025
Institution: [University Name]

================================================================================
"""
# ============================================================================
# PHASE 1: PROJECT SETUP AND LIBRARY IMPORTS
# ============================================================================

# Standard library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for academic publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Print versions for reproducibility
print("=" * 70)
print("ENVIRONMENT AND VERSION INFORMATION")
print("=" * 70)
print(f"Python Version: {sys.version}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Matplotlib Version: {plt.matplotlib.__version__}")
print(f"Seaborn Version: {sns.__version__}")
print("=" * 70)

# ============================================================================
# Install required packages (run this cell if packages are not installed)
import subprocess
import sys

# List of required packages
required_packages = [
    'numpy',
    'pandas', 
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'scipy',
    'jupyter'
]

# Check and install missing packages
for package in required_packages:
    try:
        __import__(package)
        print(f"[OK] {package} is already installed")
    except ImportError:
        print(f"[INSTALLING] {package} not found - installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# ============================================================================
# PHASE 2: LITERATURE REVIEW AND ACADEMIC JUSTIFICATION
# ============================================================================

"""
Why Gaussian Mixture Models for Public Health Research?

Theoretical Foundation
----------------------
Gaussian Mixture Models (GMM) represent a sophisticated probabilistic approach to 
clustering that overcomes significant limitations of traditional methods like K-Means. 
In public health research, where population heterogeneity is the norm rather than 
the exception, GMM's ability to model overlapping subpopulations with probabilistic 
membership estimates makes it particularly suitable for complex health data.

Key Advantages for Public Health Applications
----------------------------------------------

1. Probabilistic Cluster Assignment
   - Unlike K-Means which forces hard assignments, GMM provides posterior probabilities
   - Each individual receives a probability of belonging to each cluster
   - Critical for health decisions where uncertainty quantification matters

2. Modeling Population Heterogeneity
   - Health populations naturally exhibit continuous distributions of risk factors
   - GMM captures latent subgroups without imposing artificial boundaries
   - Reflects the biological reality of disease processes

3. Flexibility Through Covariance Structures
   - Four covariance types allow modeling of various cluster shapes
   - Full covariance captures elongated, correlated clusters
   - Diagonal and spherical options provide computational efficiency

4. Uncertainty Quantification
   - Confidence in cluster assignments can be assessed
   - Important for clinical decision-making and resource allocation

Academic Justification Statement
--------------------------------
> "Health populations rarely form hard, discrete clusters. Gaussian Mixture Models 
> provide a statistically principled approach to capturing latent subpopulations 
> with associated uncertainty, making them ideal for public health research where 
> individual risk profiles often span multiple categories."
"""

# ============================================================================
# LITERATURE REFERENCES
# ============================================================================

REFERENCES = """
REFERENCES
==========

1. McLachlan, G.J., & Peel, D. (2000). Finite Mixture Models. John Wiley & Sons.
   - Foundational text on mixture model theory and applications

2. Fraley, C., & Raftery, A.E. (2002). Model-based clustering, discriminant analysis,
   and density estimation. Journal of the American Statistical Association, 97(458), 611-631.
   - Discusses Bayesian model selection for mixture models

3. Bishop, C.M. (2006). Pattern Recognition and Machine Learning. Springer.
   - Chapter 9 covers EM algorithm for Gaussian mixtures

4. Reynolds, D.A. (2009). Gaussian Mixture Models. Encyclopedia of Biometrics, 659-663.
   - Practical applications of GMM in classification

5. Wang, X., et al. (2020). Gaussian mixture model-based clustering analysis of 
   health examination data. BMC Medical Informatics and Decision Making, 20(1), 1-12.
   - Recent public health application of GMM

6. Hasan, M.M., et al. (2021). Clustering of health-related behaviors: A systematic 
   review. Preventive Medicine, 147, 106522.
   - Review of clustering approaches in health behavior research

7. Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics, 6(2), 461-464.
   - Introduction to Bayesian Information Criterion (BIC)

8. Akaike, H. (1974). A new look at the statistical model identification. IEEE 
   Transactions on Automatic Control, 19(6), 716-723.
   - Introduction to Akaike Information Criterion (AIC)
"""

print(REFERENCES)

# ============================================================================
# PHASE 3: DATA ACQUISITION AND DESCRIPTION
# ============================================================================

"""
For this project, we will use the Behavioral Risk Factor Surveillance System (BRFSS) 
dataset. The BRFSS is the nation's premier system of health-related telephone surveys 
that collect state data about U.S. residents regarding their health-related risk behaviors, 
chronic health conditions, and use of preventive services.

Dataset Source: IEEE Dataport / CDC BRFSS
The BRFSS contains comprehensive health data including:
- Demographic characteristics
- Health status indicators
- Risk behaviors (smoking, alcohol use, physical activity)
- Chronic conditions (diabetes, heart disease, asthma)
- Healthcare access and utilization
- Preventive health practices
"""

# ============================================================================
# DATA LOADING
# ============================================================================

def load_local_data(filepath):
    """
    Load dataset from local file path.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.sav') or filepath.endswith('.dta'):
            df = pd.read_stata(filepath)
        else:
            raise ValueError("Unsupported file format")
        print(f"[OK] Successfully loaded data from: {filepath}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return None

def download_data(url, output_path):
    """
    Download dataset from URL.
    
    Parameters:
    -----------
    url : str
        URL to download data from
    output_path : str
        Local path to save the downloaded file
        
    Returns:
    --------
    pd.DataFrame
        Downloaded dataset
    """
    import urllib.request
    try:
        print(f"Downloading data from: {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"[OK] Data saved to: {output_path}")
        return load_local_data(output_path)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return None

def generate_synthetic_health_data(n_samples=3000):
    """
    Generate synthetic public health dataset for demonstration purposes.
    This simulates BRFSS-like data with realistic health indicators.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    pd.DataFrame
        Synthetic health dataset
    """
    print("Generating synthetic public health dataset...")
    
    # Define cluster parameters to simulate distinct health phenotypes
    # Cluster 1: Health-conscious individuals
    n_cluster1 = int(n_samples * 0.35)
    cluster1 = {
        'age': np.random.normal(45, 12, n_cluster1),
        'bmi': np.random.normal(23, 3, n_cluster1),
        'physical_activity': np.random.normal(5, 1.5, n_cluster1),  # days/week
        'sleep_hours': np.random.normal(7.5, 0.8, n_cluster1),
        'fruit_vegetable_intake': np.random.normal(4, 1, n_cluster1),
        'alcohol_consumption': np.random.normal(2, 2, n_cluster1),
        'smoking_status': np.random.binomial(1, 0.1, n_cluster1),
        'healthcare_visits': np.random.poisson(2, n_cluster1),
        'chronic_conditions': np.random.poisson(0.3, n_cluster1),
        'mental_health_days': np.random.normal(3, 2, n_cluster1),
        'stress_level': np.random.normal(4, 1.5, n_cluster1),
        'blood_pressure_systolic': np.random.normal(118, 10, n_cluster1),
        'cholesterol_total': np.random.normal(185, 25, n_cluster1),
        'glucose_level': np.random.normal(95, 10, n_cluster1)
    }
    
    # Cluster 2: Moderate risk individuals
    n_cluster2 = int(n_samples * 0.40)
    cluster2 = {
        'age': np.random.normal(52, 10, n_cluster2),
        'bmi': np.random.normal(27, 4, n_cluster2),
        'physical_activity': np.random.normal(2.5, 1.5, n_cluster2),
        'sleep_hours': np.random.normal(6.5, 1, n_cluster2),
        'fruit_vegetable_intake': np.random.normal(2, 0.8, n_cluster2),
        'alcohol_consumption': np.random.normal(6, 4, n_cluster2),
        'smoking_status': np.random.binomial(1, 0.25, n_cluster2),
        'healthcare_visits': np.random.poisson(4, n_cluster2),
        'chronic_conditions': np.random.poisson(1.2, n_cluster2),
        'mental_health_days': np.random.normal(8, 3, n_cluster2),
        'stress_level': np.random.normal(6, 2, n_cluster2),
        'blood_pressure_systolic': np.random.normal(128, 12, n_cluster2),
        'cholesterol_total': np.random.normal(210, 30, n_cluster2),
        'glucose_level': np.random.normal(105, 15, n_cluster2)
    }
    
    # Cluster 3: High-risk individuals
    n_cluster3 = n_samples - n_cluster1 - n_cluster2
    cluster3 = {
        'age': np.random.normal(58, 8, n_cluster3),
        'bmi': np.random.normal(32, 5, n_cluster3),
        'physical_activity': np.random.normal(1, 1, n_cluster3),
        'sleep_hours': np.random.normal(5.5, 1.5, n_cluster3),
        'fruit_vegetable_intake': np.random.normal(1, 0.5, n_cluster3),
        'alcohol_consumption': np.random.normal(10, 5, n_cluster3),
        'smoking_status': np.random.binomial(1, 0.45, n_cluster3),
        'healthcare_visits': np.random.poisson(7, n_cluster3),
        'chronic_conditions': np.random.poisson(2.5, n_cluster3),
        'mental_health_days': np.random.normal(15, 4, n_cluster3),
        'stress_level': np.random.normal(8, 1.5, n_cluster3),
        'blood_pressure_systolic': np.random.normal(145, 15, n_cluster3),
        'cholesterol_total': np.random.normal(240, 35, n_cluster3),
        'glucose_level': np.random.normal(120, 20, n_cluster3)
    }
    
    # Combine clusters with some overlap (add noise)
    df1 = pd.DataFrame(cluster1)
    df2 = pd.DataFrame(cluster2)
    df3 = pd.DataFrame(cluster3)
    
    # Add noise to create overlap between clusters
    noise_level = 0.3
    df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Add noise to make clusters less distinct (more realistic)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col] + np.random.normal(0, df[col].std() * noise_level, len(df))
    
    # Clip unrealistic values
    df['age'] = df['age'].clip(18, 100)
    df['bmi'] = df['bmi'].clip(15, 55)
    df['physical_activity'] = df['physical_activity'].clip(0, 7)
    df['sleep_hours'] = df['sleep_hours'].clip(3, 12)
    df['fruit_vegetable_intake'] = df['fruit_vegetable_intake'].clip(0, 10)
    df['smoking_status'] = (df['smoking_status'] > 0.5).astype(int)
    df['healthcare_visits'] = df['healthcare_visits'].clip(0, 30)
    df['chronic_conditions'] = df['chronic_conditions'].clip(0, 10)
    df['mental_health_days'] = df['mental_health_days'].clip(0, 30)
    df['stress_level'] = df['stress_level'].clip(1, 10)
    
    # Add demographic variables
    df['sex'] = np.random.binomial(1, 0.52, len(df))  # Slight female majority
    df['education'] = np.random.choice([1, 2, 3, 4], len(df), p=[0.1, 0.3, 0.4, 0.2])
    df['income'] = np.random.choice([1, 2, 3, 4, 5], len(df), p=[0.15, 0.25, 0.3, 0.2, 0.1])
    df['race'] = np.random.choice([1, 2, 3, 4, 5], len(df), p=[0.6, 0.13, 0.17, 0.08, 0.02])
    
    # Add true cluster labels for validation (hidden from model)
    true_labels = np.concatenate([
        np.zeros(n_cluster1),
        np.ones(n_cluster2),
        np.full(n_cluster3, 2)
    ])
    df['true_cluster'] = true_labels.astype(int)
    
    # Round numeric values for realism
    numeric_cols = ['age', 'bmi', 'physical_activity', 'sleep_hours', 
                    'fruit_vegetable_intake', 'alcohol_consumption',
                    'blood_pressure_systolic', 'cholesterol_total', 'glucose_level',
                    'stress_level', 'mental_health_days']
    
    for col in numeric_cols:
        df[col] = df[col].round(1)
    
    print(f"[OK] Synthetic dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Features: {df.shape[1] - 2} health indicators + demographics + true labels")
    
    return df

# Generate the dataset
data = generate_synthetic_health_data(n_samples=3000)

# ============================================================================
# DATASET DESCRIPTION
# ============================================================================

def describe_dataset(df):
    """
    Provide comprehensive description of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to describe
    """
    print("=" * 70)
    print("DATASET METADATA AND DESCRIPTION")
    print("=" * 70)
    
    print("\n[INFO] BASIC INFORMATION")
    print("-" * 50)
    print(f"Number of observations: {df.shape[0]:,}")
    print(f"Number of variables: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum():,} ({100*df.isnull().sum().sum()/(df.shape[0]*df.shape[1]):.2f}%)")
    
    print("\n[INFO] VARIABLE LIST")
    print("-" * 50)
    
    # Classify variables by type
    continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_vars = []
    
    print("\nContinuous/Numeric Variables:")
    for i, col in enumerate(continuous_vars, 1):
        print(f"  {i:2d}. {col:<25} Range: [{df[col].min():.1f}, {df[col].max():.1f}]")
    
    print("\n" + "=" * 70)
    
    return continuous_vars, categorical_vars

# Run dataset description
continuous_vars, categorical_vars = describe_dataset(data)

# ============================================================================
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("=" * 70)
print("PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# 4.1.1 Summary Statistics
print("\n[INFO] SECTION 4.1: SUMMARY STATISTICS")

# Select key health indicators for analysis
health_indicators = [
    'age', 'bmi', 'physical_activity', 'sleep_hours', 
    'fruit_vegetable_intake', 'alcohol_consumption',
    'healthcare_visits', 'chronic_conditions', 'mental_health_days',
    'stress_level', 'blood_pressure_systolic', 'cholesterol_total', 'glucose_level'
]

# Summary statistics for health indicators
summary_stats = data[health_indicators].describe().T
summary_stats['median'] = data[health_indicators].median()
summary_stats['skewness'] = data[health_indicators].skew()
summary_stats['kurtosis'] = data[health_indicators].kurtosis()

print("\nDescriptive Statistics for Health Indicators:")
print("-" * 70)
print(summary_stats.round(2).to_string())

# 4.2.1 Distribution Plots
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(health_indicators):
    ax = axes[i]
    ax.hist(data[col], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(data[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data[col].mean():.1f}')
    ax.axvline(data[col].median(), color='green', linestyle='-', linewidth=2, label=f'Median: {data[col].median():.1f}')
    ax.set_title(f'{col}', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.legend(fontsize=8)

# Remove extra subplots
for j in range(len(health_indicators), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Distribution of Health Indicators', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/01_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/01_distributions.png")

# 4.2.2 Correlation Heatmap
fig, ax = plt.subplots(figsize=(14, 12))

# Compute correlation matrix
correlation_matrix = data[health_indicators].corr()

# Create heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r', 
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            ax=ax)

ax.set_title('Correlation Matrix of Health Indicators', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/02_correlation_heatmap.png")

# 4.2.3 Box Plots by Key Demographics
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# BMI by Age Group
data['age_group'] = pd.cut(data['age'], bins=[18, 35, 50, 65, 100], 
                            labels=['18-35', '36-50', '51-65', '65+'])
sns.boxplot(data=data, x='age_group', y='bmi', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('BMI Distribution by Age Group', fontweight='bold')
axes[0, 0].set_xlabel('Age Group')
axes[0, 0].set_ylabel('BMI')

# Healthcare Visits by Sex
sns.boxplot(data=data, x='sex', y='healthcare_visits', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Healthcare Visits by Sex', fontweight='bold')
axes[0, 1].set_xlabel('Sex (0=Female, 1=Male)')
axes[0, 1].set_ylabel('Annual Healthcare Visits')

# Chronic Conditions by BMI Category
data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 25, 30, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
sns.boxplot(data=data, x='bmi_category', y='chronic_conditions', ax=axes[1, 0], palette='Set1')
axes[1, 0].set_title('Chronic Conditions by BMI Category', fontweight='bold')
axes[1, 0].set_xlabel('BMI Category')
axes[1, 0].set_ylabel('Number of Chronic Conditions')

# Mental Health Days by Stress Level Category
data['stress_category'] = pd.cut(data['stress_level'], bins=[0, 4, 6, 8, 10],
                                  labels=['Low', 'Moderate', 'High', 'Very High'])
sns.boxplot(data=data, x='stress_category', y='mental_health_days', ax=axes[1, 1], palette='coolwarm')
axes[1, 1].set_title('Poor Mental Health Days by Stress Level', fontweight='bold')
axes[1, 1].set_xlabel('Stress Category')
axes[1, 1].set_ylabel('Days of Poor Mental Health (Last 30 Days)')

plt.suptitle('Health Indicators by Demographics and Risk Categories', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/03_boxplots_by_demographics.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/03_boxplots_by_demographics.png")

# 4.2.4 Pairplot for Key Health Indicators
key_indicators = ['bmi', 'physical_activity', 'stress_level', 'chronic_conditions']

# Sample data for faster plotting (full dataset may be slow)
sample_data = data.sample(n=min(1000, len(data)), random_state=42)

g = sns.pairplot(sample_data[key_indicators], 
                  diag_kind='kde',
                  plot_kws={'alpha': 0.5, 's': 20},
                  diag_kws={'fill': True, 'alpha': 0.7})

g.fig.suptitle('Pairwise Relationships of Key Health Indicators', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/04_pairplot.png")

# ============================================================================
# PHASE 5: DATA PREPROCESSING
# ============================================================================

print("=" * 70)
print("PHASE 5: DATA PREPROCESSING")
print("=" * 70)

print("\n[INFO] SECTION 5.1: MISSING VALUE ANALYSIS")

# Check for missing values
missing_analysis = pd.DataFrame({
    'Variable': data.columns,
    'Missing Count': data.isnull().sum().values,
    'Missing %': (data.isnull().sum().values / len(data) * 100).round(2),
    'Data Type': data.dtypes.values
})

# Filter to show only variables with missing values
missing_analysis = missing_analysis[missing_analysis['Missing Count'] > 0]

if len(missing_analysis) > 0:
    print("\nVariables with Missing Values:")
    print(missing_analysis.to_string(index=False))
    
    # Visualization of missing values
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_pct = (data.isnull().sum() / len(data) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) > 0:
        missing_pct.plot(kind='barh', color='coral', edgecolor='white', ax=ax)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Variable', fontweight='bold')
        ax.set_xlim(0, max(100, missing_pct.max() * 1.2))
        
        for i, (idx, val) in enumerate(missing_pct.items()):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/05_missing_values.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\n[OK] Figure saved: figures/05_missing_values.png")
else:
    print("\n[OK] No missing values detected in the dataset.")

print("\n[INFO] SECTION 5.2: OUTLIER DETECTION")

def detect_outliers_iqr(df, columns, threshold=1.5):
    """
    Detect outliers using the IQR method.
    """
    outlier_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outlier_info

# Detect outliers in health indicators
outlier_info = detect_outliers_iqr(data, health_indicators)

print("\nOutlier Analysis (IQR Method, threshold=1.5):")
print("-" * 70)
outlier_df = pd.DataFrame(outlier_info).T
outlier_df = outlier_df.round(2)
print(outlier_df.to_string())

# Visualize outliers
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(health_indicators):
    ax = axes[i]
    bp = ax.boxplot(data[col].dropna(), vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('navy')
    ax.set_title(f'{col}\n({outlier_info[col]["count"]} outliers)', fontsize=10)
    ax.set_ylabel('Value')

for j in range(len(health_indicators), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Outlier Detection using Box Plots', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/06_outlier_detection.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/06_outlier_detection.png")

print("\n[INFO] SECTION 5.3: FEATURE SCALING AND TRANSFORMATION")

from sklearn.preprocessing import StandardScaler

# Select features for clustering
# Exclude demographic variables and true labels for pure health behavior clustering
feature_columns = [
    'age', 'bmi', 'physical_activity', 'sleep_hours', 
    'fruit_vegetable_intake', 'alcohol_consumption',
    'healthcare_visits', 'chronic_conditions', 'mental_health_days',
    'stress_level', 'blood_pressure_systolic', 'cholesterol_total', 'glucose_level'
]

# Create feature matrix
X = data[feature_columns].copy()

print(f"\nFeatures selected for clustering: {len(feature_columns)}")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {col}")

# Handle missing values before scaling (if any)
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.median())
    print("\n[OK] Missing values imputed with median values")

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

print("\n[OK] Feature scaling applied using StandardScaler")
print(f"  Scaled data shape: {X_scaled.shape}")

# Visualize before and after scaling
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Before scaling
ax1 = axes[0]
data[feature_columns[:6]].boxplot(ax=ax1)
ax1.set_title('Before Scaling (Original Distribution)', fontweight='bold')
ax1.set_ylabel('Value')
ax1.tick_params(axis='x', rotation=45)

# After scaling
ax2 = axes[1]
X_scaled_df[feature_columns[:6]].boxplot(ax=ax2)
ax2.set_title('After Scaling (Standardized)', fontweight='bold')
ax2.set_ylabel('Standardized Value')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/07_feature_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/07_feature_scaling.png")

# ============================================================================
# PHASE 6: DIMENSIONALITY REDUCTION FOR VISUALIZATION
# ============================================================================

print("=" * 70)
print("PHASE 6: DIMENSIONALITY REDUCTION")
print("=" * 70)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 6.1 PCA for visualization
print("\nApplying Principal Component Analysis (PCA)...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"  Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")

# 6.2 t-SNE for visualization
print("\nApplying t-SNE for nonlinear dimensionality reduction...")

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

print(f"  t-SNE completed with perplexity=30")

# Visualize both projections colored by true clusters
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA visualization
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=data['true_cluster'], cmap='viridis',
                            alpha=0.6, s=30, edgecolor='white', linewidth=0.3)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
axes[0].set_title('PCA Projection of Health Data\n(Colored by True Clusters)', fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# t-SNE visualization
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                            c=data['true_cluster'], cmap='viridis',
                            alpha=0.6, s=30, edgecolor='white', linewidth=0.3)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].set_title('t-SNE Projection of Health Data\n(Colored by True Clusters)', fontweight='bold')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('figures/08_dimensionality_reduction.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/08_dimensionality_reduction.png")

# ============================================================================
# PHASE 7: GAUSSIAN MIXTURE MODELS IMPLEMENTATION
# ============================================================================

print("=" * 70)
print("PHASE 7: GAUSSIAN MIXTURE MODELS")
print("=" * 70)

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 7.1 Basic GMM Implementation
print("\n[INFO] SECTION 7.1: BASIC GMM IMPLEMENTATION")

# Split data for model validation
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

print(f"\nData Split:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# Fit a basic GMM with 3 components (based on our known data structure)
print("\nFitting Basic GMM with 3 components...")

gmm_basic = GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=42,
    n_init=10
)

gmm_basic.fit(X_train)

# Predict cluster assignments
train_labels_basic = gmm_basic.predict(X_train)
test_labels_basic = gmm_basic.predict(X_test)

# Get probabilities
train_probs_basic = gmm_basic.predict_proba(X_train)
test_probs_basic = gmm_basic.predict_proba(X_test)

print("\n[OK] Basic GMM fitted successfully")
print(f"  Log-likelihood (train): {gmm_basic.score(X_train):.4f}")
print(f"  Log-likelihood (test): {gmm_basic.score(X_test):.4f}")
print(f"  Number of iterations: {gmm_basic.n_iter_}")
print(f"  Convergence: {gmm_basic.converged_}")

print("\n[INFO] SECTION 7.2: MODEL EVALUATION METRICS")

def evaluate_gmm(X, labels, model):
    """
    Comprehensive evaluation of GMM model performance.
    """
    metrics = {}
    
    # Internal validation indices
    metrics['silhouette'] = silhouette_score(X, labels)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    
    # Model-specific metrics
    metrics['bic'] = model.bic(X)
    metrics['aic'] = model.aic(X)
    metrics['log_likelihood'] = model.score(X)
    
    return metrics

# Evaluate basic model
train_metrics = evaluate_gmm(X_train, train_labels_basic, gmm_basic)
test_metrics = evaluate_gmm(X_test, test_labels_basic, gmm_basic)

print("\nModel Evaluation Metrics:")
print("-" * 50)
print(f"{'Metric':<25} {'Training':>12} {'Test':>12}")
print("-" * 50)
for key in train_metrics:
    print(f"{key:<25} {train_metrics[key]:>12.4f} {test_metrics[key]:>12.4f}")

print("\n[INFO] SECTION 7.3: HYPERPARAMETER TUNING")

"""
HYPERPARAMETER SPACE
====================

The GMM model offers several hyperparameters for tuning:

1. n_components : Number of mixture components (clusters)
   - Range: 2 to 10
   - Too few: underfitting, groups distinct populations
   - Too many: overfitting, spurious clusters

2. covariance_type : Type of covariance matrix
   - 'full': Each component has its own general covariance matrix
   - 'tied': All components share the same covariance matrix
   - 'diag': Each component has a diagonal covariance matrix
   - 'spherical': Each component has a single variance

3. n_init : Number of initializations
   - Higher values reduce sensitivity to initialization

4. reg_covar : Regularization added to covariance diagonal
   - Prevents singular covariance matrices
"""

# Define hyperparameter grid
param_grid = {
    'n_components': [2, 3, 4, 5, 6, 7, 8],
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'n_init': [5, 10, 15],
    'reg_covar': [1e-6, 1e-5, 1e-4]
}

print("\nHyperparameter Grid:")
print(f"  n_components: {param_grid['n_components']}")
print(f"  covariance_type: {param_grid['covariance_type']}")
print(f"  n_init: {param_grid['n_init']}")
print(f"  reg_covar: {param_grid['reg_covar']}")
print(f"\nTotal combinations: {len(param_grid['n_components']) * len(param_grid['covariance_type']) * len(param_grid['n_init']) * len(param_grid['reg_covar'])}")

# Grid Search with BIC as the optimization criterion
print("\nRunning Grid Search with BIC optimization...")

from itertools import product

def run_grid_search_gmm(X, param_grid, scoring='bic'):
    """
    Perform exhaustive grid search for GMM hyperparameters.
    """
    results = []
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    total = len(combinations)
    print(f"  Evaluating {total} model configurations...")
    
    for i, params in enumerate(combinations):
        try:
            # Fit GMM with current parameters
            gmm = GaussianMixture(
                n_components=params['n_components'],
                covariance_type=params['covariance_type'],
                n_init=params['n_init'],
                reg_covar=params['reg_covar'],
                random_state=42,
                max_iter=200
            )
            
            gmm.fit(X)
            
            # Calculate metrics
            labels = gmm.predict(X)
            
            result = {
                'n_components': params['n_components'],
                'covariance_type': params['covariance_type'],
                'n_init': params['n_init'],
                'reg_covar': params['reg_covar'],
                'bic': gmm.bic(X),
                'aic': gmm.aic(X),
                'log_likelihood': gmm.score(X),
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels),
                'converged': gmm.converged_
            }
            
            results.append(result)
            
            # Progress update
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
                
        except Exception as e:
            print(f"    Error with parameters {params}: {e}")
            continue
    
    return pd.DataFrame(results)

# Run grid search (this may take a few minutes)
grid_results = run_grid_search_gmm(X_train, param_grid, scoring='bic')

print("\n[INFO] SECTION 7.4: HYPERPARAMETER TUNING RESULTS")

# Sort by BIC to find best model
grid_results_sorted = grid_results.sort_values('bic').reset_index(drop=True)

print("\n[INFO] TOP 10 MODELS BY BIC (Best to Worst):")
print("-" * 100)
top_models = grid_results_sorted.head(10)[['n_components', 'covariance_type', 'n_init', 
                                            'bic', 'aic', 'silhouette', 'calinski_harabasz',
                                            'davies_bouldin', 'converged']]
print(top_models.to_string(index=False))

# Best model
best_idx = grid_results_sorted.index[0]
best_params = {
    'n_components': int(grid_results_sorted.loc[best_idx, 'n_components']),
    'covariance_type': grid_results_sorted.loc[best_idx, 'covariance_type'],
    'n_init': int(grid_results_sorted.loc[best_idx, 'n_init']),
    'reg_covar': grid_results_sorted.loc[best_idx, 'reg_covar']
}

print(f"\n[OK] BEST MODEL CONFIGURATION:")
print("-" * 50)
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"  BIC: {grid_results_sorted.loc[best_idx, 'bic']:.2f}")
print(f"  AIC: {grid_results_sorted.loc[best_idx, 'aic']:.2f}")
print(f"  Silhouette Score: {grid_results_sorted.loc[best_idx, 'silhouette']:.4f}")

# Visualize hyperparameter tuning results
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 7.4.1 BIC vs Number of Components (by covariance type)
ax1 = axes[0, 0]
for cov_type in param_grid['covariance_type']:
    subset = grid_results_sorted[grid_results_sorted['covariance_type'] == cov_type]
    grouped = subset.groupby('n_components')['bic'].mean()
    ax1.plot(grouped.index, grouped.values, 'o-', label=cov_type, linewidth=2, markersize=8)

ax1.set_xlabel('Number of Components', fontsize=12)
ax1.set_ylabel('BIC Score', fontsize=12)
ax1.set_title('BIC Score vs Number of Components\n(by Covariance Type)', fontweight='bold')
ax1.legend(title='Covariance Type')
ax1.grid(True, alpha=0.3)

# 7.4.2 BIC by Covariance Type (boxplot)
ax2 = axes[0, 1]
grid_results_sorted.boxplot(column='bic', by='covariance_type', ax=ax2)
ax2.set_xlabel('Covariance Type', fontsize=12)
ax2.set_ylabel('BIC Score', fontsize=12)
ax2.set_title('BIC Distribution by Covariance Type', fontweight='bold')
plt.suptitle('')  # Remove automatic title

# 7.4.3 Silhouette Score vs BIC
ax3 = axes[1, 0]
scatter = ax3.scatter(grid_results_sorted['bic'], grid_results_sorted['silhouette'], 
                       c=grid_results_sorted['n_components'], cmap='viridis', 
                       alpha=0.6, s=50, edgecolor='white')
ax3.set_xlabel('BIC Score', fontsize=12)
ax3.set_ylabel('Silhouette Score', fontsize=12)
ax3.set_title('BIC vs Silhouette Score\n(Color = Number of Components)', fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='n_components')
ax3.grid(True, alpha=0.3)

# 7.4.4 Model Performance Comparison (Top 10 models)
ax4 = axes[1, 1]
top_10 = grid_results_sorted.head(10)
x_pos = np.arange(len(top_10))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, top_10['bic'], width, label='BIC', color='steelblue', alpha=0.8)
bars2 = ax4.bar(x_pos + width/2, top_10['aic'], width, label='AIC', color='coral', alpha=0.8)

ax4.set_xlabel('Model Rank', fontsize=12)
ax4.set_ylabel('Information Criterion Score', fontsize=12)
ax4.set_title('Top 10 Models: BIC vs AIC Comparison', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"#{i+1}" for i in range(10)])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/09_hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/09_hyperparameter_tuning.png")

# ============================================================================
# PHASE 8: FINAL MODEL TRAINING AND VALIDATION
# ============================================================================

print("=" * 70)
print("PHASE 8: FINAL MODEL TRAINING AND VALIDATION")
print("=" * 70)

# 8.1.1 Train the optimal GMM model
print("\n[INFO] SECTION 8.1: TRAINING OPTIMAL MODEL")

# Initialize and train the best model
gmm_optimal = GaussianMixture(
    n_components=best_params['n_components'],
    covariance_type=best_params['covariance_type'],
    n_init=best_params['n_init'],
    reg_covar=best_params['reg_covar'],
    random_state=42,
    max_iter=500
)

# Fit on training data
gmm_optimal.fit(X_train)

print(f"\n[OK] Optimal GMM Model Trained")
print(f"  Configuration:")
print(f"    - Number of components: {best_params['n_components']}")
print(f"    - Covariance type: {best_params['covariance_type']}")
print(f"    - Number of initializations: {best_params['n_init']}")
print(f"    - Regularization: {best_params['reg_covar']}")
print(f"\n  Convergence: {gmm_optimal.converged_}")
print(f"  Number of iterations: {gmm_optimal.n_iter_}")

# 8.2.1 Performance on Training and Test Sets
print("\n[INFO] SECTION 8.2: MODEL PERFORMANCE ANALYSIS")

# Predict on training and test sets
train_labels = gmm_optimal.predict(X_train)
test_labels = gmm_optimal.predict(X_test)

train_probs = gmm_optimal.predict_proba(X_train)
test_probs = gmm_optimal.predict_proba(X_test)

# Evaluate on both sets
train_eval = evaluate_gmm(X_train, train_labels, gmm_optimal)
test_eval = evaluate_gmm(X_test, test_labels, gmm_optimal)

print("\nPerformance Comparison (Training vs Test):")
print("-" * 60)
print(f"{'Metric':<25} {'Training':>12} {'Test':>12}")
print("-" * 60)
for key in train_eval:
    print(f"{key:<25} {train_eval[key]:>12.4f} {test_eval[key]:>12.4f}")

# Calculate performance gap
print("\nPerformance Gap Analysis:")
print("-" * 60)
for key in train_eval:
    gap = abs(train_eval[key] - test_eval[key])
    print(f"{key:<25}: Gap = {gap:.4f}")

# 8.3.1 Model Stability with Multiple Initializations
print("\n[INFO] SECTION 8.3: MODEL STABILITY ANALYSIS")

# Run GMM with multiple random seeds to assess stability
n_runs = 20
stability_results = []

print(f"\nRunning {n_runs} initializations with different random seeds...")

for seed in range(n_runs):
    gmm_test = GaussianMixture(
        n_components=best_params['n_components'],
        covariance_type=best_params['covariance_type'],
        n_init=1,  # Single initialization per run
        reg_covar=best_params['reg_covar'],
        random_state=seed,
        max_iter=500
    )
    
    gmm_test.fit(X_train)
    
    stability_results.append({
        'seed': seed,
        'log_likelihood': gmm_test.score(X_train),
        'bic': gmm_test.bic(X_train),
        'aic': gmm_test.aic(X_train),
        'converged': gmm_test.converged_,
        'labels': gmm_test.predict(X_train)
    })

stability_df = pd.DataFrame(stability_results)

print("\nStability Analysis Results:")
print("-" * 50)
print(f"  Number of runs: {n_runs}")
print(f"  Converged runs: {sum(stability_df['converged'])}/{n_runs}")
print(f"\n  Log-likelihood:")
print(f"    Mean: {stability_df['log_likelihood'].mean():.4f}")
print(f"    Std:  {stability_df['log_likelihood'].std():.4f}")
print(f"    CV:   {stability_df['log_likelihood'].std()/abs(stability_df['log_likelihood'].mean())*100:.2f}%")
print(f"\n  BIC:")
print(f"    Mean: {stability_df['bic'].mean():.2f}")
print(f"    Std:  {stability_df['bic'].std():.2f}")

# ============================================================================
# PHASE 9: CLUSTER ANALYSIS AND INTERPRETATION
# ============================================================================

print("=" * 70)
print("PHASE 9: CLUSTER ANALYSIS AND INTERPRETATION")
print("=" * 70)

# 9.1.1 Cluster Distribution
print("\n[INFO] SECTION 9.1: CLUSTER DISTRIBUTION")

# Get cluster labels for full dataset
full_labels = gmm_optimal.predict(X_scaled)
data['cluster'] = full_labels

cluster_counts = pd.Series(full_labels).value_counts().sort_index()
cluster_proportions = cluster_counts / len(full_labels) * 100

print("\nCluster Distribution:")
print("-" * 40)
for cluster, count in cluster_counts.items():
    prop = cluster_proportions[cluster]
    print(f"  Cluster {cluster}: {count:,} ({prop:.1f}%)")

# Visualize cluster distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
bars = ax1.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='white', linewidth=2)
ax1.set_xlabel('Cluster', fontsize=12)
ax1.set_ylabel('Number of Individuals', fontsize=12)
ax1.set_title('Cluster Distribution', fontweight='bold')
ax1.set_xticks(cluster_counts.index)

for bar, count, prop in zip(bars, cluster_counts.values, cluster_proportions.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             f'{count}\n({prop:.1f}%)', ha='center', va='bottom', fontsize=10)

# Pie chart
ax2 = axes[1]
ax2.pie(cluster_proportions, labels=[f'Cluster {i}' for i in cluster_counts.index],
        autopct='%1.1f%%', colors=colors, explode=[0.02]*len(cluster_counts),
        shadow=True, startangle=90)
ax2.set_title('Cluster Proportions', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/10_cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/10_cluster_distribution.png")

# 9.2.1 Cluster Profiles (Mean Values)
print("\n[INFO] SECTION 9.2: CLUSTER PROFILES")

# Calculate cluster means for all variables
cluster_profiles = data.groupby('cluster')[feature_columns].mean()
cluster_profiles_std = data.groupby('cluster')[feature_columns].std()

print("\nCluster Profiles (Mean Values):")
print("-" * 100)
print(cluster_profiles.round(2).to_string())

# Visualize cluster profiles as heatmap
fig, ax = plt.subplots(figsize=(14, 8))

# Normalize for better visualization
cluster_profiles_normalized = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())

sns.heatmap(cluster_profiles_normalized.T, 
            annot=cluster_profiles.T.round(1), 
            fmt='', 
            cmap='RdYlGn',
            center=0.5,
            linewidths=0.5,
            cbar_kws={'label': 'Normalized Value'},
            ax=ax)

ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Health Indicator', fontsize=12)
ax.set_title('Cluster Profiles: Mean Values by Health Indicator\n(Values shown are actual means)', 
             fontweight='bold')

plt.tight_layout()
plt.savefig('figures/11_cluster_profiles_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/11_cluster_profiles_heatmap.png")

# 9.3.1 Detailed Cluster Interpretation
print("\n[INFO] SECTION 9.3: CLUSTER INTERPRETATION AND HEALTH PHENOTYPES")

# Calculate overall population means for comparison
overall_means = data[feature_columns].mean()

print("\n[INFO] DETAILED CLUSTER DESCRIPTIONS:")
print("=" * 70)

for cluster in sorted(data['cluster'].unique()):
    cluster_data = data[data['cluster'] == cluster][feature_columns]
    cluster_mean = cluster_data.mean()
    cluster_size = len(cluster_data)
    cluster_pct = cluster_size / len(data) * 100
    
    print(f"\n[CLUSTER] {cluster}: n={cluster_size:,} ({cluster_pct:.1f}% of population)")
    print("-" * 60)
    
    # Compare to population mean
    deviations = (cluster_mean - overall_means) / overall_means * 100
    
    # Categorize indicators
    high_risk = []
    low_risk = []
    average = []
    
    for col in feature_columns:
        if col in ['age']:  # These are expected to vary
            continue
        if abs(deviations[col]) < 10:
            average.append(col)
        elif deviations[col] > 10:
            high_risk.append((col, deviations[col]))
        else:
            low_risk.append((col, deviations[col]))
    
    print(f"\n  [HIGH] Elevated Risk Indicators:")
    for indicator, dev in sorted(high_risk, key=lambda x: x[1], reverse=True):
        print(f"     - {indicator}: {cluster_mean[indicator]:.1f} (+{dev:.0f}% vs population mean)")
    
    print(f"\n  [LOW] Favorable Indicators:")
    for indicator, dev in sorted(low_risk, key=lambda x: x[1]):
        print(f"     - {indicator}: {cluster_mean[indicator]:.1f} (-{abs(dev):.0f}% vs population mean)")
    
    print(f"\n  [AVG] Average Indicators:")
    for indicator in average[:5]:  # Show first 5
        print(f"     - {indicator}: {cluster_mean[indicator]:.1f}")
    if len(average) > 5:
        print(f"     - ... and {len(average)-5} more")

# 9.4.1 External Validation (if ground truth is available)
print("\n[INFO] SECTION 9.4: EXTERNAL VALIDATION")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score

# Compare GMM clusters to true labels
if 'true_cluster' in data.columns:
    ari = adjusted_rand_score(data['true_cluster'], data['cluster'])
    nmi = normalized_mutual_info_score(data['true_cluster'], data['cluster'])
    homogeneity = homogeneity_score(data['true_cluster'], data['cluster'])
    completeness = completeness_score(data['true_cluster'], data['cluster'])
    
    print("\nExternal Validation Metrics (vs True Cluster Labels):")
    print("-" * 50)
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Homogeneity Score: {homogeneity:.4f}")
    print(f"  Completeness Score: {completeness:.4f}")
    
    if ari > 0.7:
        print("\n  [OK] Strong agreement with true cluster structure")
    elif ari > 0.4:
        print("\n  [WARNING] Moderate agreement with true cluster structure")
    else:
        print("\n  [ERROR] Weak agreement with true cluster structure")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(data['true_cluster'], data['cluster'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'GMM {i}' for i in range(best_params['n_components'])],
                yticklabels=[f'True {i}' for i in range(3)])
    ax.set_xlabel('GMM Cluster', fontsize=12)
    ax.set_ylabel('True Cluster', fontsize=12)
    ax.set_title('Confusion Matrix: GMM Clusters vs True Labels', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/13_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[OK] Figure saved: figures/13_confusion_matrix.png")

# ============================================================================
# PHASE 10: PROBABILISTIC MEMBERSHIP ANALYSIS
# ============================================================================

print("=" * 70)
print("PHASE 10: PROBABILISTIC MEMBERSHIP ANALYSIS")
print("=" * 70)

# Get membership probabilities for all data points
membership_probs = gmm_optimal.predict_proba(X_scaled)
data_probs = data.copy()

for i in range(best_params['n_components']):
    data_probs[f'prob_cluster_{i}'] = membership_probs[:, i]

# 10.1.1 Membership Probability Distribution
print("\n[INFO] SECTION 10.1: MEMBERSHIP PROBABILITY DISTRIBUTION")

print("\nMembership Probability Statistics:")
print("-" * 60)
for i in range(best_params['n_components']):
    probs = data_probs[f'prob_cluster_{i}']
    print(f"\n  Cluster {i}:")
    print(f"    Mean:   {probs.mean():.4f}")
    print(f"    Std:    {probs.std():.4f}")
    print(f"    Min:    {probs.min():.4f}")
    print(f"    Max:    {probs.max():.4f}")
    print(f"    High confidence (>=0.8): {(probs >= 0.8).sum():,} ({(probs >= 0.8).mean()*100:.1f}%)")

# Visualize membership probability distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i in range(best_params['n_components']):
    ax = axes[i // 2, i % 2]
    ax.hist(data_probs[f'prob_cluster_{i}'], bins=50, color=plt.cm.viridis(i/best_params['n_components']),
            edgecolor='white', alpha=0.7)
    ax.axvline(data_probs[f'prob_cluster_{i}"].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {data_probs[f"prob_cluster_{i}"].mean():.2f}')
    ax.set_xlabel('Membership Probability')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Cluster {i} Membership Probability Distribution', fontweight='bold')
    ax.legend()

plt.suptitle('Distribution of Cluster Membership Probabilities', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/14_membership_probabilities.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/14_membership_probabilities.png")

# 10.2.1 Hard vs Soft Assignment Comparison
print("\n[INFO] SECTION 10.2: HARD VS SOFT ASSIGNMENT ANALYSIS")

# Identify individuals with uncertain cluster assignment
data_probs['max_prob'] = membership_probs.max(axis=1)
data_probs['entropy'] = -np.sum(membership_probs * np.log(membership_probs + 1e-10), axis=1)

print("\nCluster Assignment Certainty Analysis:")
print("-" * 60)

# Count by certainty level
high_confidence = (data_probs['max_prob'] >= 0.8).sum()
moderate_confidence = ((data_probs['max_prob'] >= 0.5) & (data_probs['max_prob'] < 0.8)).sum()
low_confidence = (data_probs['max_prob'] < 0.5).sum()

print(f"  Very High Confidence (>=0.8): {high_confidence:,} ({100*high_confidence/len(data_probs):.1f}%)")
print(f"  Moderate Confidence (0.5-0.8): {moderate_confidence:,} ({100*moderate_confidence/len(data_probs):.1f}%)")
print(f"  Low Confidence (<0.5): {low_confidence:,} ({100*low_confidence/len(data_probs):.1f}%)")

# Find most uncertain individuals
uncertain_individuals = data_probs[data_probs['max_prob'] < 0.5]

print(f"\n  [WARNING] High Uncertainty Cases (max prob < 0.5): {len(uncertain_individuals):,}")
print("\n  Top 5 Most Uncertain Individuals:")
uncertain_sorted = uncertain_individuals.sort_values('max_prob').head(5)
for idx in uncertain_sorted.index:
    row = data_probs.loc[idx]
    print(f"    Individual {idx}: Max prob = {row['max_prob']:.3f}")
    probs = [f'C{i}: {row[f"prob_cluster_{i}"]:.2f}' for i in range(best_params['n_components'])]
    print(f"      Probabilities: {', '.join(probs)}")

# ============================================================================
# PHASE 11: VISUALIZATION OF FINAL RESULTS
# ============================================================================

print("=" * 70)
print("PHASE 11: VISUALIZATION OF FINAL RESULTS")
print("=" * 70)

# 11.1.1 Final Cluster Visualization in PCA and t-SNE space
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA visualization
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=data['cluster'], cmap='viridis',
                            alpha=0.6, s=30, edgecolor='white', linewidth=0.3)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
axes[0].set_title('GMM Clusters in PCA Space', fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Add cluster centroids
for cluster in range(best_params['n_components']):
    mask = data['cluster'] == cluster
    centroid_pca = X_pca[mask].mean(axis=0)
    axes[0].scatter(centroid_pca[0], centroid_pca[1], c='red', s=200, marker='X', 
                    edgecolor='white', linewidth=2, zorder=10)

# t-SNE visualization
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                            c=data['cluster'], cmap='viridis',
                            alpha=0.6, s=30, edgecolor='white', linewidth=0.3)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].set_title('GMM Clusters in t-SNE Space', fontweight='bold')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

# Add cluster centroids
for cluster in range(best_params['n_components']):
    mask = data['cluster'] == cluster
    centroid_tsne = X_tsne[mask].mean(axis=0)
    axes[1].scatter(centroid_tsne[0], centroid_tsne[1], c='red', s=200, marker='X', 
                    edgecolor='white', linewidth=2, zorder=10)

plt.suptitle('Final GMM Clustering Results', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/15_final_clustering_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/15_final_clustering_results.png")

# 11.2.1 Silhouette Plot
from sklearn.metrics import silhouette_samples

fig, ax = plt.subplots(figsize=(12, 8))

# Calculate silhouette scores for each sample
silhouette_vals = silhouette_samples(X_scaled, data['cluster'])
y_lower = 10

for i in range(best_params['n_components']):
    cluster_silhouette_vals = silhouette_vals[data['cluster'] == i]
    cluster_silhouette_vals.sort()
    
    size_cluster = len(cluster_silhouette_vals)
    y_upper = y_lower + size_cluster
    
    color = plt.cm.viridis(i / best_params['n_components'])
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    
    # Label cluster
    ax.text(-0.05, y_lower + 0.5 * size_cluster, f'Cluster {i}', fontsize=11, fontweight='bold')
    
    y_lower = y_upper + 10

# Add vertical line for average silhouette score
avg_silhouette = silhouette_score(X_scaled, data['cluster'])
ax.axvline(x=avg_silhouette, color='red', linestyle='--', linewidth=2, 
           label=f'Average: {avg_silhouette:.3f}')

ax.set_xlabel('Silhouette Coefficient', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Silhouette Plot for Cluster Validation', fontweight='bold')
ax.legend(loc='upper right')
ax.set_xlim(-0.1, 1)

plt.tight_layout()
plt.savefig('figures/16_silhouette_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Figure saved: figures/16_silhouette_plot.png")

# ============================================================================
# PHASE 12: CONCLUSIONS AND FUTURE WORK
# ============================================================================

print("=" * 70)
print("PHASE 12: CONCLUSIONS AND FUTURE WORK")
print("=" * 70)

# Calculate summary statistics
n_clusters = best_params['n_components']
n_features = len(feature_columns)
silhouette_final = silhouette_score(X_scaled, data['cluster'])
bic_final = gmm_optimal.bic(X_scaled)
aic_final = gmm_optimal.aic(X_scaled)

print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)

print(f"""
PROJECT OVERVIEW
----------------
This project applied Gaussian Mixture Models (GMM) to identify latent 
subpopulations within a synthetic public health dataset, demonstrating 
how probabilistic clustering can capture population heterogeneity that 
traditional hard-clustering methods may miss.

METHODOLOGY
-----------
- Dataset: {len(data):,} individuals with {n_features} health indicators
- Algorithm: Gaussian Mixture Models (GMM)
- Hyperparameter Tuning: Grid search with BIC optimization
- Best Configuration:
  - Number of clusters: {n_clusters}
  - Covariance type: {best_params['covariance_type']}
  - Regularization: {best_params['reg_covar']}

KEY FINDINGS
------------
1. Optimal Number of Clusters: {n_clusters}
   - BIC Score: {bic_final:.2f}
   - AIC Score: {aic_final:.2f}
   - Silhouette Score: {silhouette_final:.4f}

2. Cluster Characteristics:
""")

for cluster in range(n_clusters):
    cluster_data = data[data['cluster'] == cluster]
    cluster_mean = cluster_data[feature_columns].mean()
    
    # Identify most distinctive features
    deviations = (cluster_mean - overall_means) / overall_means * 100
    top_deviations = deviations.abs().nlargest(3)
    
    print(f"\n   Cluster {cluster} ({len(cluster_data):,} individuals, {100*len(cluster_data)/len(data):.1f}%):")
    print(f"   - Most distinctive characteristics:")
    for feat in top_deviations.index:
        val = cluster_mean[feat]
        dev = deviations[feat]
        direction = "+" if dev > 0 else "-"
        print(f"     - {feat}: {val:.1f} ({direction}{abs(dev):.0f}% vs mean)")

print(f"""
3. Model Quality:
   - Average silhouette score: {silhouette_final:.4f}
   - {"Strong" if silhouette_final > 0.5 else "Moderate" if silhouette_final > 0.3 else "Weak"} cluster separation
   - {100*sum(data_probs['max_prob'] >= 0.8)/len(data_probs):.1f}% of individuals have high confidence assignments (>=0.8)

PUBLIC HEALTH IMPLICATIONS
--------------------------
- The identified clusters represent distinct health phenotypes with different
  risk profiles and intervention needs.
- Probabilistic cluster assignments allow for uncertainty-aware decision making.
- This approach can support targeted intervention design and resource allocation.
""")

print("\nLIMITATIONS")
print("-" * 70)
print("""
1. Data Characteristics
   - Synthetic dataset may not fully represent real-world complexity
   - Missing potential confounders (socioeconomic factors, healthcare access)
   - Cross-sectional design limits causal inference

2. Methodological Limitations
   - GMM assumes Gaussian distributions; non-Gaussian features may be misfit
   - Covariance type selection impacts cluster shapes; full covariance may overfit
   - Model selection criteria (BIC/AIC) have known limitations

3. Generalizability
   - Results may not generalize to different populations or time periods
   - External validation with real health data is recommended
   - Clinical validation required before operational deployment
""")

print("\nFUTURE WORK AND RECOMMENDATIONS")
print("-" * 70)
print("""
1. Methodological Extensions
   - Compare GMM with other mixture models (Dirichlet Process, Variational Bayes)
   - Implement semi-supervised GMM with known disease outcomes
   - Explore deep embedding for high-dimensional health data

2. Validation Studies
   - External validation with real BRFSS or NHANES data
   - Clinical validation against established risk scores
   - Longitudinal analysis to assess cluster stability over time

3. Public Health Applications
   - Develop risk stratification tools for specific conditions
   - Design targeted intervention programs based on cluster profiles
   - Integration with electronic health records for real-time clustering

4. Technical Improvements
   - Feature selection using domain knowledge
   - Handling missing data with multiple imputation
   - Scalable implementations for large datasets
""")

# ============================================================================
# FINAL CODE SUMMARY AND MODEL EXPORT
# ============================================================================

import joblib
import json

print("=" * 70)
print("MODEL EXPORT AND REPRODUCIBILITY")
print("=" * 70)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Save the trained model
model_path = 'models/gmm_optimal_model.joblib'
scaler_path = 'models/standard_scaler.joblib'

joblib.dump(gmm_optimal, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n[OK] Model saved to: {model_path}")
print(f"[OK] Scaler saved to: {scaler_path}")

# Save configuration and results
config = {
    'best_params': best_params,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_info': {
        'n_samples': len(data),
        'n_features': len(feature_columns),
        'feature_names': feature_columns
    },
    'metrics': {
        'bic': float(bic_final),
        'aic': float(aic_final),
        'silhouette': float(silhouette_final),
        'calinski_harabasz': float(calinski_harabasz_score(X_scaled, data['cluster'])),
        'davies_bouldin': float(davies_bouldin_score(X_scaled, data['cluster']))
    },
    'cluster_sizes': {f'cluster_{i}': int(count) for i, count in enumerate(cluster_counts.values)}
}

config_path = 'models/model_config.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"[OK] Configuration saved to: {config_path}")

# Export cluster profiles
profiles_path = 'results/cluster_profiles.csv'
cluster_profiles.to_csv(profiles_path)
print(f"[OK] Cluster profiles saved to: {profiles_path}")

print("\n" + "=" * 70)
print("PROJECT COMPLETE")
print("=" * 70)
print(f"""
All outputs saved to:
  - Figures: figures/
  - Models: models/
  - Results: results/
  
To reload the model:
  >>> import joblib
  >>> model = joblib.load('models/gmm_optimal_model.joblib')
  >>> scaler = joblib.load('models/standard_scaler.joblib')
  >>> predictions = model.predict(new_data)
""")

print("\n" + "=" * 70)
print("REFERENCES")
print("=" * 70)
print("""
1. McLachlan, G.J., & Peel, D. (2000). Finite Mixture Models. John Wiley & Sons.
2. Bishop, C.M. (2006). Pattern Recognition and Machine Learning. Springer.
3. Fraley, C., & Raftery, A.E. (2002). Model-based clustering, discriminant analysis,
   and density estimation. Journal of the American Statistical Association, 97(458), 611-631.
4. Reynolds, D.A. (2009). Gaussian Mixture Models. Encyclopedia of Biometrics, 659-663.
5. Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics, 6(2), 461-464.
6. Wang, X., et al. (2020). Gaussian mixture model-based clustering analysis of 
   health examination data. BMC Medical Informatics and Decision Making, 20(1), 1-12.
""")

print("\n" + "-" * 70)
print("Author: Cavin Otieno")
print("MSc Public Health Data Science - SDS6217 Advanced Machine Learning")
print("-" * 70)
