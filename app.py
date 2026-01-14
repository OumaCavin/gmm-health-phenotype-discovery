"""
================================================================================
GMM Health Phenotype Discovery - Interactive Streamlit App
================================================================================

This Streamlit application provides an interactive interface for the GMM-based
health phenotype discovery model. Users can explore cluster assignments, input
health parameters, and visualize model results.

Author: Group 6 - Cavin Otieno, Joseph Ongoro Marindi, Laura Nabalayo Kundu, Nevin Khaemba
Student IDs: SDS6/46982/2024, SDS6/46284/2024, SDS6/47543/2024, SDS6/47545/2024
MSc Public Health Data Science - SDS6217 Advanced Machine Learning
University of Nairobi

================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="GMM Health Phenotype Discovery",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# Define project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v2')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained GMM model and scaler."""
    model_path = os.path.join(MODELS_DIR, 'gmm_clustering', 'gmm_optimal_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'gmm_clustering', 'standard_scaler.joblib')
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_cluster_profiles():
    """Load cluster profiles from CSV."""
    profiles_path = os.path.join(OUTPUT_DIR, 'cluster_profiles', 'gmm_cluster_profiles.csv')
    try:
        return pd.read_csv(profiles_path)
    except FileNotFoundError:
        return None

@st.cache_data
def load_model_info():
    """Load model configuration info."""
    config_path = os.path.join(OUTPUT_DIR, 'metrics', 'project_config.json')
    try:
        return pd.read_json(config_path, orient='index')
    except FileNotFoundError:
        return None

def predict_cluster(scaler, model, input_data):
    """Predict cluster assignment for input data."""
    try:
        scaled_data = scaler.transform(input_data)
        probabilities = model.predict_proba(scaled_data)
        cluster = model.predict(scaled_data)
        return cluster[0], probabilities[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def get_cluster_name(cluster_id):
    """Get descriptive name for cluster."""
    cluster_names = {
        0: "Health-Conscious Low Risk",
        1: "Moderate Risk Profile",
        2: "High Cardiovascular Risk",
        3: "Metabolic Syndrome Group",
        4: "Mental Health Focus Group"
    }
    return cluster_names.get(cluster_id, f"Cluster {cluster_id}")

# =============================================================================
# SIDEBAR - MODEL INFORMATION
# =============================================================================

st.sidebar.title("🏥 GMM Health Phenotype")
st.sidebar.markdown("### Interactive Prediction Tool")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")

# Load model info
model_info = load_model_info()
if model_info is not None:
    st.sidebar.info(f"""
    **Best Configuration:**
    - Clusters: {model_info.loc['n_components', 0] if 'n_components' in model_info.values else 'N/A'}
    - Covariance: {model_info.loc['covariance_type', 0] if 'covariance_type' in model_info.values else 'N/A'}
    - BIC Score: {model_info.loc['bic', 0] if 'bic' in model_info.values else 'N/A'}
    """)
else:
    st.sidebar.warning("Model info not found")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Project Files")
st.sidebar.markdown(f"""
- **Notebook**: `GMM_Health_Phenotype_Discovery.ipynb`
- **Data**: `data/raw/nhanes_health_data.csv`
- **Model**: `models/gmm_clustering/`
- **Output**: `output_v2/`
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 👥 Group 6 Members
| Student ID | Name |
|------------|------|
| SDS6/46982/2024 | Cavin Otieno |
| SDS6/46284/2024 | Joseph Ongoro Marindi |
| SDS6/47543/2024 | Laura Nabalayo Kundu |
| SDS6/47545/2024 | Nevin Khaemba |
""")

# =============================================================================
# MAIN PAGE - TABS
# =============================================================================

st.title("🏥 GMM Health Phenotype Discovery")
st.markdown("### Interactive Clustering Tool for NHANES Health Data")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict Cluster",
    "📈 Cluster Profiles",
    "📊 Model Performance",
    "ℹ️ About",
    "📥 Data Download"
])

# =============================================================================
# TAB 1: PREDICT CLUSTER
# =============================================================================

with tab1:
    st.header("🔮 Cluster Prediction")
    st.markdown("Enter health parameters to predict phenotype cluster assignment.")
    
    # Load model
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.error("Model not found! Please run the notebook first to train and save the model.")
        st.info("Expected model location: `models/gmm_clustering/gmm_optimal_model.joblib`")
        st.info("Expected scaler location: `models/gmm_clustering/standard_scaler.joblib`")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Demographics")
            age = st.slider("Age (years)", 20, 80, 45, help="Participant age in years")
            sex = st.selectbox("Sex", ["Male", "Female"])
            bmi = st.number_input("BMI (kg/m²)", value=25.0, min_value=10.0, max_value=60.0, step=0.1)
            
        with col2:
            st.subheader("💓 Vital Signs")
            systolic_bp = st.number_input("Systolic BP (mmHg)", value=120, min_value=70, max_value=200)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", value=80, min_value=40, max_value=120)
            
        st.subheader("🩸 Laboratory Values")
        col3, col4 = st.columns(2)
        
        with col3:
            total_cholesterol = st.number_input("Total Cholesterol (mg/dL)", value=180, min_value=100, max_value=400)
            hdl_cholesterol = st.number_input("HDL Cholesterol (mg/dL)", value=50, min_value=20, max_value=100)
            
        with col4:
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", value=90, min_value=50, max_value=300)
            ldl_cholesterol = st.number_input("LDL Cholesterol (mg/dL)", value=100, min_value=30, max_value=250)
            
        st.subheader("🧠 Mental Health (PHQ-9)")
        phq9_score = st.slider("PHQ-9 Depression Score", 0, 27, 5, help="0-4: Minimal, 5-9: Mild, 10-14: Moderate, 15-19: Moderately Severe, 20-27: Severe")
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == "Male" else 0],
            'bmi': [bmi],
            'systolic_bp_mmHg': [systolic_bp],
            'diastolic_bp_mmHg': [diastolic_bp],
            'total_cholesterol_mg_dL': [total_cholesterol],
            'hdl_cholesterol_mg_dL': [hdl_cholesterol],
            'ldl_cholesterol_mg_dL': [ldl_cholesterol],
            'fasting_glucose_mg_dL': [fasting_glucose],
            'phq9_total_score': [phq9_score]
        })
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Phenotype Cluster", type="primary"):
            cluster, probabilities = predict_cluster(scaler, model, input_data)
            
            if cluster is not None:
                st.success(f"Predicted Cluster: **{cluster}** - **{get_cluster_name(cluster)}**")
                
                # Display probabilities
                st.subheader("📊 Membership Probabilities")
                
                prob_df = pd.DataFrame({
                    'Cluster': [f"Cluster {i}" for i in range(len(probabilities))],
                    'Probability': probabilities,
                    'Probability (%)': probabilities * 100
                })
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = plt.cm.viridis(np.linspace(0, 1, len(probabilities)))
                bars = ax.barh(prob_df['Cluster'], prob_df['Probability (%)'], color=colors)
                ax.set_xlabel('Probability (%)')
                ax.set_title('Cluster Membership Probabilities')
                ax.set_xlim(0, 100)
                
                # Add value labels
                for bar, prob in zip(bars, prob_df['Probability (%)']):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1f}%', va='center', fontsize=10)
                
                st.pyplot(fig)
                
                # Pie chart
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.pie(probabilities, labels=[f'Cluster {i}' for i in range(len(probabilities))],
                       autopct='%1.1f%%', colors=colors, startangle=90)
                ax2.set_title('Cluster Distribution')
                st.pyplot(fig2)
                
                # Risk interpretation
                st.subheader("⚠️ Risk Assessment")
                max_prob_cluster = np.argmax(probabilities)
                max_prob = probabilities[max_prob_cluster]
                
                if max_prob < 0.5:
                    st.warning("Uncertain classification - individual shows characteristics of multiple phenotypes")
                else:
                    if max_prob_cluster in [2, 3]:
                        st.error(f"High-risk phenotype: {get_cluster_name(max_prob_cluster)}")
                        st.markdown("""
                        **Recommendations:**
                        - Consult healthcare provider for comprehensive assessment
                        - Regular monitoring of cardiovascular markers
                        - Lifestyle modification counseling
                        """)
                    elif max_prob_cluster == 0:
                        st.success(f"Low-risk phenotype: {get_cluster_name(max_prob_cluster)}")
                        st.markdown("""
                        **Recommendations:**
                        - Maintain current healthy lifestyle
                        - Continue regular health screenings
                        """)
                    else:
                        st.info(f"Moderate-risk phenotype: {get_cluster_name(max_prob_cluster)}")
                        st.markdown("""
                        **Recommendations:**
                        - Periodic health monitoring
                        - Consider lifestyle improvements
                        """)

# =============================================================================
# TAB 2: CLUSTER PROFILES
# =============================================================================

with tab2:
    st.header("📈 Cluster Profiles")
    st.markdown("Explore the characteristic profiles of each identified health phenotype.")
    
    # Load profiles
    profiles = load_cluster_profiles()
    
    if profiles is not None:
        # Display cluster summary
        st.subheader("Cluster Characteristics")
        
        # Get cluster names from columns
        cluster_cols = [col for col in profiles.columns if 'Cluster_' in col or col.isdigit()]
        
        # Create comparison table
        st.dataframe(profiles, use_container_width=True)
        
        # Visualize profiles
        st.subheader("Visual Comparison")
        
        # Select features to visualize
        numeric_cols = profiles.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect(
            "Select features to compare",
            numeric_cols,
            default=numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
        )
        
        if selected_features:
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data for heatmap
            heatmap_data = profiles.set_index('Variable')[selected_features].T
            
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                       ax=ax, cbar_kws={'label': 'Value'})
            ax.set_title('Cluster Profiles Comparison')
            ax.set_ylabel('Features')
            ax.set_xlabel('Clusters')
            
            st.pyplot(fig)
    else:
        st.warning("Cluster profiles not found. Please run the notebook first.")

# =============================================================================
# TAB 3: MODEL PERFORMANCE
# =============================================================================

with tab3:
    st.header("📊 Model Performance")
    st.markdown("View the hyperparameter tuning results and model evaluation metrics.")
    
    # Load model info
    model_info = load_model_info()
    
    if model_info is not None:
        st.subheader("Optimal Model Configuration")
        
        # Display as table
        config_df = pd.DataFrame({
            'Parameter': model_info.index,
            'Value': model_info[0].values
        })
        st.table(config_df)
        
        st.subheader("Performance Metrics")
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        bic_val = model_info.loc['bic', 0] if 'bic' in model_info.index else None
        aic_val = model_info.loc['aic', 0] if 'aic' in model_info.index else None
        silhouette_val = model_info.loc['silhouette', 0] if 'silhouette' in model_info.index else None
        
        col1.metric("BIC Score", f"{bic_val:.2f}" if bic_val else "N/A", 
                   delta="Lower is better" if bic_val else None)
        col2.metric("AIC Score", f"{aic_val:.2f}" if aic_val else "N/A",
                   delta="Lower is better" if aic_val else None)
        col3.metric("Silhouette Score", f"{silhouette_val:.3f}" if silhouette_val else "N/A",
                   delta="Higher is better" if silhouette_val else None)
        
        # Interpretation
        st.info("""
        ### Metric Interpretation
        
        - **BIC (Bayesian Information Criterion)**: Balances model fit and complexity. Lower values indicate better models.
        - **AIC (Akaike Information Criterion)**: Estimates model quality. Lower values indicate better models.
        - **Silhouette Score**: Measures cluster cohesion and separation. Range: [-1, 1], higher is better.
        """)
    else:
        st.warning("Model performance metrics not found.")

# =============================================================================
# TAB 4: ABOUT
# =============================================================================

with tab4:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Project Overview
    
    This project applies **Gaussian Mixture Models (GMM)** to identify latent subpopulations 
    in NHANES health data. Unlike traditional hard-clustering methods, GMM provides 
    probabilistic cluster assignments that better reflect the continuous nature of 
    health risk factors.
    
    ## Key Features
    
    - **Probabilistic Clustering**: Captures uncertainty in cluster assignments
    - **Hyperparameter Tuning**: Systematic grid search optimization
    - **Population Phenotype Discovery**: Identifies distinct health subgroups
    - **Interactive Prediction**: Real-time cluster prediction for new inputs
    
    ## Dataset
    
    - **Source**: National Health and Nutrition Examination Survey (NHANES)
    - **Samples**: 5,000 respondents
    - **Variables**: 47 health indicators
    
    ## Technologies Used
    
    | Technology | Purpose |
    |------------|---------|
    | Python 3.12 | Programming Language |
    | scikit-learn | GMM Implementation |
    | Pandas | Data Manipulation |
    | NumPy | Numerical Computing |
    | Matplotlib/Seaborn | Visualization |
    | Streamlit | Web Application |
    
    ## Team Members
    
    | Student ID | Name | Role |
    |------------|------|------|
    | SDS6/46982/2024 | Cavin Otieno | Lead Developer |
    | SDS6/46284/2024 | Joseph Ongoro Marindi | Data Analyst |
    | SDS6/47543/2024 | Laura Nabalayo Kundu | Research Lead |
    | SDS6/47545/2024 | Nevin Khaemba | Visualization Lead |
    
    ## References
    
    1. McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models. Wiley.
    2. CDC National Health and Nutrition Examination Survey.
    3. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
    """)
    
    st.markdown("---")
    st.markdown("*© 2025 Group 6 - MSc Public Health Data Science*")

# =============================================================================
# TAB 5: DATA DOWNLOAD
# =============================================================================

with tab5:
    st.header("📥 Data Download")
    st.markdown("Download model outputs and results.")
    
    # List available files
    output_files = []
    
    # Check each output directory
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)
            output_files.append(rel_path)
    
    # Check model directory
    for root, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, PROJECT_ROOT)
            output_files.append(rel_path)
    
    if output_files:
        st.subheader("Available Files")
        
        # Create dataframe
        files_df = pd.DataFrame({
            'File': output_files,
            'Type': [os.path.splitext(f)[1] for f in output_files]
        })
        
        st.dataframe(files_df, use_container_width=True)
        
        # Download buttons
        st.subheader("Download Files")
        
        for file_path in output_files:
            full_path = os.path.join(PROJECT_ROOT, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'rb') as f:
                        file_data = f.read()
                    
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"📄 `{file_path}`")
                    col2.download_button(
                        label="Download",
                        data=file_data,
                        file_name=os.path.basename(file_path),
                        mime='application/octet-stream'
                    )
                except Exception as e:
                    st.warning(f"Could not read {file_path}: {e}")
    else:
        st.info("No output files found. Please run the notebook to generate outputs.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>GMM Health Phenotype Discovery | MSc Public Health Data Science | University of Nairobi</p>
        <p>Group 6: Cavin Otieno, Joseph Ongoro Marindi, Laura Nabalayo Kundu, Nevin Khaemba</p>
    </div>
    """,
    unsafe_allow_html=True
)
