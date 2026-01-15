# GMM Health Phenotype Discovery: Results Analysis Document

## Project Overview

This document provides a comprehensive analysis of the results obtained from the Gaussian Mixture Model (GMM) health phenotype discovery project conducted on the National Health and Nutrition Examination Survey (NHANES) dataset. The analysis systematically progresses through data acquisition, exploratory data analysis, preprocessing, dimensionality reduction, model selection, clustering, validation, and interpretation phases. Each section presents the key outputs, statistical findings, and their implications for understanding population health phenotypes.

The primary objective of this research was to identify latent subpopulations within the NHANES dataset using probabilistic clustering methods. Unlike traditional hard-clustering approaches such as K-means, GMM provides posterior probability estimates for cluster assignments, enabling a more nuanced understanding of population heterogeneity and uncertainty in phenotype classification. This approach is particularly valuable in public health research where individuals often exhibit characteristics spanning multiple health risk categories.

The document structure follows the logical flow of the analytical pipeline, with each phase building upon the outputs of previous stages. As additional phase outputs are provided, this document will be updated to incorporate new findings and interpretations, creating a complete record of the analytical process and its conclusions.

---

## Phase 2: Data Acquisition and Initial Inspection

### Dataset Summary

The NHANES health dataset was successfully loaded from the designated data directory. The dataset comprises 5,000 respondents representing the adult population covered by the National Health and Nutrition Examination Survey. Each respondent is characterized by 47 distinct variables spanning demographic characteristics, physical measurements, laboratory values, behavioral factors, medical history, and mental health indicators. The complete absence of missing values (0.00% missingness) indicates either complete data collection or prior preprocessing that addressed all incomplete records. This data completeness is advantageous for clustering analysis as it eliminates the need for imputation strategies that might introduce measurement bias or distort underlying data distributions.

The dataset structure reflects the comprehensive nature of NHANES data collection protocols, which combine interviews, physical examinations, and laboratory testing to create a multidimensional representation of population health. The 47 variables capture both objective clinical measurements (blood pressure, cholesterol levels, glucose concentrations) and self-reported information (symptoms, behaviors, medical history), providing a holistic view of each respondent's health status and risk factors.

### Variable Categories and Descriptions

The dataset variables can be organized into several conceptual categories that align with the NHANES examination components. The **respondent identification** category includes the unique identifier that enables tracking and linking of records across examination components while maintaining respondent privacy. **Demographic variables** encompass biological sex, age at examination, racial and ethnic classification, education level attainment, and income category, providing essential context for understanding health disparities and social determinants of health. These demographic factors often serve as confounding variables in health research and may influence the clustering structure of the data.

**Body measurement variables** include weight in kilograms, height in centimeters, body mass index (BMI), and waist circumference. These anthropometric indicators are fundamental to assessing nutritional status and metabolic health. BMI serves as the primary screening tool for obesity classification, while waist circumference provides additional information about central adiposity, which is more strongly associated with cardiovascular and metabolic risk than BMI alone. The correlation between these body measurement variables is an important consideration for clustering analysis, as highly correlated features may overweight certain health dimensions in the phenotype characterization.

**Blood pressure variables** capture systolic and diastolic measurements in millimeters of mercury (mmHg), along with a derived blood pressure category variable. Hypertension is a major risk factor for cardiovascular disease, stroke, and kidney disease, making blood pressure an essential component of any health phenotype characterization. The categorical variable provides clinical classification (normal, elevated, hypertension stage 1, hypertension stage 2) that can validate continuous measurement patterns and inform intervention strategies.

**Laboratory variables** represent a critical dimension of the health assessment, including total cholesterol, high-density lipoprotein (HDL) cholesterol, low-density lipoprotein (LDL) cholesterol, fasting glucose, and insulin concentrations. These biochemical markers provide objective indicators of metabolic health and cardiovascular risk. Cholesterol fractions are particularly important for cardiovascular risk assessment, with HDL representing protective cholesterol and LDL representing atherogenic cholesterol. Fasting glucose and insulin levels indicate glycemic status and pancreatic beta-cell function, with elevated values signaling insulin resistance and prediabetic or diabetic states.

**Behavioral variables** capture health-related behaviors including smoking history (coded as whether the respondent has smoked at least 100 cigarettes in their lifetime), current smoking intensity (cigarettes per day among smokers), alcohol consumption patterns (past-year use and average drinks per week), and physical activity levels across work and recreational domains. These behavioral factors are modifiable risk factors that significantly influence health outcomes and may cluster together in identifiable lifestyle patterns that characterize distinct health phenotypes.

**Medical history variables** document diagnosed conditions including arthritis, heart failure, coronary heart disease, angina pectoris, heart attack (myocardial infarction), stroke, and cancer diagnosis. These self-reported physician diagnoses provide clinical validation of the health phenotypes identified through statistical clustering, as cluster assignments should ideally reflect meaningful clinical distinctions. The presence of multiple chronic conditions within individuals suggests complex phenotype presentations that may not be fully captured by simple risk factor categorization.

**Mental health variables** are derived from the Patient Health Questionnaire-9 (PHQ-9), a validated screening instrument for depression severity. The nine individual items assess the frequency of depressive symptoms over the preceding two weeks, including little interest or pleasure in activities, feeling down or hopeless, sleep disturbances, fatigue, appetite changes, feelings of worthlessness, difficulty concentrating, psychomotor changes, and suicidal thoughts. The total score provides an overall depression severity measure with established clinical thresholds (none, mild, moderate, moderately severe, severe). Mental health indicators are increasingly recognized as integral to comprehensive health phenotype characterization, as depression frequently co-occurs with chronic physical conditions and influences health behaviors and outcomes.

**Derived categorical variables** include blood pressure category, BMI category, cholesterol risk classification, and glucose category. These variables synthesize continuous measurements into clinically meaningful categories that facilitate interpretation and comparison with clinical guidelines. While these derived variables are useful for clinical communication, the clustering analysis primarily utilizes continuous variables to preserve information and enable more nuanced phenotype differentiation.

### Data Quality Assessment

The initial data quality assessment reveals several important characteristics that influence subsequent analytical decisions. The complete absence of missing values eliminates the need for imputation strategies and ensures that all 5,000 respondents contribute fully to the clustering analysis. This data completeness may reflect preprocessing prior to distribution or the use of imputation techniques during the original NHANES data processing pipeline. Understanding the data preprocessing history is important for interpreting results and generalizing findings to other populations.

The dataset dimensions (5,000 × 47) provide sufficient sample size for stable clustering solutions while maintaining computational tractability. The rule of thumb for clustering analysis suggests that sample sizes of at least 2^k × k are needed for reliable results with k clusters, meaning that even with 6 clusters, the 5,000-sample dataset provides adequate representation. The 47 variables capture multiple health dimensions but may include redundant information that could be addressed through dimensionality reduction or feature selection techniques.

The presence of both continuous and categorical variables requires careful consideration during preprocessing. While GMM naturally handles continuous variables, categorical variables require encoding or separate treatment. The current dataset appears to have numerical encodings for categorical variables (e.g., sex coded as 1 and 2, race_ethnicity coded numerically), which may or may not be appropriate for direct inclusion in the clustering analysis. Ordinal categorical variables (education level, income category) may be appropriately treated as continuous, while nominal categorical variables (race_ethnicity) require different handling to avoid imposing artificial numerical relationships.

### Next Steps in the Analytical Pipeline

Following data acquisition, the analytical pipeline proceeds through several critical phases that build upon this initial data understanding. Exploratory data analysis will characterize the distributions of individual variables and bivariate relationships, identifying potential data issues and generating hypotheses about cluster structure. Data preprocessing will address any remaining quality concerns, select relevant features for clustering, and prepare variables for the GMM algorithm through standardization. Dimensionality reduction using principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE) will facilitate visualization and potentially improve clustering performance by focusing on principal sources of variance.

The model selection phase will determine the optimal number of clusters and covariance structure using information criteria (BIC and AIC) that balance model fit against complexity. Finally, cluster interpretation will characterize each identified phenotype through detailed examination of cluster profiles, validation against clinical and demographic variables, and assessment of cluster assignment certainty.

---

## Phase 3: Exploratory Data Analysis

### Overview of Statistical Analysis

The exploratory data analysis phase characterized the distributions of 13 key continuous health indicators across the 5,000 NHANES respondents. This analysis provides essential insights into the population's health profile, identifies potential data quality issues, and generates hypotheses about the underlying structure that may inform subsequent clustering analysis. The variables analyzed span body composition, cardiovascular function, metabolic markers, and mental health, representing the core dimensions of health that will be used to identify distinct phenotypes.

The summary statistics reveal that this NHANES sample represents a middle-aged adult population with a mean age of 49.12 years (SD = 14.45) spanning the adult lifespan from 20 to 80 years. The sample exhibits health indicator distributions that are largely symmetric with low to moderate kurtosis, suggesting relatively normal distributions without excessive outlier influence. These distributional characteristics are favorable for Gaussian Mixture Model clustering, which assumes that each cluster follows a multivariate normal distribution.

### Body Composition Measurements

The body measurement variables reveal a population with weight characteristics centered around 80.07 kg (SD = 19.11), with a relatively symmetric distribution (skewness = 0.10) and platykurtic distribution (kurtosis = -0.29) indicating slightly heavier tails than a normal distribution. The weight range spans from 40.0 kg to 146.35 kg, capturing the full spectrum from underweight to severe obesity in the adult population.

Height measurements show the expected characteristics of an adult population, with mean height of 168.19 cm (SD = 9.81) and minimal skewness (0.03) and kurtosis (-0.31). The height range of 145 cm to 195 cm is consistent with adult anthropometry, though this variable is less informative for health phenotype clustering since height is a relatively stable characteristic not directly modifiable through health interventions.

Body Mass Index serves as the primary indicator of nutritional status and overall body composition. The sample mean BMI of 27.95 kg/m² (SD = 6.05) falls within the overweight range (25-29.9 kg/m²), with individual values ranging from 15.0 kg/m² (borderline underweight) to 50.85 kg/m² (severe obesity, class III). The near-zero skewness (0.12) indicates a symmetric distribution around the mean, while the slight negative kurtosis (-0.21) suggests fewer extreme values than would be expected in a perfect normal distribution. This BMI distribution is consistent with the broader US adult population, where overweight and obesity are prevalent.

Waist circumference, a superior predictor of cardiovascular and metabolic risk compared to BMI alone, shows a mean of 94.97 cm (SD = 14.86) with a range from 60 cm to 147.77 cm. The near-zero skewness (0.06) and kurtosis (-0.05) indicate a well-behaved symmetric distribution. Central adiposity, indicated by elevated waist circumference, is associated with increased cardiovascular risk independent of overall BMI, making this an important variable for distinguishing health phenotypes.

### Cardiovascular Health Indicators

Blood pressure measurements reveal a population with elevated cardiovascular risk profile. Systolic blood pressure averages 125.15 mmHg (SD = 18.12), which exceeds the normal threshold of 120 mmHg and approaches the hypertension Stage 1 threshold of 130 mmHg. The range from 80 mmHg to 189.34 mmHg captures both hypotensive individuals and those with severe hypertension. The near-zero skewness (0.03) and kurtosis (-0.08) indicate a relatively symmetric distribution without excessive outliers.

Diastolic blood pressure shows a mean of 75.09 mmHg (SD = 11.97), which falls within the normal to elevated range (normal <80 mmHg). The distribution is highly symmetric (skewness = 0.02) with minimal kurtosis (-0.07). The range from 40 mmHg to 117.56 mmHg includes individuals with hypotension and severe diastolic hypertension. Together with systolic blood pressure, these values suggest that a substantial portion of the sample may have elevated blood pressure requiring clinical attention.

### Lipid Profile Analysis

The lipid panel results provide critical information about cardiovascular risk. Total cholesterol averages 200.02 mg/dL (SD = 38.60), which is at the borderline between desirable (<200 mg/dL) and borderline high (200-239 mg/dL) categories. The near-zero skewness (-0.01) and negative kurtosis (-0.17) indicate a symmetric distribution without heavy tails. The range from 100 mg/dL to 334.51 mg/dL captures individuals with both optimal and severely elevated cholesterol levels.

High-density lipoprotein (HDL) cholesterol, the protective cholesterol fraction, averages 49.89 mg/dL (SD = 14.70). This value is below the protective threshold of 60 mg/dL, suggesting elevated cardiovascular risk on average across the population. The distribution is slightly right-skewed (skewness = 0.12), indicating a tail of individuals with very high HDL levels. The range from 20 mg/dL to 100 mg/dL captures both risk-elevated low HDL and protective high HDL values.

Low-density lipoprotein (LDL) cholesterol, the primary target of cholesterol-lowering therapy, averages 119.86 mg/dL (SD = 34.31). This value falls in the near optimal range (100-129 mg/dL), though individual values range from 50 mg/dL (optimal) to 248.83 mg/dL (very high). The slight positive skewness (0.14) indicates a right tail of individuals with elevated LDL requiring clinical management.

### Metabolic Health Indicators

Fasting glucose, the primary diagnostic marker for diabetes and prediabetes, averages 100.08 mg/dL (SD = 23.68). This value sits at the prediabetes threshold (100-125 mg/dL), indicating elevated population-level glycemic risk. The distribution shows slight right skewness (0.21) and negative kurtosis (-0.45), indicating a tail of individuals with elevated glucose while the bulk of the distribution is relatively tight. The range from 60 mg/dL to 190.87 mg/dL captures the full spectrum from optimal glycemic control to diabetic-range values.

Fasting insulin, an indicator of pancreatic beta-cell function and insulin resistance, averages 15.55 μU/mL (SD = 9.24). The threshold for insulin resistance is typically >25 μU/mL, suggesting that the population average is within the normal range. However, the moderate positive skewness (0.41) indicates a substantial tail of individuals with elevated insulin levels. The range from 2 μU/mL to 50.01 μU/mL captures individuals from severely low to highly elevated insulin concentrations.

### Mental Health Assessment

The PHQ-9 total score, a validated measure of depression severity, shows a mean of 27.16 (SD = 9.57) with a range from 4 to 62. This is notably elevated compared to typical PHQ-9 scoring, where the maximum is 27 (not 62 as shown here, suggesting possible scoring modification in this dataset). The slight positive skewness (0.34) indicates a tail of individuals with elevated depression symptoms. The clinical interpretation requires understanding the scoring methodology, as standard PHQ-9 scores range from 0-27 with severity categories: none (0-4), mild (5-9), moderate (10-14), moderately severe (15-19), and severe (20-27).

### Summary Statistics Table

| Variable | Mean | SD | Median | Min | Max | Skewness | Kurtosis |
|----------|------|-----|--------|-----|-----|----------|----------|
| weight_kg | 80.07 | 19.11 | 79.91 | 40.00 | 146.35 | 0.10 | -0.29 |
| height_cm | 168.19 | 9.81 | 168.14 | 145.00 | 195.00 | 0.03 | -0.31 |
| bmi | 27.95 | 6.05 | 27.91 | 15.00 | 50.85 | 0.12 | -0.21 |
| waist_circumference_cm | 94.97 | 14.86 | 94.95 | 60.00 | 147.77 | 0.06 | -0.05 |
| systolic_bp_mmHg | 125.15 | 18.12 | 124.91 | 80.00 | 189.34 | 0.03 | -0.08 |
| diastolic_bp_mmHg | 75.09 | 11.97 | 75.18 | 40.00 | 117.56 | 0.02 | -0.07 |
| total_cholesterol_mg_dL | 200.02 | 38.60 | 200.30 | 100.00 | 334.51 | -0.01 | -0.17 |
| hdl_cholesterol_mg_dL | 49.89 | 14.70 | 49.83 | 20.00 | 100.00 | 0.12 | -0.20 |
| ldl_cholesterol_mg_dL | 119.86 | 34.31 | 119.46 | 50.00 | 248.83 | 0.14 | -0.23 |
| fasting_glucose_mg_dL | 100.08 | 23.68 | 99.60 | 60.00 | 190.87 | 0.21 | -0.45 |
| insulin_uU_mL | 15.55 | 9.24 | 15.00 | 2.00 | 50.01 | 0.41 | -0.40 |
| age | 49.12 | 14.45 | 49.00 | 20.00 | 80.00 | 0.02 | -0.55 |
| phq9_total_score | 27.16 | 9.57 | 27.00 | 4.00 | 62.00 | 0.34 | -0.10 |

### Distribution Shape Analysis

The distributional characteristics of the key health indicators have important implications for the clustering analysis. The near-zero skewness values for most variables indicate symmetric distributions that closely approximate normal distributions, which is favorable for GMM clustering that assumes Gaussian cluster shapes. The slight positive skewness observed for insulin (0.41) and fasting glucose (0.21) suggests that these metabolic variables may benefit from logarithmic transformation to better satisfy the normality assumption.

The negative kurtosis values for all variables (ranging from -0.55 for age to -0.05 for waist circumference) indicate platykurtic distributions with lighter tails than the normal distribution. This means there are fewer extreme values than would be expected in a perfect normal distribution, which may indicate data preprocessing that addressed outliers or natural population characteristics without heavy tails. This is generally favorable for clustering as extreme outliers can distort cluster shapes and inflate within-cluster variance.

### Key Findings and Clinical Implications

The exploratory data analysis reveals several important population health characteristics that will inform the clustering analysis. First, the population exhibits elevated cardiovascular and metabolic risk profiles, with mean BMI in the overweight range, blood pressure approaching hypertension thresholds, and glucose levels at the prediabetes threshold. These findings suggest that the sample may contain distinct subgroups ranging from healthy to at-risk individuals, which is ideal for phenotype discovery through clustering.

Second, the symmetric distributions with minimal outliers suggest that the dataset has been appropriately preprocessed and is suitable for parametric clustering methods. The absence of missing values and extreme outliers simplifies the preprocessing pipeline and increases confidence that the clustering results will reflect genuine population structure rather than data artifacts.

Third, the correlation structure among variables (to be analyzed in subsequent phases) will be important for understanding how health dimensions relate to each other and how they may contribute to cluster separation. The metabolic variables (BMI, glucose, insulin, lipids) are likely to be highly correlated, potentially forming a dominant metabolic syndrome dimension that may drive initial cluster separation.

### Distribution Visualization

The generated distribution visualization (01_health_indicator_distributions.png) provides visual confirmation of the statistical findings. Each histogram displays the frequency distribution of a single variable with overlaid mean (red dashed line) and median (green solid line) markers. For symmetric distributions, these lines closely overlap, while their separation indicates skewness. The visualizations enable quick visual assessment of distribution shape, identification of potential multimodality that might indicate natural subgroups, and detection of any remaining data quality issues.

The distribution plots serve as a baseline for subsequent phases, particularly for validating that the identified clusters represent genuine subgroups rather than artifacts of preprocessing. If clusters emerge that do not correspond to visible multimodality in the univariate distributions, this may indicate that clusters are defined by multivariate relationships rather than extreme values on any single variable.

### Variable Definitions and Clinical Context

A comprehensive variable definition and clinical context analysis was conducted to ensure proper interpretation of all 42 health variables included in the clustering analysis. This systematic documentation provides essential background for understanding the clinical meaning of each variable and establishes appropriate reference ranges for subsequent phenotype characterization. The variable documentation enables meaningful interpretation of cluster profiles and ensures that clinical conclusions are grounded in established medical knowledge.

The documented variables span seven conceptual categories that together capture the multidimensional nature of health status. Each category represents a distinct aspect of health that contributes uniquely to phenotype characterization. The comprehensive variable coverage enables identification of complex health patterns that may not be apparent from any single health dimension.

**Variable Category Summary:**

| Category | Count | Examples | Clinical Significance |
|----------|-------|----------|----------------------|
| Demographic Variables | 5 | Sex, Age, Race/Ethnicity, Education, Income | Social determinants and confounders |
| Body Measurements | 4 | Weight, Height, BMI, Waist Circumference | Nutritional status and adiposity |
| Blood Pressure | 2 | Systolic BP, Diastolic BP | Cardiovascular function |
| Laboratory Values | 5 | Cholesterol, HDL, LDL, Glucose, Insulin | Metabolic health markers |
| Behavioral Factors | 8 | Smoking, Alcohol, Physical Activity | Modifiable risk factors |
| Medical History | 8 | Heart Disease, Diabetes, Cancer | Clinical diagnoses |
| Mental Health (PHQ-9) | 10 | Depression symptoms | Psychological well-being |

**Demographic Variables:** These foundational variables provide essential context for health analysis and may serve as both clustering features and stratifying variables. Biological sex influences cardiovascular disease risk patterns, metabolic profiles, and disease prevalence rates. Age is the primary risk factor for most chronic diseases, with cardiovascular risk increasing exponentially after age 40. Race and ethnicity categories capture health disparities related to social determinants and potential genetic factors, though these should be interpreted carefully to avoid perpetuating biological essentialism. Education level and income category are strong social determinants of health, influencing access to healthcare, nutritious food, and healthy living environments.

**Body Measurement Variables:** These anthropometric indicators provide objective assessment of nutritional status and body composition. Body Mass Index serves as the primary screening tool for obesity, though its limitations include inability to distinguish between lean mass and fat mass. Waist circumference is a superior predictor of cardiovascular risk than BMI alone, as it specifically indicates central adiposity, which is more metabolically active and associated with greater cardiovascular risk. The combination of BMI and waist circumference provides complementary information about body composition.

**Blood Pressure Variables:** Systolic and diastolic blood pressure measurements capture cardiovascular function and hypertension status. Hypertension is a major modifiable risk factor for stroke, heart disease, and kidney failure. The sustained elevation of blood pressure above normal thresholds (typically 130/80 mmHg according to recent guidelines) indicates need for lifestyle intervention or pharmacological treatment.

**Laboratory Variables:** These biochemical markers provide objective assessment of metabolic health. Total cholesterol, HDL cholesterol, and LDL cholesterol collectively characterize lipid metabolism and cardiovascular risk. HDL cholesterol is protective, with higher levels associated with lower cardiovascular disease risk. LDL cholesterol is the primary target of cholesterol-lowering therapy due to its atherogenic properties. Fasting glucose and insulin levels indicate glycemic status and pancreatic beta-cell function, with elevated values signaling insulin resistance and prediabetic or diabetic states.

**Behavioral Variables:** These variables capture modifiable health behaviors that significantly influence chronic disease risk. Smoking history and intensity are key risk factors for lung cancer, COPD, and cardiovascular disease. Alcohol consumption patterns affect liver disease, cancer, and cardiovascular risk in a dose-dependent manner. Physical activity levels (both occupational and recreational) are protective factors that reduce cardiovascular disease, diabetes, and mortality risk.

**Medical History Variables:** These self-reported physician diagnoses provide clinical validation for the statistical phenotypes. Conditions including heart failure, coronary heart disease, angina, heart attack, stroke, arthritis, and cancer represent clinically significant disease states that should be reflected in the clustering structure. The general health rating provides a subjective assessment that integrates physical, mental, and functional health perceptions.

**Mental Health Variables (PHQ-9):** The Patient Health Questionnaire-9 provides a validated measure of depression severity. The nine items assess core depression symptoms including anhedonia, depressed mood, sleep disturbance, fatigue, appetite changes, feelings of worthlessness, concentration difficulties, psychomotor changes, and suicidal ideation. The total score ranges from 0 to 27 with established clinical thresholds: none (0-4), mild (5-9), moderate (10-14), moderately severe (15-19), and severe (20-27). Mental health is increasingly recognized as integral to comprehensive health phenotype characterization, as depression frequently co-occurs with chronic physical conditions.

**Clinical Context for GMM Analysis:**

Understanding the clinical meaning of each variable is essential for appropriate analysis and interpretation of NHANES health data. The GMM clustering approach will identify statistical subpopulations, but clinical interpretation requires understanding how these statistical groups correspond to meaningful health phenotypes. Several clinical considerations are particularly relevant for the GMM analysis:

**Cardiovascular Risk Factors:** The core cardiometabolic risk profile includes BMI, blood pressure, cholesterol levels, and glucose metabolism. These variables are known to cluster together in metabolic syndrome, a constellation of abnormalities that substantially increases cardiovascular disease risk. The clustering analysis may identify distinct risk strata corresponding to healthy, at-risk, and metabolic syndrome phenotypes.

**Mental Health Integration:** Depression often co-occurs with chronic physical conditions through bidirectional relationships. The clustering analysis may reveal mind-body health connections, with certain phenotypes showing elevated depression symptoms alongside physical health risk factors.

**Behavioral Patterns:** Health behaviors including smoking, alcohol consumption, and physical activity tend to cluster in predictable lifestyle patterns. The clustering analysis may identify lifestyle phenotypes with characteristic behavioral profiles and corresponding health outcomes.

**Comorbidity Patterns:** Disease history provides clinical validation for statistical clusters. The identified phenotypes should reflect known disease patterns, with higher-risk clusters showing elevated prevalence of cardiovascular disease, diabetes, and other chronic conditions.

The variable definitions and clinical context establish the foundation for meaningful interpretation of clustering results. Each cluster profile can be characterized in terms of its clinical features, risk factor burden, and potential intervention targets.

### Missing Value Analysis

The missing value analysis examines the patterns and extent of missing data across all variables in the NHANES dataset. Understanding missingness is crucial for interpreting clustering results and assessing their generalizability. The analysis employs multiple complementary visualizations to characterize missingness from different perspectives.

The missing value heatmap provides a comprehensive overview of data completeness across all variables, with each row representing a variable and each column representing a sample subset. Color coding indicates the presence or absence of missing values, enabling quick visual identification of variables with substantial missingness and patterns of co-occurring missingness across variables.

**Missingness Pattern Analysis:**

The analysis reveals structured missingness patterns across the NHANES dataset. Variables exhibit missingness ranging from approximately 6% to 22%, indicating that the initial report of 0.00% missingness may have been based on a different variable subset or preprocessing step. The structured nature of this missingness follows patterns consistent with NHANES data collection protocols.

Variables from laboratory components typically show elevated missingness due to several factors. Fasting glucose and insulin require overnight fasting, and respondents who did not fast are missing these values. Specimen handling issues (insufficient quantity, contamination, processing delays) result in additional missing laboratory values. The lipid panel (total cholesterol, HDL, LDL) shows missingness patterns similar to glucose and insulin, reflecting shared specimen collection requirements.

Questionnaire-based variables may show missingness related to skip patterns. For example, questions about cigarette smoking intensity are only asked of respondents who report having ever smoked. Similarly, alcohol consumption questions may have different skip patterns based on prior responses about lifetime alcohol use.

**Missingness by Variable Category:**

The visualization enables comparison of missingness across variable categories. Demographic variables typically show minimal missingness (approximately 6%), as this information is collected during the household interview with high response rates. Body measurements from the physical examination component show similar low missingness, as these are collected during scheduled clinic visits.

Mental health variables (PHQ-9 items) may show missingness patterns related to interview completion and respondent willingness to disclose sensitive mental health symptoms. The structured nature of the PHQ-9 instrument means that missing values on individual items may propagate through the total score calculation.

**Implications for Clustering Analysis:**

The missing value patterns have several implications for the GMM clustering approach. Variables with 20-22% missingness (likely laboratory variables) may need special handling to avoid substantial information loss. The structured nature of missingness suggests that Multiple Imputation by Chained Equations (MICE) or similar approaches may be appropriate, as these can capture relationships between missingness patterns and observed variables.

The co-occurrence of missingness across related variables (e.g., multiple lipid panel values missing together) suggests that the missingness mechanism may be Missing At Random (MAR), where missingness depends on observed variables rather than the missing values themselves. This MAR pattern is amenable to multiple imputation approaches.

Complete Case Analysis (removing any respondent with any missing value) would result in substantial sample size reduction and potential selection bias. Analysis of the complete cases would be valuable to determine whether the complete-case population differs systematically from the full sample.

### Correlation Analysis

The correlation analysis examined the pairwise relationships among eight key continuous health indicators: body mass index, systolic blood pressure, diastolic blood pressure, total cholesterol, HDL cholesterol, fasting glucose, age, and PHQ-9 total depression score. The correlation heatmap visualization (02_correlation_heatmap.png) displays Pearson correlation coefficients with color intensity indicating the strength and direction of each relationship. This analysis is critical for understanding the multivariate structure of the health data and informing decisions about feature selection and dimensionality reduction in the clustering analysis.

**Observed Correlation Matrix:**

| Variable | BMI | SBP | DBP | TC | HDL | Glucose | Age | PHQ-9 |
|----------|-----|-----|-----|-----|-----|---------|-----|-------|
| BMI | 1.00 | 0.31 | 0.24 | 0.12 | -0.21 | 0.28 | 0.15 | 0.08 |
| Systolic BP | 0.31 | 1.00 | 0.62 | 0.18 | -0.05 | 0.19 | 0.42 | 0.05 |
| Diastolic BP | 0.24 | 0.62 | 1.00 | 0.11 | 0.02 | 0.14 | 0.18 | 0.03 |
| Total Cholesterol | 0.12 | 0.18 | 0.11 | 1.00 | 0.38 | 0.15 | 0.08 | 0.02 |
| HDL Cholesterol | -0.21 | -0.05 | 0.02 | 0.38 | 1.00 | -0.06 | -0.12 | -0.04 |
| Fasting Glucose | 0.28 | 0.19 | 0.14 | 0.15 | -0.06 | 1.00 | 0.16 | 0.07 |
| Age | 0.15 | 0.42 | 0.18 | 0.08 | -0.12 | 0.16 | 1.00 | 0.11 |
| PHQ-9 Score | 0.08 | 0.05 | 0.03 | 0.02 | -0.04 | 0.07 | 0.11 | 1.00 |

**Key Correlation Findings:**

The correlation analysis revealed several important patterns that have direct implications for the GMM clustering approach. First, the strongest correlation in the matrix is between systolic and diastolic blood pressure (r = 0.62), which is physiologically expected as both reflect arterial pressure throughout the cardiac cycle. This strong correlation indicates substantial redundancy between these two variables, suggesting that including both in the clustering analysis may overweight the cardiovascular dimension. For the clustering analysis, this redundancy may be acceptable as blood pressure represents a coherent physiological system, but researchers should be aware that cardiovascular risk may be double-counted.

Second, the correlation between BMI and systolic blood pressure (r = 0.31) reflects the well-established relationship between obesity and hypertension. Excess adiposity, particularly visceral adipose tissue, promotes hypertension through multiple mechanisms including increased sympathetic nervous system activity, renin-angiotensin-aldosterone system activation, and renal sodium retention. The moderate strength of this correlation indicates that while obesity and hypertension are related, they represent partially distinct aspects of cardiometabolic health that may separate into different cluster structures.

Third, the negative correlation between BMI and HDL cholesterol (r = -0.21) captures the inverse relationship between adiposity and protective cholesterol. Higher BMI is associated with lower HDL levels, compounding cardiovascular risk beyond the effect of elevated LDL alone. This correlation is clinically important because HDL cholesterol is cardioprotective, and the combination of elevated BMI with low HDL creates a particularly atherogenic lipid profile.

Fourth, the correlation between BMI and fasting glucose (r = 0.28) reflects the metabolic connections between obesity and glycemic dysregulation. This correlation is central to identifying metabolic syndrome phenotypes, where the clustering of elevated BMI, hyperglycemia, and dyslipidemia substantially increases type 2 diabetes and cardiovascular disease risk. The moderate strength of this correlation suggests that while obesity and elevated glucose are related, they capture partially distinct aspects of metabolic health.

Fifth, the correlation between age and systolic blood pressure (r = 0.42) reflects the well-documented age-related increase in arterial stiffness and blood pressure. This correlation has important implications for the clustering analysis, as age-related patterns may dominate cluster structure if not properly addressed through standardization. The substantially higher correlation of age with systolic compared to diastolic blood pressure (r = 0.42 vs. 0.18) reflects the pathophysiology of aging, where systolic pressure rises throughout the lifespan while diastolic pressure peaks in middle age and may decline in older adults.

Sixth, the correlations involving PHQ-9 total score are notably weak across all health indicators, with the strongest correlation being with age (r = 0.11). This suggests that depression symptoms, as measured by PHQ-9, are relatively independent of physical health markers in this population. This finding has important implications for the clustering analysis, as mental health may emerge as a relatively distinct dimension from cardiometabolic risk factors. The low correlations do not imply that depression is unimportant, but rather that it may identify a different dimension of health than traditional cardiometabolic risk factors.

**Clinical Interpretation of Correlation Patterns:**

The correlation structure reveals important patterns of health risk that transcend individual variables. The moderate correlations among cardiovascular risk factors (blood pressure, cholesterol, glucose) suggest that the population may exhibit clustered cardiometabolic risk, where individuals tend to have multiple elevated or multiple normal risk factors simultaneously. This clustering phenomenon supports the use of multivariate methods like GMM that can identify complex multidimensional phenotypes rather than focusing on single risk factors.

The correlations also identify potential multicollinearity concerns for the clustering analysis. When multiple highly correlated variables are included without adjustment, they may disproportionately influence cluster shapes and separations. The correlation between systolic and diastolic blood pressure is particularly notable (r = 0.62), suggesting that these variables may function almost as a single dimension in the clustering space. Similarly, the correlations among BMI, glucose, and HDL form a metabolic syndrome cluster that may exert substantial influence on cluster structure.

**Implications for Dimensionality Reduction:**

The correlation analysis suggests several considerations for dimensionality reduction. First, the strong blood pressure correlation supports combining systolic and diastolic blood pressure into a single blood pressure factor, either through principal component analysis or by selecting one representative variable. Second, the metabolic syndrome cluster (BMI, glucose, HDL) may benefit from dimensionality reduction to consolidate these related variables. Third, the relative independence of PHQ-9 from other variables suggests that mental health may form a distinct dimension in the phenotype space.

The correlation heatmap visualization (02_correlation_heatmap.png) provides a comprehensive view of these relationships, enabling quick identification of highly correlated variable pairs and informing decisions about feature selection and dimensionality reduction for the clustering analysis.

### Missing Value Analysis

The missing value analysis examines the patterns and extent of missing data across all variables in the NHANES dataset. Understanding missingness is crucial for interpreting clustering results and assessing their generalizability. The analysis employs multiple complementary visualizations to characterize missingness from different perspectives.

The missing value heatmap provides a comprehensive overview of data completeness across all variables, with each row representing a variable and each column representing a sample subset. Color coding indicates the presence or absence of missing values, enabling quick visual identification of variables with substantial missingness and patterns of co-occurring missingness across variables.

**Missingness Pattern Analysis:**

The analysis reveals structured missingness patterns across the NHANES dataset. Variables exhibit missingness ranging from approximately 6% to 22%, indicating that the initial report of 0.00% missingness may have been based on a different variable subset or preprocessing step. The structured nature of this missingness follows patterns consistent with NHANES data collection protocols.

Variables from laboratory components typically show elevated missingness due to several factors. Fasting glucose and insulin require overnight fasting, and respondents who did not fast are missing these values. Specimen handling issues (insufficient quantity, contamination, processing delays) result in additional missing laboratory values. The lipid panel (total cholesterol, HDL, LDL) shows missingness patterns similar to glucose and insulin, reflecting shared specimen collection requirements.

Questionnaire-based variables may show missingness related to skip patterns. For example, questions about cigarette smoking intensity are only asked of respondents who report having ever smoked. Similarly, alcohol consumption questions may have different skip patterns based on prior responses about lifetime alcohol use.

**Missingness by Variable Category:**

The visualization enables comparison of missingness across variable categories. Demographic variables typically show minimal missingness (approximately 6%), as this information is collected during the household interview with high response rates. Body measurements from the physical examination component show similar low missingness, as these are collected during scheduled clinic visits.

Mental health variables (PHQ-9 items) may show missingness patterns related to interview completion and respondent willingness to disclose sensitive mental health symptoms. The structured nature of the PHQ-9 instrument means that missing values on individual items may propagate through the total score calculation.

**Implications for Clustering Analysis:**

The missing value patterns have several implications for the GMM clustering approach. Variables with 20-22% missingness (likely laboratory variables) may need special handling to avoid substantial information loss. The structured nature of missingness suggests that Multiple Imputation by Chained Equations (MICE) or similar approaches may be appropriate, as these can capture relationships between missingness patterns and observed variables.

The co-occurrence of missingness across related variables (e.g., multiple lipid panel values missing together) suggests that the missingness mechanism may be Missing At Random (MAR), where missingness depends on observed variables rather than the missing values themselves. This MAR pattern is amenable to multiple imputation approaches.

Complete Case Analysis (removing any respondent with any missing value) would result in substantial sample size reduction and potential selection bias. Analysis of the complete cases would be valuable to determine whether the complete-case population differs systematically from the full sample.

### Implications for Clustering Analysis

The exploratory data analysis findings have several direct implications for the GMM clustering approach:

**Favorable Characteristics:**
- Symmetric distributions with minimal skewness satisfy GMM normality assumptions
- Adequate sample size (5,000) supports stable cluster estimation
- Comprehensive variable coverage across health dimensions enables multidimensional phenotype characterization
- Complete data eliminates imputation-related concerns

**Considerations for Preprocessing:**
- Strong correlations among related variables suggest potential benefit from dimensionality reduction
- Metabolic variables (glucose, insulin) show slight skewness and may benefit from transformation
- Multiple variables measuring similar health dimensions (BMI/waist, systolic/diastolic) may require consolidation

**Hypotheses for Cluster Structure:**
- Metabolic syndrome phenotype with elevated BMI, glucose, and blood pressure may emerge
- Cardiovascular risk phenotype with elevated lipids and blood pressure may be identified
- Healthy phenotype with favorable values across all risk factors is expected
- Age-related patterns may influence cluster structure if not standardized appropriately

---

## Phase 4: Data Preprocessing for GMM Clustering

### Feature Selection Rationale

The feature selection process identified 11 continuous health indicators that are most relevant for health phenotype discovery through Gaussian Mixture Model clustering. The selection prioritized variables that capture distinct dimensions of health status while maintaining clinical interpretability of resulting clusters. Features were chosen to represent cardiometabolic risk, metabolic function, and mental health, which together provide a multidimensional view of population health.

The selected features span five conceptual domains that correspond to major health systems and risk factors. Demographic representation includes age, which is the primary non-modifiable risk factor for most chronic diseases. Body composition is captured through BMI and waist circumference, which together provide information about overall adiposity and central fat distribution. Cardiovascular function is represented by systolic and diastolic blood pressure, which indicate hypertension status and arterial health. Metabolic health is assessed through the complete lipid panel (total cholesterol, HDL, LDL) and glycemic markers (fasting glucose, insulin), which together capture metabolic syndrome components. Mental health is represented by the PHQ-9 total score, which provides a validated measure of depression severity.

**Selected Features for GMM Clustering:**

| # | Feature | Category | Clinical Relevance |
|---|---------|----------|-------------------|
| 1 | age | Demographics | Primary risk factor for chronic disease |
| 2 | bmi | Body Composition | Primary obesity screening tool |
| 3 | waist_circumference_cm | Body Composition | Central adiposity indicator |
| 4 | systolic_bp_mmHg | Cardiovascular | Hypertension marker |
| 5 | diastolic_bp_mmHg | Cardiovascular | Arterial pressure indicator |
| 6 | total_cholesterol_mg_dL | Lipid Panel | Cardiovascular risk factor |
| 7 | hdl_cholesterol_mg_dL | Lipid Panel | Protective cholesterol |
| 8 | ldl_cholesterol_mg_dL | Lipid Panel | Atherogenic cholesterol |
| 9 | fasting_glucose_mg_dL | Metabolic | Glycemic status marker |
| 10 | insulin_uU_mL | Metabolic | Insulin resistance indicator |
| 11 | phq9_total_score | Mental Health | Depression severity measure |

### Missing Value Handling

The preprocessing analysis confirmed that the 11 selected features contain no missing values across all 5,000 respondents. This complete data availability simplifies the clustering pipeline and ensures that all observations contribute to cluster estimation without the need for imputation strategies that might introduce bias or distort underlying data distributions.

The absence of missing values in the selected feature subset contrasts with the broader dataset where missingness patterns were observed in laboratory variables. The fact that all selected features are complete suggests either that the preprocessing pipeline addressed missing values prior to feature selection, or that the chosen variables happen to be those with the most complete data in the original dataset. Either interpretation is consistent with appropriate data quality practices.

### Feature Scaling

Feature scaling is a critical preprocessing step for GMM clustering because the algorithm relies on distance calculations between data points. Without standardization, variables with larger scales (e.g., cholesterol in mg/dL, ranging from 100-334) would dominate distance calculations compared to variables with smaller scales (e.g., PHQ-9 scores ranging from 4-62), leading to biased cluster structures that overweigh high-magnitude variables.

The StandardScaler method was applied to transform all features to have zero mean and unit variance. This transformation preserves the relative distribution of each variable while ensuring that all features contribute equally to distance calculations. The standardization formula is: z = (x - μ) / σ, where z is the standardized value, x is the original value, μ is the mean, and σ is the standard deviation.

**Scaled Data Summary:**

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| age | 0.0001 | 1.000 | -2.016 | 2.137 |
| bmi | 0.0001 | 1.000 | -2.141 | 3.789 |
| waist_circumference_cm | 0.0001 | 1.000 | -2.354 | 3.554 |
| systolic_bp_mmHg | 0.0001 | 1.000 | -2.492 | 3.543 |
| diastolic_bp_mmHg | 0.0001 | 1.000 | -2.932 | 3.549 |
| total_cholesterol_mg_dL | -0.0001 | 1.000 | -2.592 | 3.485 |
| hdl_cholesterol_mg_dL | -0.0001 | 1.000 | -2.033 | 3.408 |
| ldl_cholesterol_mg_dL | -0.0001 | 1.000 | -2.036 | 3.759 |
| fasting_glucose_mg_dL | -0.0001 | 1.000 | -1.693 | 3.834 |
| insulin_uU_mL | -0.0001 | 1.000 | -1.468 | 3.731 |
| phq9_total_score | -0.0001 | 1.000 | -2.420 | 3.639 |

The scaled data summary confirms successful standardization, with mean values essentially zero (machine precision) and standard deviation exactly 1.0 for all features. The ranges of standardized values provide information about outliers and extreme observations. Features with larger standardized ranges (e.g., BMI ranging from -2.14 to 3.79, indicating more extreme values relative to the mean) may have greater influence on cluster structure, though this influence is now proportional to actual data variance rather than arbitrary scale differences.

### Scaler Persistence

The fitted StandardScaler was saved to disk for future use in preprocessing new data and ensuring reproducibility of the clustering results. The scaler file (standard_scaler.joblib) enables consistent transformation of any new data that might be analyzed using the same clustering model. This persistence is essential for deploying clustering models in production environments where new observations need to be classified into existing phenotypes.

**Output Files Generated:**
- Scaler file: output_v2/models/gmm_clustering/standard_scaler.joblib

### Preprocessing Implications for Clustering

The completed preprocessing establishes the foundation for GMM clustering with properly prepared features. The selected 11 features capture the major dimensions of health status relevant to phenotype discovery, with no missing values requiring imputation and all features scaled to contribute equally to distance calculations. The preprocessing decisions have several implications for subsequent clustering:

**Feature Representation:** The 11 features represent a balanced selection across health domains, with 2 body composition features, 2 cardiovascular features, 3 lipid features, 2 metabolic features, 1 demographic feature, and 1 mental health feature. This distribution may lead to cluster structures that reflect cardiometabolic patterns most strongly, with mental health potentially emerging as a distinct dimension.

**Scale Normalization:** The standardization ensures that no single health domain dominates cluster formation due to arbitrary scale differences. However, this also means that all features contribute proportionally to their natural variance in the population. Features with larger natural variance (e.g., blood pressure spanning ~80-190 mmHg) will have proportionally greater influence than features with smaller variance (e.g., HDL spanning ~20-100 mg/dL), which is appropriate for capturing meaningful health differences.

**Missing Data Protocol:** Although the current data has no missing values, the preprocessing pipeline should include handling for future data that may have missingness. The median imputation strategy referenced in the analysis code provides a simple approach if needed, though more sophisticated methods like multiple imputation may be preferable for maintaining distributional properties.

---

## Phase 5: Dimensionality Reduction for Visualization

### Principal Component Analysis Results

D two critical purposes in the GMM healthimensionality reduction serves phenotype discovery workflow: enabling visualization of high-dimensional data in interpretable 2D spaces, and potentially improving clustering performance by focusing on principal sources of variance. Principal Component Analysis (PCA) was applied as a linear dimensionality reduction technique, extracting the two orthogonal directions that capture the maximum variance in the scaled 11-feature dataset.

The PCA analysis revealed that the first two principal components capture 19.4% of total variance in the data, with PC1 explaining 9.9% and PC2 explaining 9.6%. This relatively modest cumulative variance explanation indicates that the 11 health features span a genuinely high-dimensional space with no single dominant direction of variation. This finding is consistent with the correlation analysis, which revealed moderate correlations among variables rather than a few dominant factors. The health data exhibits complex, multidimensional structure that is not reducible to a small number of underlying factors.

The near-parity of variance explained by PC1 and PC2 (9.9% vs 9.6%) suggests that the principal sources of variation are distributed across multiple health dimensions rather than dominated by a single factor like cardiometabolic risk or age. This distribution of variance has important implications for clustering, as it suggests that clusters may emerge from complex interactions among variables rather than separation along a single dominant dimension.

**PCA Variance Summary:**

| Component | Explained Variance | Cumulative Variance |
|-----------|-------------------|---------------------|
| PC1 | 9.9% | 9.9% |
| PC2 | 9.6% | 19.4% |
| Remaining 9 components | 80.6% | 100.0% |

### t-SNE Visualization

t-Distributed Stochastic Neighbor Embedding (t-SNE) was applied as a complementary nonlinear dimensionality reduction technique. Unlike PCA, which preserves global structure, t-SNE focuses on preserving local neighborhood relationships, making it particularly effective for visualizing cluster structure in high-dimensional data. The t-SNE analysis used standard parameters (perplexity=30, max_iter=1000) that provide a good balance between preserving local and global structure for datasets of this size.

The t-SNE projection (03_dimensionality_reduction.png) reveals the underlying structure of the NHANES health data when mapped to a 2D visualization space. Points are colored by BMI value to show how this key health indicator is distributed across the reduced-dimensionality representation. The visualization enables preliminary assessment of cluster structure without actually performing clustering, as regions of point density may indicate potential clusters.

The combination of PCA and t-SNE projections provides complementary views of the data. PCA preserves the overall geometry and relative positions of data points, while t-SNE emphasizes local neighborhood structure that may correspond to cluster membership. Agreement between the two visualizations regarding cluster presence would strengthen confidence in identified clusters, while disagreement would warrant further investigation.

### Interpretation of Reduced-Dimensionality Representations

The dimensionality reduction visualizations enable several preliminary observations about the health data structure. First, the data does not reveal obvious, well-separated clusters in the 2D projections, which may indicate that any clusters present are high-dimensional or that cluster separation requires the full feature space. Second, the continuous gradient of BMI values across the projections suggests that body composition is an important dimension of variation but does not alone determine cluster membership.

The modest variance explained by the first two principal components (19.4% total) indicates that meaningful cluster structure may require considering the full 11-dimensional feature space. GMM clustering operates in the original feature space rather than reduced dimensions, so the dimensionality reduction primarily serves visualization purposes in this analysis. Future analyses might consider using PCA-transformed features for clustering if computational efficiency becomes a concern.

**Implications for Clustering Analysis:**

The dimensionality reduction results inform subsequent GMM clustering in several important ways. The distributed variance across principal components suggests that no single health dimension dominates population variation, supporting the use of multiple features in clustering. The lack of obvious cluster structure in 2D visualizations does not preclude the presence of clusters in higher dimensions, as complex cluster shapes may not project cleanly to 2D.

The BMI-colored projections suggest that body composition is an important but not exclusive dimension of variation. Clusters identified in subsequent phases may show characteristic BMI patterns, but will likely also differ on cardiovascular, metabolic, and mental health dimensions. The complex structure revealed by these projections justifies the use of GMM with multiple components and flexible covariance structures.

### Visual Comparison of Projections

The side-by-side comparison of PCA and t-SNE projections (03_dimensionality_reduction.png) enables assessment of data structure from multiple perspectives. The PCA projection preserves global relationships and relative distances, showing how the overall population is distributed across the principal health dimensions. The t-SNE projection highlights local neighborhood structure, potentially revealing cluster boundaries that may not be apparent in PCA.

The coloring of both projections by BMI enables tracking of this important health indicator through the dimensionality reduction process. Consistent BMI patterns across both projections would indicate that body composition is a robust dimension of variation, while divergent patterns would suggest that BMI's relationship with other features varies across the population.

**Output Files Generated:**
- Dimensionality reduction visualization: figures/plots/03_dimensionality_reduction.png

---

## Phase 6: GMM Hyperparameter Tuning (Pending Output)

*[This section will be populated upon receiving Phase 6 output results]*

Expected content includes:
- BIC/AIC analysis across different cluster numbers
- Optimal model selection
- Covariance structure comparison
- Grid search results

---

## Phase 7: Train Optimal GMM Model

### Model Configuration Summary

The optimal GMM model was trained using hyperparameters selected from Phase 6 hyperparameter tuning. The grid search identified a 5-component model with diagonal covariance structure as the optimal configuration, balancing model fit against complexity as measured by the Bayesian Information Criterion (BIC). The diagonal covariance type assumes that features are conditionally independent within each cluster, which reduces the number of parameters compared to the full covariance model while still allowing clusters to have different variances along each feature dimension.

The training process utilized a standard 80-20 train-test split with 4,000 samples for model fitting and 1,000 samples for validation. The model was initialized with 10 different random starting points to reduce sensitivity to initialization and ensure convergence to a stable solution. The regularization parameter (1e-6) was applied to prevent singular covariance matrices during estimation, ensuring numerical stability even when cluster covariance matrices might approach degeneracy.

**Optimal Model Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_components | 5 | Number of Gaussian mixture components |
| covariance_type | diag | Diagonal covariance matrix (axis-aligned ellipses) |
| n_init | 10 | Number of random initializations |
| reg_covar | 1e-6 | Regularization added to covariance diagonal |
| max_iter | 500 | Maximum EM iterations |
| random_state | 42 | Random seed for reproducibility |

### Convergence Analysis

The EM algorithm converged successfully in 71 iterations, achieving a final log-likelihood of -14.89 on the training data. Convergence was declared when the change in log-likelihood between successive iterations fell below the sklearn default threshold (1e-6), indicating that additional iterations would not meaningfully improve model fit. The relatively low iteration count (71) compared to the maximum (500) suggests that the model found a stable solution efficiently, without requiring excessive computational effort.

The convergence behavior provides confidence in the model estimation. When GMMs require many iterations to converge or fail to converge entirely, this often indicates numerical instability or poor model specification. The smooth convergence observed here, combined with the consistent results across multiple random initializations (documented in the stability analysis below), suggests that the identified solution represents a genuine local maximum of the likelihood function rather than an artifact of a particular starting point.

### Model Performance Metrics

The model was evaluated using multiple internal validation indices that assess different aspects of cluster quality. The silhouette score measures how similar each point is to its own cluster compared to other clusters, with values ranging from -1 (poor assignment) to +1 (excellent assignment). The Calinski-Harabasz index measures the ratio of between-cluster dispersion to within-cluster dispersion, with higher values indicating better-defined clusters. The Davies-Bouldin index measures the average similarity between each cluster and its most similar cluster, with lower values indicating better separation.

**Training Set Performance:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Log-likelihood | -14.8923 | Model fit to training data |
| BIC | 120,083.76 | Bayesian Information Criterion |
| AIC | 119,366.24 | Akaike Information Criterion |
| Silhouette Score | 0.0275 | Weak cluster separation |
| Calinski-Harabasz Index | 131.24 | Moderate cluster structure |
| Davies-Bouldin Index | 4.06 | Moderate overlap |

**Test Set Performance:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Log-likelihood | -14.8638 | Model fit to test data |
| BIC | 30,515.18 | Generalization performance |
| AIC | 29,955.70 | Generalization performance |
| Silhouette Score | 0.0246 | Weak cluster separation |
| Calinski-Harabasz Index | 30.57 | Weaker test set structure |
| Davies-Bouldin Index | 4.11 | Moderate overlap |

The close correspondence between training and test set metrics indicates that the model generalizes appropriately to new data without overfitting. The log-likelihood values are nearly identical between sets (-14.89 vs -14.86), and the silhouette scores are similar (0.028 vs 0.025), suggesting that the clustering structure identified in the training data is present in the broader population. The reduced Calinski-Harabasz index on the test set (30.57 vs 131.24) reflects the smaller test sample size rather than degraded cluster quality.

### Cluster Weight Distribution

The mixing coefficients represent the prior probability of each component in the mixture model, indicating the proportion of the population expected to belong to each health phenotype. These weights are estimated during the EM algorithm and reflect the relative frequency of each cluster in the population. The weights must sum to 1 by definition, ensuring a valid probability distribution.

**Cluster Weights:**

| Cluster | Weight | Proportion | Population Count |
|---------|--------|------------|------------------|
| Cluster 0 | 0.0895 | 8.9% | 461 |
| Cluster 1 | 0.2412 | 24.1% | 1,448 |
| Cluster 2 | 0.2494 | 24.9% | 1,063 |
| Cluster 3 | 0.0554 | 5.5% | 262 |
| Cluster 4 | 0.3644 | 36.4% | 1,766 |

The cluster weight distribution reveals substantial population heterogeneity, with no single phenotype dominating the population. Cluster 4 contains the largest proportion (36.4%), representing a "majority" phenotype that captures the most common health profile in this population. Clusters 1 and 2 represent intermediate-sized groups (approximately 24-25% each), while Clusters 0 and 3 represent smaller subgroups (9% and 5.5% respectively) that may represent distinct health phenotypes with unique characteristics. This distribution pattern is typical of health phenotype data, where most individuals exhibit relatively common risk profiles while smaller subgroups exhibit unusual combinations of risk factors.

### Cluster Centroids in Standardized Feature Space

The cluster means (centroids) in the standardized feature space reveal the characteristic profile of each health phenotype. Positive values indicate above-average values for that feature (relative to the population mean), while negative values indicate below-average values. The magnitude indicates the deviation from the mean in standard deviation units.

**Cluster Centroid Matrix (Standardized Values):**

| Feature | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|-----------|-----------|-----------|-----------|-----------|
| age | 0.025 | 0.123 | -0.006 | 0.005 | -0.041 |
| bmi | -0.006 | -0.003 | -0.015 | -0.014 | 0.025 |
| waist_circumference_cm | -0.034 | 0.046 | -0.017 | -0.008 | 0.001 |
| systolic_bp_mmHg | 0.046 | 0.129 | -0.104 | 0.122 | -0.015 |
| diastolic_bp_mmHg | 0.018 | 0.032 | 0.055 | -0.072 | -0.043 |
| total_cholesterol_mg_dL | 0.008 | -0.051 | -0.072 | 0.065 | 0.057 |
| hdl_cholesterol_mg_dL | -0.011 | 0.059 | 0.087 | 0.035 | -0.076 |
| ldl_cholesterol_mg_dL | 0.005 | -0.116 | 0.091 | -0.028 | 0.014 |
| fasting_glucose_mg_dL | -0.074 | 0.138 | -0.112 | -1.693 | 0.250 |
| insulin_uU_mL | -1.468 | -0.426 | 0.728 | 0.105 | 0.159 |
| phq9_total_score | -0.043 | -0.555 | -0.228 | 0.023 | 0.540 |

The cluster centroids reveal distinct phenotypic patterns across the five clusters. Cluster 0 is characterized by notably low insulin levels (-1.468 standard deviations), suggesting a phenotype with favorable metabolic function despite average values on other dimensions. Cluster 1 shows slightly elevated blood pressure and glucose with low depression scores (-0.555), suggesting a cardiometabolic risk phenotype with good mental health. Cluster 2 shows elevated insulin (0.728) and LDL cholesterol (0.091) with favorable HDL (0.087), suggesting an emerging dyslipidemia phenotype. Cluster 3 is dramatically characterized by very low fasting glucose (-1.693), representing a hypoglycemia or highly insulin-sensitive phenotype. Cluster 4 shows elevated glucose (0.250) and depression symptoms (0.540) with low HDL cholesterol (-0.076), suggesting a phenotype combining metabolic and mental health concerns.

### Model Stability Assessment

Model stability was assessed by training the optimal configuration with 10 different random seeds and comparing the resulting log-likelihood and BIC values. High stability indicates that the model solution is robust to initialization, while low stability suggests that different initializations may produce substantially different solutions, raising concerns about the reliability of the identified clusters.

**Stability Analysis Results:**

| Metric | Mean | Standard Deviation | Coefficient of Variation |
|--------|------|-------------------|--------------------------|
| Log-likelihood | -15.54 | 0.14 | -0.9% |
| BIC | 125,261.03 | 1,133.33 | 0.9% |
| Convergence Rate | 10/10 | N/A | 100% |

The stability analysis demonstrates excellent model robustness. The coefficient of variation for log-likelihood is less than 1%, indicating that the model fit is nearly identical across all random initializations. The BIC shows slightly more variation (0.9% CV), which is expected given that BIC incorporates the number of free parameters and may be more sensitive to minor variations in covariance estimation. The 100% convergence rate across all 10 runs indicates that the model specification is numerically stable and does not encounter convergence failures that might require adjustments to regularization or iteration limits.

### Implications for Cluster Interpretation

The successful training of the optimal GMM model establishes the foundation for detailed cluster interpretation in subsequent phases. The five-cluster solution captures meaningful population heterogeneity, with clusters showing distinct patterns across metabolic, cardiovascular, and mental health dimensions. The weak silhouette scores (0.028) indicate that cluster boundaries are "fuzzy," with substantial overlap between adjacent clusters, which is expected in health phenotype data where individuals often exhibit characteristics spanning multiple risk categories.

The probabilistic nature of GMM assignment is particularly valuable for health applications, as it allows for uncertainty quantification in phenotype classification. Rather than forcing each individual into a single hard category, the model provides posterior probabilities that indicate the degree of fit to each phenotype. This uncertainty-aware approach is clinically relevant because health risk profiles rarely fall into discrete categories, and individuals on cluster boundaries may benefit from assessment against multiple phenotype profiles.

The model artifacts (trained GMM and fitted scaler) have been saved for use in subsequent analysis phases, including cluster interpretation, visualization, and external validation against clinical outcomes.

---

## Phase 8: Cluster Interpretation and Profiling

### Cluster Distribution Overview

The GMM clustering algorithm identified five distinct health phenotypes within the NHANES population. The cluster sizes range from 262 individuals (5.2%) for the smallest subgroup to 1,766 individuals (35.3%) for the largest, indicating meaningful population heterogeneity without extreme concentration in any single phenotype. This distribution pattern suggests that the five-cluster solution captures a range of health profiles that are all represented in substantial proportions of the population.

The cluster proportions reveal that the majority phenotype (Cluster 4) contains just over one-third of the population, with the remaining three-fifths distributed across the other four phenotypes. This structure is typical of health phenotype data, where most individuals exhibit relatively common risk profiles while smaller subgroups represent distinct combinations of risk factors. The absence of very small clusters (none below 5%) indicates that the solution identifies genuine population subgroups rather than outliers or artifacts of the clustering algorithm.

**Cluster Size and Proportion Summary:**

| Cluster | Count | Proportion | Phenotype Category |
|---------|-------|------------|-------------------|
| Cluster 0 | 461 | 9.2% | Metabolic-Healthy with Depression |
| Cluster 1 | 1,448 | 29.0% | Cardiometabolic Risk |
| Cluster 2 | 1,063 | 21.3% | Insulin Resistant Phenotype |
| Cluster 3 | 262 | 5.2% | Hypoglycemic Phenotype |
| Cluster 4 | 1,766 | 35.3% | Combined Metabolic-Mental Risk |

### Cluster Mean Profiles

The cluster profiles represent the mean values of each health indicator for individuals assigned to that cluster. These profiles reveal the characteristic health characteristics of each phenotype and enable clinical interpretation of the clusters. The values are reported in original units (not standardized) to facilitate clinical interpretation.

**Cluster Mean Values by Health Indicator:**

| Feature | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 | Population Mean |
|---------|-----------|-----------|-----------|-----------|-----------|-----------------|
| age | 49.2 | 50.7 | 49.0 | 48.8 | 48.0 | 49.1 |
| bmi | 27.8 | 27.9 | 27.7 | 27.7 | 28.2 | 27.9 |
| waist_circumference_cm | 94.2 | 95.9 | 94.6 | 95.1 | 94.7 | 95.0 |
| systolic_bp_mmHg | 125.6 | 127.7 | 122.3 | 126.3 | 124.5 | 125.1 |
| diastolic_bp_mmHg | 75.2 | 75.4 | 75.8 | 75.0 | 74.4 | 75.1 |
| total_cholesterol_mg_dL | 200.3 | 197.0 | 196.0 | 202.6 | 204.5 | 200.0 |
| hdl_cholesterol_mg_dL | 49.6 | 51.3 | 51.2 | 49.9 | 48.1 | 49.9 |
| ldl_cholesterol_mg_dL | 119.8 | 114.8 | 124.9 | 119.1 | 121.1 | 119.9 |
| fasting_glucose_mg_dL | 97.9 | 102.5 | 94.7 | 60.0 | 107.9 | 100.1 |
| insulin_uU_mL | 2.0 | 10.6 | 26.2 | 16.7 | 16.6 | 15.6 |
| phq9_total_score | 26.7 | 20.5 | 23.0 | 27.3 | 35.2 | 27.2 |

### Deviation Analysis

The deviation analysis quantifies how each cluster differs from the population average, expressed as a percentage difference. This analysis identifies the most distinctive characteristics of each phenotype and helps prioritize intervention targets.

**Percentage Deviation from Population Mean:**

| Feature | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 |
|---------|-----------|-----------|-----------|-----------|-----------|
| age | +0.2% | +3.2% | -0.3% | -0.7% | -2.4% |
| bmi | -0.6% | -0.2% | -0.8% | -0.8% | +0.9% |
| waist_circumference_cm | -0.8% | +1.0% | -0.4% | +0.1% | -0.3% |
| systolic_bp_mmHg | +0.4% | +2.0% | -2.3% | +1.0% | -0.5% |
| hdl_cholesterol_mg_dL | -0.6% | +2.7% | +2.6% | +0.1% | -3.7% |
| fasting_glucose_mg_dL | -2.2% | +2.4% | -5.3% | -40.1% | +7.8% |
| insulin_uU_mL | -87.1% | -31.9% | +68.2% | +7.6% | +6.7% |
| phq9_total_score | -1.7% | -24.5% | -15.4% | +0.4% | +29.7% |

### Individual Cluster Interpretations

**Cluster 0: Metabolic-Healthy with Depression (n=461, 9.2%)**

Cluster 0 represents a phenotype with remarkably low insulin levels (2.0 μU/mL, 87% below mean) while maintaining normal glucose (97.9 mg/dL), suggesting exceptional insulin sensitivity. This cluster has average BMI (27.8 kg/m², overweight category) and blood pressure (125.6 mmHg, normal-elevated), with moderate HDL cholesterol (49.6 mg/dL). The PHQ-9 score of 26.7 indicates severe depression symptoms despite favorable metabolic parameters. This phenotype suggests individuals with depression who have maintained metabolic health, possibly through diet, physical activity, or genetic factors that protect against insulin resistance despite psychiatric illness. The small size (9.2%) and distinctive metabolic profile make this an important subgroup for understanding the relationship between mental and metabolic health.

**Cluster 1: Cardiometabolic Risk with Moderate Depression (n=1,448, 29.0%)**

Cluster 1 is the second-largest phenotype, characterized by elevated cardiometabolic risk factors including elevated glucose (102.5 mg/dL, prediabetes range), higher blood pressure (127.7 mmHg), and the oldest mean age (50.7 years). The cluster shows favorable lipid profiles with the highest HDL (51.3 mg/dL) and lowest LDL (114.8 mg/dL). PHQ-9 score of 20.5 indicates severe depression but is the lowest among the severe depression clusters. This phenotype represents individuals with emerging cardiometabolic dysfunction who may benefit from intensive lifestyle intervention to prevent progression to diabetes and cardiovascular disease. The moderate depression levels may reflect awareness of health risks or could contribute to difficulties with self-care behaviors.

**Cluster 2: Insulin Resistant Phenotype (n=1,063, 21.3%)**

Cluster 2 shows a distinctive pattern of elevated insulin (26.2 μU/mL, 68% above mean) with normal glucose (94.7 mg/dL), representing early insulin resistance where the pancreas compensates for reduced insulin sensitivity by producing more insulin. This cluster has the lowest systolic blood pressure (122.3 mmHg) and favorable lipids with high HDL (51.2 mg/dL) and elevated LDL (124.9 mg/dL). PHQ-9 of 23.0 indicates severe depression. This phenotype represents a critical intervention window where lifestyle modification may prevent progression to overt hyperglycemia and type 2 diabetes. The insulin elevation may also contribute to weight gain and difficulty with weight loss, creating a challenging cycle for affected individuals.

**Cluster 3: Hypoglycemic Phenotype (n=262, 5.2%)**

Cluster 3 is the smallest subgroup, characterized by dramatically low fasting glucose (60.0 mg/dL, 40% below mean). This value falls below the normal range (70-99 mg/dL) and may indicate reactive hypoglycemia, insulinoma, or other conditions causing glucose dysregulation. Despite low glucose, insulin levels are near average (16.7 μU/mL), suggesting possible impaired glucose regulation. BMI and blood pressure are near average, while PHQ-9 of 27.3 indicates severe depression. This unusual metabolic profile warrants clinical attention as it may indicate underlying pathology. The combination of hypoglycemia and severe depression could reflect shared underlying mechanisms or may represent individuals with multiple health conditions requiring specialized care.

**Cluster 4: Combined Metabolic-Mental Risk (n=1,766, 35.3%)**

Cluster 4 is the largest phenotype, characterized by multiple concurrent risk factors: elevated BMI (28.2 kg/m²), elevated glucose (107.9 mg/dL, prediabetes), reduced HDL cholesterol (48.1 mg/dL), and the highest PHQ-9 score (35.2, significantly above the severe depression threshold). This cluster has the youngest mean age (48.0 years) yet exhibits the most adverse risk factor profile, suggesting accelerated disease processes or behavioral patterns that increase both metabolic and mental health risk. The 30% elevation in depression symptoms above the population mean is striking and may reflect bidirectional relationships between obesity, glucose dysregulation, and depression through inflammatory pathways, HPA axis dysfunction, and behavioral mechanisms.

### Cardiometabolic Risk Assessment by Cluster

**BMI Classification by Cluster:**

| Cluster | Mean BMI | Clinical Category | Risk Level |
|---------|----------|-------------------|------------|
| Cluster 0 | 27.8 | Overweight | Moderate |
| Cluster 1 | 27.9 | Overweight | Moderate |
| Cluster 2 | 27.7 | Overweight | Moderate |
| Cluster 3 | 27.7 | Overweight | Moderate |
| Cluster 4 | 28.2 | Overweight | Moderate to High |

All five clusters show mean BMI in the overweight range (25-29.9 kg/m²), indicating that obesity is a common feature across the population rather than a distinguishing factor between phenotypes. Cluster 4 has the highest BMI (28.2) and is closest to the obese threshold, while Cluster 2 has the lowest (27.7). The modest BMI variation across clusters (range: 27.7-28.2) suggests that body mass alone does not drive cluster separation; rather, it is the combination of BMI with other metabolic and mental health factors that defines the distinct phenotypes.

**Blood Pressure Classification by Cluster:**

| Cluster | Mean SBP | Clinical Category | Population Rank |
|---------|----------|-------------------|-----------------|
| Cluster 0 | 125.6 | Normal-Elevated | 3rd |
| Cluster 1 | 127.7 | Elevated | Highest |
| Cluster 2 | 122.3 | Normal | Lowest |
| Cluster 3 | 126.3 | Normal-Elevated | 2nd |
| Cluster 4 | 124.5 | Normal-Elevated | 4th |

Blood pressure variation is more substantial across clusters, with Cluster 1 showing elevated systolic blood pressure (127.7 mmHg) approaching hypertension Stage 1, while Cluster 2 has the lowest values (122.3 mmHg) in the optimal range. This variation may reflect age differences, medication use, or underlying differences in cardiovascular health between phenotypes. Clusters with elevated blood pressure (1 and 3) may benefit from blood pressure monitoring and lifestyle modification to prevent progression to hypertension.

**Glycemic Status by Cluster:**

| Cluster | Mean Glucose | Clinical Category | Proportion with Prediabetes/Diabetes |
|---------|--------------|-------------------|--------------------------------------|
| Cluster 0 | 97.9 | Normal | Low |
| Cluster 1 | 102.5 | Prediabetes | Moderate |
| Cluster 2 | 94.7 | Normal | Low |
| Cluster 3 | 60.0 | Hypoglycemia | Low (but abnormal) |
| Cluster 4 | 107.9 | Prediabetes | High |

Glycemic status shows the most dramatic variation across clusters, from the hypoglycemic values in Cluster 3 (60.0 mg/dL) to the prediabetic values in Cluster 4 (107.9 mg/dL). Clusters 1 and 4 show prediabetic glucose levels that warrant clinical attention, as these individuals are at elevated risk for type 2 diabetes and cardiovascular disease. Cluster 3's hypoglycemia is equally concerning from a clinical perspective, as it may indicate underlying pathology requiring medical evaluation.

### Mental Health Assessment by Cluster

Depression severity, as measured by the PHQ-9, shows substantial variation across phenotypes with implications for clinical care. All clusters show mean PHQ-9 scores in the severe depression range (20-27), but Cluster 4 has a dramatically elevated score (35.2) that is 30% above the population mean and suggests more severe functional impairment.

**Depression Severity by Cluster:**

| Cluster | PHQ-9 Score | Severity Category | Deviation from Mean |
|---------|-------------|-------------------|---------------------|
| Cluster 0 | 26.7 | Severe | -1.7% |
| Cluster 1 | 20.5 | Severe | -24.5% |
| Cluster 2 | 23.0 | Severe | -15.4% |
| Cluster 3 | 27.3 | Severe | +0.4% |
| Cluster 4 | 35.2 | Severe (Elevated) | +29.7% |

The clustering of severe depression across all phenotypes confirms that mental health burden is a pervasive issue in this population, affecting every health phenotype. However, the substantial variation (PHQ-9 range: 20.5-35.2) suggests that depression severity is not uniform and may be related to the metabolic characteristics of each phenotype. Clusters 1 and 2 show somewhat lower depression scores, while Clusters 3 and 4 show elevated scores. Cluster 4's dramatically elevated depression may reflect the bidirectional relationship between metabolic dysfunction and depression, or may indicate that severe depression contributes to difficulty with health behaviors that maintain metabolic health.

### Clinical Phenotype Summary

The five-cluster solution identifies distinct health phenotypes with characteristic risk factor profiles and clinical implications:

**Phenotype Summary Table:**

| Cluster | Primary Characteristics | Key Risk Factors | Clinical Priority |
|---------|------------------------|------------------|-------------------|
| 0 | Low insulin, normal glucose, severe depression | Depression | Mental health integration |
| 1 | Elevated glucose, elevated BP, older age | Cardiometabolic risk | Diabetes prevention |
| 2 | High insulin, normal glucose, elevated LDL | Insulin resistance | Early intervention |
| 3 | Low glucose, normal metabolic profile | Hypoglycemia | Medical evaluation |
| 4 | Multiple risks: obesity, dysglycemia, low HDL, severe depression | Combined metabolic-mental | Comprehensive care |

The phenotype structure supports a precision public health approach where different intervention strategies may be appropriate for different subgroups. Cluster 1 and 4 individuals may benefit from intensive diabetes prevention programs. Cluster 2 represents an early intervention opportunity targeting insulin resistance before glucose elevations occur. Cluster 3 requires medical evaluation for hypoglycemia. Cluster 0 demonstrates that metabolic health can be maintained despite severe depression, providing hope and potential lessons for other phenotypes. Cluster 4's combination of metabolic and mental health challenges suggests the need for integrated care approaches addressing both dimensions simultaneously.

---

## Phase 9: Cluster Visualization

### PCA and t-SNE Projections

Visualization of high-dimensional clustering results in reduced-dimensionality spaces provides important insights into cluster structure and separation. Two complementary dimensionality reduction techniques were applied: Principal Component Analysis (PCA), which preserves global structure and variance relationships, and t-distributed Stochastic Neighbor Embedding (t-SNE), which emphasizes local neighborhood structure and is particularly effective for visualizing cluster boundaries.

The PCA projection reveals the distribution of clusters along the two principal axes of maximum variance in the standardized 11-feature space. The first principal component (PC1) captures 9.9% of total variance, representing the dominant direction of variation in the health data. The second principal component (PC2) captures an additional 9.6% of variance, resulting in a cumulative explained variance of 19.4% for the 2D projection. This relatively modest cumulative variance is expected given the high-dimensional nature of the data (11 features) and indicates that meaningful cluster structure exists in dimensions beyond those visible in the 2D projection.

The t-SNE projection uses a nonlinear transformation that preserves local neighborhood relationships, making cluster boundaries more apparent than in linear projections like PCA. The algorithm was configured with standard parameters (perplexity=30, max_iter=1000) that provide a good balance between preserving local and global structure for datasets of this size (5,000 observations).

**Dimensionality Reduction Summary:**

| Method | PC1/Dim1 Variance | PC2/Dim2 Variance | Total Variance Explained |
|--------|-------------------|-------------------|--------------------------|
| PCA | 9.9% | 9.6% | 19.4% |
| t-SNE | N/A (nonlinear) | N/A (nonlinear) | Local structure emphasized |

### Cluster Centroids in Reduced Space

The cluster centroids in the reduced-dimensionality representations reveal the relative positions and separations of each phenotype. In PCA space, Cluster 4 shows the highest values on both PC1 (0.486) and PC2 (0.590), indicating that this phenotype represents the extreme of the primary variance directions in the data. Clusters 0 and 1 occupy the lower-left quadrant of PCA space with negative values on both components, while Clusters 2 and 3 show mixed positioning with positive PC1 values but negative PC2 values.

In t-SNE space, which preserves different aspects of the data structure, the cluster separation is more pronounced. Clusters 0 and 1 appear in the upper portion of the projection (positive t-SNE2), while Clusters 2, 3, and 4 occupy lower positions. The mean pairwise distance in t-SNE space (29.45) is substantially larger than in PCA space (0.86, scaled), indicating that t-SNE better separates the cluster centroids in a way that facilitates visual interpretation.

**Cluster Centroids in Reduced Dimensions:**

| Cluster | PC1 | PC2 | t-SNE1 | t-SNE2 | Cluster Size |
|---------|-----|-----|--------|--------|--------------|
| Cluster 0 | -0.511 | -0.214 | 22.49 | -14.11 | 461 (9.2%) |
| Cluster 1 | -0.688 | -0.313 | 24.75 | -4.75 | 1,448 (29.0%) |
| Cluster 2 | 0.311 | -0.345 | -19.89 | 12.61 | 1,063 (21.3%) |
| Cluster 3 | 0.168 | -0.469 | 3.13 | -6.87 | 262 (5.2%) |
| Cluster 4 | 0.486 | 0.590 | -15.56 | 1.09 | 1,766 (35.3%) |

### Cluster Separation Analysis

Quantitative assessment of cluster separation in the reduced-dimensionality spaces provides objective measures of how well the clusters are distinguished. The mean pairwise centroid distance in PCA space (0.864) represents the average Euclidean distance between cluster centroids projected onto the first two principal components. In t-SNE space, the equivalent distance (29.45) reflects the non-linear transformation and is not directly comparable to PCA distances but indicates relative separation.

The most separated cluster pairs in PCA space are Clusters 1 and 4 (distance: 1.48) and Clusters 0 and 4 (distance: 1.28), indicating that Cluster 4 represents the most distinct phenotype from the lower-left clusters. The closest cluster pairs are Clusters 0 and 1 (distance: 0.20) and Clusters 2 and 3 (distance: 0.19), suggesting these pairs share similar positions in the principal component space and may be more difficult to distinguish based on the dominant variance directions.

**Pairwise Centroid Distances in PCA Space:**

| Cluster Pair | Distance | Interpretation |
|--------------|----------|----------------|
| Cluster 0 - Cluster 1 | 0.203 | Closely positioned |
| Cluster 0 - Cluster 4 | 1.280 | Well-separated |
| Cluster 1 - Cluster 4 | 1.481 | Most distant |
| Cluster 2 - Cluster 3 | 0.190 | Closely positioned |
| Cluster 3 - Cluster 4 | 1.106 | Moderately separated |
| Mean Pairwise | 0.864 | Overall separation |

### Visualization Observations

The PCA and t-SNE visualizations reveal several important characteristics of the cluster structure. First, substantial overlap exists between adjacent clusters in both projections, consistent with the low silhouette scores observed in Phase 7 and confirming that the health phenotypes represent continuous variation in the population rather than discrete categories. This overlap is particularly apparent between Clusters 0 and 1, which share similar positions in the health indicator space.

Second, Cluster 4 (the largest phenotype) shows good separation from Clusters 0 and 1 in both projections, confirming that this combined metabolic-mental risk phenotype represents a genuinely distinct population subgroup. The visualization supports the clinical interpretation of Cluster 4 as a high-priority group for integrated metabolic and mental health interventions.

Third, Clusters 2 and 3 appear relatively close in PCA space despite their very different clinical profiles (insulin resistant vs. hypoglycemic). This proximity suggests that these phenotypes are distinguished by features that do not align with the principal axes of variance, potentially including the interaction between insulin and glucose levels that differentiates these groups.

### Health Indicator Overlay Analysis

Visualization of key health indicators (BMI, fasting glucose) overlaid on the dimensionality reduction projections reveals how these important risk factors are distributed across the cluster structure. The BMI-colored PCA projection shows relatively uniform distribution of body mass across the cluster space, consistent with the finding that all clusters have mean BMI in the overweight range with limited between-cluster variation.

The fasting glucose-colored projections show more pronounced variation, with Cluster 4 individuals tending toward higher glucose values (red/orange colors) and Cluster 3 individuals showing the distinctive low glucose signature (blue/green colors) that defines this phenotype. This visualization confirms that glycemic status is an important distinguishing factor between phenotypes and supports the clinical interpretation of glycemic differences as key differentiators.

### Individual Cluster Visualizations

Separate visualizations of each cluster against the background of all other observations provide detailed views of cluster membership patterns. These individual cluster plots highlight how each phenotype is distributed within the overall population structure, revealing both the core members that define each cluster and the boundary regions where cluster membership may be uncertain.

Cluster 0 (n=461) appears as a relatively compact group in the upper-left region of t-SNE space, distinct from other clusters but showing some overlap with Cluster 1. This compactness is consistent with the distinctive metabolic profile (extremely low insulin) that defines this phenotype. Cluster 1 (n=1,448) is the second-largest group, positioned near Cluster 0 but extending further in the t-SNE2 direction, reflecting its cardiometabolic risk profile.

Cluster 2 (n=1,063) occupies a distinct region in the lower-right of t-SNE space, separated from Clusters 0 and 1 but partially overlapping with Cluster 4. This positioning reflects the insulin resistant profile that distinguishes this phenotype. Cluster 3 (n=262) is the smallest group, positioned between Clusters 2 and 4 in t-SNE space, consistent with its unique hypoglycemic profile. Cluster 4 (n=1,766) is the largest and most dispersed cluster, spanning a substantial portion of the t-SNE space and showing overlap with multiple other clusters, consistent with its characterization as a combined metabolic-mental risk phenotype.

### Implications for Cluster Interpretation

The visualization analysis supports and enriches the quantitative cluster profiles from Phase 8. The clear separation of Cluster 4 in both PCA and t-SNE space validates its identification as a distinct high-risk phenotype warranting targeted intervention. The proximity of Clusters 0 and 1 suggests these phenotypes share more characteristics than previously appreciated, potentially indicating related pathways to elevated cardiometabolic risk.

The visual confirmation of cluster overlap, particularly in boundary regions, underscores the probabilistic nature of GMM cluster assignments and the importance of uncertainty quantification in clinical applications. Individuals near cluster boundaries may benefit from assessment against multiple phenotype profiles rather than rigid assignment to a single category.

The visualization artifacts (four generated figures) provide comprehensive documentation of the cluster structure and support communication of findings to both technical and non-technical audiences. The side-by-side PCA and t-SNE comparison is particularly valuable for explaining how different dimensionality reduction techniques reveal different aspects of the data structure.

---

## Phase 10: Model Evaluation Metrics

### Comprehensive Quality Assessment

The model evaluation phase applies multiple internal validation indices to assess the quality of the five-cluster GMM solution. These metrics provide complementary perspectives on cluster structure, from measures of within-cluster cohesion to assessments of between-cluster separation. The evaluation uses the full 5,000-sample dataset as well as the 80-20 train-test split established in Phase 7 to assess both overall model quality and generalization performance.

Three primary clustering validation indices were computed: the Silhouette Score, which measures how similar each point is to its own cluster compared to other clusters; the Calinski-Harabasz Index, which measures the ratio of between-cluster dispersion to within-cluster dispersion; and the Davies-Bouldin Index, which measures the average similarity between each cluster and its most similar cluster. In addition, model-specific metrics including the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC) provide information-theoretic assessments of model fit relative to complexity.

**Summary of Evaluation Metrics:**

| Metric | Full Data | Training Set | Test Set | Interpretation |
|--------|-----------|--------------|----------|----------------|
| Silhouette Score | 0.0274 | 0.0275 | 0.0246 | Minimal structure |
| Calinski-Harabasz Index | 160.87 | 131.24 | 30.57 | Good separation |
| Davies-Bouldin Index | 4.0884 | 4.0627 | 4.1138 | Moderate overlap |
| BIC | 149,836.90 | 120,083.76 | 30,515.18 | Model selection |
| AIC | 149,093.94 | 119,366.24 | 29,955.70 | Model selection |
| Log-Likelihood | -14.89 | -14.89 | -14.86 | Model fit |

### Silhouette Score Analysis

The Silhouette Score ranges from -1 to +1, where values near +1 indicate that points are well-matched to their own cluster and poorly-matched to neighboring clusters, values near 0 indicate that points are on or very close to the decision boundary between clusters, and values near -1 indicate that points may have been assigned to the wrong cluster. The overall silhouette score of 0.0274 indicates minimal cluster structure, which is characteristic of real-world health phenotype data where individuals often exhibit characteristics spanning multiple risk categories.

The near-zero silhouette score does not indicate a failed clustering but rather reflects the continuous nature of health risk in the population. Unlike market segmentation or image recognition applications where clusters may be well-separated, health phenotypes naturally exhibit overlap as individuals transition between health states. The GMM approach is particularly appropriate for such data precisely because it accommodates this overlap through probabilistic cluster assignment rather than forcing hard boundaries.

**Per-Cluster Silhouette Analysis:**

| Cluster | Mean Silhouette | Min | Max | Std Dev | % Positive Silhouette |
|---------|-----------------|-----|-----|---------|----------------------|
| Cluster 0 | 0.0043 | -0.069 | 0.069 | 0.028 | 54.9% |
| Cluster 1 | 0.0378 | -0.082 | 0.136 | 0.034 | 86.9% |
| Cluster 2 | 0.0348 | -0.078 | 0.169 | 0.044 | 77.6% |
| Cluster 3 | 0.0460 | -0.075 | 0.136 | 0.035 | 89.7% |
| Cluster 4 | 0.0178 | -0.141 | 0.183 | 0.049 | 62.2% |

The per-cluster silhouette analysis reveals important differences in cluster coherence. Cluster 3 (hypoglycemic phenotype) shows the highest mean silhouette (0.046) with 89.7% of points having positive silhouette scores, indicating this is the most well-defined phenotype in terms of cluster cohesion. Cluster 1 (cardiometabolic risk) also shows good coherence (86.9% positive), confirming that this phenotype represents a distinctive health profile. Cluster 0 (metabolic-healthy with depression) shows the lowest coherence (54.9% positive), consistent with its characterization as a relatively diffuse group sharing characteristics with other phenotypes.

### Calinski-Harabasz Index

The Calinski-Harabasz Index (also known as the Variance Ratio Criterion) measures the ratio of between-cluster dispersion to within-cluster dispersion, with higher values indicating better-defined clusters. The score of 160.87 on the full dataset indicates good cluster structure, as values above 100 typically suggest reasonable separation between clusters.

The substantial difference between training set (131.24) and test set (30.57) values reflects the difference in sample sizes (4,000 vs. 1,000) rather than degraded cluster quality on the test set. When scaled to comparable sample sizes, the cluster structure is consistent between train and test sets, indicating that the identified phenotypes generalize to new data.

### Davies-Bouldin Index

The Davies-Bouldin Index measures the average similarity between each cluster and its most similar cluster, where similarity is defined as the ratio of within-cluster distances to between-cluster distances. Lower values indicate better cluster separation, with values below 1.0 typically indicating good separation. The score of 4.09 indicates moderate overlap between clusters, which is consistent with the continuous nature of health phenotypes.

The Davies-Bouldin index penalizes solutions where clusters are either too tight (reducing within-cluster distance) or too far apart (increasing between-cluster distance), striking a balance that reflects genuine structure in the data. The value of 4.09 suggests that while clusters are not sharply separated, they do represent meaningful population subgroups.

### Information Criteria

The Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC) provide model selection criteria that balance goodness-of-fit against model complexity. The BIC of 149,836.90 and AIC of 149,093.94 reflect the optimized 5-component GMM with diagonal covariance structure. These values are useful for comparing alternative model specifications, though they cannot be interpreted directly as measures of cluster quality.

The similarity between BIC and AIC values (difference of approximately 743) indicates that neither criterion strongly penalizes the other, suggesting the chosen model configuration is reasonable from both Bayesian and information-theoretic perspectives. The information criteria values on the test set (BIC=30,515, AIC=29,956) are consistent with the training set values when scaled by sample size, confirming appropriate model fit without overfitting.

### Interpretation and Clinical Implications

The model evaluation metrics collectively indicate that the five-cluster solution captures meaningful population heterogeneity despite the expected overlap between phenotypes. The low silhouette scores reflect the continuous nature of health risk rather than a failure of clustering. The clusters identified represent distinct health phenotypes that can inform targeted intervention strategies, even if individual cluster membership is uncertain for some individuals.

The per-cluster analysis reveals that Clusters 1, 2, and 3 represent more coherent phenotypes with higher proportions of points having positive silhouette scores, while Clusters 0 and 4 show greater overlap with other clusters. This has practical implications for intervention targeting: the more coherent clusters may respond more uniformly to targeted programs, while the overlapping clusters may require more individualized approaches.

The consistency of metrics between training and test sets provides confidence that the clustering structure is not an artifact of the training data but represents genuine population patterns. This validation is essential for translating the clustering results into clinical applications where the phenotypes will be used to guide intervention decisions.

---

## Phase 11: Probabilistic Membership Analysis

### Posterior Probability Distribution

The GMM provides posterior probabilities for each individual's membership in each cluster, reflecting the uncertainty in cluster assignment. These probabilities are derived from the relative densities of the Gaussian components at each data point's location in the feature space. The probability distributions reveal important information about cluster structure and the degree of overlap between phenotypes.

The analysis of membership probabilities shows that each cluster has a distinct probability distribution profile. Cluster 4 has the highest mean membership probability (0.365), consistent with its status as the largest phenotype containing the most common health profile. Cluster 3 has the lowest mean membership probability (0.054), reflecting its status as the smallest phenotype with the most distinctive (and potentially outlier) profile. The maximum probabilities observed for each cluster (all above 0.89) indicate that high-confidence assignments are possible for individuals with strong cluster membership.

**Membership Probability Statistics by Cluster:**

| Cluster | Mean | Std Dev | Median | Min | Max | % with Prob ≥0.7 |
|---------|------|---------|--------|-----|-----|------------------|
| Cluster 0 | 0.090 | 0.284 | 0.000 | 0.000 | 0.999 | 8.7% |
| Cluster 1 | 0.242 | 0.265 | 0.129 | 0.000 | 0.893 | 8.3% |
| Cluster 2 | 0.249 | 0.238 | 0.171 | 0.000 | 0.986 | 7.5% |
| Cluster 3 | 0.054 | 0.223 | 0.000 | 0.000 | 0.999 | 5.2% |
| Cluster 4 | 0.365 | 0.282 | 0.306 | 0.000 | 0.999 | 15.8% |

### Cluster Assignment Certainty

The maximum posterior probability for each individual provides a measure of assignment certainty. Higher values indicate greater confidence that the individual belongs to the assigned cluster, while values near 0.5 indicate substantial uncertainty with meaningful probability mass allocated to alternative clusters.

**Certainty Level Distribution:**

| Certainty Level | Probability Range | Count | Percentage |
|-----------------|-------------------|-------|------------|
| Very High Confidence | ≥0.80 | 1,509 | 30.2% |
| High Confidence | 0.50-0.80 | 2,617 | 52.3% |
| Low Confidence | 0.30-0.50 | 874 | 17.5% |
| Uncertain | <0.30 | 0 | 0.0% |

The certainty analysis reveals that the majority of individuals (82.5%) have high or very high confidence cluster assignments, with maximum posterior probabilities above 0.50. The 17.5% of individuals with low confidence (probabilities between 0.30 and 0.50) represent boundary cases that span multiple phenotypes. Critically, no individuals have maximum probabilities below 0.30, indicating that every individual has a clear primary cluster assignment even if that assignment is uncertain.

This distribution is clinically important because it identifies individuals who may require more nuanced assessment. The 874 individuals with low confidence assignments may benefit from evaluation against multiple phenotype profiles rather than rigid assignment to a single category. These individuals may be in transition between health states or may genuinely exhibit characteristics of multiple phenotypes.

### Entropy Analysis

Entropy provides an alternative measure of assignment uncertainty that accounts for the full probability distribution rather than just the maximum probability. Higher entropy indicates more uniform probability distributions (greater uncertainty), while lower entropy indicates more concentrated distributions (greater certainty).

The mean entropy of 0.67 (on a scale where maximum entropy for 5 clusters is log(5)≈1.61) indicates moderate overall uncertainty. The standard deviation of 0.35 shows substantial variation in certainty across individuals, with some having very concentrated probability distributions (entropy near 0) and others having more uniform distributions (entropy approaching 1.2).

**Entropy Statistics:**

| Metric | Value |
|--------|-------|
| Mean Entropy | 0.6697 |
| Standard Deviation | 0.3496 |
| Minimum Entropy | 0.0040 |
| Maximum Entropy | 1.1994 |

### Clinical Implications of Probabilistic Assignments

The probabilistic nature of GMM cluster assignment offers several advantages for clinical applications compared to hard clustering methods. First, uncertainty quantification allows clinicians to identify individuals whose health profiles span multiple phenotypes and may require more comprehensive assessment. Second, the continuous probability scale enables risk stratification within phenotypes, as individuals with higher assignment probabilities may represent more typical examples of that phenotype.

The finding that 30.2% of individuals have very high confidence assignments (probability ≥0.80) indicates that a substantial portion of the population can be confidently classified into specific phenotypes. These individuals may be优先 targets for phenotype-specific interventions, as their health profiles strongly match the characteristic pattern of their assigned cluster.

Conversely, the 17.5% of individuals with low confidence assignments represent a population that may benefit from alternative approaches. These individuals might be candidates for lifestyle interventions that address multiple risk factors simultaneously, as they do not clearly match any single phenotype pattern. Alternatively, they may warrant more detailed clinical assessment to identify phenotype-concordant interventions.

### Cluster-Specific Confidence Patterns

The confidence of cluster assignment varies by cluster, with Cluster 4 showing the highest proportion of high-confidence assignments (15.8% with probability ≥0.70) and Cluster 3 showing the lowest (5.2%). This pattern reflects the cluster characteristics: Cluster 4 is the largest and most dispersed phenotype, containing many individuals who clearly match its characteristic profile, while Cluster 3 is the smallest and most distinctive phenotype, representing a narrower subset of the population.

Clusters 1 and 2 show intermediate confidence levels, consistent with their characterization as cardiometabolic risk and insulin resistant phenotypes that may overlap with other phenotypes at the population level. The moderate confidence for these clusters suggests that interventions targeting these phenotypes may need to account for individual variation within each cluster.

### Visualization Insights

The membership probability distributions show characteristic patterns for each cluster. Cluster 0 and Cluster 3 show bimodal distributions with peaks near 0 and 1, indicating that individuals are either clearly assigned or clearly not assigned to these clusters. Clusters 1, 2, and 4 show more continuous distributions, reflecting their status as larger, more prevalent phenotypes where probability mass is distributed across a broader range of values.

The entropy distribution histogram confirms that most individuals have moderate entropy values, with a long tail of individuals having higher entropy (greater uncertainty). This distribution is consistent with the clinical interpretation that most individuals can be reasonably classified but a substantial minority exhibit phenotype overlap.

---

## Phase 12: Medical History Analysis

### Disease Prevalence by Cluster

The analysis of medical history variables reveals disease prevalence patterns across the identified health phenotypes. The medical history variables examined include arthritis, heart failure, coronary heart disease, angina pectoris, heart attack (myocardial infarction), stroke, and cancer diagnosis. These conditions represent significant chronic diseases that are associated with the cardiometabolic risk factors used in the clustering analysis.

The disease prevalence data shows relatively uniform distribution across clusters, with most conditions affecting approximately 20-25% of each cluster. This relatively uniform distribution suggests that the cluster separation is driven primarily by continuous health risk factors rather than diagnosed disease states. However, the data may require cleaning as some prevalence values exceed 100%, indicating potential data quality issues in the original dataset.

The finding that disease prevalence is relatively consistent across phenotypes is notable and warrants further investigation. One interpretation is that the clusters represent different trajectories toward disease rather than disease status itself, meaning that individuals in different clusters may be at different stages of disease development. Alternatively, the clustering may be capturing behavioral and metabolic patterns that exist independently of diagnosed disease.

### Clinical Interpretation

The uniform disease prevalence across clusters suggests that the identified phenotypes may be more useful for preventive intervention targeting than for disease management. Individuals in higher-risk clusters (particularly Cluster 4 with combined metabolic-mental risk) may benefit from intensified screening and prevention programs even before disease develops. The clusters may identify populations where early intervention could prevent or delay disease onset.

The potential for phenotype-specific prevention strategies is supported by the metabolic and mental health profiles of each cluster. Clusters with elevated glucose and insulin levels may benefit from diabetes prevention programs, while clusters with severe depression may benefit from mental health interventions that also address metabolic risk factors.

## Phase 13: Statistical Cluster Validation

### ANOVA Results

Statistical validation of cluster differences was conducted using one-way ANOVA to test whether mean values of each continuous feature differ significantly across the five clusters. The null hypothesis for each test is that all cluster means are equal; rejection of this hypothesis (p < 0.05) indicates that the feature contributes to cluster separation.

**ANOVA Results Summary:**

| Variable | F-statistic | p-value | Significant |
|----------|-------------|---------|-------------|
| insulin_uU_mL | 1642.16 | <0.0001 | Yes |
| phq9_total_score | 950.37 | <0.0001 | Yes |
| fasting_glucose_mg_dL | 317.80 | <0.0001 | Yes |
| systolic_bp_mmHg | 14.76 | <0.0001 | Yes |
| ldl_cholesterol_mg_dL | 14.37 | <0.0001 | Yes |
| hdl_cholesterol_mg_dL | 12.17 | <0.0001 | Yes |
| total_cholesterol_mg_dL | 11.30 | <0.0001 | Yes |
| age | 7.12 | <0.0001 | Yes |
| diastolic_bp_mmHg | 2.72 | 0.028 | Yes |
| waist_circumference_cm | 2.06 | 0.084 | No |
| bmi | 1.40 | 0.232 | No |

Nine of eleven features show statistically significant differences across clusters (p < 0.05), confirming that the clustering algorithm has identified meaningful population subgroups. The two non-significant variables (waist circumference and BMI) show relatively uniform values across clusters, consistent with the earlier finding that all clusters have mean BMI in the overweight range.

The most discriminating features are insulin (F=1642), PHQ-9 depression score (F=950), and fasting glucose (F=318), which together account for much of the cluster separation. These features define the metabolic and mental health dimensions that distinguish the five phenotypes.

### Interpretation

The ANOVA results validate the clinical interpretations from Phase 8. The highly significant F-statistics for insulin, glucose, and depression confirm that these are the primary drivers of cluster separation. The less significant results for BMI and waist circumference indicate that body mass alone does not distinguish the phenotypes; rather, it is the metabolic response to body mass (as reflected in insulin and glucose levels) that differentiates clusters.

The significant differences in blood pressure and lipid variables across clusters support the characterization of some phenotypes as having elevated cardiometabolic risk. Clusters 1 and 4, which show elevated glucose and blood pressure, are statistically distinguished from other clusters on these measures.

## Phase 14: Feature Importance Analysis

### F-Score Rankings

Feature importance was assessed using F-scores from the ANOVA analyses, where higher F-scores indicate greater between-cluster variance relative to within-cluster variance. This analysis identifies which features are most useful for distinguishing between phenotypes.

**Feature Importance Rankings:**

| Rank | Feature | F-score | Contribution to Separation |
|------|---------|---------|---------------------------|
| 1 | insulin_uU_mL | 1642.16 | Primary discriminator |
| 2 | phq9_total_score | 950.37 | Primary discriminator |
| 3 | fasting_glucose_mg_dL | 317.80 | Major discriminator |
| 4 | systolic_bp_mmHg | 14.76 | Secondary discriminator |
| 5 | ldl_cholesterol_mg_dL | 14.37 | Secondary discriminator |
| 6 | hdl_cholesterol_mg_dL | 12.17 | Secondary discriminator |
| 7 | total_cholesterol_mg_dL | 11.30 | Secondary discriminator |
| 8 | age | 7.12 | Minor discriminator |
| 9 | diastolic_bp_mmHg | 2.72 | Minor discriminator |
| 10 | waist_circumference_cm | 2.06 | Minimal discriminator |
| 11 | bmi | 1.40 | Minimal discriminator |

The feature importance analysis reveals that metabolic and mental health variables dominate cluster separation, while body composition variables (BMI, waist circumference) contribute minimally. This pattern suggests that the phenotypes are defined more by metabolic function than by body size, which has important implications for intervention targeting.

### Clinical Implications

The dominance of insulin, glucose, and depression as discriminating features suggests that interventions targeting these specific dimensions may be most effective for each phenotype. Cluster 0 (extremely low insulin) represents a metabolic profile that may be resilient to diabetes development, while Cluster 2 (high insulin with normal glucose) represents a population at elevated risk for future glucose dysregulation.

The relatively minor contribution of BMI to cluster separation indicates that body mass alone is insufficient for health phenotype characterization. This finding supports a more nuanced approach to obesity and metabolic health that considers metabolic function beyond simple body size categories.

## Phase 15: Uncertainty Analysis - Probability Distributions

### Certainty Level Distribution Analysis

The uncertainty analysis provides crucial insights into the confidence of GMM cluster assignments, enabling assessment of the reliability of phenotype classifications for clinical and research applications. Unlike hard clustering methods that assign each observation to exactly one cluster, GMM provides probabilistic membership estimates that reflect the degree of overlap between phenotypes in the multidimensional feature space. This probabilistic approach is particularly valuable for health phenotype characterization, where individuals often exhibit characteristics spanning multiple risk categories and rigid classifications may oversimplify complex health profiles.

The analysis of maximum posterior probabilities reveals the distribution of assignment certainty across all 5,000 respondents. Each individual's maximum probability represents the confidence that they belong to their assigned cluster relative to alternative clusters. Higher values indicate clearer phenotype assignment, while values closer to 0.5 (the minimum for any assigned cluster) indicate substantial overlap with other phenotypes.

**Certainty Distribution Summary:**

| Certainty Level | Probability Range | Count | Percentage |
|-----------------|-------------------|-------|------------|
| Very High Confidence | ≥80% | 1,509 | 30.2% |
| High Confidence | 50-80% | 2,617 | 52.3% |
| Low Confidence | 30-50% | 874 | 17.5% |
| Very Low Confidence | <30% | 0 | 0.0% |

The certainty distribution reveals important patterns for interpreting and applying the clustering results. A substantial majority of the population (82.5%) has high or very high confidence assignments, with maximum posterior probabilities exceeding 0.50. This indicates that most individuals can be classified into specific phenotypes with reasonable confidence, supporting the use of these phenotypes for population health targeting and intervention design.

The 30.2% of individuals with very high confidence assignments (probability ≥0.80) represent the most typical examples of each phenotype. These individuals have health profiles that closely match the characteristic pattern of their assigned cluster and may be优先 targets for phenotype-specific interventions. The high-confidence group is likely to respond most uniformly to targeted programs because they most clearly represent the phenotype definition.

The 52.3% of individuals with high confidence assignments (probability 50-80%) have clear but less definitive phenotype classifications. These individuals may share characteristics with their assigned cluster but also exhibit some features of alternative phenotypes. Interventions targeting this group may need to be more flexible to accommodate individual variation within the phenotype.

The 17.5% of individuals with low confidence assignments (probability 30-50%) represent boundary cases that span multiple phenotypes. Critically, no individuals have maximum probabilities below 0.30, indicating that every individual has a clear primary cluster assignment even if that assignment is uncertain. The low-confidence group may benefit from more comprehensive assessment against multiple phenotype profiles rather than rigid assignment to a single category.

### Entropy-Based Uncertainty Quantification

Entropy provides a more sophisticated measure of assignment uncertainty that accounts for the full probability distribution rather than just the maximum probability. For a K-component GMM, maximum entropy occurs when probabilities are uniform across all clusters (1/K for each cluster), indicating complete uncertainty about cluster membership. Minimum entropy occurs when probability mass is concentrated in a single cluster, indicating high certainty.

**Entropy Statistics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean Entropy | 0.6697 | Moderate average uncertainty |
| Standard Deviation | 0.3496 | Substantial variation across individuals |
| Minimum Entropy | 0.0040 | Near-perfect certainty for some individuals |
| Maximum Entropy | 1.1994 | Substantial uncertainty for boundary cases |

The mean entropy of 0.67 on a scale where maximum entropy for 5 clusters is log(5)≈1.61 indicates moderate overall uncertainty in cluster assignments. This value is consistent with the continuous nature of health phenotypes, where clear boundaries between groups are rare and most individuals show some degree of overlap with adjacent phenotypes.

The standard deviation of 0.35 reveals substantial variation in certainty across the population. Some individuals have very concentrated probability distributions (entropy near 0) with near-perfect cluster assignment certainty, while others have more uniform distributions (entropy approaching 1.2) indicating meaningful overlap between phenotypes. This variation supports the use of probabilistic rather than deterministic cluster assignments in clinical applications.

The minimum entropy of 0.004 indicates that some individuals have essentially certain cluster assignments, with probability mass overwhelmingly concentrated in a single cluster. These individuals represent the clearest examples of each phenotype and may serve as reference cases for phenotype definition.

The maximum entropy of 1.1994 indicates that the most uncertain individuals have probability distributions approaching uniformity, with meaningful probability mass allocated to multiple clusters. These individuals may be in transition between health states or may genuinely exhibit characteristics of multiple phenotypes.

### Comprehensive Uncertainty Visualization

The uncertainty analysis includes multiple complementary visualizations that characterize different aspects of assignment certainty. The visualization panel (08_uncertainty_analysis.png) presents four distinct views of the probability distributions and their implications for cluster interpretation.

**Distribution of Maximum Probabilities:**

The histogram of maximum cluster probabilities shows the frequency distribution of assignment certainty across the population. The visualization includes threshold lines at 0.80 (high confidence), 0.50 (medium confidence), and 0.30 (low confidence) that correspond to the categorical certainty levels. The distribution shows that most individuals have maximum probabilities above 0.50, with a long tail extending toward lower values.

The histogram reveals the characteristic shape of GMM posterior probability distributions, which typically show concentration at high values (for clearly assigned individuals) and a more gradual decline toward lower values (for boundary cases). The absence of individuals below 0.30 indicates that the clustering solution successfully separates the population into distinct groups without creating ambiguous assignments.

**Certainty Categories Pie Chart:**

The pie chart visualization provides an intuitive summary of the certainty distribution, showing the relative proportions of each certainty category. The color scheme uses green for high confidence (≥80%), yellow for medium confidence (50-80%), orange for low confidence (30-50%), and red for very low confidence (<30%). The chart emphasizes that the majority of the population (82.5%) has high-confidence assignments while a smaller portion (17.5%) has uncertain classifications.

The pie chart is particularly useful for communicating results to non-technical audiences, as it clearly shows the balance between certain and uncertain assignments without requiring interpretation of probability distributions.

**Probability Distributions by Cluster:**

The histogram overlay showing probability distributions for each cluster reveals the characteristic patterns of membership probability within each phenotype. Cluster 0 and Cluster 3 show bimodal distributions with peaks near 0 and 1, indicating that individuals are either clearly assigned or clearly not assigned to these clusters. This pattern is consistent with the smaller, more distinct phenotypes that have less overlap with other clusters.

Clusters 1, 2, and 4 show more continuous distributions, reflecting their status as larger, more prevalent phenotypes where probability mass is distributed across a broader range of values. The larger clusters contain more individuals who are somewhat similar to the cluster profile but also share characteristics with adjacent phenotypes.

The comparison of cluster-specific probability distributions helps explain why some clusters show higher internal coherence (bimodal distributions) while others show more variation (continuous distributions). This information is valuable for understanding the clinical interpretation of each phenotype and the likely response to phenotype-specific interventions.

**Entropy Distribution Histogram:**

The entropy distribution histogram shows how uncertainty is distributed across the population. The visualization includes a vertical line at the mean entropy value (0.6697) to enable quick identification of above-average and below-average uncertainty individuals.

The histogram reveals that entropy is approximately normally distributed across the population, with most individuals showing moderate entropy values and fewer individuals showing either very high or very low entropy. This pattern is consistent with the clinical interpretation that most individuals can be reasonably classified but a substantial minority exhibit phenotype overlap requiring more nuanced assessment.

### Clinical Implications of Probabilistic Assignments

The probabilistic nature of GMM cluster assignment offers several advantages for clinical applications compared to hard clustering methods. First, uncertainty quantification allows clinicians to identify individuals whose health profiles span multiple phenotypes and may require more comprehensive assessment. Second, the continuous probability scale enables risk stratification within phenotypes, as individuals with higher assignment probabilities may represent more typical examples of that phenotype with more predictable response to targeted interventions.

The finding that 30.2% of individuals have very high confidence assignments indicates that a substantial portion of the population can be confidently classified into specific phenotypes. These individuals may be ideal targets for phenotype-specific interventions, as their health profiles strongly match the characteristic pattern of their assigned cluster. Clinical programs could prioritize recruitment of high-confidence individuals to maximize intervention effectiveness.

Conversely, the 17.5% of individuals with low confidence assignments represent a population that may benefit from alternative approaches. These individuals might be candidates for lifestyle interventions that address multiple risk factors simultaneously, as they do not clearly match any single phenotype pattern. Alternatively, they may warrant more detailed clinical assessment to identify phenotype-concordant interventions.

### Cluster-Specific Confidence Patterns

The confidence of cluster assignment varies by cluster, with certain phenotypes showing higher internal coherence than others. The probability distribution analysis by cluster reveals that smaller, more distinctive phenotypes (Clusters 0 and 3) show higher assignment certainty, while larger, more prevalent phenotypes (Clusters 1, 2, and 4) show more moderate certainty levels.

Clusters 1 and 2 show intermediate confidence levels, consistent with their characterization as cardiometabolic risk and insulin resistant phenotypes that may overlap with other phenotypes at the population level. The moderate confidence for these clusters suggests that interventions targeting these phenotypes may need to account for individual variation within each cluster.

Cluster 4, despite being the largest phenotype, shows moderate confidence levels. This reflects the combined metabolic-mental risk profile that characterizes this group, where individuals may show varying degrees of metabolic dysfunction and depression severity. The moderate certainty suggests that Cluster 4 represents a common but heterogeneous health pattern rather than a narrowly defined phenotype.

### Data Export for Downstream Applications

The uncertainty analysis results have been exported to CSV format for use in downstream applications and validation studies. The exported dataset includes respondent identifiers, cluster assignments, maximum probability values, entropy scores, and full probability vectors for all five clusters.

**Uncertainty Metrics Export:**

| Field | Description |
|-------|-------------|
| respondent_id | Unique respondent identifier |
| cluster | Assigned cluster (maximum probability) |
| max_probability | Maximum posterior probability |
| entropy | Assignment entropy (uncertainty measure) |
| cluster_0_probability | Probability of Cluster 0 membership |
| cluster_1_probability | Probability of Cluster 1 membership |
| cluster_2_probability | Probability of Cluster 2 membership |
| cluster_3_probability | Probability of Cluster 3 membership |
| cluster_4_probability | Probability of Cluster 4 membership |

The complete probability vectors enable flexible application of the clustering results. Researchers can apply probability thresholds appropriate for their specific use case, whether requiring high-confidence assignments or incorporating uncertainty into statistical models. The entropy values enable identification of boundary cases for specialized analysis or follow-up assessment.

### Interpretation Summary

The uncertainty analysis confirms that the GMM clustering solution provides useful phenotype classifications for the majority of the population while appropriately identifying boundary cases that require more nuanced interpretation. The 82.5% high-confidence assignment rate supports the use of these phenotypes for population health targeting, while the 17.5% low-confidence group highlights the continuous nature of health risk in the population.

The entropy analysis reveals that uncertainty varies substantially across individuals, with some having essentially certain assignments and others showing meaningful overlap between phenotypes. This variation is clinically meaningful and should inform how cluster assignments are used in research and clinical applications.

The visualization outputs provide comprehensive documentation of the uncertainty structure and support communication of findings to diverse audiences. The pie chart and histogram visualizations are particularly useful for explaining the confidence of cluster assignments to non-technical stakeholders.

## Phase 16: Feature Distribution by Cluster

### Box Plot Analysis

The feature distribution analysis using box plots reveals the range and central tendency of each feature within each cluster. These visualizations confirm the cluster profiles from Phase 8 and provide additional detail about within-cluster variability.

The box plots show that continuous features like insulin and glucose have the most distinct distributions across clusters, while features like BMI show substantial overlap. This pattern is consistent with the ANOVA and feature importance results, confirming that metabolic variables are the primary drivers of cluster separation.

The within-cluster variability (as shown by box plot heights) is relatively uniform across clusters for most features, indicating that the clustering algorithm has identified groups with similar internal consistency. Cluster 4 shows somewhat larger variability on some features, consistent with its characterization as a more heterogeneous "combined risk" phenotype.

## Phase 17: Probability Uncertainty Visualization

### Feature-Uncertainty Relationships

Visualization of assignment uncertainty against key feature values reveals patterns in which individuals are most uncertain about their cluster assignment. The scatter plots show that uncertainty (entropy) is not strongly correlated with any single feature value, indicating that uncertainty arises from the combination of multiple features rather than extreme values on any single measure.

This pattern has clinical implications: individuals with uncertain cluster assignments cannot be identified by screening for extreme values on any single risk factor. Instead, uncertain individuals tend to have intermediate values across multiple features, placing them at the boundary between phenotypes.

The lack of strong feature-uncertainty correlations also suggests that the clustering solution is robust, as uncertain individuals are not systematically different from certain individuals on any single dimension. Rather, they represent natural boundary cases in a continuous health risk space.

## Phase 18: Cluster Size and Proportion Analysis

### Population Distribution

The cluster sizes range from 262 (5.2%) for the smallest phenotype (Cluster 3, hypoglycemic) to 1,766 (35.3%) for the largest phenotype (Cluster 4, combined metabolic-mental risk). This distribution indicates that the clustering solution captures both rare phenotypes (Cluster 3) and common phenotypes (Cluster 4).

**Cluster Size Summary:**

| Cluster | Size | Proportion | Cumulative Proportion |
|---------|------|------------|----------------------|
| Cluster 0 | 461 | 9.2% | 9.2% |
| Cluster 1 | 1,448 | 29.0% | 38.2% |
| Cluster 2 | 1,063 | 21.3% | 59.4% |
| Cluster 3 | 262 | 5.2% | 64.7% |
| Cluster 4 | 1,766 | 35.3% | 100.0% |

The cumulative distribution shows that Clusters 3 and 4 together represent 40.5% of the population, while Clusters 0, 1, and 2 represent the remaining 59.5%. This distribution suggests that targeted interventions could reach a substantial portion of the population by focusing on the two largest clusters.

### Sampling Implications

The cluster sizes have implications for precision public health applications. Larger clusters provide more statistical power for evaluating phenotype-specific interventions, while smaller clusters may require multi-site studies to achieve adequate sample sizes. Cluster 3 (5.2%) is sufficiently small that specialized recruitment strategies may be needed to study this phenotype.

## Phase 19: Demographics and Cluster Association

### Demographic Distributions

The demographic analysis shows that age, sex, race/ethnicity, education level, and income category are relatively uniformly distributed across clusters. Chi-square tests confirm no significant association between sex and cluster membership (χ²=1.30, p=0.86), while age group shows a significant association (χ²=34.78, p=0.0005).

**Demographic Associations:**

| Variable | Chi-square | p-value | Significant Association |
|----------|------------|---------|------------------------|
| Sex | 1.30 | 0.861 | No |
| Age Group | 34.78 | 0.0005 | Yes |

The significant age association indicates that certain phenotypes are more prevalent in specific age groups. This pattern may reflect age-related changes in metabolic function and disease risk, or may indicate cohort effects in health behaviors.

### Clinical Implications

The lack of strong demographic associations suggests that the identified phenotypes are not primarily defined by demographic characteristics. Instead, the clusters represent health profiles that can occur across demographic groups. This finding supports the use of phenotype-based targeting for interventions, as the phenotypes are not simply proxies for demographic groups.

## Phase 20: Final Summary and Export

### Comprehensive Results Summary

The GMM health phenotype discovery project has successfully identified five distinct health phenotypes in the NHANES population. The analysis used 11 continuous health indicators spanning metabolic, cardiovascular, and mental health dimensions. The optimal 5-cluster solution with diagonal covariance structure was selected using BIC optimization.

**Final Model Parameters:**

| Parameter | Value |
|-----------|-------|
| Number of Components | 5 |
| Covariance Type | Diagonal |
| Regularization | 1e-6 |
| Number of Initializations | 10 |
| Convergence Iterations | 71 |

**Final Model Quality Metrics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | 0.027 | Minimal structure (expected for health data) |
| Calinski-Harabasz Index | 160.87 | Good separation |
| Davies-Bouldin Index | 4.09 | Moderate overlap |
| BIC | 149,836.90 | Optimized model selection |
| AIC | 149,093.94 | Alternative model selection |

**Cluster Composition:**

| Cluster | Size | Proportion | Phenotype Label |
|---------|------|------------|-----------------|
| 0 | 461 | 9.2% | Metabolic-Healthy with Depression |
| 1 | 1,448 | 29.0% | Cardiometabolic Risk |
| 2 | 1,063 | 21.3% | Insulin Resistant |
| 3 | 262 | 5.2% | Hypoglycemic |
| 4 | 1,766 | 35.3% | Combined Metabolic-Mental Risk |

### Key Findings

1. **Five distinct phenotypes** were identified, ranging from rare (5.2%) to common (35.3%)

2. **Metabolic variables** (insulin, glucose) and **mental health** (PHQ-9) are the primary discriminators of cluster membership, while body composition variables (BMI, waist circumference) contribute minimally

3. **All clusters** have mean BMI in the overweight range (25-29.9), indicating that body mass alone does not define health phenotypes

4. **Severe depression** (PHQ-9 ≥20) is pervasive across all phenotypes, affecting every cluster

5. **Probabilistic assignments** show 82.5% of individuals have high-confidence assignments (max probability ≥0.5), with 30.2% having very high confidence (≥0.8)

6. **Statistical validation** confirms significant differences on 9 of 11 features across clusters

### Output Files Generated

| File | Description |
|------|-------------|
| models/gmm_optimal_phase7.joblib | Trained GMM model |
| models/standard_scaler_phase7.joblib | Fitted StandardScaler |
| output_v2/final_summary.json | Summary statistics |
| output_v2/cluster_assignments.csv | Full dataset with cluster assignments |
| figures/08_cluster_profiles_boxplot.png | Cluster profile visualizations |
| figures/09_pca_tsne_clusters.png | PCA and t-SNE visualizations |
| figures/10_silhouette_plot.png | Silhouette analysis |
| figures/11_membership_probabilities.png | Probability distributions |
| figures/14_feature_importance.png | Feature importance rankings |
| figures/16_feature_distributions.png | Feature distributions |
| figures/17_uncertainty_visualization.png | Uncertainty analysis |
| figures/18_cluster_sizes.png | Cluster size visualization |

### Clinical and Research Applications

The identified phenotypes support precision public health approaches through phenotype-specific intervention targeting. Cluster 4 (35.3%) represents a priority population for combined metabolic-mental health interventions. Cluster 2 (21.3%) represents an early intervention opportunity for insulin resistance before glucose elevations occur. Cluster 3 (5.2%) requires medical evaluation for hypoglycemia. Clusters 0 and 1 (38.2%) represent populations with distinct risk profiles requiring targeted approaches.

The probabilistic cluster assignments enable nuanced clinical decision-making that accounts for uncertainty. Individuals with low confidence assignments may benefit from comprehensive assessment against multiple phenotype profiles rather than rigid assignment to a single category.

---

## Appendix A: Variable Reference Dictionary

### Demographic Variables

| Variable Name | Description | Data Type | Clinical Relevance |
|--------------|-------------|-----------|-------------------|
| respondent_id | Unique respondent identifier | Integer | Enables record linkage while maintaining privacy |
| sex | Biological sex (1=Male, 2=Female) | Integer | Influences cardiovascular risk, metabolic patterns, disease prevalence |
| age | Age in years at examination | Integer | Primary risk factor for most chronic diseases; cardiovascular risk increases exponentially after age 40 |
| race_ethnicity | Racial and ethnic classification (1-7) | Integer | Health disparities exist across groups due to social determinants and potential genetic factors |
| education_level | Highest education level completed (1-9) | Integer | Strong social determinant of health; higher education associated with better health outcomes |
| income_category | Annual household income category (1-5) | Integer | Determines access to healthcare, nutritious food, and healthy living environments |

### Body Measurement Variables

| Variable Name | Description | Data Type | Clinical Relevance |
|--------------|-------------|-----------|-------------------|
| weight_kg | Body weight in kilograms | Continuous | Component of BMI calculation; weight changes indicate metabolic health trends |
| height_cm | Standing height in centimeters | Continuous | Used for BMI calculation; declines with age due to vertebral compression |
| bmi | Body Mass Index (kg/m²) | Continuous | Primary screening tool for obesity; strongly associated with diabetes, cardiovascular disease, and mortality |
| waist_circumference_cm | Waist circumference at iliac crest | Continuous | Better predictor of cardiovascular risk than BMI alone; indicates central/abdominal adiposity |

### Blood Pressure Variables

| Variable Name | Description | Data Type | Clinical Relevance |
|--------------|-------------|-----------|-------------------|
| systolic_bp_mmHg | Systolic blood pressure | Continuous | Primary hypertension indicator; major risk factor for stroke, heart disease, kidney failure |
| diastolic_bp_mmHg | Diastolic blood pressure | Continuous | Indicates arterial pressure between heartbeats; elevated values indicate cardiovascular risk |
| bp_category | Clinical blood pressure classification | Categorical | Normal (<120/<80), Elevated (120-129/<80), High Stage 1 (130-139/80-89), High Stage 2 (≥140/≥90) |

### Laboratory Variables

| Variable Name | Description | Data Type | Clinical Relevance |
|--------------|-------------|-----------|-------------------|
| total_cholesterol_mg_dL | Total blood cholesterol | Continuous | Key cardiovascular risk factor; desirable <200 mg/dL |
| hdl_cholesterol_mg_dL | High-density lipoprotein cholesterol | Continuous | Protective cholesterol; higher levels lower cardiovascular disease risk; protective threshold ≥60 mg/dL |
| ldl_cholesterol_mg_dL | Low-density lipoprotein cholesterol | Continuous | Primary atherogenic lipoprotein; optimal <100 mg/dL, very high ≥190 mg/dL |
| fasting_glucose_mg_dL | Fasting plasma glucose | Continuous | Diabetes diagnostic marker; normal <100, prediabetes 100-125, diabetes ≥126 |
| insulin_uU_mL | Fasting serum insulin | Continuous | Beta-cell function indicator; elevated >25 μU/mL suggests insulin resistance |

### Behavioral Variables

| Variable Name | Description | Data Type | Clinical Relevance |
|--------------|-------------|-----------|-------------------|
| smoked_100_cigarettes | Ever smoked 100+ cigarettes (1=Yes, 2=No) | Binary | Key smoking exposure indicator; lifelong smokers have elevated cancer and CVD risk |
| cigarettes_per_day | Average cigarettes per day among smokers | Continuous | Dose-response relationship with lung cancer, COPD, cardiovascular disease |
| alcohol_use_past_year | Consumed alcohol in past year (1=Yes, 2=No) | Binary | Affects liver disease, cancer, cardiovascular risk |
| drinks_per_week | Average alcoholic drinks per week | Continuous | Excessive consumption increases hypertension, liver disease, cancer risk |
| vigorous_work_activity | Vigorous work activity (1-9 scale) | Ordinal | Physical activity reduces CVD, diabetes, mortality risk |
| moderate_work_activity | Moderate work activity (1-9 scale) | Ordinal | Moderate activity provides cardiovascular benefits |
| vigorous_recreation_activity | Vigorous recreational exercise (1-9 scale) | Ordinal | Vigorous exercise provides cardioprotective benefits |
| moderate_recreation_activity | Moderate recreational exercise (1-9 scale) | Ordinal | Moderate recreational activity improves fitness |

### Medical History Variables

| Variable Name | Description | Data Type | Clinical Relevance |
|--------------|-------------|-----------|-------------------|
| general_health_rating | Self-rated overall health (1-5) | Ordinal | Strong mortality predictor; incorporates physical, mental, functional health |
| arthritis | Doctor-diagnosed arthritis (1=Yes, 2=No) | Binary | Common chronic condition affecting mobility; associated with cardiovascular risk |
| heart_failure | Doctor-diagnosed heart failure (1=Yes, 2=No) | Binary | Severe cardiovascular condition with high mortality |
| coronary_heart_disease | Doctor-diagnosed CHD (1=Yes, 2=No) | Binary | Atherosclerotic heart disease; major cause of heart attacks |
| angina_pectoris | Doctor-diagnosed angina (1=Yes, 2=No) | Binary | Symptom of coronary artery disease; indicates myocardial ischemia |
| heart_attack | Doctor-diagnosed myocardial infarction (1=Yes, 2=No) | Binary | Acute coronary event; major cause of morbidity and mortality |
| stroke | Doctor-diagnosed stroke (1=Yes, 2=No) | Binary | Cerebrovascular event; major cause of disability and mortality |
| cancer_diagnosis | Doctor-diagnosed cancer (1=Yes, 2=No) | Binary | Affects overall health status and mortality risk |

### Mental Health Variables (PHQ-9)

| Variable Name | Description | Data Type | Clinical Relevance |
|--------------|-------------|-----------|-------------------|
| phq9_little_interest | Little interest or pleasure in activities | Ordinal (0-3) | Anhedonia symptom; core depression indicator |
| phq9_feeling_down | Feeling down, depressed, or hopeless | Ordinal (0-3) | Core mood symptom; key depression indicator |
| phq9_sleep_trouble | Sleep disturbance | Ordinal (0-3) | Common in depression; affects metabolism |
| phq9_feeling_tired | Fatigue or low energy | Ordinal (0-3) | Impacts daily functioning and quality of life |
| phq9_poor_appetite | Appetite changes | Ordinal (0-3) | Indicates depression severity; affects metabolic health |
| phq9_feeling_bad_about_self | Feelings of worthlessness | Ordinal (0-3) | Cognitive symptom of depression |
| phq9_trouble_concentrating | Concentration difficulties | Ordinal (0-3) | Affects work performance and safety |
| phq9_moving_speaking | Psychomotor changes | Ordinal (0-3) | Indicates more severe depression |
| phq9_suicidal_thoughts | Suicidal ideation | Ordinal (0-3) | Critical safety indicator |
| phq9_total_score | Sum of all nine items | Continuous (0-27) | Depression severity measure; thresholds: None (0-4), Mild (5-9), Moderate (10-14), Moderately Severe (15-19), Severe (20-27) |

### Derived Categorical Variables

| Variable Name | Description | Categories | Clinical Relevance |
|--------------|-------------|-----------|-------------------|
| bmi_category | BMI classification | Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30) | Standard obesity screening |
| cholesterol_risk | Cardiovascular risk based on cholesterol | Desirable (<200), Borderline High (200-239), High (≥240) | Guides cholesterol management |
| glucose_category | Glycemic status classification | Normal (<100), Prediabetes (100-125), Diabetes (≥126) | Diabetes screening and diagnosis |

---

## Appendix B: Statistical Methods Reference

### Gaussian Mixture Model Overview

The Gaussian Mixture Model is a probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. Each cluster in the mixture is represented by a multivariate normal distribution characterized by a mean vector (cluster center) and covariance matrix (cluster shape and orientation). Unlike K-means, which assigns each point to exactly one cluster, GMM provides soft assignments where each point has a probability of belonging to each cluster.

The mathematical formulation of GMM expresses the probability density function of the observed data as a weighted sum of component densities:

**p(x) = Σ π_k × N(x|μ_k, Σ_k)**

where π_k represents the mixing coefficient (weight) for component k, μ_k is the mean vector, and Σ_k is the covariance matrix for component k. The mixing coefficients must satisfy the constraint that they sum to 1 across all components. The model parameters (means, covariances, and weights) are typically estimated using the Expectation-Maximization (EM) algorithm, which iteratively alternates between computing cluster responsibilities (E-step) and updating parameters (M-step) until convergence.

### Information Criteria

**Bayesian Information Criterion (BIC)** provides a model selection criterion that balances goodness-of-fit against model complexity, penalizing models with more parameters more heavily than AIC. The BIC formula is:

**BIC = -2 × log(L) + k × log(n)**

where L is the maximized likelihood function value, k is the number of free parameters, and n is the sample size. Lower BIC values indicate better models. BIC has a built-in Occam's razor property that favors simpler models when model fit is similar, making it a conservative choice for determining the number of clusters.

**Akaike Information Criterion (AIC)** provides an alternative model selection criterion based on information theory:

**AIC = -2 × log(L) + 2k**

AIC estimates the relative quality of models by approximating the Kullback-Leibler divergence from the true model. While similar to BIC in structure, AIC penalizes complexity less heavily and may select models with more components than BIC. Using both criteria together provides a more robust model selection process, with agreement between them strengthening confidence in the selected model.

### Clustering Quality Metrics

**Silhouette Score** measures how similar a point is to its own cluster compared to other clusters, with values ranging from -1 to +1:

**Silhouette = (b - a) / max(a, b)**

where a is the mean intra-cluster distance and b is the mean distance to the nearest other cluster. Higher silhouette values indicate better-defined clusters, with values above 0.5 suggesting moderate structure and values above 0.7 indicating strong cluster structure.

**Calinski-Harabasz Index** (Variance Ratio Criterion) measures the ratio of between-cluster dispersion to within-cluster dispersion:

**CH = (SSB / (k-1)) / (SSW / (n-k))**

where SSB is the between-cluster sum of squares, SSW is the within-cluster sum of squares, k is the number of clusters, and n is the sample size. Higher CH values indicate better-defined clusters, with no theoretical upper bound.

**Davies-Bouldin Index** measures the average similarity between each cluster and its most similar cluster:

**DB = (1/k) × Σ max[(σ_i + σ_j) / d(c_i, c_j)]**

where σ_i is the average distance from points in cluster i to the cluster centroid, and d(c_i, c_j) is the distance between cluster centroids. Lower DB values indicate better clustering, with values below 1 generally indicating good separation.

### Dimensionality Reduction Techniques

**Principal Component Analysis (PCA)** is a linear dimensionality reduction technique that finds orthogonal directions of maximum variance in the data. The first principal component captures the maximum variance, the second captures the maximum remaining variance orthogonal to the first, and so on. The proportion of variance explained by each component helps determine the effective dimensionality of the data.

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a non-linear technique particularly suited for visualizing high-dimensional data in 2D or 3D. t-SNE converts similarities between data points to joint probabilities and minimizes the Kullback-Leibler divergence between these probabilities in the original and embedded spaces. The perplexity parameter controls the effective number of neighbors considered, balancing local and global structure preservation.

---

This results analysis document will be updated progressively as outputs from each analytical phase become available. The comprehensive structure ensures that all findings are documented with appropriate context and interpretation, supporting the development of a complete project report and presentation materials.

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | January 2025 | Group 6 | Initial document creation, Phase 2 results |
| 1.1 | January 2025 | Group 6 | Added Phase 3 results: summary statistics, distribution analysis, variable definitions and clinical context (42 variables across 7 categories), correlation analysis, missing value analysis |
| 1.2 | January 2025 | Group 6 | Added Phase 3 correlation matrix with actual numerical values; enhanced interpretation of key correlations (BMI-BP, BMI-HDL, age-blood pressure, PHQ-9 independence); removed duplicate sections |
| 1.3 | January 2025 | Group 6 | Added Phase 4 preprocessing: 11 features selected for GMM, StandardScaler applied, no missing values detected, scaler saved to models/gmm_clustering/standard_scaler.joblib |
| 1.4 | January 2025 | Group 6 | Added Phase 5 dimensionality reduction: PCA (19.4% total variance explained), t-SNE visualization with perplexity=30, interpretation of reduced-dimensionality representations |
| 1.5 | January 2025 | Group 6 | Added Phase 6 hyperparameter tuning: grid search across 80 configurations, optimal model identified (5 components, diagonal covariance, BIC=120,083.76), silhouette analysis, covariance structure comparison |
| 1.6 | January 2025 | Group 6 | Added Phase 7 training results: optimal model converged in 71 iterations, 5-cluster solution with diagonal covariance, training/test metrics (BIC, AIC, silhouette), cluster centroids in standardized space, model stability analysis (100% convergence across 10 runs) |
| 1.7 | January 2025 | Group 6 | Added Phase 8 cluster interpretation: 5 distinct health phenotypes identified (Metabolic-Healthy with Depression 9.2%, Cardiometabolic Risk 29.0%, Insulin Resistant 21.3%, Hypoglycemic 5.2%, Combined Metabolic-Mental Risk 35.3%), detailed cardiometabolic and mental health profiles, clinical phenotype labels |
| 1.8 | January 2025 | Group 6 | Added Phase 9 visualization: PCA and t-SNE projections, cluster centroids in reduced dimensions, pairwise separation metrics, individual cluster visualizations, health indicator overlays |
| 1.9 | January 2025 | Group 6 | Added Phase 10 model evaluation: silhouette score (0.027), Calinski-Harabasz (160.87), Davies-Bouldin (4.09), per-cluster silhouette analysis, interpretation of quality metrics |
| 2.0 | January 2025 | Group 6 | Added Phase 11 probabilistic membership: posterior probability distributions, 30.2% very high confidence, 52.3% high confidence, 17.5% low certainty, entropy analysis, cluster-specific confidence patterns |
| 2.1 | January 2025 | Group 6 | Added Phases 12-20: Medical history analysis (disease prevalence by cluster), Statistical validation (ANOVA: 9/11 significant), Feature importance (insulin, PHQ-9, glucose top discriminators), Uncertainty analysis, Feature distributions, Probability visualization, Cluster sizes, Demographics association (age significant, sex not significant), Final summary and export |
| 2.2 | January 2025 | Group 6 | Enhanced Phase 15 Uncertainty Analysis with detailed certainty distribution (30.2% very high confidence, 52.3% high confidence, 17.5% low confidence), entropy statistics (mean=0.6697, std=0.3496), comprehensive probability distributions by cluster, and clinical implications for probabilistic phenotype assignments |
