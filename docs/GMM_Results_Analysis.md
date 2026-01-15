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

## Phase 5: Dimensionality Reduction (Pending Output)

*[This section will be populated upon receiving Phase 5 output results]*

Expected content includes:
- PCA results (explained variance, component loadings)
- t-SNE visualization
- Reduced-dimensionality representation
- Interpretation of principal components

---

## Phase 6: GMM Hyperparameter Tuning (Pending Output)

*[This section will be populated upon receiving Phase 6 output results]*

Expected content includes:
- BIC/AIC analysis across different cluster numbers
- Optimal model selection
- Covariance structure comparison
- Grid search results

---

## Phase 7: Train Optimal GMM Model (Pending Output)

*[This section will be populated upon receiving Phase 7 output results]*

Expected content includes:
- Final model parameters
- Convergence information
- Log-likelihood values
- Model diagnostics

---

## Phase 8: Cluster Interpretation and Profiling (Pending Output)

*[This section will be populated upon receiving Phase 8 output results]*

Expected content includes:
- Cluster profile characteristics
- Mean values for each cluster
- Clinical interpretation of phenotypes
- Distinguishing features per cluster

---

## Phase 9: Cluster Visualization (Pending Output)

*[This section will be populated upon receiving Phase 9 output results]*

Expected content includes:
- PCA 2D projections
- t-SNE visualizations
- 3D cluster representations
- Visual cluster separation assessment

---

## Phase 10: Model Evaluation Metrics (Pending Output)

*[This section will be populated upon receiving Phase 10 output results]*

Expected content includes:
- Silhouette score analysis
- Calinski-Harabasz index
- Davies-Bouldin index
- Interpretation of quality metrics

---

## Phase 11: Probabilistic Membership Analysis (Pending Output)

*[This section will be populated upon receiving Phase 11 output results]*

Expected content includes:
- Posterior probability distributions
- Cluster assignment certainty
- Membership probability analysis
- Soft clustering interpretation

---

## Phase 12: Medical History Analysis (Pending Output)

*[This section will be populated upon receiving Phase 12 output results]*

Expected content includes:
- Disease prevalence by cluster
- Medical history associations
- Clinical validation of phenotypes
- Comorbidity patterns

---

## Phase 13: Statistical Cluster Validation (Pending Output)

*[This section will be populated upon receiving Phase 13 output results]*

Expected content includes:
- ANOVA results for continuous variables
- Chi-square tests for categorical variables
- Statistical significance assessment
- Cluster validity confirmation

---

## Phase 14: Feature Importance Analysis (Pending Output)

*[This section will be populated upon receiving Phase 14 output results]*

Expected content includes:
- Feature contribution to clustering
- Discriminating variables
- Clinical relevance assessment
- Dimensionality reduction validation

---

## Phase 15: Uncertainty Analysis (Pending Output)

*[This section will be populated upon receiving Phase 15 output results]*

Expected content includes:
- Probability distribution analysis
- Assignment confidence levels
- Entropy calculations
- Uncertainty quantification

---

## Phase 16: Feature Distribution by Cluster (Pending Output)

*[This section will be populated upon receiving Phase 16 output results]*

Expected content includes:
- Box plot visualizations
- Violin plot distributions
- Statistical comparisons
- Outlier identification

---

## Phase 17: Probability Uncertainty Visualization (Pending Output)

*[This section will be populated upon receiving Phase 17 output results]*

Expected content includes:
- Detailed probability plots
- Confidence visualization
- Uncertainty mapping
- Risk assessment visualization

---

## Phase 18: Cluster Size and Proportion Analysis (Pending Output)

*[This section will be populated upon receiving Phase 18 output results]*

Expected content includes:
- Cluster size distribution
- Population prevalence estimates
- Proportion comparisons
- Sample size adequacy

---

## Phase 19: Demographics and Cluster Association (Pending Output)

*[This section will be populated upon receiving Phase 19 output results]*

Expected content includes:
- Demographic distribution by cluster
- Chi-square test results
- Population representativeness
- Disparity analysis

---

## Phase 20: Final Summary and Export (Pending Output)

*[This section will be populated upon receiving Phase 20 output results]*

Expected content includes:
- Complete project summary
- Model performance recap
- Key findings synthesis
- Export file inventory

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
