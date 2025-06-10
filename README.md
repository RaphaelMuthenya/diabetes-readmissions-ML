# Patient Readmission Risk Prediction

## Overview
This project aims to predict the likelihood of patients being readmitted to the hospital within 30 days of discharge. Using machine learning techniques on patient data, we identify key risk factors and generate models that can assist healthcare providers in improving patient care and reducing readmission rates.

## Dataset
- Source: Diabetes patient dataset
- Size: ~100,000 records
- Target: `readmit_30` (binary: 1 = readmitted within 30 days, 0 = not readmitted)

## OSEMN Framework
### 1. **Obtain**
- Collected structured patient data including demographics, lab tests, medications, diagnoses, and hospital visit info.

### 2. **Scrub**
- Removed duplicates, irrelevant features (e.g. patient ID), and handled missing values.
- Applied `get_dummies` to encode categorical variables.

### 3. **Explore**
- Investigated class imbalance: Only ~11% of patients were readmitted.
- Plotted correlations and distributions of key features (e.g. lab procedures, medications).
  
### 4. **Model**
- **Models used**: Logistic Regression, Decision Tree, Random Forest
- **Techniques**:
  - Train/test split (80/20)
  - RandomOverSampler to balance classes
  - StandardScaler for numeric features
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score, AUC
  - Confusion matrices and ROC curves visualized
- **Hyperparameter tuning**: GridSearchCV for Logistic Regression and Random Forest

### 5. **iNterpret**
- Top Features:
  - `num_lab_procedures`, `num_medications`, `time_in_hospital`, `age`
- **Clinical Implications**:
  - Patients with high lab tests or long stays have greater risk
  - Prioritize patients with high predicted risk for follow-up care
- **Dashboard-ready** output: individual patient risk scores, prediction probabilities

## Deployment
- The model can be deployed in a hospital dashboard to:
  - Flag high-risk patients
  - Guide discharge planning
  - Improve resource allocation

## Files Included
- `Readmission_Risk_Presentation.pptx`: Summary for stakeholders
- `diabetes-ml.ipynb`: Full modeling and evaluation pipeline
- `models/`: Saved model objects for deployment
- `data/`: Cleaned dataset used in modeling

## Author
Created by [Raphael Muthenya]  
Date: June 2025
