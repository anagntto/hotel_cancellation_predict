# 🏨 HotelSmart - Booking Cancellation Prediction

This repository contains the complete data science pipeline used to build the machine learning model behind the **HotelSmart App**, a predictive system designed to estimate whether a hotel reservation will be canceled.

## 🔍 Business Case

The CEO of HotelSmart is concerned about the high cancellation rate, which incurs an average cost of R$3500 per cancellation. With an estimated 50,000 reservations for the upcoming year, this represents a significant operational and financial challenge.

**Objective**: Develop a machine learning model that accurately predicts whether a reservation will be canceled, allowing the business to take proactive measures to reduce losses, optimize room occupancy, and increase profitability.

## 🧪 Project Pipeline

This notebook contains all the steps of a complete data science workflow:

### ✅ Complete Pipeline Included:
- **Business Case** – Context and goals
- **Imports** – All necessary libraries
- **Functions** – Helper functions (`cramer_v`, `calculate_metrics`, etc.)
- **Data Load** – Dataset loading and initial overview
- **Data Description** – Full descriptive analysis
- **Feature Engineering** – Creation of new variables
- **EDA (Exploratory Data Analysis)**:
  - Univariate analysis
  - Bivariate analysis (based on business hypotheses)
  - Multivariate analysis
- **Data Modeling** – Preprocessing steps:
  - Scaling using `StandardScaler`
  - Encoding categorical variables using `LabelEncoder` or `OrdinalEncoder`
- **Feature Selection**:
  - Tree-based feature importance
  - Lasso Regularization
  - Boruta Algorithm
- **Machine Learning**:
  - Training and evaluation of multiple models (Random Forest, XGBoost, LightGBM, Logistic Regression, etc.)
  - Performance comparison using precision, recall, accuracy
  - Hyperparameter tuning using Bayesian Optimization

## 🚀 App Deployment

The final model was deployed via a predictive application built with **Streamlit**, which allows users to simulate booking scenarios and receive predictions with recommendations.

You can check the project at:  
👉 https://github.com/anagntto/app_predict


## 🧠 Author

Developed as part of a postgraduate data science course challenge.

---
