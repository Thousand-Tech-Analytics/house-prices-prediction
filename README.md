# house-prices-prediction

A Structured Regression Pipeline for Tabular Data

## Overview
This project builds a reproducible and well-documented regression pipeline using the
Ames Housing dataset (Kaggle House Prices competition). Rather than focusing on
leaderboard optimization, the goal is to systematically analyze, model, and evaluate
regression approaches for tabular data, with emphasis on data understanding, feature
engineering, model comparison, and interpretability.

The project is organized into three stages (Week 7–9), each producing auditable
artifacts and documented design decisions.

## Project Objectives
- Build a clean regression workflow for mixed numerical and categorical data
- Compare linear and tree-based models under consistent evaluation metrics
- Emphasize model stability, interpretability, and reproducibility
- Produce artifacts suitable for professional review and audit

## Dataset
- Source: Kaggle – House Prices: Advanced Regression Techniques
- Training samples: 1,460
- Features: 79 explanatory variables
- Target variable: SalePrice

## Evaluation Metric
All models are evaluated using RMSE on log(SalePrice), which aligns with the Kaggle
competition metric and reduces the influence of extreme values.

## Project Structure
```text
├── house-prices-prediction.ipynb
├── outputs/
│   ├── eda/
│   │   ├── feature_type_summary.csv
│   │   ├── missing_rate.csv
│   │   ├── target_summary.csv
│   │   └── top_numeric_correlations.csv
│   └── models/
│       ├── baseline_ridge_cv.json
│       ├── fe_ridge_cv.json
│       ├── week_8_regularization_cv_table.csv
│       ├── week_8_regularization_selection.json
│       ├── week_9_model_family_comparison.csv
│       └── week_c_gbdt_feature_importance_top20.csv
```

## Week 7 — Data Understanding & Feature Engineering

### Exploratory Data Analysis
The dataset was examined to understand target distribution, missing value patterns,
and feature composition. SalePrice was found to be right-skewed, motivating the use
of a log-scale evaluation metric.

Artifacts:
- missing_rate.csv
- feature_type_summary.csv
- target_summary.csv
- top_numeric_correlations.csv

### Baseline Model
A regularized linear regression model (Ridge) was used as a baseline with:
- Median imputation for numerical features
- Mode imputation and one-hot encoding for categorical features
- 5-fold cross-validation

Artifact:
- baseline_ridge_cv.json

### Feature Engineering
The following interpretable features were engineered:
- TotalSF: total above- and below-ground living area
- HouseAge: year sold minus year built
- IsRemodeled: indicator for post-construction remodeling
- TotalBath: weighted sum of full and half bathrooms

These features produced a modest but consistent improvement without increasing
model variance.

Artifact:
- fe_ridge_cv.json

## Week 8 — Regularization & Model Selection

### Models Evaluated
- Ridge Regression
- Lasso Regression
- ElasticNet Regression

Each model was evaluated across multiple regularization strengths using
cross-validation.

### Model Selection
Although Lasso and ElasticNet occasionally achieved slightly lower error at specific
hyperparameter values, their performance was more sensitive to regularization
strength. Ridge regression was selected due to its stability and robustness in a
high-dimensional one-hot encoded feature space.

Artifacts:
- week_8_regularization_cv_table.csv
- week_8_regularization_selection.json

## Week 9 — Tree-based Models & Model Evaluation

### Tree-based Models
- Random Forest Regressor
- Gradient Boosting Regressor

Both models were evaluated using the same preprocessing pipeline and cross-validation
strategy as the linear models.

### Performance Comparison
Tree-based models substantially outperformed the linear baseline:
- Ridge Regression: RMSE ≈ 0.195
- Random Forest: RMSE ≈ 0.143
- Gradient Boosting: RMSE ≈ 0.132

Artifact:
- week_9_model_family_comparison.csv

### Feature Importance
Feature importance analysis from the Gradient Boosting model shows that aggregated
size and quality-related features dominate predictions. Engineered features such as
TotalSF and TotalBath rank among the most influential, validating earlier design
choices.

Artifact:
- week_c_gbdt_feature_importance_top20.csv

## Design Decisions & Tradeoffs
- Linear models were prioritized early for interpretability and stability
- Regularization was used to manage high-dimensional feature space
- Tree-based models were introduced to capture nonlinear effects and interactions
- Kaggle leaderboard submission was intentionally deferred to focus on methodological
  comparison rather than competitive tuning

## Reproducibility
- Fixed random seeds
- Cross-validation for all evaluations
- Explicit preprocessing pipelines
- Saved artifacts for each modeling stage

## Tools
- Python
- pandas, NumPy
- scikit-learn
- Kaggle Notebook environment

## Notes
This project emphasizes clarity, reproducibility, and evaluation rigor over
leaderboard performance, and is intended as a reference workflow for tabular
regression problems.
