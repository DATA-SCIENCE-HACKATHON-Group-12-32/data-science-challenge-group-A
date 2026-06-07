# Seasonal Flu Vaccine Prediction — Data Science Challenge

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-006400?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.52.0-9B59B6?style=for-the-badge)
![pandas](https://img.shields.io/badge/pandas-2.2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.0.2-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.0-11557C?style=for-the-badge&logo=python&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-0.13.2-4C72B0?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

> Group submission for the **7PAM2015 Research Methods in Data Science** hackathon at the University of Hertfordshire. We were given an anonymised US public-health survey and asked to predict, for each respondent, whether they had received the seasonal flu vaccine.

🔗 **[Google Colab](https://colab.research.google.com/drive/1N5NdY9TZHqCb9ES8sUKtLZ6i9eyxhKhj)**

---

## Table of contents

- [Problem](#problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Repository structure](#repository-structure)
- [Results](#results)
- [Tech stack](#tech-stack)
- [Team](#team)

---

## Problem

Binary classification — predict the value of `seasonal_vaccine` (0 = did not vaccinate, 1 = vaccinated) for each respondent in a held-out test set, using their demographic, behavioural, and attitudinal survey responses. Submissions are scored by the marking team against a hidden target column using F1 score.

## Dataset

- **Training set:** 4,756 rows × 29 features + target
- **Test set:** 4,749 rows × 29 features (target hidden)
- **Class balance:** 54% vaccinated / 46% not — mildly imbalanced
- **Feature types:** 13 numeric (Likert-scale opinions, behavioural binaries, household counts) and 10 categorical (age group, education, race, employment, etc.)

## Approach

Our workflow is documented end-to-end in `notebooks/dsc_main_group_A.ipynb`:

1. **Exploratory data analysis** — target balance, missing values, numeric correlations with the target, and per-category vaccination rates for every categorical feature.
2. **Preprocessing experiments** — we benchmarked six techniques covered in the module (median imputation, MICE, KNN imputation, Yeo-Johnson, StandardScaler, PCA) on the same 5-fold cross-validation splits, and adopted only the techniques that produced measurable lift. Yeo-Johnson was the single winner.
3. **Five-model progression** — each model exists to test one specific idea:
   - M1: Logistic Regression (no scaling) — baseline
   - M2: Logistic Regression + StandardScaler — isolates the effect of scaling
   - M3: Random Forest (defaults) — tests a non-linear method
   - M4: Random Forest (CV-tuned via GridSearchCV, with Yeo-Johnson)
   - M5: XGBoost (with Yeo-Johnson) — modern gradient boosting
4. **Explainable AI** — three independent XAI techniques applied to the XGBoost model: SHAP summary plots (global feature importance), SHAP waterfall plots (single-prediction explanations for both confident and borderline cases), and permutation importance (model-agnostic sanity check).
5. **Submission** — five prediction files, ranked by their 5-fold cross-validated F1 score from best to worst.

## Repository structure

```
.
├── dataset/
│   ├── dataset_A_training.csv                       # 4,756 labelled training rows
│   └── dataset_A_testing.csv                        # 4,749 unlabelled test rows
├── models/
├── dsc_main_group_A.ipynb                           # the main project notebook
├── submissions/
│   ├── Challenge_submission-group_A_order_1.csv     # best F1  — Logistic Regression (unscaled)
│   ├── Challenge_submission-group_A_order_2.csv     # Logistic Regression (scaled)
│   ├── Challenge_submission-group_A_order_3.csv     # XGBoost
│   ├── Challenge_submission-group_A_order_4.csv     # Random Forest (tuned)
│   └── Challenge_submission-group_A_order_5.csv     # worst F1 — Random Forest (default)
└── README.md
```

## Results

Five-fold cross-validated metrics on the training set:

| Rank | Model | F1 | ROC-AUC | Accuracy |
|------|---|---|---|---|
| 1 | Logistic Regression (unscaled) | 0.7973 | 0.8470 | 0.7811 |
| 2 | Logistic Regression (scaled) | 0.7967 | 0.8471 | 0.7805 |
| 3 | XGBoost | 0.7909 | 0.8462 | 0.7740 |
| 4 | Random Forest (tuned) | 0.7875 | 0.8472 | 0.7702 |
| 5 | Random Forest (default) | 0.7822 | 0.8383 | 0.7664 |

**Key findings:**

- The top four models tie within statistical noise. Linear models are competitive with XGBoost on this dataset because the opinion features have monotonic relationships with the target.
- Yeo-Johnson is the only preprocessing technique that gives a real (small) lift on this data. StandardScaler, MICE, KNN, and PCA either did nothing or hurt.
- SHAP and permutation importance agree on the top predictors: opinion of vaccine effectiveness, perceived flu risk, doctor's recommendation, age group, and health-worker status.

## Tech stack

- **Language:** Python 3.10+
- **Data handling:** pandas, NumPy
- **Modelling:** scikit-learn (LogisticRegression, RandomForestClassifier, GridSearchCV, StratifiedKFold, ColumnTransformer, Pipeline, SimpleImputer, IterativeImputer, KNNImputer, OneHotEncoder, StandardScaler, PowerTransformer, PCA), XGBoost
- **Explainability:** SHAP (TreeExplainer, summary & waterfall plots), scikit-learn permutation_importance
- **Visualisation:** Matplotlib, seaborn
- **Environment:** Google Colab (primary), Jupyter (local)
- **Collaboration:** GitHub Organisation with shared Colab notebook

## Team

Group A — University of Hertfordshire, MSc Data Science with Advanced Research.
Module: 7PAM2015 Research Methods in Data Science.
