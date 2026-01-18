
# Heart Disease Training Notebook

This bundle contains a Jupyter notebook that trains multiple classifiers on the
Heart Disease dataset (Kaggle: `fedesoriano/heart-failure-prediction`).

# Heart Disease Classification — End-to-End ML & Streamlit App

## Problem Statement
Predict the presence of **heart disease** (binary: 0/1) using clinical and demographic features, and deploy an interactive web app for evaluation.

## Dataset Description
- Use the public *Heart Failure Prediction* dataset (Kaggle). Place `heart.csv` at project root.  
- Target column: `HeartDisease` (0/1).  
- Example feature columns: `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`.

## Files
- `Heart_Disease.ipynb` — main notebook
- `requirements.txt` — Python dependencies (xgboost optional)
- `data/README.txt` — where to place `heart.csv`
- `models/` — outputs (created when you run the notebook)

## Quick Start
1. Create and activate a Python 3.9+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download `heart.csv` from Kaggle and put it into `data/` folder.
4. Open the notebook and run all cells:
   ```bash
   jupyter lab  # or jupyter notebook / code . (VS Code)
   ```
5. Trained models and metrics will be written to `models/`.

## Notes
- If installing `xgboost` is problematic on your system, you can comment it out
  in `requirements.txt`. The notebook will still run and skip the XGBoost model.

## Models Used
- Logistic Regression
- Decision Tree Classifier
- k-Nearest Neighbors (kNN)
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)
- XGBoost (Ensemble)
<!--METRICS_TABLE_START-->

**ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC**
---|---|---|---|---|---|---
Logistic Regression | 0.8859 | 0.9299 | 0.8716 | 0.9314 | 0.9005 | 0.7694
Decision Tree | 0.7935 | 0.7910 | 0.8137 | 0.8137 | 0.8137 | 0.5820
kNN | 0.9130 | 0.9503 | 0.9135 | 0.9314 | 0.9223 | 0.8238
Naive Bayes (Gaussian) | 0.8859 | 0.9118 | 0.8932 | 0.9020 | 0.8976 | 0.7688
Random Forest | 0.9022 | 0.9331 | 0.8962 | 0.9314 | 0.9135 | 0.8018
XGBoost | 0.8696 | 0.9284 | 0.8980 | 0.8627 | 0.8800 | 0.7380


| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression performed consistently well with high recall and strong ROC-AUC. It generalized well after feature scaling and provided a stable linear baseline for heart disease prediction. |
| Decision Tree | Decision Tree showed comparatively lower accuracy due to overfitting. Although it captured non-linear relationships, it did not generalize as well as ensemble models. |
| kNN | kNN achieved high accuracy and F1-score after feature scaling. It effectively captured local data patterns but is sensitive to the choice of k and computationally expensive. |
| Naive Bayes | Naive Bayes delivered balanced precision and recall despite its feature independence assumption. It showed robust and reliable performance on this dataset. |
| Random Forest (Ensemble) | Random Forest achieved strong, consistent performance across all metrics. The ensemble approach reduced overfitting and captured complex feature interactions effectively. |
| XGBoost (Ensemble) | XGBoost demonstrated high AUC and precision with strong predictive power. Performance can be further improved with hyperparameter tuning. |

<!--METRICS_TABLE_END-->
## Repo Structure
```
project/
├── app.py
├── requirements.txt
├── heart.csv              # training data (add this)
├── models/                # saved pipelines & metrics appear here after training
└── README.md
```

## Notes
- On Streamlit free tier, upload **only TEST** data in the app (smaller files) to conserve resources.
- If `xgboost` fails to install on your platform, the app will still work with the other five models.
