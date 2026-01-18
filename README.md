
# Heart Disease Training Notebook

This bundle contains a Jupyter notebook that trains multiple classifiers on the
Heart Disease dataset (Kaggle: `fedesoriano/heart-failure-prediction`).

# Heart Disease Classification — End-to-End ML & Streamlit App

## Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease in a patient based on clinical and demographic features. The project also includes deployment of the trained models using a Streamlit web application to demonstrate model performance interactively.

## Dataset Description
- Source: Kaggle
- Type: Binary classification dataset
- Target Variable: HeartDisease
- 0 → No heart disease
- 1 → Presence of heart disease
- Number of Instances: 918
- Number of Features: 11 input features + 1 targe 
- Age – Age of the patient
- Sex – Gender (M/F)
- ChestPainType – Type of chest pain
- RestingBP – Resting blood pressure
- Cholesterol – Serum cholesterol
- FastingBS – Fasting blood sugar
- RestingECG – Resting ECG results
- MaxHR – Maximum heart rate achieved
- ExerciseAngina – Exercise induced angina
- Oldpeak – ST depression
- ST_Slope – Slope of the ST segment
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


Final Verdict
Based on the comparative evaluation of all six machine learning models on the Heart Disease dataset, ensemble‑based models and distance‑based classifiers demonstrated superior performance over simpler baseline models.
Among all the models, k‑Nearest Neighbors (kNN) achieved the highest overall accuracy and F1‑score, indicating its strong ability to capture local patterns in the data after proper feature scaling. Random Forest, an ensemble model, provided the most stable and consistent performance across all evaluation metrics, including MCC, highlighting its robustness and reduced susceptibility to overfitting.
Logistic Regression proved to be a reliable and interpretable baseline model with strong recall and ROC‑AUC, making it suitable for medical decision‑support scenarios where sensitivity is critical. Naive Bayes performed reasonably well despite its simplifying independence assumptions, showing robustness and efficiency.
On the other hand, Decision Tree showed the weakest performance due to overfitting, emphasizing the limitation of single‑tree models on complex datasets. XGBoost delivered competitive results with high precision and AUC, and its performance could be further improved through hyperparameter tuning.
Overall, the results indicate that ensemble models and well‑tuned algorithms are more suitable for heart disease prediction, as they effectively balance accuracy, robustness, and generalization capability.
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
