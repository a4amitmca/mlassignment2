
# Heart Disease Classification — Streamlit App

> End-to-end ML mini-project for classification with a Streamlit UI.

## Problem Statement
Predict whether a person has heart disease from clinical attributes using supervised learning (binary classification).

## Dataset Description
- Use the public *Heart Failure Prediction* dataset (Kaggle). Place `heart.csv` at project root.  
- Target column: `HeartDisease` (0/1).  
- Example feature columns: `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`.

## Models Used
- Logistic Regression
- Decision Tree Classifier
- k-Nearest Neighbors (kNN)
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)
- XGBoost (Ensemble)

### Metrics Comparison Table (auto-generated after training)
<!--METRICS_TABLE_START-->
<!-- Table will be injected here -->
<!--METRICS_TABLE_END-->

## How to run locally
```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate

# 2) Install deps
pip install -r requirements.txt

# 3) Place training data as heart.csv in the project root

# 4) Launch app
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this repo to GitHub (include `app.py`, `requirements.txt`, and `heart.csv`).
2. Go to https://streamlit.io/cloud, sign in with GitHub.
3. Click **New app** > pick your repo > branch (main) > file path `app.py` > **Deploy**.

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
