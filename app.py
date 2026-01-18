
# app.py
# Streamlit app for Heart Disease Classification — aligns with ML Assignment 2
# Author: AMIT KUMAR . (generated helper)
#
# Features (per assignment):
#  - CSV upload (test data) ✅
#  - Train & evaluate 6 models on the same dataset ✅
#  - Show metrics: Accuracy, AUC, Precision, Recall, F1, MCC ✅
#  - Confusion matrix & classification report ✅
#  - Model selection dropdown ✅
#  - Saves trained pipelines in ./models/*.pkl and metrics in ./models/metrics.csv ✅
#
# Notes:
#  - Put training dataset as 'heart.csv' in the project root (Kaggle: fedesoriano/heart-failure-prediction).
#  - Uploaded CSV in the sidebar is treated as TEST data (may or may not include target column).

import os
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False
    XGBClassifier = None  # type: ignore

st.set_page_config(page_title="Heart Disease Classification", layout="wide")

# ------------------ Constants ------------------
TARGET_COL = "HeartDisease"
TRAIN_PATH = "heart.csv"  # expected to be present in repo
MODELS_DIR = "models"
RANDOM_STATE = 42

# ------------------ Helpers ------------------

def infer_column_types(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(df: pd.DataFrame):
    num_cols, cat_cols = infer_column_types(df)
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        # older scikit-learn
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', ohe, cat_cols),
        ],
        remainder='drop'
    )
    return pre, num_cols, cat_cols


def get_model_zoo():
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'kNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes (Gaussian)': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    if XGB_OK:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
        )
    return models


def evaluate_all_models(df: pd.DataFrame):
    os.makedirs(MODELS_DIR, exist_ok=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pre, _, _ = build_preprocessor(df)

    records = []
    reports = {}

    for name, model in get_model_zoo().items():
        pipe = Pipeline(steps=[('pre', pre), ('clf', model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        # Probabilities for AUC
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps['clf'], 'decision_function'):
            y_prob = pipe.decision_function(X_test)
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float('nan')

        records.append({
            'ML Model Name': name,
            'Accuracy': acc,
            'AUC': auc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'MCC': mcc,
        })

        # Save pipeline
        safe = name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
        dump(pipe, os.path.join(MODELS_DIR, f"{safe}.pkl"))

        # Store reports for quick viewing
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        reports[name] = {'cm': cm, 'cr': cr}

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(os.path.join(MODELS_DIR, 'metrics.csv'), index=False)
    return metrics_df.sort_values('Accuracy', ascending=False), reports


def fmt_val(v):
    if v is None:
        return '-'
    try:
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            return f"{float(v):.4f}"
        return str(v)
    except Exception:
        return str(v)


def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)


def apply_model_to_df(model_name: str, df: pd.DataFrame):
    """Load trained pipeline and apply to df. If target present, also compute metrics."""
    safe = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    model_path = os.path.join(MODELS_DIR, f"{safe}.pkl")
    if not os.path.exists(model_path):
        st.error(f"Trained model not found: {model_path}. Please train models first.")
        return None
    pipe = load(model_path)

    has_target = TARGET_COL in df.columns
    X = df.drop(columns=[TARGET_COL]) if has_target else df.copy()

    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None

    result_df = df.copy()
    result_df['Prediction'] = preds
    if proba is not None:
        result_df['Probability'] = proba

    if has_target:
        y_true = df[TARGET_COL]
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        mcc = matthews_corrcoef(y_true, preds)
        auc = roc_auc_score(y_true, proba) if proba is not None else float('nan')
        cm = confusion_matrix(y_true, preds)
        cr = classification_report(y_true, preds)
        metrics = {
            'Accuracy': acc, 'AUC': auc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'MCC': mcc
        }
        return result_df, metrics, cm, cr

    return result_df, None, None, None


# ------------------ UI ------------------
st.title("Heart Disease Classification — Streamlit App")

with st.sidebar:
    st.header("Upload Test Data (CSV)")
    up_file = st.file_uploader("Upload CSV (same schema as training). Target column optional.", type=["csv"]) 
    st.caption("Tip: On Streamlit free tier, prefer uploading only TEST data, as requested.")

    st.markdown("---")
    st.subheader("Model Selection")
    available_models = list(get_model_zoo().keys())
    if not XGB_OK:
        st.info("xgboost not available in this environment — will skip it unless installed.")
    model_choice = st.selectbox("Choose a model for prediction", available_models)

    st.markdown("---")
    st.subheader("Training Dataset")
    st.write("By default, the app expects **heart.csv** in project root for training.")
    st.write("If missing, you can upload a full dataset (with target) below to train.")
    train_upload = st.file_uploader("(Optional) Upload FULL training CSV (must include 'HeartDisease')", type=["csv"], key="train_uploader")


# Tabs for workflow
train_tab, predict_tab, about_tab = st.tabs(["🔧 Train & Evaluate", "📝 Predict", "ℹ️ About"])  # wrench, memo

with train_tab:
    st.subheader("1) Load dataset")
    train_df = None
    if train_upload is not None:
        train_df = pd.read_csv(train_upload)
        st.success(f"Loaded uploaded training data: {train_df.shape}")
    elif os.path.exists(TRAIN_PATH):
        train_df = pd.read_csv(TRAIN_PATH)
        st.success(f"Loaded {TRAIN_PATH}: {train_df.shape}")
    else:
        st.warning("No training data found. Upload a full dataset with target in the sidebar to proceed.")

    if train_df is not None:
        st.dataframe(train_df.head(), use_container_width=True)
        st.caption(f"Columns: {list(train_df.columns)}")

        st.subheader("2) Train 6 models and evaluate")
        if st.button("Train & Evaluate", type="primary"):
            with st.spinner("Training models... (this may take ~10–20s) "):
                metrics_df, reports = evaluate_all_models(train_df)
            st.success("Training complete. Models saved in ./models/")
            st.dataframe(metrics_df.reset_index(drop=True), use_container_width=True)

            # Download metrics
            buf = io.StringIO()
            metrics_df.to_csv(buf, index=False)
            st.download_button("Download metrics.csv", buf.getvalue(), file_name="metrics.csv", mime="text/csv")

            # Show confusion matrix & report for a chosen model from the table
            st.markdown("---")
            st.subheader("3) Confusion Matrix & Classification Report (validation split)")
            model_for_report = st.selectbox("Select model to inspect", metrics_df['ML Model Name'].tolist(), key="report_selector")
            rep = reports.get(model_for_report)
            if rep:
                plot_confusion_matrix(rep['cm'])
                st.text("Classification Report:" + rep['cr'])

with predict_tab:
    st.subheader("Use a trained model to predict on your TEST CSV")
    if up_file is None:
        st.info("Upload a CSV in the sidebar to run predictions here.")
    else:
        test_df = pd.read_csv(up_file)
        st.write("Test data loaded:")
        st.dataframe(test_df.head(), use_container_width=True)

        result = apply_model_to_df(model_choice, test_df)
        if result is not None:
            pred_df, metrics, cm, cr = result
            st.markdown("### Predictions")
            st.dataframe(pred_df.head(20), use_container_width=True)

            # Allow download of predictions
            out_buf = io.StringIO()
            pred_df.to_csv(out_buf, index=False)
            st.download_button("Download predictions.csv", out_buf.getvalue(), file_name="predictions.csv", mime="text/csv")

            # If ground truth present, show evaluation
            if metrics is not None:
                st.markdown("### Evaluation on your uploaded data")
                nice = {k: fmt_val(v) for k, v in metrics.items()}
                st.json(nice)
                if cm is not None:
                    plot_confusion_matrix(cm)
                if cr is not None:
                    st.text("Classification Report:" + cr)
        else:
            st.warning("Could not run predictions. Make sure models are trained first.")

with about_tab:
    st.markdown(
        """
        **About this app**  
        - Trains and evaluates six classifiers on the same dataset: Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes, Random Forest, and XGBoost (if installed).  
        - Metrics reported: Accuracy, ROC-AUC, Precision, Recall, F1, and MCC.  
        - Upload only **test** CSV in the sidebar (target optional). If target column is present, the app computes metrics as well.  
        - The app expects a training file named **heart.csv** at the project root or you can upload a full dataset (with the 'HeartDisease' target) in the sidebar.
        
        **Data schema (typical for Heart Disease dataset):**  
        Columns include numeric and categorical features such as: `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease`.
        
        **Reproducibility:** Random seed is fixed to 42 for splits and models where applicable.
        """
    )
