import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =====================================================
# Advanced Feature Engineering (Hardened for 7k Data)
# =====================================================

def clean_numeric(value):
    """Handles KES prefixes, commas, and malformed numeric strings safely."""
    if pd.isna(value): return 0.0
    if isinstance(value, (int, float)): return float(value)
    try:
        value_str = str(value).replace("KES", "").replace(",", "").strip()
        match = re.search(r'[-+]?\d*\.?\d+', value_str)
        return float(match.group(0)) if match else 0.0
    except: return 0.0

def engineer_features(df):
    # 1. Robust Cleaning for M-Shwari Columns
    df["principal_cleaned"] = df["PRINCIPAL AMOUNT"].apply(clean_numeric)
    df["balance_cleaned"] = df["BALANCE"].apply(clean_numeric)
    df["amt_paid_cleaned"] = df["AMOUNT PAID"].apply(clean_numeric) if "AMOUNT PAID" in df.columns else 0.0
    df["dpd_cleaned"] = pd.to_numeric(df["DPD"], errors="coerce").fillna(0)
    df["loan_count"] = pd.to_numeric(df["LOANS COUNTER"], errors="coerce").fillna(0)

    # 2. Behavioral Ratios (Daniel Muuo Logic)
    # Exposure: How much of the original loan is still outstanding
    df["exposure_ratio"] = df["balance_cleaned"] / (df["principal_cleaned"] + 1)
    # Intensity: Number of loans relative to delinquency duration
    df["loan_intensity"] = df["loan_count"] / (df["dpd_cleaned"] + 1)
    # Velocity: Speed of delinquency
    df["delinquency_velocity"] = df["dpd_cleaned"] / (df["loan_count"] + 1)
    # Log transform to normalize skewed DPD distributions
    df["log_dpd"] = np.log1p(df["dpd_cleaned"])
    
    # 3. Target Definition (The 'Will Pay' Label)
    # Using actual payment history if available, else balance reduction
    if "AMOUNT PAID" in df.columns:
        df["target"] = (df["amt_paid_cleaned"] > 0).astype(int)
    else:
        df["target"] = (df["balance_cleaned"] < df["principal_cleaned"]).astype(int)
        
    return df

# =====================================================
# UI Setup & Subtle Branding
# =====================================================

st.set_page_config(page_title="Predict-Pay AI Enterprise", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #004a99; color: white; font-weight: bold; border: none;
    }
    .footer { position: fixed; bottom: 0; width: 100%; text-align: center; color: #6c757d; font-size: 11px; padding: 10px; background: white; border-top: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True)

st.title("Predict-Pay: Enterprise Recovery Engine")
st.markdown("**Built by:** Daniel Muuo")

with st.expander("System Documentation & Data Requirements"):
    st.markdown("### 1. Required CSV Columns")
    st.markdown("""
    Ensure your CSV has these exact headers (Case Sensitive):
    
    | Column Name | Purpose |
    | :--- | :--- |
    | **CFID** | Unique Case ID |
    | **DEBTOR NAMES** | Full Name for Call Lists |
    | **PRINCIPAL AMOUNT** | Original loan value |
    | **BALANCE** | Current outstanding amount |
    | **DPD** | Days Past Due |
    | **LOANS COUNTER** | Total previous loans |
    | **AMOUNT PAID** | (Optional) Past payment history |
    """)
    st.divider()
    st.markdown("### 2. Technical Methodology")
    st.markdown("""
    * **Calibration:** Isotonic (Large Data) or Sigmoid (Small Data).
    * **Oversampling:** SMOTE is used to balance payment/non-payment classes.
    * **Explainability:** SHAP TreeExplainer for feature attribution.
    """)

st.divider()

# =====================================================
# Step 2: Controls & Data Ingestion
# =====================================================

col_a, col_b = st.columns([2, 1])
with col_a:
    st.subheader("Data Ingestion")
    upload_file = st.file_uploader("Upload csv file", type=["csv"])
with col_b:
    st.subheader("Engine Tuning")
    test_size = st.slider("Validation Hold-out (%)", 10, 40, 20)

# =====================================================
# Step 3: Core Logic & Execution
# =====================================================

# Model Features
features = ["principal_cleaned", "balance_cleaned", "dpd_cleaned", "loan_count", 
            "exposure_ratio", "loan_intensity", "delinquency_velocity", "log_dpd"]

if upload_file is not None:
    df_raw = pd.read_csv(upload_file)
    
    # Validation
    required = ["CFID", "DEBTOR NAMES", "PRINCIPAL AMOUNT", "DPD", "LOANS COUNTER", "BALANCE"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        st.error(f"❌ Missing Columns: {', '.join(missing)}"); st.stop()
        
    df = engineer_features(df_raw)
    
    # Select calibration method based on 7k data size
    calib_method = 'isotonic' if len(df) > 1000 else 'sigmoid'
    
    st.success(f"Data Ingested: {len(df)} records. Logic: {calib_method.capitalize()} Calibration.")
else:
    st.info("Please upload your CSV file to begin.")
    st.stop()

if st.button("Run Recovery Prediction Engine"):
    # Target Check
    X, y = df[features], df["target"]
    if len(np.unique(y)) < 2:
        st.error("❌ The data has no variation in 'Paid' vs 'Unpaid' statuses. Cannot train model."); st.stop()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, stratify=y, random_state=42)

    # 1. Pipeline for Validation
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric="logloss", random_state=42))
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    # 2. Calibrated Production Model
    base_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric="logloss", random_state=42)
    calibrated_model = CalibratedClassifierCV(base_model, method=calib_method, cv=cv)
    
    # SMOTE the training set
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    calibrated_model.fit(X_train_res, y_train_res)

    # =====================================================
    # Step 4: Enterprise Analytics Dashboard
    # =====================================================
    st.divider()
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        y_probs = calibrated_model.predict_proba(X_test)[:, 1]
        st.metric("Model Confidence (AUC)", f"{roc_auc_score(y_test, y_probs):.3f}")
        st.write(f"**CV Stability (Mean):** {cv_scores.mean():.3f}")
        
        # Drivers from the first calibrated estimator
        importance_vals = calibrated_model.calibrated_classifiers_[0].estimator.feature_importances_
        importance_df = pd.DataFrame({"Feature": features, "Weight": importance_vals}).sort_values("Weight", ascending=False)
        st.write("### Portfolio Risk Drivers")
        st.bar_chart(importance_df.set_index("Feature"))

    with res_col2:
        st.write("###Priority Recovery List")
        # Global Probability
        df["Recovery_Probability"] = (calibrated_model.predict_proba(df[features])[:, 1] * 100).round(2)
        
        # Rank by probability - filtering out those who have already paid
        ranking = df[df["target"] == 0][["CFID", "DEBTOR NAMES", "Recovery_Probability", "BALANCE"]].sort_values("Recovery_Probability", ascending=False)
        
        st.dataframe(ranking, use_container_width=True)
        st.download_button("Export Results", ranking.to_csv(index=False), "Recovery_List.csv")

    # SHAP Explainability
    st.divider()
    st.subheader("Case-Level Transparency (SHAP)")
    explainer = shap.TreeExplainer(calibrated_model.calibrated_classifiers_[0].estimator)
    shap_values = explainer(X_test)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(plt.gcf(), bbox_inches='tight'); plt.clf()

    # Model Persistence
    joblib.dump({
        "model_object": calibrated_model,
        "feature_columns": features,
        "metadata": {"author": "Daniel Muuo", "version": "3.2", "samples": len(df)}
    }, "predict_pay.pkl")
    st.success("Model Persisted: predict_pay_pkl")

st.markdown("<div class='footer'>Predict-Pay Enterprise v3.2 | Built by Daniel Muuo for M-Shwari Analysis</div>", unsafe_allow_html=True)
