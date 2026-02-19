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
# SAFE FEATURE ENGINEERING (Leakage-Free)
# =====================================================

def clean_numeric(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        value_str = str(value).replace("KES", "").replace(",", "").strip()
        match = re.search(r'[-+]?\d*\.?\d+', value_str)
        return float(match.group(0)) if match else 0.0
    except:
        return 0.0

def engineer_features(df):
    df["principal_cleaned"] = df["PRINCIPAL AMOUNT"].apply(clean_numeric)
    df["dpd_cleaned"] = pd.to_numeric(df["DPD"], errors="coerce").fillna(0)
    df["loan_count"] = pd.to_numeric(df["LOANS COUNTER"], errors="coerce").fillna(0)

    # Behavioral Features ONLY (No balance usage)
    df["loan_intensity"] = df["loan_count"] / (df["dpd_cleaned"] + 1)
    df["delinquency_velocity"] = df["dpd_cleaned"] / (df["loan_count"] + 1)
    df["log_dpd"] = np.log1p(df["dpd_cleaned"])

    # Target must be based ONLY on actual payment history
    if "AMOUNT PAID" not in df.columns:
        raise ValueError("AMOUNT PAID column required for modeling.")

    df["amt_paid_cleaned"] = df["AMOUNT PAID"].apply(clean_numeric)
    df["target"] = (df["amt_paid_cleaned"] > 0).astype(int)

    return df

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="Predict-Pay Recovery Engine ", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #004a99; color: white; font-weight: bold; border: none;
    }
    /* Expanding the file uploader and slider visual impact */
    .stFileUploader { padding: 20px; border: 2px dashed #004a99; border-radius: 10px; background: #ffffff; }
    .stSlider { padding: 20px; background: #ffffff; border-radius: 10px; box-shadow: 0px 2px 4px rgba(0,0,0,0.05); }
    
    .footer { position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; color: #6c757d; font-size: 11px; padding: 10px; background: white; border-top: 1px solid #dee2e6; z-index: 100; }
    </style>
    """, unsafe_allow_html=True)

st.title("Predict-Pay: Recovery Engine")
st.markdown("**Built by:** Daniel Muuo")

with st.expander("System Documentation & Data Requirements"):
    st.markdown("### 1. Required CSV Columns")
    st.markdown("""
    Ensure your CSV has these exact headers (Case Sensitive):
    
    | Column Name | Purpose |
    | :--- | :--- |
    | **CFID** | Unique Case ID |
    | **IDENTIFICATION** | National ID or Passport |
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

# Massive input area for better UX
col_file, col_tune = st.columns([1.5, 1])

with col_file:
    st.subheader("Data Ingestion")
    upload_file = st.file_uploader("Upload CSV Portfolio", type=["csv"], help="Maximum file size 200MB")

with col_tune:
    st.subheader("Engine Tuning")
    test_size = st.slider("Validation Hold-out (%)", 10, 40, 20, help="Higher % ensures more robust evaluation")

# =====================================================
# MODEL FEATURES (SAFE)
# =====================================================

features = [
    "principal_cleaned",
    "dpd_cleaned",
    "loan_count",
    "loan_intensity",
    "delinquency_velocity",
    "log_dpd"
]

if upload_file is not None:

    df_raw = pd.read_csv(upload_file)

    required = [
        "CFID",
        "DEBTOR NAMES",
        "PRINCIPAL AMOUNT",
        "DPD",
        "LOANS COUNTER",
        "AMOUNT PAID"
    ]

    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    df = engineer_features(df_raw)

    calib_method = 'isotonic' if len(df) > 1000 else 'sigmoid'
    st.success(f"Data Loaded: {len(df)} rows | Calibration: {calib_method.capitalize()}")

else:
    st.stop()

# =====================================================
# TRAINING
# =====================================================

if st.button("Run Recovery Engine"):

    X, y = df[features], df["target"]

    if len(np.unique(y)) < 2:
        st.error("No class variation detected.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size/100,
        stratify=y,
        random_state=42
    )

    # CV PIPELINE (Leakage-Free)
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    # FINAL CALIBRATED MODEL
    base_model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="logloss",
        random_state=42
    )

    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=calib_method,
        cv=cv
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    calibrated_model.fit(X_train_res, y_train_res)

    # =====================================================
    # RESULTS
    # =====================================================

    y_probs = calibrated_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)

    st.metric("Model AUC", f"{auc:.3f}")
    st.write(f"CV Mean AUC: {cv_scores.mean():.3f}")

    # FEATURE IMPORTANCE
    importance_vals = calibrated_model.calibrated_classifiers_[0].estimator.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Weight": importance_vals
    }).sort_values("Weight", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

    # =====================================================
    # PRIORITY LIST
    # =====================================================

    df["Recovery_Probability"] = (
        calibrated_model.predict_proba(df[features])[:, 1] * 100
    ).round(2)

    ranking = df[df["target"] == 0][
        ["CFID", "DEBTOR NAMES", "IDENTIFICATION", "Recovery_Probability"]
    ].sort_values("Recovery_Probability", ascending=False)

    st.dataframe(ranking, use_container_width=True)
    st.download_button(
        "Download Recovery List",
        ranking.to_csv(index=False),
        "Recovery_List.csv"
    )

    # =====================================================
    # SHAP (Safe Sampling)
    # =====================================================

    st.subheader("SHAP Explainability")

    sample_X = X_test.sample(min(300, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(
        calibrated_model.calibrated_classifiers_[0].estimator
    )

    shap_values = explainer(sample_X)

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    # =====================================================
    # SAVE MODEL
    # =====================================================

    joblib.dump({
        "model": calibrated_model,
        "features": features,
        "version": "4.0-leakage-safe",
        "samples": len(df)
    }, "predict_pay.pkl")

    st.success("Predict-Pay model saved successfully.")

st.markdown("<div class='footer'>Predict-Pay Engine | Built by Daniel Muuo</div>", unsafe_allow_html=True)
