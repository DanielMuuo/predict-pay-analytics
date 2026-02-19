import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import base64
import re

# -- ADVANCED CLEANING FROM YOUR PREDICTOR SCRIPT --
def clean_currency(value):
    if pd.isna(value): return 0.0
    try:
        value_str = str(value).strip().replace('KES', '').replace(',', '').strip()
        match = re.search(r'[-+]?\d*\.?\d+', value_str)
        return float(match.group(0)) if match else 0.0
    except: return 0.0

# -- PAGE CONFIG --
st.set_page_config(page_title="Predict-Pay AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("Predict-Pay: Smart Collections Engine")
st.write("Targeted Debt Recovery using XGBoost Machine Learning logic from Daniel Muuo.")

# --- DATA REQUIREMENTS GUIDE (UPDATED) ---
with st.expander("ℹ️ View CSV Data Requirements"):
    st.markdown("""
    ### Required Columns (Exact Names Needed)
    Ensure your CSV has the following headers for the AI to process correctly:
    
    | Column Name | Data Type | Purpose |
    | :--- | :--- | :--- |
    | **CFID** | Number/ID | Unique Case File ID for system matching. |
    | **IDENTIFICATION** | Number | Debtor's National ID or Passport. |
    | **DEBTOR NAMES** | Text | Used for the final prioritized call list. |
    | **PRINCIPAL AMOUNT** | Currency | Original loan amount. |
    | **DPD** | Number | Days Past Due - a primary driver for prediction. |
    | **LOANS COUNTER** | Number | Total previous loans - measures customer loyalty. |
    | **BALANCE** | Currency | Current debt used to calculate repayment behavior. |
    
    *Tip: If your columns have different names, please rename them in Excel before uploading.*
    """)

st.divider()

## Sidebar
st.sidebar.header("Step 1: Data Input")
upload_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

st.sidebar.header("Step 2: Model Tuning")
test_size = st.sidebar.slider("Validation Split (%)", 10, 50, 20)

## Main Logic
if upload_file is not None:
    df = pd.read_csv(upload_file)
    
    # Check for required columns including new identifiers
    required = ['CFID', 'IDENTIFICATION', 'DEBTOR NAMES', 'PRINCIPAL AMOUNT', 'DPD', 'LOANS COUNTER', 'BALANCE']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}. Please check the guide above.")
        st.stop()

    # 1. Clean & Map Real Columns
    df['principal_cleaned'] = df['PRINCIPAL AMOUNT'].apply(clean_currency)
    df['dpd_cleaned'] = pd.to_numeric(df['DPD'], errors='coerce').fillna(0)
    df['loan_count'] = pd.to_numeric(df['LOANS COUNTER'], errors='coerce').fillna(0)
    df['balance_cleaned'] = df['BALANCE'].apply(clean_currency)
    
    # Target Variable logic
    df['will_pay'] = (df['balance_cleaned'] < df['principal_cleaned']).astype(int)
    
    features = ['principal_cleaned', 'dpd_cleaned', 'loan_count']
    # Preview with identifiers
    st.write("### Live Data Preview", df[['CFID', 'IDENTIFICATION', 'DEBTOR NAMES'] + features].head(5))
else:
    st.warning("⬅️ Upload your csv file to start.")
    # Fallback dummy data with identifiers
    df = pd.DataFrame({
        'CFID': [2951000 + i for i in range(200)],
        'IDENTIFICATION': [12345678 + i for i in range(200)],
        'DEBTOR NAMES': ['Sample Client ' + str(i) for i in range(200)],
        'principal_cleaned': np.random.randint(5000, 100000, 200),
        'dpd_cleaned': np.random.randint(1, 180, 200),
        'loan_count': np.random.randint(1, 10, 200),
        'will_pay': np.random.choice([0, 1], 200, p=[0.7, 0.3])
    })
    features = ['principal_cleaned', 'dpd_cleaned', 'loan_count']

if st.button("Execute Prediction Engine"):
    X = df[features]
    y = df['will_pay']
    
    if len(np.unique(y)) > 1:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size/100)
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{model.score(X_test, y_test):.2%}")
            importance = pd.DataFrame({'Driver': features, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=False)
            st.write("### Decision Drivers")
            st.bar_chart(importance.set_index('Driver'))
            
        with col2:
            st.write("### Collection Priority List")
            probs = model.predict_proba(X)[:, 1]
            df['Recovery_Probability'] = (probs * 100).round(2)
            
            # UPDATED: Included CFID and IDENTIFICATION in display and export
            display_cols = ['CFID', 'IDENTIFICATION', 'DEBTOR NAMES', 'principal_cleaned', 'Recovery_Probability']
            st.dataframe(df[display_cols].sort_values('Recovery_Probability', ascending=False))
            
            csv = df[display_cols].sort_values('Recovery_Probability', ascending=False).to_csv(index=False)
            st.download_button(
                label="Download Priority List for Field Team",
                data=csv,
                file_name="mshwari_priority_list.csv",
                mime="text/csv",
            )
    else:
        st.error("Not enough data variety in the uploaded file to train the AI.")
