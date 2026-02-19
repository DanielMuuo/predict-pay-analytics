import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import base64

# -- PAGE CONFIG --
st.set_page_config(page_title="Predict-Pay AI", layout="wide")

# Custom CSS for a sleek look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

## Header Section
st.title("📈 Predict-Pay: Smart Collections Engine")
st.write("Targeted Debt Recovery using XGBoost Machine Learning.")

---

## Sidebar: Data Source & Configuration
st.sidebar.header("Step 1: Data Input")
upload_file = st.sidebar.file_uploader("Upload your debtor CSV", type=["csv"])

st.sidebar.header("Step 2: Model Tuning")
test_size = st.sidebar.slider("Validation Split (%)", 10, 50, 20)

## Main Logic
if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("### Data Preview", df.head(5))
else:
    st.warning("👈 Upload a CSV to start. Using simulated data for demonstration below.")
    # Create sophisticated dummy data
    df = pd.DataFrame({
        'loan_amount': np.random.randint(5000, 100000, 200),
        'days_past_due': np.random.randint(1, 180, 200),
        'credit_score': np.random.randint(300, 850, 200),
        'last_payment_ratio': np.random.uniform(0, 1, 200),
        'will_pay': np.random.choice([0, 1], 200, p=[0.7, 0.3]) # Realistic imbalance
    })

if st.button("🚀 Execute Prediction Engine"):
    # 1. Preprocessing
    X = df.drop('will_pay', axis=1) if 'will_pay' in df.columns else df
    y = df['will_pay'] if 'will_pay' in df.columns else None
    
    if y is not None:
        # 2. SMOTE for Imbalanced Classes
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size/100)
        
        # 3. Training
        model = XGBClassifier(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        # 4. Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{model.score(X_test, y_test):.2%}")
            # Feature Importance
            importance = pd.DataFrame({'Driver': X.columns, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=False)
            st.write("### Decision Drivers")
            st.bar_chart(importance.set_index('Driver'))
            
        with col2:
            st.write("### Collection Priority List")
            # Predict probability
            probs = model.predict_proba(X)[:, 1]
            df['Recovery_Probability'] = (probs * 100).round(2)
            st.dataframe(df[['loan_amount', 'days_past_due', 'Recovery_Probability']].sort_values('Recovery_Probability', ascending=False))
            
            # Download link
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prioritized_collections.csv">📥 Download Priority List</a>'
            st.markdown(href, unsafe_allow_html=True)

st.divider()
st.caption("Developed by Daniel Muuo | Fintech Operations & Data Analyst")