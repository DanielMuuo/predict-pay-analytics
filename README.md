# Predict-Pay: AI Collection Prioritization Engine

## Live Demo
[Click here to view the Live Web App](https://predict-pay-analytics-hd8nw36wzwxxopdzhsgdun.streamlit.app/)
### Quick Start / Testing
Don't have a dataset? [Download this sample_data.csv](sample_data.csv) to test the engine immediately.

## Business Overview
In high-volume debt collection, the primary challenge is **resource allocation**. Calling every debtor is inefficient. This project provides a data-driven solution to identify "High-Propensity" debtors, allowing teams to focus efforts where recovery is most likely.

## Key Features
- **Predictive Scoring:** Uses an XGBoost Classifier to assign a 0-100% repayment probability to each account.
- **Handling Data Imbalance:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns effectively even if "paid" records are rare.
- **Dynamic Dashboard:** A Streamlit interface that allows managers to upload CSV files and get instant priority lists.
- **Automated Feature Importance:** Visualizes which factors (e.g., Loan Amount vs. Days Past Due) are driving repayment behavior.



## Tech Stack
- **Language:** Python 3.9+
- **Machine Learning:** XGBoost, Scikit-Learn
- **Data Handling:** Pandas, NumPy, Imbalanced-Learn
- **Deployment:** Streamlit Cloud

## Project Structure
```text
├── app.py              # Main Streamlit Web Application
├── requirements.txt    # Production Dependencies
├── LICENSE             # MIT License

└── README.md           # Project Documentation

