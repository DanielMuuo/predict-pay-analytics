# Predict-Pay: AI Collection Prioritization Engine

## Live Demo
[Click here to view the Live Web App](https://predict-pay-analytics-hd8nw36wzwxxopdzhsgdun.streamlit.app/)
### Quick Start / Testing
Don't have a dataset? [Download this sample_data.csv](sample_data.csv) to test the above engine.

## Business Overview
In high-volume debt collection, the primary challenge is **resource allocation**. Calling every debtor is inefficient. This project provides a data-driven solution to identify "High-Propensity" debtors, allowing teams to focus efforts where recovery is most likely.

## Key Features
- **Predictive Scoring:** Uses an XGBoost Classifier to assign a 0-100% repayment probability to each account.
- **Handling Data Imbalance:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns effectively even if "paid" records are rare.
- **Dynamic Dashboard:** A Streamlit interface that allows managers to upload CSV files and get instant priority lists.
- **Automated Feature Importance:** Visualizes which factors (e.g., Loan Amount vs. Days Past Due) are driving repayment behavior.



## Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## Project Structure
```text
├── app.py              # Main Streamlit Web Application
├── requirements.txt    # Production Dependencies
├── LICENSE             # MIT License
├── sample_data.csv     # Sample dataset for testing (New)
└── README.md           # Project Documentation
