## Customer Churn Prediction — Project Summary

**Objective.** Build a ranking model that identifies customers most likely to churn so retention efforts can focus on the highest-risk group under a limited outreach budget.

**Data.** Kaggle — Telco Customer Churn: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Approach.** 
- Cleaned the dataset (normalized “No internet/phone service,” fixed `TotalCharges` blanks that occur when `tenure=0`, trimmed text, dropped `customerID` from features).
- Used a leak-proof scikit-learn pipeline (impute → scale → one-hot encode with `drop='first'`).
- Compared Logistic Regression, Random Forest, and a calibrated RF. Chose Logistic Regression for similar accuracy, better calibration, and clear drivers.

**Results.**
- Cross-validation: ROC-AUC 0.846 ± 0.005, PR-AUC 0.659 ± 0.013  
- Holdout set: ROC-AUC 0.835, PR-AUC 0.619 (baseline positive rate ≈ 0.266)
- Operating at Top-25% coverage (threshold ≈ 0.719): 352 customers flagged  
  Precision 0.631, Recall 0.594, F1 0.612

**Key signals (directional).**
- Higher churn risk: Month-to-month contract, Fiber optic service, Electronic check payments, higher TotalCharges  
- Lower churn risk: Longer tenure, Two-year contracts

**Business view (example scenario used in the notebook).**
Using $20 margin per month, 3 months retained if saved, $15 offer cost, and a 25% success rate, the Top-25% program on the holdout cohort yields ≈ $3,330 gross savings vs. $5,280 in offer cost (net ≈ –$1,950, or about –$1,386 per 1,000 customers). Levers to reach break-even: raise success to ~40%, reduce offer cost to ≤ $9.50, or increase expected months retained to ~4.8+.

**Deliverables.**
- `top25_contacts.csv` — ranked customers with churn probabilities and a Top-25% flag  
- `churn_lr_v2.pkl` — trained Logistic Regression pipeline  
- `threshold_top25.npy` — saved decision threshold for the chosen operating point
