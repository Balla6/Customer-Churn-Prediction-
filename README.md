**Customer Churn Prediction — Executive Summary**

- **Business goal:** Rank customers by churn risk so marketing can focus limited outreach on the highest-risk segment.

- **Data source:** Kaggle — [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

- **Data & prep:** Normalized “No internet/phone service”, coerced numeric `TotalCharges` (blanks only where `tenure=0`), dropped `customerID`. End-to-end scikit-learn **Pipeline** (impute → scale → OHE with `drop='first'`) to prevent leakage and keep training reproducible.

- **Models tested:** Logistic Regression (class-weighted), Random Forest, Calibrated RF.  
  **Chosen:** **Logistic Regression** — similar accuracy, better calibration, and interpretable drivers.

- **Validation:**  
  - **CV:** ROC-AUC **0.846±0.005**, PR-AUC **0.659±0.013**  
  - **Holdout:** ROC-AUC **0.835**, PR-AUC **0.619**

- **Operating point (Top-25% contacted):** threshold **0.719**, **352** flagged → Precision **0.631**, Recall **0.594**, F1 **0.612**.

- **Lift:** Top 10% of customers capture **≈26%** of churners (**≈2.6×** lift vs random).

- **Key drivers (LogReg):**  
  ↑ **Month-to-month**, **Fiber optic**, **Electronic check**, higher **TotalCharges**  
  ↓ **Longer tenure**, **Two-year contracts** (and, to a lesser extent, One-year)

- **ROI (example assumptions: $20 margin/mo, 3 months saved, $15 offer, 25% success):**  
  Test cohort **1,407** customers → **222** true churners flagged → expected saves **55.5**  
  **Gross savings:** $3,330 **Offer cost:** $5,280 **Net impact:** **–$1,950** (≈ **–$1,386 per 1,000 customers**)

- **Break-even levers (at 25% coverage):**  
  Raise success rate to **~40%**, **or** cut offer cost to **≤$9.50**, **or** keep a saved customer **≥4.8 months**.

- **Action plan:** Target **Month-to-month + high charge + Fiber** segments first; A/B test lower-cost bundles/credits; promote longer contracts & autopay; track conversion by segment and re-tune threshold.

- **Artifacts:** `top25_contacts.csv`, `churn_lr_v2.pkl`, `threshold_top25.npy`.

