## Customer Churn Prediction — Plain-English Summary

**What I built**  
A model that ranks customers by their risk of churning so the team can focus retention offers on the people who are most likely to leave.

**Data**  
Kaggle’s Telco Customer Churn dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**How it works (in simple terms)**  
- Cleaned up the data (unified “No internet/phone service”, fixed `TotalCharges` blanks that occur when `tenure=0`, dropped `customerID`).  
- Trained several models inside a leak-proof scikit-learn Pipeline (impute → scale → one-hot encode with `drop='first'`).  
- Compared Logistic Regression, Random Forest, and a calibrated RF. Chose **Logistic Regression** because it performed as well as the others, was better calibrated, and gave clear, business-friendly drivers.

**Does it work? (results)**  
- Cross-validation: **ROC-AUC 0.846 ± 0.005**, **PR-AUC 0.659 ± 0.013**  
- Final holdout set: **ROC-AUC 0.835**, **PR-AUC 0.619**
- If we contact the **top 25%** highest-risk customers (threshold ≈ **0.719**):  
  **352** people are flagged → **Precision ~0.631**, **Recall ~0.594**, **F1 ~0.612**  
- Lift: the **top 10%** bucket captures **≈26%** of all churners (**~2.6×** better than random).

**What’s driving churn (directional)**  
- Higher risk: **Month-to-month contracts**, **Fiber optic**, **Electronic check**, higher **TotalCharges**  
- Lower risk: **Longer tenure**, **Two-year contracts** (one-year helps too, but less)

**Business impact (example math from the notebook)**  
Using simple, editable assumptions: $20 margin per month, 3 months saved if retained, $15 per offer, and 25% success rate.  
- On the test cohort (1,407 customers), that yields **~$3,330** gross savings, **~$5,280** in offer cost → **~$-1,950** net (≈ **$-1,386 per 1,000 customers**).  
- Break-even levers:
  - Raise success to **~40%**, **or**
  - Reduce offer cost to **≤ $9.50**, **or**
  - Increase expected months saved to **≈ 4.8+**.

**What to do next**  
- Prioritize **Month-to-month + high-charge + Fiber** segments.  
- A/B test lower-cost incentives or longer-term value offers.  
- Promote **longer contracts** and **autopay**.  
- Monitor conversion by segment and re-tune the contact threshold.

**Artifacts you can reuse**  
- `top25_contacts.csv` — the top 25% customers to contact (with scores)  
- `churn_lr_v2.pkl` — the trained Logistic Regression pipeline  
- `threshold_top25.npy` — the saved decision threshold for the 25% operating point

