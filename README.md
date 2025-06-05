# insurance-claim-triage-ml

This project demonstrates a machine learning pipeline for predicting the approval status of synthetic health insurance claims and identifying which claims should be flagged for manual review.

## 🔍 Objective
Build a binary classification model to predict whether a synthetic insurance claim will be approved or denied, and triage uncertain cases for manual review.

## 📊 Dataset
- **Source**: [Enhanced Health Insurance Claims Dataset on Kaggle](https://www.kaggle.com/datasets/leandrenash/enhanced-health-insurance-claims-dataset)
- **Type**: Fully synthetic, non-identifiable healthcare claims data
To run this notebook, download the dataset manually from Kaggle and place the CSV in the /data folder as enhanced_health_insurance_claims.csv.

## 📁 Project Structure
```
insurance-claim-triage-ml/
├── data/                  # Dataset file (claims.csv)
├── notebooks/            # Jupyter notebook for EDA and modeling
│   └── 01_eda_model_triage.ipynb
├── src/                  # Helper functions (e.g., preprocessing)
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md             # Project overview
```

## 🧠 Features Used
- ClaimAmount
- DiagnosisCode
- ProcedureCode
- PatientAge
- PatientGender
- ProviderSpecialty
- PatientIncome
- PatientMaritalStatus
- PatientEmploymentStatus
- ProviderLocation
- ClaimType
- ClaimSubmissionMethod

*(Identifiers and claim dates are excluded from modeling)*

## 🔧 ML Pipeline
1. Preprocessing
2. Train/test split
3. Model training (Random Forest, Logistic Regression)
4. Prediction and probability estimation
5. **Triage Logic**:
   - If model prediction probability ∈ [0.4, 0.6], flag as `Needs Review`
   - Else, mark as confidently approved/denied

## 📈 Evaluation
- Accuracy, Precision, Recall, ROC-AUC
- Confusion matrix
- Feature importances

## ⚖️ Ethical Use
This project uses synthetic data and is for educational purposes only. It does not represent a deployable model and is not intended to guide real insurance or clinical decisions.

## 🚀 Future Enhancements
- Add Streamlit demo for triage decision interface
- Simulate multiple denial reasons (multi-class model)
- Incorporate SHAP for interpretability

---
*Maintained by [efall-dev](https://github.com/efall-dev)*