"""
Loan Default Prediction Inference Engine
Supports all 4 required classification models:
1. Logistic Regression (Tuned Best Model)
2. K-Nearest Neighbors (KNN)
3. Naive Bayes (GaussianNB)
4. Decision Tree Classifier
Plus:
- Scratch Logistic Regression (pure NumPy)
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from scratch_logistic_regression import ScratchLogisticRegression
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd


class LoanDefaultPredictor:
    def __init__(self, model_name: str = "logistic_regression", artifacts_dir: str = None):
        if artifacts_dir is None:
            # Check root artifacts or ml_pipeline/artifacts
            candidates = [
                "/app/applet/artifacts",
                "/artifacts",
                os.path.join(os.path.dirname(__file__), "..", "artifacts"),
                os.path.join(os.path.dirname(__file__), "artifacts"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    artifacts_dir = os.path.abspath(c)
                    break

        self.artifacts_dir = artifacts_dir or "/app/applet/artifacts"
        self.model_name = model_name

        # Resolve paths
        prep_candidates = [
            os.path.join(self.artifacts_dir, "preprocessing", "preprocessor.joblib"),
            os.path.join(self.artifacts_dir, "preprocessing", "preprocessor.pkl"),
            os.path.join(self.artifacts_dir, "preprocessor.joblib"),
            "/app/applet/artifacts/preprocessing/preprocessor.joblib",
        ]
        self.preprocessor_path = next((p for p in prep_candidates if os.path.exists(p)), None)

        model_candidates = [
            os.path.join(self.artifacts_dir, "models", f"{model_name}.joblib"),
            os.path.join(self.artifacts_dir, "models", f"{model_name}.pkl"),
            os.path.join(self.artifacts_dir, f"{model_name}.joblib"),
            f"/app/applet/artifacts/models/{model_name}.joblib",
            f"/app/applet/artifacts/models/{model_name}.pkl",
        ]
        self.model_path = next((p for p in model_candidates if os.path.exists(p)), None)

        if not self.preprocessor_path or not self.model_path:
            raise FileNotFoundError(
                f"Artifacts not found for {model_name} in {self.artifacts_dir}. (preprocessor: {self.preprocessor_path}, model: {self.model_path})"
            )

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)

        # Population baseline means
        self.num_baseline = {
            "Age": 43.5, "Income": 82500, "LoanAmount": 127500, "CreditScore": 574,
            "MonthsEmployed": 59.5, "NumCreditLines": 2.5, "InterestRate": 13.5,
            "LoanTerm": 36, "DTIRatio": 0.50
        }

    def predict_single(self, input_dict: dict) -> dict:
        """
        Run inference on a single applicant's raw data dictionary.
        """
        df_single = pd.DataFrame([input_dict])

        # Ensure correct numeric types
        num_fields = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
        for field in num_fields:
            if field in df_single:
                df_single[field] = pd.to_numeric(df_single[field], errors="coerce")

        # Transform using preprocessor
        X_trans = self.preprocessor.transform(df_single)

        # Predict probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_trans)[0]
            default_prob = float(probs[1])
        else:
            default_prob = float(self.model.predict(X_trans)[0])

        risk_score = round(default_prob * 100, 1)

        # Risk Tier Classification
        if default_prob < 0.15:
            risk_tier = "Low"
            recommendation = "Low default probability. Applicant profile demonstrates strong financial fundamentals."
        elif default_prob < 0.35:
            risk_tier = "Moderate"
            recommendation = "Standard risk profile. Standard underwriting covenants and routine verification recommended."
        elif default_prob < 0.55:
            risk_tier = "Elevated"
            recommendation = "Elevated default risk. Enhanced documentation, co-signer verification, or lower LTV suggested."
        else:
            risk_tier = "High"
            recommendation = "High risk of default detected. Manual committee review or credit restructuring strongly advised."

        # Decision threshold (0.50 standard, or 0.30 for high sensitivity)
        prediction = 1 if default_prob >= 0.50 else 0

        # Feature contributions relative to baseline
        feature_contributions = []
        for feat in num_fields:
            val = float(input_dict.get(feat, self.num_baseline[feat]))
            base = self.num_baseline[feat]
            if feat in ["InterestRate", "LoanAmount", "DTIRatio", "NumCreditLines"]:
                impact = (val - base) / base * 0.20
            else:
                impact = -(val - base) / base * 0.20

            direction = "Risk Increasing" if impact > 0 else "Risk Mitigating"
            feature_contributions.append({
                "feature": feat,
                "value": val,
                "impact": round(impact, 4),
                "direction": direction
            })

        feature_contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return {
            "prediction": prediction,
            "prediction_label": "Default" if prediction == 1 else "No Default",
            "default_probability": round(default_prob, 4),
            "probability_percent": round(default_prob * 100, 2),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "recommendation": recommendation,
            "model_used": self.model_name,
            "feature_contributions": feature_contributions[:6]
        }

    def predict_batch(self, records: list) -> list:
        return [self.predict_single(r) for r in records]


if __name__ == "__main__":
    predictor = LoanDefaultPredictor("logistic_regression")
    sample_borrower = {
        "Age": 56,
        "Income": 85994,
        "LoanAmount": 50587,
        "CreditScore": 520,
        "MonthsEmployed": 80,
        "NumCreditLines": 4,
        "InterestRate": 15.23,
        "LoanTerm": 36,
        "DTIRatio": 0.44,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Divorced",
        "HasMortgage": "Yes",
        "HasDependents": "Yes",
        "LoanPurpose": "Other",
        "HasCoSigner": "Yes"
    }
    res = predictor.predict_single(sample_borrower)
    print("Sample Prediction Result using Logistic Regression:")
    print(json.dumps(res, indent=2))
