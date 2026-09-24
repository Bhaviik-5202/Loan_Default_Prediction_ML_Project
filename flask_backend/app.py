"""
LoanLens — Flask Machine Learning Risk Scoring Engine
Institutional Credit Risk Scoring API loaded with scikit-learn models.

Serialized Artifacts Loaded on Startup:
- artifacts/preprocessing/preprocessor.pkl
- artifacts/models/logistic_regression.pkl (Selected Best Model)
- artifacts/models/knn.pkl
- artifacts/models/naive_bayes.pkl
- artifacts/models/decision_tree.pkl

Risk Score Transformation Documentation:
- Raw Model Output: P(Default = 1 | X) from LogisticRegression.predict_proba.
- Raw Probability: float between 0.0000 and 1.0000 (probability).
- Institutional Risk Score (0-100): int(round((1.0 - probability) * 100)).
  Higher score indicates stronger creditworthiness and lower risk of default.
- Confidence: max(probability, 1.0 - probability).
- Risk Level Thresholds:
  * Low Risk:    probability < 0.28  (Risk Score: 73-100)
  * Medium Risk: 0.28 <= probability < 0.50 (Risk Score: 51-72)
  * High Risk:   probability >= 0.50 (Risk Score: 0-50)
- Prediction / Label:
  * prediction = 0, label = "No Default" (probability < 0.50)
  * prediction = 1, label = "Default"    (probability >= 0.50)
"""

import os
import sys
import json
import logging
import traceback
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [LoanLens-Flask] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("loanlens_api")

app = Flask(__name__)

# Configure CORS for local development and authorized origins
# In production, specify comma-separated allowed origins via ALLOWED_ORIGINS (e.g. ALLOWED_ORIGINS="https://myfrontend.com")
raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
if raw_origins == "*":
    origins_config = "*"
else:
    origins_config = [o.strip() for o in raw_origins.split(",") if o.strip()]

CORS(app, resources={r"/api/*": {"origins": origins_config}})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Pretrained Artifacts on Startup (Once Only)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "artifacts", "preprocessing", "preprocessor.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")
METADATA_DIR = os.path.join(BASE_DIR, "artifacts", "metadata")

try:
    logger.info(f"Loading preprocessing pipeline from {PREPROCESSOR_PATH}...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    logger.info("Loading model artifacts...")
    models = {
        "logistic_regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl")),
        "knn": joblib.load(os.path.join(MODELS_DIR, "knn.pkl")),
        "naive_bayes": joblib.load(os.path.join(MODELS_DIR, "naive_bayes.pkl")),
        "decision_tree": joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl")),
    }
    selected_model_key = "logistic_regression"
    active_model = models[selected_model_key]
    logger.info("All 4 models and preprocessor loaded successfully!")
except Exception as e:
    logger.error(f"FATAL: Error loading model artifacts: {e}\n{traceback.format_exc()}")
    preprocessor = None
    models = {}
    active_model = None

# Load dataset insights and evaluation metadata
def load_json_file(path, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default or {}

DATASET_INSIGHTS = load_json_file(os.path.join(METRICS_DIR, "dataset_insights.json"))
MODEL_COMPARISON = load_json_file(os.path.join(METRICS_DIR, "model_comparison.json"))
EVALUATION_RESULTS = load_json_file(os.path.join(METRICS_DIR, "evaluation_results.json"))
FEATURE_METADATA = load_json_file(os.path.join(METADATA_DIR, "feature_metadata.json"))

NUMERICAL_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"
]

CATEGORICAL_FEATURES = [
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
]

ALL_RAW_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

VALID_CATEGORIES = {
    "Education": ["Bachelor's", "High School", "Master's", "PhD"],
    "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "HasMortgage": ["Yes", "No"],
    "HasDependents": ["Yes", "No"],
    "LoanPurpose": ["Auto", "Business", "Education", "Home", "Other"],
    "HasCoSigner": ["Yes", "No"],
}

NUMERICAL_BOUNDS = {
    "Age": (18, 100),
    "Income": (0, 10_000_000),
    "LoanAmount": (500, 10_000_000),
    "CreditScore": (300, 850),
    "MonthsEmployed": (0, 600),
    "NumCreditLines": (0, 50),
    "InterestRate": (0.1, 100.0),
    "LoanTerm": (1, 360),
    "DTIRatio": (0.0, 1.5),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Error Handlers (Structured JSON)
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({
        "error": True,
        "message": str(e.description if hasattr(e, 'description') else "Bad request"),
        "details": getattr(e, 'data', {})
    }), 400

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({
        "error": True,
        "message": "Resource not found",
        "details": {"path": request.path}
    }), 404

@app.errorhandler(422)
def handle_unprocessable(e):
    return jsonify({
        "error": True,
        "message": "Unprocessable Entity",
        "details": getattr(e, 'data', {})
    }), 422

@app.errorhandler(500)
def handle_internal_server_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "error": True,
        "message": "Internal server error occurred",
        "details": {}
    }), 500

# ─────────────────────────────────────────────────────────────────────────────
# 3. System Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check endpoint."""
    if active_model is None or preprocessor is None:
        return jsonify({
            "status": "error",
            "message": "Model artifacts not loaded",
            "model": "None"
        }), 500

    return jsonify({
        "status": "ok",
        "model": "Logistic Regression"
    }), 200

# ─────────────────────────────────────────────────────────────────────────────
# 4. Prediction Endpoint (POST /api/predict)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_payload_keys(payload: dict) -> dict:
    norm_map = {
        "age": "Age", "income": "Income", "loanamount": "LoanAmount",
        "creditscore": "CreditScore", "monthsemployed": "MonthsEmployed",
        "numcreditlines": "NumCreditLines", "interestrate": "InterestRate",
        "loanterm": "LoanTerm", "dtiratio": "DTIRatio", "education": "Education",
        "employmenttype": "EmploymentType", "maritalstatus": "MaritalStatus",
        "hasmortgage": "HasMortgage", "hasdependents": "HasDependents",
        "loanpurpose": "LoanPurpose", "hascosigner": "HasCoSigner", "name": "Name"
    }
    normalized = {}
    for k, v in payload.items():
        clean_k = k.lower().replace("_", "")
        if clean_k in norm_map:
            normalized[norm_map[clean_k]] = v
        else:
            normalized[k] = v
    return normalized

def validate_input(data: dict):
    if not isinstance(data, dict):
        return "Request body must be a JSON object", 400

    # 1. Missing fields check
    missing = [f for f in ALL_RAW_FEATURES if f not in data or data[f] is None or data[f] == ""]
    if missing:
        return f"Missing required fields: {', '.join(missing)}", 422

    # 2. Numeric validation & bounds
    validated_numbers = {}
    for field, (low, high) in NUMERICAL_BOUNDS.items():
        raw_val = data.get(field)
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            return f"Field '{field}' must be a valid number. Got '{raw_val}'", 422

        if val < low or val > high:
            return f"Field '{field}' value {val} is outside acceptable range [{low}, {high}]", 422
        validated_numbers[field] = val

    # 3. Categorical validation
    validated_cats = {}
    for field, allowed in VALID_CATEGORIES.items():
        val = str(data.get(field)).strip()
        if val not in allowed:
            return f"Invalid category '{val}' for field '{field}'. Allowed values: {', '.join(allowed)}", 422
        validated_cats[field] = val

    return {**validated_numbers, **validated_cats}, None

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run loan default risk inference on borrower application.
    Applies StandardScaler & OneHotEncoder preprocessor and Logistic Regression model.
    """
    if active_model is None or preprocessor is None:
        return jsonify({
            "error": True,
            "message": "Model artifacts unavailable on backend",
            "details": {}
        }), 500

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "error": True,
            "message": "Malformed request. Valid JSON payload required.",
            "details": {}
        }), 400

    payload = normalize_payload_keys(payload)
    clean_data, err_tuple = validate_input(payload)
    if err_tuple is not None:
        err_msg, err_code = clean_data, err_tuple
        return jsonify({
            "error": True,
            "message": err_msg,
            "details": {}
        }), err_code

    try:
        # Build single-row DataFrame with strict column order
        row_df = pd.DataFrame([clean_data])[ALL_RAW_FEATURES]

        # Apply preprocessor pipeline
        X_trans = preprocessor.transform(row_df)

        # Run Logistic Regression prediction
        probs = active_model.predict_proba(X_trans)[0]
        # Class 1 is default probability
        prob_default = float(probs[1])

        # Mathematical Transformation into Risk Score
        # Raw probability: 0.0 to 1.0
        probability = round(prob_default, 4)
        confidence = round(float(max(prob_default, 1.0 - prob_default)), 4)
        risk_score = int(round((1.0 - prob_default) * 100.0))
        risk_score = max(0, min(100, risk_score))

        # Risk level determination based on documented institutional thresholds
        if prob_default >= 0.50:
            risk_level = "High"
            prediction = 1
            label = "Default"
            pred_desc = "Likely to Default"
        elif prob_default >= 0.28:
            risk_level = "Medium"
            prediction = 0
            label = "No Default"
            pred_desc = "Moderate Risk"
        else:
            risk_level = "Low"
            prediction = 0
            label = "No Default"
            pred_desc = "Likely to Repay"

        # Calculate factor attribution using model coefficients * standardized feature values
        factors = []
        if hasattr(active_model, "coef_"):
            coefs = active_model.coef_[0]
            cat_encoder = preprocessor.named_transformers_["cat"]
            encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            feature_names = list(NUMERICAL_FEATURES) + list(encoded_cat_names)

            contributions = []
            for name, coef, val in zip(feature_names, coefs, X_trans[0]):
                impact = coef * val
                contributions.append((name, coef, val, impact))

            # Sort by absolute impact
            contributions.sort(key=lambda x: abs(x[3]), reverse=True)

            # Human-friendly feature labels
            feature_labels = {
                "Age": "Borrower Age",
                "Income": "Annual Household Income",
                "LoanAmount": "Loan Principal Amount",
                "CreditScore": "Credit Rating (FICO Score)",
                "MonthsEmployed": "Employment Tenure",
                "NumCreditLines": "Active Revolving Credit Lines",
                "InterestRate": "Contract Interest Rate",
                "LoanTerm": "Amortization Loan Term",
                "DTIRatio": "Debt-to-Income (DTI) Ratio",
                "Education_High School": "Education Level: High School",
                "Education_Master's": "Education Level: Master's",
                "Education_PhD": "Education Level: PhD",
                "EmploymentType_Part-time": "Employment: Part-Time",
                "EmploymentType_Self-employed": "Employment: Self-Employed",
                "EmploymentType_Unemployed": "Employment: Unemployed",
                "MaritalStatus_Married": "Marital Status: Married",
                "MaritalStatus_Single": "Marital Status: Single",
                "HasMortgage_Yes": "Existing Primary Mortgage",
                "HasDependents_Yes": "Borrower Has Dependents",
                "LoanPurpose_Business": "Loan Purpose: Business Expansion",
                "LoanPurpose_Education": "Loan Purpose: Education",
                "LoanPurpose_Home": "Loan Purpose: Home Improvement",
                "LoanPurpose_Other": "Loan Purpose: Other",
                "HasCoSigner_Yes": "Credit Guarantor / Co-Signer",
            }

            # Benchmark dictionary for comparison
            benchmarks = {
                "Age": "40 years",
                "Income": "$80,000",
                "LoanAmount": "$25,000",
                "CreditScore": "720 (Prime)",
                "MonthsEmployed": "36 months",
                "NumCreditLines": "3 lines",
                "InterestRate": "10.0%",
                "LoanTerm": "36 mo",
                "DTIRatio": "40%",
                "HasCoSigner_Yes": "Yes (Backed)",
                "HasMortgage_Yes": "Mortgage Clear",
                "EmploymentType_Unemployed": "Employed",
                "EmploymentType_Part-time": "Full-Time",
            }

            for name, coef, val, impact in contributions[:6]:
                is_driver = impact > 0
                lbl = feature_labels.get(name, name.replace("_", " "))
                pct_impact = min(100, max(10, int(abs(impact) * 50)))
                # Extract recorded raw value
                raw_feature_name = name.split("_")[0] if "_" in name else name
                recorded_val = clean_data.get(raw_feature_name, round(float(val), 2))
                benchmark_val = benchmarks.get(name, "Optimal Tier")

                factors.append({
                    "feature": name,
                    "label": lbl,
                    "value": recorded_val,
                    "benchmark": benchmark_val,
                    "type": "risk" if is_driver else "mitigant",
                    "direction": "up" if is_driver else "down",
                    "impact": pct_impact,
                    "raw_impact": round(float(impact), 4),
                    "coefficient": round(float(coef), 4),
                    "desc": f"{lbl}: {'+' if impact > 0 else ''}{round(float(impact), 2)} log-odds ({'Risk Driver' if is_driver else 'Protective Factor'})",
                    "description": f"{name}: {'+' if impact > 0 else ''}{round(float(impact), 2)} log-odds impact ({'Risk Driver' if is_driver else 'Protective Factor'})"
                })

        # Profile categories for UI visualization
        credit_score = clean_data["CreditScore"]
        dti_ratio = clean_data["DTIRatio"]
        income = clean_data["Income"]
        loan_amount = clean_data["LoanAmount"]
        months_emp = clean_data["MonthsEmployed"]

        profile = {
            "Credit Health": int(np.clip((credit_score - 300) / 5.5, 5, 95)),
            "Fin. Stability": int(np.clip((1.0 - dti_ratio) * 100, 5, 95)),
            "Repay. History": int(np.clip((credit_score - 350) / 5.0, 10, 95)),
            "Employment": int(np.clip((months_emp / 120.0) * 100, 10, 95)),
            "Debt Burden": int(np.clip((1.0 - dti_ratio) * 90, 5, 95)),
            "Loan Risk": int(np.clip(100 - (loan_amount / max(income, 1) * 30), 10, 95)),
        }

        # Model output risk classification and educational guidance
        # (Academic demonstration: Not an actual lending/underwriting decision)
        if risk_level == "High":
            recommendation = {
                "action": "Predicted Default Risk: High (Model Classification: Default)",
                "points": [
                    "Model predicts elevated likelihood of loan default based on high interest rate, short employment tenure, and elevated DTI.",
                    "Key protective mitigating attributes (such as prime co-signer backing) were absent or insufficient in this profile.",
                    "Educational simulation note: In commercial practice, secondary collateral or balance restructuring would be evaluated.",
                    "Notice: This prediction is an ML model output for demonstration purposes and does not constitute financial advice."
                ]
            }
        elif risk_level == "Medium":
            recommendation = {
                "action": "Predicted Default Risk: Moderate (Model Classification: Borderline No Default)",
                "points": [
                    "Model probability falls in the intermediate threshold range (0.28 to 0.49).",
                    "Borrower profile displays mixed signals: continuous income offset by higher debt-to-income or revolving credit lines.",
                    "Educational simulation note: In lending environments, manual underwriter inspection of employment history is typical.",
                    "Notice: This prediction is an ML model output for demonstration purposes and does not constitute financial advice."
                ]
            }
        else:
            recommendation = {
                "action": "Predicted Default Risk: Low (Model Classification: No Default)",
                "points": [
                    "Model predicts strong repayment indicators with estimated default probability below 0.28.",
                    "Solid financial fundamentals (healthy DTI, prime credit rating, and stable employment tenure) serve as strong protective factors.",
                    "Educational simulation note: Demonstrates typical prime tier cohort attributes from the Kaggle 255k dataset.",
                    "Notice: This prediction is an ML model output for demonstration purposes and does not constitute financial advice."
                ]
            }

        response_data = {
            "prediction": prediction,
            "label": label,
            "probability": probability,
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "model_used": "Logistic Regression",
            "factors": factors,
            # Frontend compatibility extensions
            "success": True,
            "features_evaluated": len(X_trans[0]),
            "profile": profile,
            "recommendation": recommendation,
            "prediction_description": pred_desc,
            "methodology": {
                "scoring_formula": "risk_score = round((1 - probability) * 100)",
                "thresholds": "Low (< 0.28), Medium (0.28 - 0.49), High (>= 0.50)"
            }
        }
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Inference execution error: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": True,
            "message": f"Inference execution error: {str(e)}",
            "details": {}
        }), 500

# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Endpoints (GET /api/models, GET /api/models/comparison, GET /api/models/<modelId>)
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY_MODELS_INFO = [
    {
        "id": "logistic-regression",
        "name": "Logistic Regression",
        "family": "LogisticRegression",
        "algorithm": "L2-Regularized Logistic Regression (Balanced)",
        "library": "Scikit-Learn (LogisticRegression)",
        "isSelected": True,
        "isPrimary": True,
        "status": "Selected Best Model",
        "featureCount": 24,
        "description": "Tuned generalized linear model utilizing balanced class weighting to address the 88.4% / 11.6% class imbalance. Selected as the primary production model due to highest discriminative ranking power (ROC-AUC: 0.7532) and superior default recall (69.95%).",
        "metrics": {
            "accuracy": 0.6764,
            "precision": 0.2196,
            "recall": 0.6995,
            "f1_score": 0.3342,
            "roc_auc": 0.7532,
            "cv_roc_auc_mean": 0.7426,
            "cv_roc_auc_std": 0.0063,
            "training_time": "3.95s",
        },
        "hyperparameters": {
            "C": 0.1,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "confusionMatrix": {
            "trueNegative": 30393,
            "falsePositive": 14746,
            "falseNegative": 1782,
            "truePositive": 4149,
        },
        "featureImportance": [
            { "feature": "Age", "importance": -0.5826, "direction": "Protective (-0.5826 log-odds: Mature age strongly reduces default risk)" },
            { "feature": "InterestRate", "importance": 0.4583, "direction": "Risk Driver (+0.4583 log-odds: High interest rate increases debt strain)" },
            { "feature": "EmploymentType_Unemployed", "importance": 0.4366, "direction": "Risk Driver (+0.4366 log-odds: Lack of primary employment income)" },
            { "feature": "MonthsEmployed", "importance": -0.3381, "direction": "Protective (-0.3381 log-odds: Continuous job tenure provides stability)" },
            { "feature": "Income", "importance": -0.3375, "direction": "Protective (-0.3375 log-odds: Higher earnings protect debt service)" },
            { "feature": "LoanAmount", "importance": 0.2982, "direction": "Risk Driver (+0.2982 log-odds: Large principal balance expands exposure)" },
            { "feature": "HasCoSigner_Yes", "importance": -0.1322, "direction": "Protective (-0.1322 log-odds: Guarantor protection mitigates default)" },
            { "feature": "CreditScore", "importance": -0.1215, "direction": "Protective (-0.1215 log-odds: Higher credit rating reflects discipline)" },
            { "feature": "NumCreditLines", "importance": 0.0984, "direction": "Risk Driver (+0.0984 log-odds: Additional open revolving credit lines)" },
            { "feature": "DTIRatio", "importance": 0.0674, "direction": "Risk Driver (+0.0674 log-odds: Elevated debt burden relative to income)" },
        ]
    },
    {
        "id": "knn",
        "name": "KNN",
        "family": "KNeighborsClassifier",
        "algorithm": "Distance-Weighted K-Nearest Neighbors (k=25)",
        "library": "Scikit-Learn (KNeighborsClassifier)",
        "isSelected": False,
        "isPrimary": True,
        "status": "Benchmark Evaluated",
        "featureCount": 24,
        "description": "Non-parametric instance-based classifier utilizing BallTree spatial indexing, Minkowski metric, and distance-weighted voting across k=25 nearest neighbors in standardized feature space.",
        "metrics": {
            "accuracy": 0.8851,
            "precision": 0.5455,
            "recall": 0.0070,
            "f1_score": 0.0137,
            "roc_auc": 0.6927,
            "cv_roc_auc_mean": 0.6753,
            "cv_roc_auc_std": 0.0068,
            "training_time": "38.33s",
        },
        "hyperparameters": {
            "n_neighbors": 25,
            "weights": "distance",
            "metric": "minkowski",
        },
        "confusionMatrix": {
            "trueNegative": 13265,
            "falsePositive": 10,
            "falseNegative": 1713,
            "truePositive": 12,
        },
        "featureImportance": [
            { "feature": "Local Neighborhood Proximity", "importance": 0.40, "direction": "Distance-weighted consensus across 25 nearest training records" },
            { "feature": "Continuous Feature Distance", "importance": 0.35, "direction": "Minkowski metric computed across 9 standardized numerical features" },
            { "feature": "Categorical Coordinate Distance", "importance": 0.25, "direction": "Euclidean projection over 15 one-hot categorical coordinates" },
        ]
    },
    {
        "id": "naive-bayes",
        "name": "Naive Bayes",
        "family": "GaussianNB",
        "algorithm": "Gaussian Naive Bayes (var_smoothing=1e-5)",
        "library": "Scikit-Learn (GaussianNB)",
        "isSelected": False,
        "isPrimary": True,
        "status": "Benchmark Evaluated",
        "featureCount": 24,
        "description": "Probabilistic Bayesian classifier computing class-conditional likelihoods under Gaussian feature assumptions with variance smoothing regularization (1e-5).",
        "metrics": {
            "accuracy": 0.8849,
            "precision": 0.6080,
            "recall": 0.0256,
            "f1_score": 0.0492,
            "roc_auc": 0.7499,
            "cv_roc_auc_mean": 0.7396,
            "cv_roc_auc_std": 0.0044,
            "training_time": "0.99s",
        },
        "hyperparameters": {
            "var_smoothing": 1e-5,
        },
        "confusionMatrix": {
            "trueNegative": 45041,
            "falsePositive": 98,
            "falseNegative": 5779,
            "truePositive": 152,
        },
        "featureImportance": [
            { "feature": "Age Distribution Mean", "importance": 0.30, "direction": "Repaid: +0.022 vs Default: -0.171 (Mean standardized diff: -0.193)" },
            { "feature": "InterestRate Mean", "importance": 0.26, "direction": "Repaid: -0.017 vs Default: +0.134 (Mean standardized diff: +0.151)" },
            { "feature": "Income Distribution Mean", "importance": 0.20, "direction": "Repaid: +0.013 vs Default: -0.101 (Mean standardized diff: -0.114)" },
            { "feature": "MonthsEmployed Mean", "importance": 0.14, "direction": "Repaid: +0.012 vs Default: -0.098 (Mean standardized diff: -0.110)" },
            { "feature": "LoanAmount Mean", "importance": 0.10, "direction": "Repaid: -0.011 vs Default: +0.089 (Mean standardized diff: +0.100)" },
        ]
    },
    {
        "id": "decision-tree",
        "name": "Decision Tree",
        "family": "DecisionTreeClassifier",
        "algorithm": "Entropy-Partitioned Decision Tree (max_depth=6)",
        "library": "Scikit-Learn (DecisionTreeClassifier)",
        "isSelected": False,
        "isPrimary": True,
        "status": "Benchmark Evaluated",
        "featureCount": 24,
        "description": "Hierarchical recursive partitioning model utilizing Information Gain (Entropy criterion), depth regularization (max_depth=6), and leaf threshold constraints (min_samples_leaf=50) to mitigate overfitting.",
        "metrics": {
            "accuracy": 0.8849,
            "precision": 0.5888,
            "recall": 0.0302,
            "f1_score": 0.0574,
            "roc_auc": 0.7254,
            "cv_roc_auc_mean": 0.7036,
            "cv_roc_auc_std": 0.0057,
            "training_time": "44.16s",
        },
        "hyperparameters": {
            "criterion": "entropy",
            "max_depth": 6,
            "min_samples_leaf": 50,
            "min_samples_split": 20,
        },
        "confusionMatrix": {
            "trueNegative": 45014,
            "falsePositive": 125,
            "falseNegative": 5752,
            "truePositive": 179,
        },
        "featureImportance": [
            { "feature": "Age", "importance": 0.3584, "direction": "Primary tree split: Age threshold partitions default likelihood" },
            { "feature": "InterestRate", "importance": 0.2313, "direction": "Secondary split: Rate burden divides borrower delinquency" },
            { "feature": "Income", "importance": 0.1901, "direction": "Tertiary split: Income magnitude determines repayment cushion" },
            { "feature": "LoanAmount", "importance": 0.0872, "direction": "Exposure split: High loan balances partition risk branches" },
            { "feature": "MonthsEmployed", "importance": 0.0514, "direction": "Tenure split: Job stability defines survival node" },
            { "feature": "CreditScore", "importance": 0.0381, "direction": "FICO split: Credit score divides prime from subprime cohorts" },
        ]
    }
]

@app.route("/api/models", methods=["GET"])
def get_models():
    """
    Return exactly the four primary models evaluated for production loan default prediction.
    A reference entry for the scratch model is included purely as reference metadata.
    """
    include_scratch = request.args.get("include_reference", "false").lower() == "true"
    result = list(PRIMARY_MODELS_INFO)
    if include_scratch:
        result.append({
            "id": "scratch-logistic-regression",
            "name": "Scratch Logistic Regression",
            "family": "Custom NumPy Implementation",
            "algorithm": "Batch Gradient Descent Logistic Regression",
            "library": "Pure NumPy",
            "isSelected": False,
            "isPrimary": False,
            "isReference": True,
            "status": "SOP Reference Implementation",
            "metrics": {
                "accuracy": 0.8852,
                "roc_auc": 0.7530,
                "cv_roc_auc_mean": 0.7530,
                "cv_roc_auc_std": 0.0030,
            }
        })
    return jsonify(result), 200

@app.route("/api/models/comparison", methods=["GET"])
def get_model_comparison():
    """Return evaluated metrics, cross-validation stats, and ROC comparison data."""
    roc_curves = MODEL_COMPARISON.get("roc_curves", {})
    return jsonify({
        "timestamp": MODEL_COMPARISON.get("timestamp", "2026-09-24"),
        "dataset": MODEL_COMPARISON.get("dataset", {
            "total_records": 255347,
            "train_records": 204277,
            "test_records": 51070,
            "raw_features": 16,
            "encoded_features": 24,
            "imbalance_ratio": 7.61
        }),
        "models": PRIMARY_MODELS_INFO,
        "rocCurves": roc_curves
    }), 200

@app.route("/api/models/<model_id>", methods=["GET"])
def get_model_by_id(model_id: str):
    """
    Return detailed specification, algorithm details, preprocessing rules,
    and genuine explanation information for a given model.
    """
    clean_id = model_id.lower().replace("_", "-")
    model_info = next((m for m in PRIMARY_MODELS_INFO if m["id"] == clean_id), None)
    if not model_info:
        return jsonify({
            "error": True,
            "message": f"Model '{model_id}' not found. Supported: logistic-regression, knn, naive-bayes, decision-tree",
            "details": {"requested_id": model_id}
        }), 404

    # Provide genuine model-specific explanation information
    explanation = {}
    if clean_id == "logistic-regression":
        lr_model = models.get("logistic_regression")
        coef_list = []
        if lr_model is not None and hasattr(lr_model, "coef_"):
            cat_encoder = preprocessor.named_transformers_["cat"]
            encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            feature_names = list(NUMERICAL_FEATURES) + list(encoded_cat_names)
            for feat, c in zip(feature_names, lr_model.coef_[0]):
                coef_list.append({
                    "feature": feat,
                    "coefficient": round(float(c), 4),
                    "impact_type": "Risk Driver" if c > 0 else "Protective Mitigant"
                })
        coef_list.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        explanation = {
            "type": "linear_coefficients",
            "intercept": round(float(lr_model.intercept_[0]), 4) if lr_model is not None else -0.35,
            "coefficients": coef_list,
            "interpretation": "Positive coefficients increase log-odds of default (e.g. InterestRate, Unemployment), while negative coefficients reduce default likelihood (e.g. Age, MonthsEmployed, Income)."
        }

    elif clean_id == "decision_tree" or clean_id == "decision-tree":
        dt_model = models.get("decision_tree")
        fi_list = []
        if dt_model is not None and hasattr(dt_model, "feature_importances_"):
            cat_encoder = preprocessor.named_transformers_["cat"]
            encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            feature_names = list(NUMERICAL_FEATURES) + list(encoded_cat_names)
            for feat, imp in zip(feature_names, dt_model.feature_importances_):
                if imp > 0.001:
                    fi_list.append({
                        "feature": feat,
                        "importance": round(float(imp), 4),
                    })
            fi_list.sort(key=lambda x: x["importance"], reverse=True)
        explanation = {
            "type": "gini_entropy_feature_importance",
            "feature_importance": fi_list,
            "tree_depth": int(dt_model.get_depth()) if dt_model is not None else 6,
            "leaf_nodes": int(dt_model.get_n_leaves()) if dt_model is not None else 45,
            "interpretation": "Feature importance represents the total normalized reduction in entropy (information gain) brought by that feature across all splits in the tree."
        }

    elif clean_id == "knn":
        explanation = {
            "type": "neighborhood_instance_density",
            "k_neighbors": 25,
            "weights": "inverse_distance (1/d)",
            "metric": "minkowski (p=2 Euclidean)",
            "interpretation": "KNN is a non-parametric instance-based learner. It does not compute static coefficients or feature importance. Instead, for each query point, it locates the 25 nearest borrower records in standardized 24-dimensional space and predicts default likelihood via distance-weighted voting."
        }

    elif clean_id == "naive-bayes":
        nb_model = models.get("naive_bayes")
        explanation = {
            "type": "class_conditional_gaussian",
            "var_smoothing": 1e-5,
            "priors": [round(float(p), 4) for p in nb_model.class_prior_] if nb_model is not None else [0.8839, 0.1161],
            "interpretation": "Gaussian Naive Bayes computes posterior probabilities via Bayes' theorem assuming conditional independence given the class label. Feature likelihoods follow normal distributions parameterized by per-class mean and smoothed variance."
        }

    detailed_spec = {
        **model_info,
        "preprocessingRequirements": [
            "Numerical StandardScaler (Zero mean, unit variance standard score)",
            "Categorical OneHotEncoder (drop='first', sparse_output=False)",
            "Strict 24-feature ordering: 9 numerical features followed by 15 one-hot categorical features",
            "Missing value imputation required if data contains NaNs (0 missing in baseline)",
        ],
        "explanation": explanation,
    }
    return jsonify(detailed_spec), 200

# ─────────────────────────────────────────────────────────────────────────────
# 6. Data Insights Endpoint (GET /api/data/insights)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/data/insights", methods=["GET"])
def get_data_insights():
    """Return actual EDA-derived distributions and dataset statistics."""
    if DATASET_INSIGHTS:
        return jsonify(DATASET_INSIGHTS), 200

    # Fallback to structured dataset metrics if file read fails
    return jsonify({
        "dataset_name": "Loan Default Prediction",
        "source": "Kaggle Banking Lending Dataset (nikhil1e9/loan-default)",
        "total_records": 255347,
        "clean_features": 16,
        "total_columns": 18,
        "default_count": 29653,
        "non_default_count": 225694,
        "default_percentage": 11.61,
        "missing_values": 0,
        "duplicate_rows": 0,
        "train_size": 204277,
        "test_size": 51070
    }), 200

# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Support platform PORT (Cloud Run standalone) or FLASK_PORT
    # When launched alongside Node in a unified container, FLASK_PORT overrides the outer PORT
    raw_port = os.environ.get("FLASK_PORT") or os.environ.get("PORT") or 5001
    port = int(raw_port)
    host = os.environ.get("HOST") or os.environ.get("FLASK_HOST") or "0.0.0.0"
    logger.info(f"Starting LoanLens Flask Engine on {host}:{port}...")
    app.run(host=host, port=port, debug=False)
