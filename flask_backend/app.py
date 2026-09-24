"""
LoanLens Flask API Service
Production Machine Learning Inference and Model Governance Engine
Serving 4 primary classification models:
  1. Logistic Regression (Tuned Production Candidate & Selected Best Model)
  2. K-Nearest Neighbors (KNN)
  3. Naive Bayes (GaussianNB)
  4. Decision Tree Classifier
"""

import os
import sys
import json
import logging
import pickle
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("LoanLensAPI")

app = Flask(__name__)

# Configure CORS
cors_origins = os.environ.get("CORS_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": cors_origins.split(",") if cors_origins != "*" else "*"}})

# Global artifact storage — loaded once at application startup
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts")))

MODELS: Dict[str, Any] = {}
PREPROCESSOR: Optional[Any] = None
METRICS_DATA: Dict[str, Any] = {}
DATASET_INSIGHTS: Dict[str, Any] = {}
FEATURE_METADATA: Dict[str, Any] = {}

# Allowed categories and domain constraints
ALLOWED_CATEGORIES = {
    "Education": ["Bachelor's", "High School", "Master's", "PhD"],
    "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "HasMortgage": ["Yes", "No"],
    "HasDependents": ["Yes", "No"],
    "LoanPurpose": ["Auto", "Business", "Education", "Home", "Other"],
    "HasCoSigner": ["Yes", "No"]
}

NUMERICAL_BOUNDS = {
    "Age": (18, 100),
    "Income": (1000.0, 50000000.0),
    "LoanAmount": (500.0, 50000000.0),
    "CreditScore": (300, 850),
    "MonthsEmployed": (0, 600),
    "NumCreditLines": (0, 50),
    "InterestRate": (0.1, 50.0),
    "LoanTerm": (1, 360),
    "DTIRatio": (0.0, 1.5)
}


def load_artifacts_once():
    """Load serialized model and preprocessing artifacts once at application startup."""
    global MODELS, PREPROCESSOR, METRICS_DATA, DATASET_INSIGHTS, FEATURE_METADATA
    logger.info("Initializing artifact loader from: %s", ARTIFACTS_DIR)

    # 1. Load Preprocessor
    prep_path = os.path.join(ARTIFACTS_DIR, "preprocessing", "preprocessor.pkl")
    if not os.path.exists(prep_path):
        prep_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    
    if os.path.exists(prep_path):
        with open(prep_path, "rb") as f:
            PREPROCESSOR = pickle.load(f)
        logger.info("Loaded preprocessor artifact from %s", prep_path)
    else:
        logger.error("Preprocessor artifact not found at %s", prep_path)

    # 2. Load Models
    model_files = {
        "logistic_regression": "logistic_regression.pkl",
        "knn": "knn.pkl",
        "naive_bayes": "naive_bayes.pkl",
        "decision_tree": "decision_tree.pkl"
    }

    for key, filename in model_files.items():
        path = os.path.join(ARTIFACTS_DIR, "models", filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                MODELS[key] = pickle.load(f)
            logger.info("Loaded model artifact [%s] from %s", key, path)
        else:
            logger.warning("Model file %s not found in %s/models", filename, ARTIFACTS_DIR)

    # 3. Load Metrics & Comparison JSON
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics", "model_comparison.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            METRICS_DATA = json.load(f)
        logger.info("Loaded model comparison metrics from %s", metrics_path)

    # 4. Load Dataset Insights JSON
    insights_path = os.path.join(ARTIFACTS_DIR, "metrics", "dataset_insights.json")
    if os.path.exists(insights_path):
        with open(insights_path, "r") as f:
            DATASET_INSIGHTS = json.load(f)
        logger.info("Loaded dataset insights from %s", insights_path)

    # 5. Load Feature Metadata
    feat_meta_path = os.path.join(ARTIFACTS_DIR, "metadata", "feature_metadata.json")
    if os.path.exists(feat_meta_path):
        with open(feat_meta_path, "r") as f:
            FEATURE_METADATA = json.load(f)
        logger.info("Loaded feature metadata from %s", feat_meta_path)


# Load artifacts during module import/startup
load_artifacts_once()


# ─── Structured Error Responses ───────────────────────────────────────────────

def error_response(message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None):
    payload = {
        "error": True,
        "message": message,
        "details": details or {}
    }
    return make_response(jsonify(payload), status_code)


@app.errorhandler(400)
def handle_bad_request(e):
    return error_response("Bad Request: Malformed JSON or invalid parameter format", 400)


@app.errorhandler(404)
def handle_not_found(e):
    return error_response("Resource not found", 404)


@app.errorhandler(405)
def handle_method_not_allowed(e):
    return error_response("HTTP Method Not Allowed", 405)


@app.errorhandler(422)
def handle_unprocessable_entity(e):
    return error_response("Unprocessable Entity: Input validation constraints failed", 422)


@app.errorhandler(500)
def handle_internal_server_error(e):
    logger.error("Internal Server Error: %s", str(e))
    return error_response("An internal server error occurred. Please contact system administrator.", 500)


# ─── Health Endpoint ──────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check confirming operational readiness and active model."""
    artifacts_ready = PREPROCESSOR is not None and "logistic_regression" in MODELS
    return jsonify({
        "status": "ok",
        "model": "Logistic Regression",
        "artifacts_loaded": artifacts_ready,
        "models_available": [
            "Logistic Regression",
            "KNN",
            "Naive Bayes",
            "Decision Tree"
        ]
    })


# ─── Input Validation Helper ──────────────────────────────────────────────────

def normalize_and_validate_borrower(data: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    """
    Validate and normalize the 16 borrower features against schema and range boundaries.
    Returns (cleaned_dict, errors_dict).
    """
    if not isinstance(data, dict):
        return None, {"body": "Expected a JSON object"}

    errors = {}
    cleaned = {}

    # 1. Validate Numerical Features
    for field, (min_val, max_val) in NUMERICAL_BOUNDS.items():
        if field not in data or data[field] is None or data[field] == "":
            errors[field] = f"Missing required numerical field: '{field}'"
            continue
        try:
            val = float(data[field])
            if np.isnan(val) or np.isinf(val):
                errors[field] = f"Field '{field}' must be a finite real number"
                continue
            if val < min_val or val > max_val:
                errors[field] = f"Field '{field}' value {val} is outside allowed range [{min_val}, {max_val}]"
                continue
            # Integer casts for count / age fields
            if field in ["Age", "CreditScore", "MonthsEmployed", "NumCreditLines", "LoanTerm"]:
                cleaned[field] = int(round(val))
            else:
                cleaned[field] = val
        except (ValueError, TypeError):
            errors[field] = f"Field '{field}' must be a valid number"

    # 2. Validate Categorical Features
    for field, allowed in ALLOWED_CATEGORIES.items():
        if field not in data or data[field] is None or data[field] == "":
            errors[field] = f"Missing required categorical field: '{field}'"
            continue

        raw_val = data[field]
        # Support boolean flags for Yes/No fields
        if field in ["HasMortgage", "HasDependents", "HasCoSigner"]:
            if isinstance(raw_val, bool):
                raw_val = "Yes" if raw_val else "No"
            elif isinstance(raw_val, (int, float)):
                raw_val = "Yes" if int(raw_val) == 1 else "No"

        str_val = str(raw_val).strip()

        # Case-insensitive match against allowed categories
        matched = next((c for c in allowed if c.lower() == str_val.lower()), None)
        if matched:
            cleaned[field] = matched
        else:
            errors[field] = f"Field '{field}' value '{str_val}' is invalid. Allowed choices: {allowed}"

    if errors:
        return None, errors

    return cleaned, None


# ─── POST /api/predict ────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict_loan_default():
    """
    Execute real model inference using the preprocessor pipeline and tuned Logistic Regression model.
    Converts default probability into an institutional risk score and transparent factor attributions.
    """
    if PREPROCESSOR is None or "logistic_regression" not in MODELS:
        return error_response("Model or preprocessor artifacts are not loaded on server", 500)

    payload = request.get_json(silent=True)
    if payload is None:
        return error_response("Invalid or missing JSON payload in request body", 400)

    cleaned_input, validation_errors = normalize_and_validate_borrower(payload)
    if validation_errors:
        return error_response("Validation failed for borrower attributes", 422, validation_errors)

    try:
        # Create single-row DataFrame
        df_input = pd.DataFrame([cleaned_input])

        # Apply ColumnTransformer (StandardScaler on 9 numerics + OHE on 7 categoricals)
        X_trans = PREPROCESSOR.transform(df_input)

        # Predict probability using tuned Logistic Regression
        model = MODELS["logistic_regression"]
        probs = model.predict_proba(X_trans)[0]
        prob_default = float(probs[1])

        # Mathematical Transformation of Probability to Risk Score
        # Risk Score is calibrated on a 0-100 scale: risk_score = round(p * 100, 1)
        risk_score = round(prob_default * 100.0, 1)

        # Institutional Risk Level Boundaries:
        # Low Risk:    p < 0.20  (risk_score < 20.0) -> Prime borrower, strong repayment capacity
        # Medium Risk: 0.20 <= p < 0.50 (20.0 <= risk_score < 50.0) -> Standard risk, routine covenants
        # High Risk:   p >= 0.50 (risk_score >= 50.0) -> Elevated delinquency probability, review required
        if prob_default < 0.20:
            risk_level = "Low"
            recommendation_action = "Standard Approval / Prime Terms"
            recommendation_text = "Applicant demonstrates strong financial fundamentals and low default likelihood."
        elif prob_default < 0.50:
            risk_level = "Medium"
            recommendation_action = "Standard Terms with Routine Verification"
            recommendation_text = "Standard credit risk tier. Regular employment verification and debt covenants advised."
        else:
            risk_level = "High"
            recommendation_action = "Credit Committee Review / Enhanced Due Diligence"
            recommendation_text = "Elevated risk of default detected. Consider lowering loan-to-value or requiring additional guarantor."

        prediction = 1 if prob_default >= 0.50 else 0
        label = "Default" if prediction == 1 else "No Default"
        confidence = round(max(prob_default, 1.0 - prob_default), 2)

        # Transparent Factor Attribution via Parametric Log-Odds Contributions
        # z = beta_0 + sum(beta_j * x_trans_j)
        factors = []
        try:
            feature_names = PREPROCESSOR.get_feature_names_out()
            coefficients = model.coef_[0]
            x_row = X_trans[0]

            for feat_name, coef, x_val in zip(feature_names, coefficients, x_row):
                # Clean feature name prefix from ColumnTransformer
                clean_name = feat_name.replace("num__", "").replace("cat__", "")
                contribution = float(coef * x_val)

                # Focus on significant non-zero contributions
                if abs(contribution) >= 0.01:
                    is_risk = contribution > 0
                    direction = "Risk Driver" if is_risk else "Risk Mitigating"
                    factors.append({
                        "feature": clean_name,
                        "coefficient": round(float(coef), 4),
                        "transformed_value": round(float(x_val), 3),
                        "impact": round(contribution, 4),
                        "direction": direction,
                        "description": f"{clean_name}: {'Increases' if is_risk else 'Reduces'} default log-odds by {abs(contribution):.3f}"
                    })

            # Sort factors by absolute magnitude
            factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
        except Exception as factor_err:
            logger.warning("Could not calculate linear factor contributions: %s", factor_err)

        response_payload = {
            "prediction": prediction,
            "label": label,
            "probability": round(prob_default, 4),
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "model_used": "Logistic Regression",
            "factors": factors[:8],
            "recommendation": {
                "action": recommendation_action,
                "notes": recommendation_text
            },
            "features_evaluated": int(X_trans.shape[1])
        }

        return jsonify(response_payload)

    except Exception as exc:
        logger.error("Inference execution failed: %s", exc)
        return error_response(f"Inference execution failed: {str(exc)}", 500)


# ─── GET /api/models ──────────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def get_primary_models():
    """
    Return exactly the four primary classification models evaluated in the pipeline.
    Scratch implementation is exposed only under an optional reference entry.
    """
    # Primary 4 models definition
    primary_models = [
        {
            "id": "logistic-regression",
            "name": "Logistic Regression",
            "family": "LogisticRegression",
            "algorithm": "L2-Regularized Logistic Regression (Balanced)",
            "library": "Scikit-Learn (LogisticRegression)",
            "featureCount": 24,
            "status": "Selected Best Model",
            "isSelected": True,
            "description": "Tuned generalized linear model utilizing balanced class weighting to overcome the 88.4% / 11.6% class imbalance. Selected as the best primary model due to highest discriminative ranking power (Test ROC-AUC: 0.7532, 5-Fold CV AUC: 0.7426 ± 0.0063) and outstanding default recall (69.95%), ensuring credit institutions capture potential defaults.",
            "metrics": {
                "accuracy": 0.6764,
                "precision": 0.2196,
                "recall": 0.6995,
                "f1_score": 0.3342,
                "roc_auc": 0.7532,
                "cv_roc_auc_mean": 0.7426,
                "cv_roc_auc_std": 0.0063,
                "training_time": "3.95s"
            },
            "hyperparameters": {
                "C": 0.1,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 1000
            },
            "confusionMatrix": {
                "trueNegative": 30393,
                "falsePositive": 14746,
                "falseNegative": 1782,
                "truePositive": 4149
            }
        },
        {
            "id": "knn",
            "name": "K-Nearest Neighbors (KNN)",
            "family": "KNeighborsClassifier",
            "algorithm": "Distance-Weighted K-Nearest Neighbors (k=25)",
            "library": "Scikit-Learn (KNeighborsClassifier)",
            "featureCount": 24,
            "status": "Benchmark Evaluated",
            "isSelected": False,
            "description": "Non-parametric instance-based learning classifying loans through local density proximity in standardized feature space. Utilizes BallTree spatial indexing, Minkowski metric, and distance-weighted voting across k=25 nearest neighbors.",
            "metrics": {
                "accuracy": 0.8851,
                "precision": 0.5455,
                "recall": 0.0070,
                "f1_score": 0.0137,
                "roc_auc": 0.6927,
                "cv_roc_auc_mean": 0.6753,
                "cv_roc_auc_std": 0.0068,
                "training_time": "38.33s"
            },
            "hyperparameters": {
                "n_neighbors": 25,
                "weights": "distance",
                "metric": "minkowski",
                "algorithm": "ball_tree"
            },
            "confusionMatrix": {
                "trueNegative": 13265,
                "falsePositive": 10,
                "falseNegative": 1713,
                "truePositive": 12
            }
        },
        {
            "id": "naive-bayes",
            "name": "Naive Bayes",
            "family": "GaussianNB",
            "algorithm": "Gaussian Naive Bayes (var_smoothing=1e-5)",
            "library": "Scikit-Learn (GaussianNB)",
            "featureCount": 24,
            "status": "Benchmark Evaluated",
            "isSelected": False,
            "description": "Probabilistic classifier applying Bayes' theorem with conditional independence assumptions. Gaussian likelihood represents standardized continuous financial metrics while variance smoothing (1e-5) provides stability across one-hot encoded variables.",
            "metrics": {
                "accuracy": 0.8849,
                "precision": 0.6080,
                "recall": 0.0256,
                "f1_score": 0.0492,
                "roc_auc": 0.7499,
                "cv_roc_auc_mean": 0.7396,
                "cv_roc_auc_std": 0.0044,
                "training_time": "0.99s"
            },
            "hyperparameters": {
                "var_smoothing": 1e-05,
                "priors": "Empirical Class Distribution (88.4% / 11.6%)"
            },
            "confusionMatrix": {
                "trueNegative": 45041,
                "falsePositive": 98,
                "falseNegative": 5779,
                "truePositive": 152
            }
        },
        {
            "id": "decision-tree",
            "name": "Decision Tree",
            "family": "DecisionTreeClassifier",
            "algorithm": "CART (Classification and Regression Tree)",
            "library": "Scikit-Learn (DecisionTreeClassifier)",
            "featureCount": 24,
            "status": "Benchmark Evaluated",
            "isSelected": False,
            "description": "Single-tree recursive partitioning using entropy criterion, tuned with max_depth=6 and min_samples_leaf=50 to prevent overfitting while capturing non-linear interactions across borrower risk tiers.",
            "metrics": {
                "accuracy": 0.8849,
                "precision": 0.5888,
                "recall": 0.0302,
                "f1_score": 0.0574,
                "roc_auc": 0.7254,
                "cv_roc_auc_mean": 0.7036,
                "cv_roc_auc_std": 0.0057,
                "training_time": "44.16s"
            },
            "hyperparameters": {
                "criterion": "entropy",
                "max_depth": 6,
                "min_samples_split": 20,
                "min_samples_leaf": 50
            },
            "confusionMatrix": {
                "trueNegative": 45014,
                "falsePositive": 125,
                "falseNegative": 5752,
                "truePositive": 179
            }
        }
    ]

    return jsonify(primary_models)


# ─── GET /api/models/comparison ───────────────────────────────────────────────

@app.route("/api/models/comparison", methods=["GET"])
def get_model_comparison():
    """
    Return empirical comparison metrics for all 4 primary classification models,
    including ROC curves and cross-validation standard deviations.
    """
    if METRICS_DATA and "models_comparison" in METRICS_DATA:
        models_comp = METRICS_DATA["models_comparison"]
        roc_curves = METRICS_DATA.get("roc_curves", {})
        scratch_audit = METRICS_DATA.get("scratch_vs_sklearn_lr", {})

        summary_table = []
        for name, m in models_comp.items():
            summary_table.append({
                "model_name": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"],
                "cv_roc_auc_mean": m["cv_roc_auc_mean"],
                "cv_roc_auc_std": m["cv_roc_auc_std"],
                "training_time": m["training_time"],
                "hyperparameters": m["hyperparameters"],
                "selected_status": "Selected Best Model" if name == "Logistic Regression" else "Benchmark Evaluated"
            })

        return jsonify({
            "models": summary_table,
            "rocCurves": roc_curves,
            "scratchAudit": scratch_audit,
            "evaluationDataset": {
                "totalRecords": 255347,
                "trainRecords": 204277,
                "testRecords": 51070,
                "splitRatio": "80/20 Stratified"
            }
        })

    # Fallback to structured static metrics if file unreadable
    return error_response("Model comparison metrics data is not available", 500)


# ─── GET /api/models/<modelId> ────────────────────────────────────────────────

@app.route("/api/models/<model_id>", methods=["GET"])
def get_model_details(model_id: str):
    """
    Return detailed specification, actual hyperparameters, preprocessing requirements,
    and algorithm-specific explanations for a requested model.
    """
    norm_id = model_id.lower().replace("_", "-")

    if norm_id in ["logistic-regression", "lr"]:
        # Extract actual coefficients from loaded model
        lr_coefs = []
        if "logistic_regression" in MODELS and PREPROCESSOR is not None:
            try:
                lr_mod = MODELS["logistic_regression"]
                feat_names = PREPROCESSOR.get_feature_names_out()
                for name, coef in zip(feat_names, lr_mod.coef_[0]):
                    clean_name = name.replace("num__", "").replace("cat__", "")
                    impact_label = "Increases Default Risk (Risk Driver)" if coef > 0 else "Decreases Default Risk (Protective Factor)"
                    lr_coefs.append({
                        "feature": clean_name,
                        "coefficient": round(float(coef), 4),
                        "direction": impact_label
                    })
                lr_coefs.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
            except Exception as e:
                logger.warning("Could not extract live coefficients: %s", e)

        return jsonify({
            "id": "logistic-regression",
            "name": "Logistic Regression",
            "family": "LogisticRegression",
            "algorithm": "L2-Regularized Logistic Regression (Balanced)",
            "library": "Scikit-Learn (LogisticRegression)",
            "status": "Selected Best Model",
            "isSelected": True,
            "preprocessingRequirements": [
                "Continuous Feature Standardization (StandardScaler, mu=0, sigma=1)",
                "Categorical Dummy Encoding (OneHotEncoder with drop_first=True)"
            ],
            "hyperparameters": {
                "C": 0.1,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "penalty": "l2",
                "max_iter": 1000
            },
            "metrics": {
                "accuracy": 0.6764,
                "precision": 0.2196,
                "recall": 0.6995,
                "f1_score": 0.3342,
                "roc_auc": 0.7532,
                "cv_roc_auc_mean": 0.7426,
                "cv_roc_auc_std": 0.0063,
                "training_time": "3.95s"
            },
            "modelExplanation": {
                "method": "Parametric Standardized Log-Odds Coefficients (beta)",
                "interpretation": "Coefficients reflect change in log-odds of loan default per one standard deviation increase in normalized feature value.",
                "featureAttributions": lr_coefs
            }
        })

    elif norm_id == "decision-tree":
        # Extract actual Gini / Entropy feature importances from loaded tree
        dt_importances = []
        if "decision_tree" in MODELS and PREPROCESSOR is not None:
            try:
                dt_mod = MODELS["decision_tree"]
                feat_names = PREPROCESSOR.get_feature_names_out()
                for name, imp in zip(feat_names, dt_mod.feature_importances_):
                    clean_name = name.replace("num__", "").replace("cat__", "")
                    if imp > 0.001:
                        dt_importances.append({
                            "feature": clean_name,
                            "importance": round(float(imp), 4),
                            "direction": "Entropy Impurity Reduction Gain"
                        })
                dt_importances.sort(key=lambda x: x["importance"], reverse=True)
            except Exception as e:
                logger.warning("Could not extract tree feature importances: %s", e)

        return jsonify({
            "id": "decision-tree",
            "name": "Decision Tree",
            "family": "DecisionTreeClassifier",
            "algorithm": "CART (Classification and Regression Tree)",
            "library": "Scikit-Learn (DecisionTreeClassifier)",
            "status": "Benchmark Evaluated",
            "isSelected": False,
            "preprocessingRequirements": [
                "Categorical Dummy Encoding (OneHotEncoder with drop_first=True)",
                "Monotonically invariant to numeric scaling, but evaluated on standardized 24D matrix for consistency"
            ],
            "hyperparameters": {
                "criterion": "entropy",
                "max_depth": 6,
                "min_samples_split": 20,
                "min_samples_leaf": 50,
                "random_state": 42
            },
            "metrics": {
                "accuracy": 0.8849,
                "precision": 0.5888,
                "recall": 0.0302,
                "f1_score": 0.0574,
                "roc_auc": 0.7254,
                "cv_roc_auc_mean": 0.7036,
                "cv_roc_auc_std": 0.0057,
                "training_time": "44.16s"
            },
            "modelExplanation": {
                "method": "Gini / Shannon Entropy Impurity Reduction",
                "interpretation": "Normalized total reduction of the criterion brought by that feature across tree splits.",
                "featureAttributions": dt_importances
            }
        })

    elif norm_id == "knn":
        return jsonify({
            "id": "knn",
            "name": "K-Nearest Neighbors (KNN)",
            "family": "KNeighborsClassifier",
            "algorithm": "Distance-Weighted K-Nearest Neighbors (k=25)",
            "library": "Scikit-Learn (KNeighborsClassifier)",
            "status": "Benchmark Evaluated",
            "isSelected": False,
            "preprocessingRequirements": [
                "Strict Standardization (StandardScaler) essential for Euclidean/Minkowski distance computation",
                "OneHotEncoder for categorical dimensions"
            ],
            "hyperparameters": {
                "n_neighbors": 25,
                "weights": "distance",
                "metric": "minkowski",
                "algorithm": "ball_tree"
            },
            "metrics": {
                "accuracy": 0.8851,
                "precision": 0.5455,
                "recall": 0.0070,
                "f1_score": 0.0137,
                "roc_auc": 0.6927,
                "cv_roc_auc_mean": 0.6753,
                "cv_roc_auc_std": 0.0068,
                "training_time": "38.33s"
            },
            "modelExplanation": {
                "method": "Instance-Based Distance-Weighted Local Consensus",
                "interpretation": "KNN makes predictions based on local neighborhood proximity in 24-dimensional standardized feature space rather than global parametric feature coefficients. Weights are inversely proportional to Minkowski distance (w = 1 / d).",
                "neighborhoodDetails": {
                    "k_neighbors": 25,
                    "distance_metric": "minkowski (p=2 Euclidean)",
                    "weighting": "distance-weighted consensus",
                    "spatial_index": "BallTree"
                }
            }
        })

    elif norm_id == "naive-bayes":
        nb_priors = {"No Default (Class 0)": 0.8839, "Default (Class 1)": 0.1161}
        if "naive_bayes" in MODELS:
            try:
                nb_mod = MODELS["naive_bayes"]
                nb_priors = {
                    "No Default (Class 0)": round(float(np.exp(nb_mod.class_log_prior_[0])), 4),
                    "Default (Class 1)": round(float(np.exp(nb_mod.class_log_prior_[1])), 4)
                }
            except Exception as e:
                logger.warning("Could not read NB priors: %s", e)

        return jsonify({
            "id": "naive-bayes",
            "name": "Naive Bayes",
            "family": "GaussianNB",
            "algorithm": "Gaussian Naive Bayes (var_smoothing=1e-5)",
            "library": "Scikit-Learn (GaussianNB)",
            "status": "Benchmark Evaluated",
            "isSelected": False,
            "preprocessingRequirements": [
                "StandardScaler applied to continuous variables for Gaussian assumption stability",
                "Variance smoothing (1e-5) added to prevent zero-variance divisions on binary indicators"
            ],
            "hyperparameters": {
                "var_smoothing": 1e-05,
                "priors": "Empirical Class Distribution"
            },
            "metrics": {
                "accuracy": 0.8849,
                "precision": 0.6080,
                "recall": 0.0256,
                "f1_score": 0.0492,
                "roc_auc": 0.7499,
                "cv_roc_auc_mean": 0.7396,
                "cv_roc_auc_std": 0.0044,
                "training_time": "0.99s"
            },
            "modelExplanation": {
                "method": "Bayesian Posterior Probability with Conditional Independence",
                "interpretation": "P(Default | X) proportional to P(Default) * Product(P(x_i | Default)). Gaussian density modeling on normalized continuous features.",
                "classPriors": nb_priors
            }
        })

    return error_response(f"Model '{model_id}' not found. Supported models: logistic-regression, knn, naive-bayes, decision-tree", 404)


# ─── GET /api/data/insights ───────────────────────────────────────────────────

@app.route("/api/data/insights", methods=["GET"])
def get_data_insights():
    """
    Return real Exploratory Data Analysis (EDA) metrics derived from the 255,347 record dataset.
    """
    if DATASET_INSIGHTS:
        return jsonify(DATASET_INSIGHTS)

    # If dataset_insights.json was not loaded, read directly from artifacts
    insights_path = os.path.join(ARTIFACTS_DIR, "metrics", "dataset_insights.json")
    if os.path.exists(insights_path):
        with open(insights_path, "r") as f:
            data = json.load(f)
            return jsonify(data)

    return error_response("Dataset insights data is currently unavailable", 500)


# ─── Application Main Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5001"))
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    logger.info("Starting LoanLens Flask Backend on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False)
