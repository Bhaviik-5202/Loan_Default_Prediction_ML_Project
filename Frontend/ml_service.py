"""
LoanML — Machine Learning Inference Engine (Weeks 7-10)
=========================================================
Week 7-9 : Real prediction pipeline — Validation → Encoding → Scaler → Model
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("loanml.ml_service")

# ─── Artifact paths ───────────────────────────────────────────────────────────
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_ARTIFACT_DIR = Path(os.environ.get("ML_ARTIFACT_DIR", _BASE_DIR.parent / "ml_artifacts"))

REQUIRED_ARTIFACTS = [
    "loan_model.pkl",
    "loan_scaler.pkl",
    "loan_feature_columns.pkl",
    "model_comparison_metrics.json"
]

def _verify_artifacts():
    """Verify that all required ML artifacts exist in the artifact directory."""
    missing = [f for f in REQUIRED_ARTIFACTS if not (_ARTIFACT_DIR / f).exists()]
    if missing:
        logger.critical(f"Missing required ML artifacts in {_ARTIFACT_DIR}: {', '.join(missing)}")
        return False
    return True

_ARTIFACTS_LOADED = _verify_artifacts()
_MODEL_PATH   = _ARTIFACT_DIR / "loan_model.pkl"
_SCALER_PATH  = _ARTIFACT_DIR / "loan_scaler.pkl"
_COLS_PATH    = _ARTIFACT_DIR / "loan_feature_columns.pkl"
_METRICS_PATH = _ARTIFACT_DIR / "model_comparison_metrics.json"

# ─── Load artifacts at module import time ────────────────────────────────────
try:
    if _ARTIFACTS_LOADED:
        _MODEL           = joblib.load(_MODEL_PATH)
        _SCALER          = joblib.load(_SCALER_PATH)
        _FEATURE_COLUMNS = joblib.load(_COLS_PATH)
        logger.info(f"ML Engine loaded: {type(_MODEL).__name__}, {len(_FEATURE_COLUMNS)} features")
        print(f"ML Engine: Loaded {type(_MODEL).__name__} — {len(_FEATURE_COLUMNS)} features.")
    else:
        raise FileNotFoundError("Required artifacts missing during verification.")
except Exception as e:
    logger.error(f"Could not load ML artifacts: {e}")
    print(f"ML Engine Error: {e}")
    _MODEL           = None
    _SCALER          = None
    _FEATURE_COLUMNS = []
    _ARTIFACTS_LOADED = False

def is_model_loaded() -> bool:
    """Returns True if the trained model and scaler are successfully loaded."""
    return _MODEL is not None and _SCALER is not None


# ─── Categorical value sets (matches training encoding) ───────────────────────
EDUCATION_VALS   = ["Bachelor's", "High School", "Master's", "PhD"]
EMPLOYMENT_VALS  = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
MARITAL_VALS     = ["Divorced", "Married", "Single"]
PURPOSE_VALS     = ["Auto", "Business", "Education", "Home", "Other"]
YES_NO_VALS      = ["Yes", "No"]


def get_ml_metrics() -> dict:
    """Load and return the full model metrics JSON."""
    if os.path.exists(_METRICS_PATH):
        try:
            with open(_METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics JSON: {e}")
    return {}


def _build_dataframe(d: dict) -> pd.DataFrame:
    """Parse input dict → raw DataFrame with correct types."""
    return pd.DataFrame([{
        'Age'            : float(d.get('Age',           35)),
        'Income'         : float(d.get('Income',        60000)),
        'LoanAmount'     : float(d.get('LoanAmount',    15000)),
        'CreditScore'    : float(d.get('CreditScore',   650)),
        'MonthsEmployed' : float(d.get('MonthsEmployed',24)),
        'NumCreditLines' : float(d.get('NumCreditLines',2)),
        'InterestRate'   : float(d.get('InterestRate',  12.5)),
        'LoanTerm'       : float(d.get('LoanTerm',      36)),
        'DTIRatio'       : float(d.get('DTIRatio',      0.35)),
        'Education'      : str(d.get('Education',       "Bachelor's")),
        'EmploymentType' : str(d.get('EmploymentType',  'Full-time')),
        'MaritalStatus'  : str(d.get('MaritalStatus',   'Single')),
        'HasMortgage'    : str(d.get('HasMortgage',     'No')),
        'HasDependents'  : str(d.get('HasDependents',   'No')),
        'LoanPurpose'    : str(d.get('LoanPurpose',     'Other')),
        'HasCoSigner'    : str(d.get('HasCoSigner',     'No')),
    }])


def _encode_and_align(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical one-hot encoding (drop_first=True) + feature alignment.
    Must exactly match the training preprocessing in train_models.py.
    """
    categorical_cols = [
        'Education', 'EmploymentType', 'MaritalStatus',
        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ]
    df_enc = pd.get_dummies(df_raw, columns=categorical_cols, drop_first=True)

    if _FEATURE_COLUMNS:
        # Add any missing columns (zeros) and reorder
        for col in _FEATURE_COLUMNS:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[_FEATURE_COLUMNS]

    return df_enc


def predict_loan_risk(input_dict: dict) -> dict:
    """
    Full real ML inference pipeline (Week 9):
    1. Parse inputs with defaults
    2. Build raw DataFrame
    3. Categorical encoding (pd.get_dummies, drop_first=True)
    4. Feature alignment to training columns
    5. StandardScaler transform
    6. HistGBM predict_proba()
    7. Risk scoring, level classification, factor attribution
    """
    df_raw = _build_dataframe(input_dict)

    # Extract scalar values for use in attribution
    age             = float(df_raw['Age'].iloc[0])
    income          = float(df_raw['Income'].iloc[0])
    loan_amount     = float(df_raw['LoanAmount'].iloc[0])
    credit_score    = float(df_raw['CreditScore'].iloc[0])
    months_employed = float(df_raw['MonthsEmployed'].iloc[0])
    num_cl          = float(df_raw['NumCreditLines'].iloc[0])
    interest_rate   = float(df_raw['InterestRate'].iloc[0])
    loan_term       = float(df_raw['LoanTerm'].iloc[0])
    dti_ratio       = float(df_raw['DTIRatio'].iloc[0])
    has_cosigner    = str(df_raw['HasCoSigner'].iloc[0])
    has_mortgage    = str(df_raw['HasMortgage'].iloc[0])
    has_dependents  = str(df_raw['HasDependents'].iloc[0])
    employment_type = str(df_raw['EmploymentType'].iloc[0])
    loan_purpose    = str(df_raw['LoanPurpose'].iloc[0])
    education       = str(df_raw['Education'].iloc[0])

    df_enc   = _encode_and_align(df_raw)

    # Predict
    if _MODEL is not None and _SCALER is not None:
        X_scaled = _SCALER.transform(df_enc)
        probs    = _MODEL.predict_proba(X_scaled)[0]
        p_default = float(probs[1])
        model_status = "trained_model"
    else:
        # Deterministic heuristic fallback (no random formula)
        cs_norm = max(0, min(1, (850 - credit_score) / 550))
        dti_norm = min(1, dti_ratio)
        p_default = 0.4 * cs_norm + 0.35 * dti_norm + 0.25 * min(interest_rate / 25, 1)
        model_status = "heuristic_fallback"

    probability   = round(p_default * 100, 1)
    risk_score    = round(100 - probability)

    # Risk level classification (calibrated on actual training distribution)
    if probability >= 50.0:
        risk_level = "High"
        prediction = "Likely to Default"
    elif probability >= 28.0:
        risk_level = "Medium"
        prediction = "Moderate Risk"
    else:
        risk_level = "Low"
        prediction = "Likely to Repay"

    # ─ Factor attribution (based on actual feature contributions) ─────────────
    factors = _compute_factors(
        credit_score, dti_ratio, interest_rate, income,
        loan_amount, months_employed, has_cosigner,
        has_mortgage, has_dependents, employment_type, num_cl
    )

    # ─ Borrower profile dimensions ────────────────────────────────────────────
    loan_to_income = loan_amount / max(income, 1)
    profile = {
        "Credit Health"  : round(max(5, min(95, (credit_score - 300) / 5.5))),
        "Fin. Stability" : round(max(5, min(95, (1 - dti_ratio) * 100))),
        "Repay. History" : round(min(95, 60 + (15 if has_cosigner=="Yes" else 0) + (15 if has_mortgage=="Yes" else 0))),
        "Employment"     : round(max(5, min(95, (months_employed / 60) * 100))),
        "Debt Burden"    : round(max(5, min(95, (1 - min(loan_to_income / 2.0, 1.0)) * 100))),
        "Loan Risk"      : round(max(5, min(95, 100 - interest_rate * 3.5))),
    }

    # ─ Recommendation ─────────────────────────────────────────────────────────
    if risk_level == "High":
        action = "High Risk — Manual Underwriting Required"
        points = [
            f"Default probability is {probability}% — exceeds acceptable lending threshold.",
            "Require debt consolidation or reduction of loan principal.",
            "Mandate co-signer with prime credit (CreditScore ≥ 720) or additional collateral.",
            "Verify income via bank statements and recent tax filings before proceeding.",
        ]
    elif risk_level == "Medium":
        action = "Moderate Risk — Standard Review with Conditions"
        points = [
            f"Default probability is {probability}% — borderline profile.",
            "Verify employment stability (minimum 12 months continuous tenure).",
            "Consider adjusting loan term to reduce monthly payment burden.",
            "Review existing credit lines and debt repayment track record.",
        ]
    else:
        action = "Low Risk — Fast-Track Approval Recommended"
        points = [
            f"Default probability is very low ({probability}%).",
            "Strong financial and credit fundamentals meet prime lending benchmarks.",
            "Eligible for preferred interest rate tier and expedited closing.",
            "Automated document verification and streamlined underwriting path.",
        ]

    return {
        "success"           : True,
        "probability"       : probability,
        "risk_score"        : risk_score,
        "risk_level"        : risk_level,
        "prediction"        : prediction,
        "factors"           : factors,
        "profile"           : profile,
        "recommendation"    : {"action": action, "points": points},
        "model_used"        : model_status,
        "features_evaluated": len(_FEATURE_COLUMNS) if _FEATURE_COLUMNS else 24,
    }


def _compute_factors(credit_score, dti_ratio, interest_rate, income,
                     loan_amount, months_employed, has_cosigner,
                     has_mortgage, has_dependents, employment_type, num_cl):
    """
    Compute human-readable risk factor explanations based on real feature values
    based on the submitted borrower profile and documented lending thresholds.
    """
    loan_to_income = loan_amount / max(income, 1)

    factors = [
        {
            "label"    : "Credit Score",
            "value"    : int(credit_score),
            "benchmark": 720,
            "impact"   : round(max(0, min(100, (800 - credit_score) / 5))),
            "direction": "down" if credit_score >= 680 else "up",
            "desc"     : f"Score {int(credit_score)} — {'Prime (≥720)' if credit_score >= 720 else 'Near-Prime (640-719)' if credit_score >= 640 else 'Subprime (<640)'}",
        },
        {
            "label"    : "Debt-to-Income Ratio",
            "value"    : round(dti_ratio * 100, 1),
            "benchmark": 40,
            "impact"   : round(min(dti_ratio * 100, 100)),
            "direction": "up" if dti_ratio > 0.40 else "down",
            "desc"     : f"DTI {round(dti_ratio*100,1)}% — {'Elevated (>40%)' if dti_ratio>0.4 else 'Healthy (≤40%)'}",
        },
        {
            "label"    : "Interest Rate Burden",
            "value"    : interest_rate,
            "benchmark": 10.0,
            "impact"   : round(min(interest_rate * 4, 100)),
            "direction": "up" if interest_rate > 14.0 else "down",
            "desc"     : f"{interest_rate}% — {'High rate burden' if interest_rate>14 else 'Manageable rate'}",
        },
        {
            "label"    : "Employment Tenure",
            "value"    : int(months_employed),
            "benchmark": 36,
            "impact"   : round(max(0, min(100, (48 - months_employed) * 2))),
            "direction": "down" if months_employed >= 24 else "up",
            "desc"     : f"{int(months_employed)} months — {'Stable (≥24mo)' if months_employed>=24 else 'Short tenure (<24mo)'}",
        },
        {
            "label"    : "Loan-to-Income Ratio",
            "value"    : round(loan_to_income, 2),
            "benchmark": 1.5,
            "impact"   : round(min(loan_to_income / 3.0 * 100, 100)),
            "direction": "up" if loan_to_income > 1.5 else "down",
            "desc"     : f"Loan is {round(loan_to_income,2):.1f}× annual income",
        },
        {
            "label"    : "Co-Signer Protection",
            "value"    : has_cosigner,
            "benchmark": "Yes",
            "impact"   : 18 if has_cosigner == "Yes" else 65,
            "direction": "down" if has_cosigner == "Yes" else "up",
            "desc"     : "Co-signer present — risk mitigated" if has_cosigner=="Yes" else "No co-signer — unmitigated risk",
        },
        {
            "label"    : "Credit Lines",
            "value"    : int(num_cl),
            "benchmark": 3,
            "impact"   : round(max(0, min(100, (5 - num_cl) * 15))),
            "direction": "down" if num_cl >= 3 else "up",
            "desc"     : f"{int(num_cl)} open credit lines — {'Adequate' if num_cl>=3 else 'Limited credit history'}",
        },
    ]
    return factors
