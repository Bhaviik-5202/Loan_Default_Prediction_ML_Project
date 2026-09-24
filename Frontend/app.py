"""
Loan Default Prediction — Production Flask Application
======================================================
Institutional Credit Risk Scoring Engine with Real ML Joblib Inference.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for

from constants import NAV
from icons import icon_svg
from data.store import initialize_store, record_prediction, get_predictions, get_dashboard_stats, get_prediction_trend
from ml_service import predict_loan_risk, get_ml_metrics, EDUCATION_VALS, EMPLOYMENT_VALS, MARITAL_VALS, PURPOSE_VALS, YES_NO_VALS

# ─── App Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4 MB max

# Production Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("loanml")

app.jinja_env.globals["icon"] = icon_svg
initialize_store()

_REQUIRED_PREDICTION_FIELDS = {
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines",
    "InterestRate", "LoanTerm", "DTIRatio", "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner",
}


def _normalize_payload(payload: dict) -> dict:
    """Normalize payload keys to match _REQUIRED_PREDICTION_FIELDS (Title Case)."""
    if not isinstance(payload, dict):
        return payload

    # Mapping of common lowercase/snake_case names to TitleCase keys
    norm_map = {
        "age": "Age", "income": "Income", "loanamount": "LoanAmount",
        "creditscore": "CreditScore", "monthsemployed": "MonthsEmployed",
        "numcreditlines": "NumCreditLines", "interestrate": "InterestRate",
        "loanterm": "LoanTerm", "dtiratio": "DTIRatio", "education": "Education",
        "employmenttype": "EmploymentType", "maritalstatus": "MaritalStatus",
        "hasmortgage": "HasMortgage", "hasdependents": "HasDependents",
        "loanpurpose": "LoanPurpose", "hascosigner": "HasCoSigner"
    }

    normalized = {}
    for k, v in payload.items():
        low_k = k.lower().replace("_", "")
        if low_k in norm_map:
            normalized[norm_map[low_k]] = v
        else:
            normalized[k] = v
    return normalized


def _validate_prediction_payload(payload):
    """Reject incomplete or out-of-range form submissions before inference."""
    missing = sorted(field for field in _REQUIRED_PREDICTION_FIELDS if field not in payload)
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    # Numeric range validation
    limits = {
        "Age": (18, 100), "Income": (0, 10_000_000), "LoanAmount": (0, 10_000_000),
        "CreditScore": (300, 850), "MonthsEmployed": (0, 600), "NumCreditLines": (0, 100),
        "InterestRate": (0, 100), "LoanTerm": (1, 600), "DTIRatio": (0, 1),
    }
    for field, (minimum, maximum) in limits.items():
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be numeric.") from None
        if not minimum <= value <= maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum}.")

    # Categorical value validation
    categorical_limits = {
        "Education": EDUCATION_VALS,
        "EmploymentType": EMPLOYMENT_VALS,
        "MaritalStatus": MARITAL_VALS,
        "LoanPurpose": PURPOSE_VALS,
        "HasMortgage": YES_NO_VALS,
        "HasDependents": YES_NO_VALS,
        "HasCoSigner": YES_NO_VALS,
    }
    for field, allowed in categorical_limits.items():
        value = payload.get(field)
        if value not in allowed:
            raise ValueError(f"Invalid value '{value}' for field '{field}'. Allowed values: {', '.join(allowed)}")

@app.context_processor
def inject_nav():
    return {"nav": NAV}

# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    applications = get_predictions()
    metrics = get_ml_metrics()
    best_model = metrics.get("model_results", {}).get("Tuned Gradient Boosting (Best)", {})
    accuracy = best_model.get("test_accuracy")
    stats = get_dashboard_stats(applications, round(accuracy * 100, 1) if accuracy is not None else None)
    trend = get_prediction_trend(applications)
    recent = applications[:8]
    return render_template("dashboard.html", stats=stats, trend=trend, recent=recent)


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/simulator")
def simulator_page():
    return render_template("simulator.html")


@app.route("/predictions")
def history_page():
    return render_template("history.html", apps=get_predictions())


@app.route("/model/analytics")
def model_analytics_page():
    metrics = get_ml_metrics()
    return render_template("model_analytics.html", metrics=metrics)


@app.route("/feature/importance")
def feature_importance_page():
    metrics = get_ml_metrics()
    return render_template("feature_importance.html", metrics=metrics)


@app.route("/dataset/explorer")
def dataset_explorer_page():
    metrics = get_ml_metrics()
    return render_template("dataset_explorer.html", metrics=metrics)


# ─── REST API Routes ──────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict_api():
    """
    Real ML Inference API — Loan Default Prediction.
    Pipeline: Normalization → Validation → DataFrame → Encoding → Scaler → Model → JSON.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required.", "success": False}), 400

    # Normalize keys to ensure consistency (e.g., 'income' -> 'Income')
    payload = _normalize_payload(payload)

    logger.info(f"POST /api/predict request received: {list(payload.keys())}")
    try:
        _validate_prediction_payload(payload)
        result = predict_loan_risk(payload)

        # Save only actual model outputs. Simulation requests intentionally do not persist.
        try:
            record_prediction(payload, result)
        except Exception as log_err:
            logger.warning(f"Could not append to application log: {log_err}")

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e), "success": False}), 400
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {str(e)}", "success": False}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate_api():
    """Risk Simulator API — live pipeline."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required.", "success": False}), 400

    # Normalize keys to ensure consistency
    payload = _normalize_payload(payload)

    try:
        result = predict_loan_risk(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400


@app.route("/api/metrics/ml", methods=["GET"])
def ml_metrics_api():
    """Returns full model metrics JSON."""
    return jsonify(get_ml_metrics())


@app.route("/api/health", methods=["GET"])
def health_api():
    """Production health check endpoint."""
    from ml_service import _MODEL, _SCALER, _FEATURE_COLUMNS
    return jsonify({
        "status"       : "ok",
        "model_loaded" : _MODEL is not None,
        "model_name"   : type(_MODEL).__name__ if _MODEL else None,
        "scaler_loaded": _SCALER is not None,
        "features"     : len(_FEATURE_COLUMNS) if _FEATURE_COLUMNS else 0,
        "version"      : "1.0.0"
    })


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Resource not found", "success": False}), 404
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("Server error")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error", "success": False}), 500
    # For non-API requests, we could return a custom 500.html, but sticking to JSON for API
    return jsonify({"error": "Internal server error", "success": False}), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port       = int(os.environ.get("PORT", 5000))
    host       = os.environ.get("HOST", "127.0.0.1")
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info(f"Starting Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug_mode, use_reloader=False)
