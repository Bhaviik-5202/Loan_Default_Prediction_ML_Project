"""
LoanML — Loan Default Risk Assessment
Frontend + placeholder scoring. Replace predict() body with the real
model pipeline (LogisticRegression + StandardScaler) when ready.
"""

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


# --------------------------------------------------------------------------
# Page routes
# --------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html", active="home")


@app.route("/assess")
def assessment_form():
    return render_template("form.html", active="assess")


@app.route("/model-info")
def model_info():
    return render_template("model-info.html", active="model")


@app.route("/data-insights")
def data_insights():
    return render_template("data-insights.html", active="insights")


@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html", active="disclaimer")


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_placeholder_risk(data):
    """Transparent placeholder scoring — replace with the real model later."""
    credit_score = _safe_float(data.get("CreditScore"), 650)
    dti_ratio = _safe_float(data.get("DTIRatio"), 0.35)
    income = _safe_float(data.get("Income"), 60000)
    loan_amount = _safe_float(data.get("LoanAmount"), 15000)
    interest_rate = _safe_float(data.get("InterestRate"), 12.5)
    months_employed = _safe_float(data.get("MonthsEmployed"), 24)

    risk_score = (
        (1 - min(credit_score, 850) / 850) * 0.40
        + min(dti_ratio, 1) * 0.25
        + min(loan_amount / max(income, 1), 1) * 0.15
        + min(interest_rate / 40, 1) * 0.10
        + (1 - min(months_employed / 120, 1)) * 0.10
    )
    probability = round(min(max(risk_score, 0), 1) * 100, 1)

    # Build factor contributions for the result page
    factors = [
        {
            "label": "Credit score",
            "value": f"{credit_score:.0f}",
            "impact": "negative" if credit_score < 670 else "positive",
            "weight": round((1 - min(credit_score, 850) / 850) * 100),
        },
        {
            "label": "Debt-to-income ratio",
            "value": f"{dti_ratio:.2f}",
            "impact": "negative" if dti_ratio > 0.36 else "positive",
            "weight": round(min(dti_ratio, 1) * 100),
        },
        {
            "label": "Annual income",
            "value": f"${income:,.0f}",
            "impact": "positive" if income >= 60000 else "negative",
            "weight": round(min(income / 200000, 1) * 100),
        },
        {
            "label": "Loan amount",
            "value": f"${loan_amount:,.0f}",
            "impact": "negative" if loan_amount > income else "positive",
            "weight": round(min(loan_amount / max(income, 1), 1) * 100),
        },
        {
            "label": "Interest rate",
            "value": f"{interest_rate:.2f}%",
            "impact": "negative" if interest_rate > 15 else "positive",
            "weight": round(min(interest_rate / 40, 1) * 100),
        },
        {
            "label": "Months employed",
            "value": f"{months_employed:.0f}",
            "impact": "positive" if months_employed >= 24 else "negative",
            "weight": round((1 - min(months_employed / 120, 1)) * 100),
        },
        {
            "label": "Employment type",
            "value": data.get("EmploymentType", "—"),
            "impact": "neutral",
            "weight": 0,
        },
        {
            "label": "Loan purpose",
            "value": data.get("LoanPurpose", "—"),
            "impact": "neutral",
            "weight": 0,
        },
    ]

    return probability, factors


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    probability, factors = compute_placeholder_risk(data)
    risk_label = "Likely to default" if probability >= 35 else "Likely to repay"

    return render_template(
        "result.html",
        probability=probability,
        risk_label=risk_label,
        factors=factors,
        data=data,
        active="assess",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint for programmatic access / future model integration."""
    data = request.get_json(silent=True) or request.form.to_dict()
    probability, factors = compute_placeholder_risk(data)
    return jsonify(
        {
            "probability": probability,
            "risk_label": "Likely to default" if probability >= 35 else "Likely to repay",
            "factors": factors,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
