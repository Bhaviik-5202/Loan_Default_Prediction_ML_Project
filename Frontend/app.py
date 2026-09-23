"""
Week 8 — Frontend only.

/api/predict below is a transparent placeholder scoring rule, NOT the
trained model. It exists so the wizard's animated analysis -> result flow
is fully clickable end-to-end right now.

Week 9 swap-in: load the saved LogisticRegression + StandardScaler from
Loan_Default_Prediction.ipynb inside predict_api(), one-hot encode the
incoming JSON the same way df_encoded was built, scale it, and return
model.predict_proba() in place of the risk_score formula below. The
response shape (probability, risk_level, risk_score, prediction, factors,
profile, recommendation) is what predict.js already expects, so the
frontend needs no changes.
"""

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", active="home")


@app.route("/predict")
def predict_page():
    return render_template("predict.html", active="predict")


@app.route("/api/predict", methods=["POST"])
def predict_api():
    d = request.get_json(force=True) or {}

    credit_score = float(d.get("CreditScore", 650))
    dti = float(d.get("DTIRatio", 0.35))
    income = max(float(d.get("Income", 60000)), 1)
    loan_amount = float(d.get("LoanAmount", 15000))
    interest_rate = float(d.get("InterestRate", 12.5))
    months_employed = float(d.get("MonthsEmployed", 24))
    has_mortgage = d.get("HasMortgage", "No") == "Yes"
    has_cosigner = d.get("HasCoSigner", "No") == "Yes"

    loan_to_income = min(loan_amount / income, 2) / 2  # 0..1

    # --- placeholder risk formula, replace with model.predict_proba() in Week 9 ---
    risk = (
        (1 - min(credit_score, 850) / 850) * 0.35
        + dti * 0.30
        + loan_to_income * 0.15
        + min(interest_rate / 40, 1) * 0.10
        + (1 - min(months_employed / 60, 1)) * 0.10
    )
    probability = round(min(max(risk, 0), 1) * 100, 1)
    risk_score = round(100 - probability)
    risk_level = "High" if probability >= 45 else "Medium" if probability >= 20 else "Low"
    prediction = "Likely to Default" if probability >= 45 else "Likely to Repay"

    factors = [
        {"label": "Debt-to-Income Ratio", "impact": round(dti * 100), "direction": "up" if dti > 0.35 else "down"},
        {"label": "Credit Score", "impact": round((credit_score - 300) / 550 * 100), "direction": "down" if credit_score >= 650 else "up"},
        {"label": "Employment Stability", "impact": round(min(months_employed / 60, 1) * 100), "direction": "down" if months_employed >= 24 else "up"},
        {"label": "Loan-to-Income Ratio", "impact": round(loan_to_income * 100), "direction": "up" if loan_to_income > 0.4 else "down"},
    ]

    profile = {
        "Credit Health": round((credit_score - 300) / 550 * 100),
        "Fin. Stability": round((1 - dti) * 100),
        "Repay. History": round(60 + (15 if has_cosigner else 0) + (15 if has_mortgage else 0)),
        "Employment": round(min(months_employed / 60, 1) * 100),
        "Debt Burden": round((1 - loan_to_income) * 100),
        "Loan Risk": round(max(0, 100 - interest_rate * 4)),
    }

    if risk_level == "High":
        action, points = "Review Required", [
            "Consider a lower loan amount relative to income",
            "Request additional income documentation",
            "Re-evaluate the applicant's debt-to-income ratio",
        ]
    elif risk_level == "Medium":
        action, points = "Additional Review Suggested", [
            "Verify employment tenure and income stability",
            "Consider requiring a co-signer if not already present",
        ]
    else:
        action, points = "Standard Approval Path", [
            "Profile is consistent with low historical default rates",
            "No additional documentation flagged by the model",
        ]

    return jsonify({
        "probability": probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": prediction,
        "factors": factors,
        "profile": profile,
        "recommendation": {"action": action, "points": points},
    })


if __name__ == "__main__":
    app.run(debug=True)
