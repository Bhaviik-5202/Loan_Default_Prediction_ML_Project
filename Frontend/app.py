"""
Week 8 — Frontend only.

This app serves the full site so the frontend can be clicked through
end-to-end: Home, Predict form, Result, Model Info, Data Insights, and
Disclaimer. The /predict route below does NOT use the trained model yet —
it applies a simple, transparent placeholder rule (credit score + DTI
ratio) purely so the result page has something to display.

Week 9 will replace the body of predict() with the real pipeline:
load the saved LogisticRegression + StandardScaler from
Loan_Default_Prediction.ipynb, one-hot encode the incoming form the
same way df_encoded was built, scale it, and call model.predict_proba().
"""

from flask import Flask, render_template, request

app = Flask(__name__)


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


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    credit_score = float(data.get("CreditScore", 650))
    dti_ratio = float(data.get("DTIRatio", 0.35))
    income = float(data.get("Income", 60000))
    loan_amount = float(data.get("LoanAmount", 15000))

    # --- Placeholder scoring only, replace with the real model in Week 9 ---
    risk_score = (
        (1 - min(credit_score, 850) / 850) * 0.5
        + dti_ratio * 0.4
        + min(loan_amount / max(income, 1), 1) * 0.1
    )
    probability = round(min(max(risk_score, 0), 1) * 100, 1)
    risk_label = "Likely to default" if probability >= 35 else "Likely to repay"

    factors = [
        {"label": "Credit score", "value": data.get("CreditScore")},
        {"label": "Debt-to-income ratio", "value": data.get("DTIRatio")},
        {"label": "Annual income", "value": f"${income:,.0f}"},
        {"label": "Loan amount", "value": f"${loan_amount:,.0f}"},
        {"label": "Employment type", "value": data.get("EmploymentType")},
        {"label": "Loan purpose", "value": data.get("LoanPurpose")},
    ]

    return render_template(
        "result.html",
        probability=probability,
        risk_label=risk_label,
        factors=factors,
        active="assess",
    )


if __name__ == "__main__":
    app.run(debug=True)
