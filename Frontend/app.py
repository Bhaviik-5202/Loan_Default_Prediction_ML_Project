"""
LoanML — Loan Default Prediction & Credit Risk Analytics Platform.

Architecture notes (read this before adding a page):

- constants.py is the single source of truth for the sidebar. Every nav
  item lists a route and a status ("live" or "soon").
- Routes marked "live" are wired to a real view function below.
- Routes marked "soon" are auto-registered at startup (see the loop near
  the bottom) to render templates/coming_soon.html — so nothing in the
  sidebar ever 404s, and turning a placeholder into a real page later is
  just: build the template, add a real view function, flip its status.
- data/mock.py is a clearly-separated mock data layer for Dashboard and
  Prediction History. Swap its functions for real queries when a
  predictions store exists; every caller already expects the same shape.
- /api/predict is a placeholder scoring formula, not the trained model —
  see the comment on predict_api() for exactly what Week 9 replaces.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for

from constants import NAV, find_nav_item
from icons import icon_svg
from data.mock import get_applications, get_dashboard_stats, get_prediction_trend

app = Flask(__name__)
app.jinja_env.globals["icon"] = icon_svg


@app.context_processor
def inject_nav():
    return {"nav": NAV}


# ---------------------------------------------------------------- mock data (cached once)

_APPLICATIONS = get_applications()


# ---------------------------------------------------------------- live pages

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    stats = get_dashboard_stats(_APPLICATIONS)
    trend = get_prediction_trend(_APPLICATIONS)
    recent = _APPLICATIONS[:8]
    return render_template("dashboard.html", stats=stats, trend=trend, recent=recent)


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/simulator")
def simulator_page():
    return render_template("simulator.html")


@app.route("/predictions")
def history_page():
    return render_template("history.html", apps=_APPLICATIONS)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """
    Placeholder scoring formula — NOT the trained model.

    Week 9 swap-in: load the saved LogisticRegression + StandardScaler
    from Loan_Default_Prediction.ipynb here, one-hot encode the incoming
    JSON the same way df_encoded was built, scale it, and return
    model.predict_proba() in place of the `risk` formula below. The
    response shape (probability, risk_level, risk_score, prediction,
    factors, profile, recommendation) is what predict.js, simulator.js
    already expect — no frontend changes needed.
    """
    d = request.get_json(force=True) or {}

    credit_score = float(d.get("CreditScore", 650))
    dti = float(d.get("DTIRatio", 0.35))
    income = max(float(d.get("Income", 60000)), 1)
    loan_amount = float(d.get("LoanAmount", 15000))
    interest_rate = float(d.get("InterestRate", 12.5))
    months_employed = float(d.get("MonthsEmployed", 24))
    has_mortgage = d.get("HasMortgage", "No") == "Yes"
    has_cosigner = d.get("HasCoSigner", "No") == "Yes"

    loan_to_income = min(loan_amount / income, 2) / 2

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


# ---------------------------------------------------------------- auto-registered "Coming Soon" pages

def _make_soon_view(item):
    def view():
        return render_template("coming_soon.html", item=item)
    view.__name__ = f"soon_{item['key']}"
    return view


for group in NAV:
    for nav_item in group["items"]:
        if nav_item["status"] == "soon":
            app.add_url_rule(nav_item["route"], view_func=_make_soon_view(nav_item))


if __name__ == "__main__":
    app.run(debug=True)
