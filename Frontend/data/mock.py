"""
Centralized mock data for Dashboard / Prediction History.

MOCK — not from the trained model. This generates a deterministic,
seeded set of past "predictions" so the Dashboard and History pages have
real content to render against. When a real predictions store exists
(e.g. a database logging each /api/predict call), replace the body of
these functions with real queries — every caller already expects this
exact shape, so nothing upstream needs to change.
"""

import random
from datetime import datetime, timedelta

_NAMES = [
    "A. Sharma", "R. Patel", "M. Gupta", "S. Rao", "K. Nair", "J. Chen",
    "L. Fernandez", "T. Okafor", "N. Kim", "P. Singh", "D. Costa", "E. Novak",
    "V. Reddy", "H. Yamada", "C. Moreau", "B. Osei", "F. Rossi", "G. Muller",
    "I. Petrov", "O. Jansen",
]
_PURPOSES = ["Auto", "Business", "Education", "Home", "Other"]
_EMPLOYMENT = ["Full-time", "Part-time", "Self-employed", "Unemployed"]

random.seed(42)


def _risk_level(prob):
    return "High" if prob >= 45 else "Medium" if prob >= 20 else "Low"


def get_applications(n=48):
    apps = []
    base_date = datetime(2026, 9, 23)
    for i in range(n):
        credit_score = random.randint(400, 830)
        income = random.randint(22000, 140000)
        loan_amount = random.randint(3000, 60000)
        dti = round(random.uniform(0.08, 0.72), 2)
        employment = random.choice(_EMPLOYMENT)
        months_employed = random.randint(0, 240)

        risk = (
            (1 - min(credit_score, 850) / 850) * 0.35
            + dti * 0.30
            + min(loan_amount / max(income, 1), 2) / 2 * 0.15
            + random.uniform(0, 0.15)
        )
        probability = round(min(max(risk, 0), 1) * 100, 1)

        apps.append({
            "id": f"APP-{1000 + i}",
            "name": _NAMES[i % len(_NAMES)],
            "credit_score": credit_score,
            "income": income,
            "loan_amount": loan_amount,
            "dti": dti,
            "employment": employment,
            "months_employed": months_employed,
            "purpose": random.choice(_PURPOSES),
            "probability": probability,
            "risk_level": _risk_level(probability),
            "prediction": "Likely to Default" if probability >= 45 else "Likely to Repay",
            "date": (base_date - timedelta(days=random.randint(0, 29))).strftime("%Y-%m-%d"),
        })
    apps.sort(key=lambda a: a["date"], reverse=True)
    return apps


def get_dashboard_stats(apps):
    total = len(apps)
    high = sum(1 for a in apps if a["risk_level"] == "High")
    low = sum(1 for a in apps if a["risk_level"] == "Low")
    med = total - high - low
    avg_prob = round(sum(a["probability"] for a in apps) / total, 1)
    avg_score = round(sum(a["credit_score"] for a in apps) / total)
    default_rate = round(100 * sum(1 for a in apps if a["prediction"] == "Likely to Default") / total, 1)

    return {
        "total_applications": total,
        "predictions_made": total,
        "high_risk": high,
        "low_risk": low,
        "medium_risk": med,
        "avg_probability": avg_prob,
        "avg_credit_score": avg_score,
        "default_rate": default_rate,
        "model_accuracy": 88.5,  # real figure, from Loan_Default_Prediction.ipynb
    }


def get_prediction_trend(apps, days=14):
    base_date = datetime(2026, 9, 23)
    buckets = []
    for d in range(days - 1, -1, -1):
        day = (base_date - timedelta(days=d)).strftime("%Y-%m-%d")
        day_apps = [a for a in apps if a["date"] == day]
        buckets.append({
            "date": day[5:],
            "count": len(day_apps),
            "defaults": sum(1 for a in day_apps if a["prediction"] == "Likely to Default"),
        })
    return buckets
