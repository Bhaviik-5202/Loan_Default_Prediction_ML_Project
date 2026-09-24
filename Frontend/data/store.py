"""Persistent, local prediction history for the Flask application.

The application deliberately starts with no records.  Entries are written only
after a successful real-model prediction, so the dashboard never presents
seeded or fabricated portfolio data as live banking information.
"""

import os
import sqlite3
from datetime import datetime, timezone


_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.environ.get("LOANML_DATABASE", os.path.join(_DATA_DIR, "predictions.db"))


def _connection():
    connection = sqlite3.connect(_DB_PATH)
    connection.row_factory = sqlite3.Row
    # Enable WAL mode to improve concurrency and prevent 'database is locked' errors
    connection.execute("PRAGMA journal_mode=WAL;")
    return connection


def initialize_store():
    """Create the minimal prediction-history table if it does not exist."""
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applicant_name TEXT NOT NULL,
                credit_score INTEGER NOT NULL,
                income REAL NOT NULL,
                loan_amount REAL NOT NULL,
                dti REAL NOT NULL,
                employment TEXT NOT NULL,
                months_employed INTEGER NOT NULL,
                purpose TEXT NOT NULL,
                probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def record_prediction(payload: dict, result: dict) -> None:
    """Persist one successful prediction from the real inference endpoint."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    applicant_name = str(payload.get("Name", "Applicant")).strip() or "Applicant"
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                applicant_name, credit_score, income, loan_amount, dti,
                employment, months_employed, purpose, probability,
                risk_level, prediction, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                applicant_name[:120], int(float(payload.get("CreditScore", 650))),
                float(payload.get("Income", 60000)), float(payload.get("LoanAmount", 15000)),
                float(payload.get("DTIRatio", 0.35)), str(payload.get("EmploymentType", "Full-time")),
                int(float(payload.get("MonthsEmployed", 24))), str(payload.get("LoanPurpose", "Other")),
                float(result["probability"]), result["risk_level"], result["prediction"], timestamp,
            ),
        )


def get_predictions(limit=None) -> list[dict]:
    """Return saved prediction records, newest first, in template-ready form."""
    query = "SELECT * FROM predictions ORDER BY id DESC"
    params = ()
    if limit:
        query += " LIMIT ?"
        params = (int(limit),)
    with _connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        {
            "id": f"APP-{row['id']:06d}", "name": row["applicant_name"],
            "credit_score": row["credit_score"], "income": row["income"],
            "loan_amount": row["loan_amount"], "dti": row["dti"],
            "employment": row["employment"], "months_employed": row["months_employed"],
            "purpose": row["purpose"], "probability": row["probability"],
            "risk_level": row["risk_level"], "prediction": row["prediction"],
            "date": row["created_at"][:10],
        }
        for row in rows
    ]


def get_dashboard_stats(applications: list[dict], model_accuracy: float | None) -> dict:
    """Compute dashboard values from stored predictions only."""
    total = len(applications)
    counts = {level: sum(a["risk_level"] == level for a in applications) for level in ("Low", "Medium", "High")}
    return {
        "total_applications": total,
        "high_risk": counts["High"], "low_risk": counts["Low"], "medium_risk": counts["Medium"],
        "avg_probability": round(sum(a["probability"] for a in applications) / total, 1) if total else 0,
        "avg_credit_score": round(sum(a["credit_score"] for a in applications) / total) if total else 0,
        "default_rate": round(100 * counts["High"] / total, 1) if total else 0,
        "model_accuracy": model_accuracy,
    }


def get_prediction_trend(applications: list[dict], days=14) -> list[dict]:
    """Build a complete recent-date series for a chart, including zero days."""
    from datetime import timedelta

    today = datetime.now(timezone.utc).date()
    counts = {a["date"]: 0 for a in applications}
    for application in applications:
        counts[application["date"]] = counts.get(application["date"], 0) + 1
    return [
        {"date": (today - timedelta(days=offset)).strftime("%b %d"),
         "count": counts.get((today - timedelta(days=offset)).isoformat(), 0)}
        for offset in range(days - 1, -1, -1)
    ]
