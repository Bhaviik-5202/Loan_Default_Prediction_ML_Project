"""
Central navigation registry — Loan Default Prediction.
Clean Production Navigation Structure.
"""

NAV = [
    {
        "group": "Main",
        "items": [
            {"key": "dashboard",  "label": "Dashboard",          "route": "/dashboard",   "icon": "grid",     "status": "live"},
            {"key": "predict",    "label": "Predict Risk",       "route": "/predict",     "icon": "target",   "status": "live"},
            {"key": "simulator",  "label": "Risk Simulator",     "route": "/simulator",   "icon": "sliders",  "status": "live"},
            {"key": "history",    "label": "Prediction History", "route": "/predictions", "icon": "clock",    "status": "live"},
        ],
    },
    {
        "group": "Analytics & Models",
        "items": [
            {"key": "model-analytics",   "label": "Model Performance",     "route": "/model/analytics",    "icon": "cpu",       "status": "live"},
            {"key": "feature-importance", "label": "Feature Importance",    "route": "/feature/importance", "icon": "bar-chart",  "status": "live"},
            {"key": "dataset-explorer",  "label": "Dataset Explorer (EDA)", "route": "/dataset/explorer",   "icon": "database",   "status": "live"},
        ],
    },
]


def find_nav_item(route):
    for group in NAV:
        for item in group["items"]:
            if item["route"] == route:
                return item
    return None


def flat_items():
    for group in NAV:
        for item in group["items"]:
            yield item
