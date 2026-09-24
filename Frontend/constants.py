"""
Central nav/module registry.

Every sidebar item, its route, and whether it's a real page or a
"Coming Soon" placeholder is defined here — nowhere else. To add a new
module later: add one entry below, then (when ready) create its real
template and swap status to "live". Nothing else in the app needs to
change; app.py auto-registers a coming-soon route for anything not
explicitly wired to a real view function.
"""

NAV = [
    {
        "group": "Overview",
        "items": [
            {"key": "dashboard", "label": "Dashboard", "route": "/dashboard", "icon": "grid", "status": "live"},
        ],
    },
    {
        "group": "Prediction",
        "items": [
            {"key": "predict", "label": "Predict Risk", "route": "/predict", "icon": "target", "status": "live"},
            {"key": "simulator", "label": "Risk Simulator", "route": "/simulator", "icon": "sliders", "status": "live"},
            {"key": "history", "label": "Prediction History", "route": "/predictions", "icon": "clock", "status": "live"},
        ],
    },
    {
        "group": "Analytics",
        "items": [
            {"key": "risk-analytics", "label": "Risk Analytics", "route": "/analytics/risk", "icon": "activity", "status": "soon",
             "desc": "Deeper cross-cuts of risk by segment, region, and loan purpose over time."},
            {"key": "app-analytics", "label": "Application Analytics", "route": "/analytics/applications", "icon": "bar-chart", "status": "soon",
             "desc": "Volume, approval, and funnel analytics across all applications."},
            {"key": "data-viz", "label": "Data Visualization", "route": "/analytics/visualize", "icon": "pie-chart", "status": "soon",
             "desc": "A free-form chart builder over the applications and predictions dataset."},
        ],
    },
    {
        "group": "Applications",
        "items": [
            {"key": "applications", "label": "Applications", "route": "/applications", "icon": "folder", "status": "soon",
             "desc": "Manage loan applications end-to-end, beyond individual predictions."},
            {"key": "applicants", "label": "Applicant Profiles", "route": "/applicants", "icon": "users", "status": "soon",
             "desc": "A searchable directory of applicants with their prediction history."},
        ],
    },
    {
        "group": "Model & AI",
        "items": [
            {"key": "model-perf", "label": "Model Performance", "route": "/model/performance", "icon": "cpu", "status": "soon",
             "desc": "Accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix for the live model."},
            {"key": "feature-importance", "label": "Feature Importance", "route": "/model/features", "icon": "layers", "status": "soon",
             "desc": "Ranked feature impact and direction across the full training set."},
            {"key": "model-comparison", "label": "Model Comparison", "route": "/model/comparison", "icon": "git-branch", "status": "soon",
             "desc": "Benchmark Logistic Regression against Decision Tree, Random Forest, and XGBoost."},
            {"key": "model-monitoring", "label": "Model Monitoring", "route": "/model/monitoring", "icon": "radio", "status": "soon",
             "desc": "Prediction drift, data drift, and live performance monitoring."},
        ],
    },
    {
        "group": "Data",
        "items": [
            {"key": "dataset", "label": "Dataset Explorer", "route": "/dataset", "icon": "database", "status": "soon",
             "desc": "Browse the training dataset: summary stats, distributions, and missing values."},
        ],
    },
    {
        "group": "Reports",
        "items": [
            {"key": "reports", "label": "Reports", "route": "/reports", "icon": "file-text", "status": "soon",
             "desc": "Generate prediction, applicant, analytics, and model performance reports."},
            {"key": "export", "label": "Export Center", "route": "/reports/export", "icon": "download", "status": "soon",
             "desc": "Bulk export applications, predictions, and reports in CSV or PDF."},
        ],
    },
    {
        "group": "System",
        "items": [
            {"key": "notifications", "label": "Notifications", "route": "/notifications", "icon": "bell", "status": "soon",
             "desc": "High-risk alerts, model updates, and system notifications in one place."},
            {"key": "activity", "label": "Activity Log", "route": "/activity", "icon": "list", "status": "soon",
             "desc": "A timeline of predictions, exports, and configuration changes."},
            {"key": "settings", "label": "Settings", "route": "/settings", "icon": "settings", "status": "soon",
             "desc": "Appearance, notification, and prediction preferences."},
        ],
    },
    {
        "group": "Project",
        "items": [
            {"key": "docs", "label": "Documentation", "route": "/documentation", "icon": "book", "status": "soon",
             "desc": "How prediction works, input features, risk levels, and system architecture."},
            {"key": "about", "label": "About Project", "route": "/about", "icon": "info", "status": "soon",
             "desc": "Project objective, technology stack, and future scope."},
            {"key": "help", "label": "Help / FAQ", "route": "/help", "icon": "help-circle", "status": "soon",
             "desc": "Answers to common questions about predictions and risk scoring."},
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
