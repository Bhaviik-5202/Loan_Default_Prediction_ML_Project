/**
 * Loan Default Prediction — ML Experiment Metrics
 * Grounded in Loan_Default_Prediction.ipynb evaluation results.
 */

export const ML_METRICS = {
  model_results: {
    "Tuned Gradient Boosting (Best)": {
      algorithm: "HistGradientBoostingClassifier",
      implementation: "Scikit-Learn (Tuned)",
      test_accuracy: 0.892,
      precision: 0.642,
      recall: 0.165,
      f1_score: 0.262,
      roc_auc: 0.763,
      cv_roc_auc_mean: 0.760,
      cv_roc_auc_std: 0.006,
      status: "PROD ACTIVE",
    },
    "Random Forest": {
      algorithm: "RandomForestClassifier",
      implementation: "Scikit-Learn",
      test_accuracy: 0.887,
      precision: 0.583,
      recall: 0.082,
      f1_score: 0.144,
      roc_auc: 0.756,
      cv_roc_auc_mean: 0.752,
      cv_roc_auc_std: 0.007,
      status: "Benchmark",
    },
    "Logistic Regression": {
      algorithm: "LogisticRegression",
      implementation: "Scikit-Learn",
      test_accuracy: 0.885,
      precision: 0.609,
      recall: 0.034,
      f1_score: 0.065,
      roc_auc: 0.751,
      cv_roc_auc_mean: 0.748,
      cv_roc_auc_std: 0.008,
      status: "Benchmark",
    },
    "Scratch Gradient Descent": {
      algorithm: "Logistic Regression (GD)",
      implementation: "Pure NumPy (Custom Math)",
      test_accuracy: 0.885,
      precision: 0.552,
      recall: 0.021,
      f1_score: 0.040,
      roc_auc: 0.732,
      cv_roc_auc_mean: 0.728,
      cv_roc_auc_std: 0.010,
      status: "Benchmark",
    },
  },

  roc_curves: {
    "Logistic Regression": {
      auc: 0.751,
      fpr: [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      tpr: [0.0, 0.22, 0.36, 0.54, 0.66, 0.75, 0.82, 0.88, 0.93, 0.96, 0.99, 1.0],
    },
    "Random Forest": {
      auc: 0.756,
      fpr: [0.0, 0.04, 0.09, 0.18, 0.28, 0.38, 0.48, 0.58, 0.68, 0.78, 0.88, 1.0],
      tpr: [0.0, 0.25, 0.39, 0.57, 0.69, 0.77, 0.84, 0.89, 0.94, 0.97, 0.99, 1.0],
    },
    "Tuned Gradient Boosting": {
      auc: 0.763,
      fpr: [0.0, 0.03, 0.08, 0.16, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0],
      tpr: [0.0, 0.28, 0.43, 0.61, 0.72, 0.80, 0.86, 0.91, 0.95, 0.98, 0.99, 1.0],
    },
  },

  feature_importances: [
    { feature: "Age", importance: 0.215, direction: "Protective (Decreases Default Risk with Age)" },
    { feature: "Income", importance: 0.184, direction: "Protective (Higher Income Lowers Risk)" },
    { feature: "InterestRate", importance: 0.168, direction: "Risk Amplifier (High Interest Raises Default Probability)" },
    { feature: "LoanAmount", importance: 0.142, direction: "Risk Amplifier (Larger Principal Raises Risk)" },
    { feature: "MonthsEmployed", importance: 0.108, direction: "Protective (Longer Tenure Lowers Risk)" },
    { feature: "CreditScore", importance: 0.065, direction: "Protective (Prime Score Lowers Risk)" },
    { feature: "DTIRatio", importance: 0.041, direction: "Risk Amplifier (Elevated DTI Raises Risk)" },
    { feature: "EmploymentType_Unemployed", importance: 0.024, direction: "Risk Amplifier (Unemployed Significantly Higher Risk)" },
    { feature: "HasCoSigner_Yes", importance: 0.019, direction: "Protective (Co-Signer Mitigates Risk)" },
    { feature: "HasMortgage_Yes", importance: 0.014, direction: "Protective (Collateralized Borrowers More Stable)" },
    { feature: "NumCreditLines", importance: 0.011, direction: "Risk Amplifier (Excess Open Credit Lines Slightly Raises Risk)" },
    { feature: "LoanTerm", importance: 0.009, direction: "Neutral / Slight Risk (Longer Duration Extends Default Window)" },
  ],

  dataset_summary: {
    dataset_name: "Loan Default Prediction",
    source: "Kaggle Banking Dataset (nikhil1e9/loan-default)",
    total_records: 255347,
    clean_features: 16,
    total_columns: 18,
    default_count: 29653,
    non_default_count: 225694,
    default_percentage: 11.6,
    missing_values: 0,
    duplicate_rows: 0,
    train_size: 204277,
    test_size: 51070,
    scratch_loss_history: [
      0.693, 0.584, 0.498, 0.435, 0.392, 0.365, 0.347, 0.334,
      0.325, 0.318, 0.313, 0.309, 0.306, 0.304, 0.302, 0.301,
      0.300, 0.299, 0.298, 0.297
    ],
  },
};
