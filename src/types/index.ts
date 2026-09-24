/**
 * LoanLens — Domain Type Definitions & API Contracts
 * Strictly focused on Loan Default Prediction & Risk Intelligence.
 */

export interface BorrowerAssessmentInput {
  Name?: string;
  Age: number;
  Income: number;
  LoanAmount: number;
  CreditScore: number;
  MonthsEmployed: number;
  NumCreditLines: number;
  InterestRate: number;
  LoanTerm: number;
  DTIRatio: number;
  Education: "High School" | "Bachelor's" | "Master's" | "PhD";
  EmploymentType: "Full-time" | "Part-time" | "Self-employed" | "Unemployed";
  MaritalStatus: "Single" | "Married" | "Divorced";
  HasMortgage: "Yes" | "No";
  HasDependents: "Yes" | "No";
  LoanPurpose: "Auto" | "Business" | "Education" | "Home" | "Other";
  HasCoSigner: "Yes" | "No";
}

export interface RiskFactor {
  label: string;
  value: string | number;
  benchmark: string | number;
  impact: number; // 0 - 100
  direction: "up" | "down";
  desc: string;
}

export interface BorrowerProfileRadar {
  "Credit Health": number;
  "Fin. Stability": number;
  "Repay. History": number;
  "Employment": number;
  "Debt Burden": number;
  "Loan Risk": number;
}

export interface UnderwritingRecommendation {
  action: string;
  points: string[];
}

export interface PredictionResult {
  success: boolean;
  prediction: "Likely to Repay" | "Moderate Risk" | "Likely to Default";
  label: "Default" | "No Default";
  risk_level: "Low" | "Medium" | "High";
  risk_score: number; // 0 - 100
  probability: number; // 0.0 - 100.0%
  confidence: number; // 0.0 - 1.0
  model_used: string;
  features_evaluated: number;
  factors: RiskFactor[];
  profile: BorrowerProfileRadar;
  recommendation: UnderwritingRecommendation;
  timestamp: string;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  cv_roc_auc_mean: number;
  cv_roc_auc_std: number;
  training_time?: string;
}

export interface ModelSpecification {
  id: string;
  name: string;
  family: string;
  algorithm: string;
  library: string;
  trainingDate: string;
  featureCount: number;
  status: "PROD ACTIVE" | "Benchmark" | "Scratch Math";
  isSelected: boolean;
  metrics: ModelMetrics;
  hyperparameters: Record<string, string | number | boolean>;
  description: string;
  confusionMatrix: {
    trueNegative: number;
    falsePositive: number;
    falseNegative: number;
    truePositive: number;
  };
  featureImportance: Array<{
    feature: string;
    importance: number;
    direction?: string;
  }>;
}

export interface DatasetInsights {
  dataset_name: string;
  source: string;
  total_records: number;
  clean_features: number;
  total_columns: number;
  default_count: number;
  non_default_count: number;
  default_percentage: number;
  missing_values: number;
  duplicate_rows: number;
  train_size: number;
  test_size: number;
  categorical_distributions: Record<string, Record<string, number>>;
  numerical_summaries: Record<string, { min: number; max: number; mean: number; median: number }>;
}
