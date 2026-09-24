/**
 * LoanLens — Model Service Abstraction
 * Manages model specifications, 4-model benchmark matrix, and detailed hyperparameter records.
 * Designed to connect to Flask /api/models endpoints once trained.
 */

import { ModelSpecification } from '../types/index.js';
import { DEV_EVALUATED_MODELS, DEV_ROC_CURVES } from '../data/developmentMockData.js';

export class ModelService {
  private useRemoteApi: boolean;
  private apiBase: string;

  constructor() {
    this.useRemoteApi = process.env.USE_REMOTE_BACKEND === 'true';
    this.apiBase = process.env.FLASK_BACKEND_URL || '/api';
  }

  async getModels(): Promise<ModelSpecification[]> {
    if (this.useRemoteApi && typeof fetch !== 'undefined') {
      try {
        const res = await fetch(`${this.apiBase}/models`);
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn('Remote /api/models unavailable, using development model definitions');
      }
    }
    return DEV_EVALUATED_MODELS;
  }

  async getModelById(id: string): Promise<ModelSpecification | null> {
    const models = await this.getModels();
    return models.find(m => m.id === id || m.name.toLowerCase() === id.toLowerCase()) || null;
  }

  async getSelectedModel(): Promise<ModelSpecification> {
    const models = await this.getModels();
    return models.find(m => m.isSelected) || models[0];
  }

  async getModelComparison(): Promise<{ models: ModelSpecification[]; rocCurves: any }> {
    const models = await this.getModels();
    return {
      models,
      rocCurves: DEV_ROC_CURVES,
    };
  }

  async getModelContext() {
    return {
      productName: "LoanLens",
      tagline: "Loan Default Prediction & Risk Intelligence",
      domain: "Institutional Lending & Banking Credit Risk",
      problemType: "Binary Classification",
      targetColumn: "Default (0 = Paid / No Default, 1 = Default)",
      datasetName: "Kaggle Loan Default Prediction (nikhil1e9/loan-default)",
      totalRecords: 255347,
      cleanFeatures: 16,
      encodedFeatures: 24,
      split: {
        trainRatio: 0.80,
        trainRecords: 204277,
        testRatio: 0.20,
        testRecords: 51070,
        stratified: true,
      },
      classDistribution: {
        negativeCases: 225694,
        negativePct: 88.4,
        positiveCases: 29653,
        positivePct: 11.6,
      },
      preprocessing: [
        "Numerical Standardization using StandardScaler (zero mean, unit variance)",
        "Categorical One-Hot Encoding with drop_first=True for binary and multi-class columns",
        "Unique Identifier removal (LoanID dropped)",
        "Zero missing values or duplicates after initial cleaning",
      ],
      scratchAlgorithm: "Logistic Regression with Gradient Descent implemented from scratch using NumPy (Binary Cross-Entropy Loss & Analytical Gradients)",
      primarySelectionMetric: "Validation ROC-AUC & F1-Score (chosen because 88.4% raw accuracy can be trivially obtained by always predicting non-default)",
    };
  }
}

export const modelService = new ModelService();
