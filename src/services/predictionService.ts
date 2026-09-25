/**
 * LoanLens — Prediction Service Abstraction
 * Manages loan default inference and risk score estimation.
 * Designed to connect to the real Flask/Python backend via /api/predict without UI redesign.
 */

import { BorrowerAssessmentInput, PredictionResult } from '../types/index.js';
import { predictLoanRisk as runCalibratedInference } from '../mlService.js';
import { recordPrediction } from '../store.js';

export class PredictionService {
  private get useRemoteApi(): boolean {
    return process.env.USE_REMOTE_BACKEND !== 'false';
  }

  private get apiEndpoint(): string {
    const rawUrl = (process.env.FLASK_BACKEND_URL || 'http://127.0.0.1:5001').replace(/\/+$/, '');
    return rawUrl.endsWith('/api/predict') ? rawUrl : `${rawUrl}/api/predict`;
  }

  /**
   * Run loan default assessment on borrower profile.
   * If remote Flask backend is configured, calls POST /api/predict.
   * Otherwise runs isolated calibrated inference engine with full factor attribution.
   */
  async assessLoanRisk(input: BorrowerAssessmentInput): Promise<PredictionResult> {
    if (this.useRemoteApi && typeof fetch !== 'undefined') {
      try {
        const response = await fetch(this.apiEndpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(input),
        });

        if (!response.ok) {
          const errBody = await response.json().catch(() => ({}));
          const message = errBody.message || errBody.error || `Remote inference error: ${response.statusText}`;
          const err = new Error(message);
          (err as any).statusCode = response.status;
          (err as any).details = errBody.details || {};
          throw err;
        }

        const data = await response.json();
        const normalized = this.normalizeResult(data, input);

        // Persist to history store
        try {
          recordPrediction(input, normalized);
        } catch (e) {
          // ignore recording error
        }

        return normalized;
      } catch (err: any) {
        if (err.statusCode && err.statusCode < 500) {
          throw err;
        }
        console.warn('Remote backend connection error in predictionService:', err.message);
        throw err;
      }
    }

    // Local calibrated model inference engine fallback
    const rawResult = runCalibratedInference(input);
    const normalized = this.normalizeResult(rawResult, input);

    // Persist to history store
    try {
      recordPrediction(input, normalized);
    } catch (e) {
      console.warn('Could not record prediction:', e);
    }

    return normalized;
  }

  private normalizeResult(data: any, input: BorrowerAssessmentInput): PredictionResult {
    const rawProb = typeof data.probability === 'number' ? data.probability : 0.25;
    // Format probability as decimal 0.XX as required
    const probability = rawProb > 1.0 ? Math.round(rawProb * 100) / 10000 : Math.round(rawProb * 10000) / 10000;
    const riskScore = typeof data.risk_score === 'number' ? data.risk_score : Math.round((1 - probability) * 100);
    const riskLevel = data.risk_level || (probability >= 0.50 ? 'High' : probability >= 0.28 ? 'Medium' : 'Low');
    const label = data.label || (probability >= 0.50 ? 'Default' : 'No Default');
    const confidence = typeof data.confidence === 'number' ? data.confidence : Math.round(Math.max(probability, 1 - probability) * 10000) / 10000;
    const prediction = data.prediction !== undefined ? data.prediction : (label === 'Default' ? 1 : 0);

    return {
      success: true,
      prediction,
      label,
      probability,
      confidence,
      risk_score: riskScore,
      risk_level: riskLevel,
      model_used: data.model_used || "Logistic Regression",
      features_evaluated: data.features_evaluated || 24,
      factors: data.factors || [],
      profile: data.profile || {
        "Credit Health": 70,
        "Fin. Stability": 65,
        "Repay. History": 75,
        "Employment": 60,
        "Debt Burden": 80,
        "Loan Risk": 65,
      },
      recommendation: data.recommendation || {
        action: riskLevel === 'High' ? "Predicted Default Risk: High (Model Classification: Default)" : riskLevel === 'Medium' ? "Predicted Default Risk: Moderate (Model Classification: Borderline No Default)" : "Predicted Default Risk: Low (Model Classification: No Default)",
        points: [
          "Model evaluation completed based on standardized credit and financial attributes.",
          "Notice: This prediction is an ML model output for demonstration purposes and does not constitute financial advice."
        ],
      },
      timestamp: new Date().toISOString(),
    };
  }
}

export const predictionService = new PredictionService();
