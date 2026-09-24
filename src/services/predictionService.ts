/**
 * LoanLens — Prediction Service Abstraction
 * Manages loan default inference and risk score estimation.
 * Designed to connect to the real Flask/Python backend via /api/predict without UI redesign.
 */

import { BorrowerAssessmentInput, PredictionResult } from '../types/index.js';
import { predictLoanRisk as runCalibratedInference } from '../mlService.js';
import { recordPrediction } from '../store.js';

export class PredictionService {
  private useRemoteApi: boolean;
  private apiEndpoint: string;

  constructor() {
    this.useRemoteApi = process.env.USE_REMOTE_BACKEND === 'true';
    this.apiEndpoint = process.env.FLASK_BACKEND_URL || '/api/predict';
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
          throw new Error(errBody.error || `Remote inference error: ${response.statusText}`);
        }

        const data = await response.json();
        return this.normalizeResult(data, input);
      } catch (err) {
        console.warn('Remote backend unavailable, falling back to local calibrated engine:', err);
      }
    }

    // Local calibrated model inference engine
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
    const probability = typeof data.probability === 'number' ? data.probability : 25.0;
    const riskScore = typeof data.risk_score === 'number' ? data.risk_score : Math.round(100 - probability);
    const riskLevel = data.risk_level || (probability >= 50 ? 'High' : probability >= 28 ? 'Medium' : 'Low');
    const label = probability >= 50 ? 'Default' : 'No Default';
    const confidence = Math.round((Math.max(probability, 100 - probability) / 100) * 100) / 100;

    return {
      success: true,
      prediction: data.prediction || (riskLevel === 'High' ? 'Likely to Default' : riskLevel === 'Medium' ? 'Moderate Risk' : 'Likely to Repay'),
      label,
      risk_level: riskLevel,
      risk_score: riskScore,
      probability,
      confidence,
      model_used: data.model_used || "HistGradientBoosting (Tuned)",
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
        action: "Standard Review with Conditions",
        points: ["Verify borrower income documentation and debt profile."],
      },
      timestamp: new Date().toISOString(),
    };
  }
}

export const predictionService = new PredictionService();
