/**
 * LoanLens — Analytics & EDA Service Abstraction
 * Manages dataset overview, feature statistics, distributions, and correlation signals.
 * Prepares the contract for GET /api/data/insights.
 */

import { DatasetInsights } from '../types/index.js';
import { DEV_DATASET_INSIGHTS } from '../data/developmentMockData.js';

export class AnalyticsService {
  private useRemoteApi: boolean;
  private apiEndpoint: string;

  constructor() {
    this.useRemoteApi = process.env.USE_REMOTE_BACKEND === 'true';
    this.apiEndpoint = process.env.FLASK_BACKEND_URL || '/api/data/insights';
  }

  async getDataInsights(): Promise<DatasetInsights> {
    if (this.useRemoteApi && typeof fetch !== 'undefined') {
      try {
        const res = await fetch(this.apiEndpoint);
        if (res.ok) return await res.json();
      } catch (err) {
        console.warn('Remote /api/data/insights unavailable, using baseline dataset analytics');
      }
    }
    return DEV_DATASET_INSIGHTS;
  }
}

export const analyticsService = new AnalyticsService();
