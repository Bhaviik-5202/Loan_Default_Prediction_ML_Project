/**
 * LoanLens — Analytics & EDA Service Abstraction
 * Manages dataset overview, feature statistics, distributions, and correlation signals.
 * Prepares the contract for GET /api/data/insights.
 */

import { DatasetInsights } from '../types/index.js';
import { DEV_DATASET_INSIGHTS } from '../data/developmentMockData.js';

export class AnalyticsService {
  private get useRemoteApi(): boolean {
    return process.env.USE_REMOTE_BACKEND !== 'false';
  }

  private get apiEndpoint(): string {
    const rawUrl = (process.env.FLASK_BACKEND_URL || 'http://127.0.0.1:5001').replace(/\/+$/, '');
    return rawUrl.endsWith('/api/data/insights') ? rawUrl : `${rawUrl}/api/data/insights`;
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
