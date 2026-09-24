/**
 * Persistent in-memory prediction history for the web application.
 * Starts empty by default. Populated as actual predictions are performed.
 */

export interface PredictionRecord {
  id: string;
  name: string;
  credit_score: number;
  income: number;
  loan_amount: number;
  dti: number;
  employment: string;
  months_employed: number;
  purpose: string;
  probability: number;
  risk_level: 'Low' | 'Medium' | 'High';
  prediction: string;
  date: string;
}

let nextId = 1;
const predictions: PredictionRecord[] = [];

export function recordPrediction(payload: Record<string, any>, result: Record<string, any>): PredictionRecord {
  const timestamp = new Date().toISOString();
  const rawName = String(payload.Name || 'Applicant').trim();
  const applicantName = (rawName || 'Applicant').slice(0, 120);

  const idFormatted = `APP-${String(nextId++).padStart(6, '0')}`;
  const record: PredictionRecord = {
    id: idFormatted,
    name: applicantName,
    credit_score: parseInt(String(payload.CreditScore || 650), 10),
    income: parseFloat(String(payload.Income || 60000)),
    loan_amount: parseFloat(String(payload.LoanAmount || 15000)),
    dti: parseFloat(String(payload.DTIRatio || 0.35)),
    employment: String(payload.EmploymentType || 'Full-time'),
    months_employed: parseInt(String(payload.MonthsEmployed || 24), 10),
    purpose: String(payload.LoanPurpose || 'Other'),
    probability: parseFloat(String(result.probability)),
    risk_level: result.risk_level as 'Low' | 'Medium' | 'High',
    prediction: String(result.prediction),
    date: timestamp.slice(0, 10),
  };

  // Add to front so newest is first
  predictions.unshift(record);
  return record;
}

export function getPredictions(limit?: number): PredictionRecord[] {
  if (limit && limit > 0) {
    return predictions.slice(0, limit);
  }
  return [...predictions];
}

export function getDashboardStats(applications: PredictionRecord[], modelAccuracy: number | null) {
  const total = applications.length;
  const counts = {
    High: applications.filter(a => a.risk_level === 'High').length,
    Medium: applications.filter(a => a.risk_level === 'Medium').length,
    Low: applications.filter(a => a.risk_level === 'Low').length,
  };

  return {
    total_applications: total,
    high_risk: counts.High,
    medium_risk: counts.Medium,
    low_risk: counts.Low,
    avg_probability: total ? Math.round((applications.reduce((s, a) => s + a.probability, 0) / total) * 10) / 10 : 0,
    avg_credit_score: total ? Math.round(applications.reduce((s, a) => s + a.credit_score, 0) / total) : 0,
    default_rate: total ? Math.round((100 * counts.High / total) * 10) / 10 : 0,
    model_accuracy: modelAccuracy,
  };
}

export function getPredictionTrend(applications: PredictionRecord[], days = 14) {
  const today = new Date();
  const counts: Record<string, number> = {};

  for (const app of applications) {
    counts[app.date] = (counts[app.date] || 0) + 1;
  }

  const trend: { date: string; count: number }[] = [];
  for (let offset = days - 1; offset >= 0; offset--) {
    const d = new Date(today);
    d.setUTCDate(today.getUTCDate() - offset);
    const isoDate = d.toISOString().slice(0, 10);
    const displayDate = d.toLocaleDateString('en-US', { month: 'short', day: '2-digit', timeZone: 'UTC' });
    trend.push({
      date: displayDate,
      count: counts[isoDate] || 0,
    });
  }

  return trend;
}
