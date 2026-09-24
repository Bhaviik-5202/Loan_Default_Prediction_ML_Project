/**
 * LoanLens — Production Web Server & API Layer
 * Loan Default Prediction & Risk Intelligence.
 */

import path from 'path';
import { fileURLToPath } from 'url';
import express, { Request, Response, NextFunction } from 'express';
import { NAV } from './src/constants.js';
import { icon } from './src/icons.js';
import {
  getPredictions,
  getDashboardStats,
  getPredictionTrend,
} from './src/store.js';
import { predictionService } from './src/services/predictionService.js';
import { modelService } from './src/services/modelService.js';
import { analyticsService } from './src/services/analyticsService.js';
import { normalizePayload, validatePredictionPayload, predictLoanRisk } from './src/mlService.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// View Engine & Static Assets
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use('/static', express.static(path.join(__dirname, 'public')));

// Request Body Parsing
app.use(express.json({ limit: '4mb' }));
app.use(express.urlencoded({ extended: true, limit: '4mb' }));

// Template Context Injection
app.use((req: Request, res: Response, next: NextFunction) => {
  res.locals.nav = NAV;
  res.locals.icon = icon;
  res.locals.currentPath = req.path;
  next();
});

// ─── Frontend Page Routes ─────────────────────────────────────────────────────

// Page 1: Overview
app.get(['/', '/overview', '/dashboard'], async (req: Request, res: Response) => {
  const applications = getPredictions();
  const selectedModel = await modelService.getSelectedModel();
  const accuracyPct = Math.round(selectedModel.metrics.accuracy * 1000) / 10;
  const stats = getDashboardStats(applications, accuracyPct);
  const trend = getPredictionTrend(applications);
  const recent = applications.slice(0, 8);

  res.render('dashboard', { stats, trend, recent });
});

// Page 2: New Assessment Form
app.get(['/assess', '/predict'], (req: Request, res: Response) => {
  res.render('predict');
});

// Page 3: Model Context
app.get('/model/context', async (req: Request, res: Response) => {
  const context = await modelService.getModelContext();
  res.render('model_context', { context });
});

// Page 4: Model Comparison
app.get(['/model/comparison', '/model/analytics'], async (req: Request, res: Response) => {
  const { models, rocCurves } = await modelService.getModelComparison();
  res.render('model_comparison', { models, rocCurves });
});

// Page 5: Data Insights
app.get(['/data/insights', '/dataset/explorer'], async (req: Request, res: Response) => {
  const insights = await analyticsService.getDataInsights();
  res.render('data_insights', { insights });
});

// Page 6: Model Details
app.get('/model/details', async (req: Request, res: Response) => {
  const allModels = await modelService.getModels();
  const requestedId = typeof req.query.model === 'string' ? req.query.model : 'gradient-boosting';
  const currentModel = (await modelService.getModelById(requestedId)) || allModels[0];

  res.render('model_details', { allModels, currentModel });
});

// History & Simulator
app.get('/predictions', (req: Request, res: Response) => {
  const apps = getPredictions();
  res.render('history', { apps });
});

app.get('/simulator', (req: Request, res: Response) => {
  res.render('simulator');
});

// Backward compatibility alias for feature importance
app.get('/feature/importance', async (req: Request, res: Response) => {
  const model = await modelService.getSelectedModel();
  res.redirect(`/model/details?model=${model.id}`);
});

// ─── API Contracts ────────────────────────────────────────────────────────────

// POST /api/predict — Run loan default risk assessment
app.post('/api/predict', async (req: Request, res: Response) => {
  let payload = req.body;
  if (!payload || typeof payload !== 'object') {
    return res.status(400).json({ error: 'A JSON request body is required.', success: false });
  }

  payload = normalizePayload(payload);

  try {
    validatePredictionPayload(payload);
    const result = await predictionService.assessLoanRisk(payload);
    return res.json(result);
  } catch (err: any) {
    return res.status(400).json({ error: err.message || 'Assessment failed', success: false });
  }
});

// POST /api/simulate — Quick sensitivity simulation
app.post('/api/simulate', (req: Request, res: Response) => {
  let payload = req.body;
  if (!payload || typeof payload !== 'object') {
    return res.status(400).json({ error: 'A JSON request body is required.', success: false });
  }

  payload = normalizePayload(payload);

  try {
    const result = predictLoanRisk(payload);
    return res.json(result);
  } catch (err: any) {
    return res.status(400).json({ error: err.message || 'Simulation failed', success: false });
  }
});

// GET /api/models — List all evaluated model definitions
app.get('/api/models', async (req: Request, res: Response) => {
  const models = await modelService.getModels();
  return res.json(models);
});

// GET /api/models/comparison — Comparison matrix and ROC curves
app.get('/api/models/comparison', async (req: Request, res: Response) => {
  const comparison = await modelService.getModelComparison();
  return res.json(comparison);
});

// GET /api/models/:modelId — Specific model details
app.get('/api/models/:modelId', async (req: Request, res: Response) => {
  const model = await modelService.getModelById(req.params.modelId);
  if (!model) {
    return res.status(404).json({ error: 'Model not found', success: false });
  }
  return res.json(model);
});

// GET /api/data/insights — Dataset EDA and statistical distributions
app.get('/api/data/insights', async (req: Request, res: Response) => {
  const insights = await analyticsService.getDataInsights();
  return res.json(insights);
});

// GET /api/health — System health check
app.get('/api/health', (req: Request, res: Response) => {
  return res.json({
    status: 'ok',
    product: 'LoanLens',
    service: 'Loan Default Prediction & Risk Intelligence',
    version: '1.0.0',
    models_configured: 4,
    features: 16,
  });
});

// ─── Error Handlers ───────────────────────────────────────────────────────────

app.use((req: Request, res: Response) => {
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({ error: 'Resource not found', success: false });
  }
  res.status(404).render('404');
});

app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  console.error('Server error:', err);
  if (req.path.startsWith('/api/')) {
    return res.status(500).json({ error: 'Internal server error', success: false });
  }
  res.status(500).render('404');
});

// ─── Server Entry ─────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.PORT || '3000', 10);
const HOST = process.env.HOST || '0.0.0.0';

app.listen(PORT, HOST, () => {
  console.log(`LoanLens server running on http://${HOST}:${PORT}`);
});
