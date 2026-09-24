/**
 * LoanLens — Production Web Server & API Layer
 * Loan Default Prediction & Risk Intelligence.
 */

import path from 'path';
import { fileURLToPath } from 'url';
import { spawn, ChildProcess } from 'child_process';
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

// Ensure remote backend environment configuration is active as required
process.env.USE_REMOTE_BACKEND = 'true';
process.env.FLASK_BACKEND_URL = process.env.FLASK_BACKEND_URL && process.env.FLASK_BACKEND_URL.trim() !== ''
  ? process.env.FLASK_BACKEND_URL
  : 'http://127.0.0.1:5001';

// Manage Python Flask Inference Engine Process
let flaskProcess: ChildProcess | null = null;

function ensureFlaskBackend(): void {
  const flaskUrl = (process.env.FLASK_BACKEND_URL || 'http://127.0.0.1:5001').replace(/\/+$/, '');
  fetch(`${flaskUrl}/api/health`)
    .then((res) => {
      if (res.ok) {
        console.log(`[Flask Backend] Active and healthy at ${flaskUrl}`);
      } else {
        throw new Error(`Health status: ${res.status}`);
      }
    })
    .catch(() => {
      console.log(`[Flask Backend] Spawning Python Flask service on ${flaskUrl}...`);
      const childEnv: Record<string, string | undefined> = { ...process.env, FLASK_PORT: '5001', FLASK_HOST: '127.0.0.1' };
      // Delete outer PORT from child so Flask doesn't try to bind to the outer Node port
      delete childEnv['PORT'];

      flaskProcess = spawn('python3', [path.join(__dirname, 'flask_backend', 'app.py')], {
        env: childEnv,
        stdio: ['ignore', 'inherit', 'inherit'],
      });

      flaskProcess.on('error', (err) => {
        console.error('[Flask Backend] Spawning error:', err);
      });

      flaskProcess.on('exit', (code, signal) => {
        if (code !== 0 && code !== null) {
          console.warn(`[Flask Backend] Exited with code ${code} signal ${signal}`);
        }
      });
    });
}

ensureFlaskBackend();

process.on('SIGINT', () => {
  flaskProcess?.kill();
  process.exit(0);
});
process.on('SIGTERM', () => {
  flaskProcess?.kill();
  process.exit(0);
});

const app = express();

// View Engine & Static Assets
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use('/static', express.static(path.join(__dirname, 'public')));

// Request Body Parsing
app.use(express.json({ limit: '4mb' }));
app.use(express.urlencoded({ extended: true, limit: '4mb' }));

// Handle JSON parse errors gracefully as 400 Bad Request
app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  if (err instanceof SyntaxError && 'body' in err) {
    return res.status(400).json({
      error: true,
      message: 'Malformed request. Valid JSON payload required.',
      details: {}
    });
  }
  next(err);
});

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
  const requestedId = typeof req.query.model === 'string' ? req.query.model : 'logistic-regression';
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
    return res.status(400).json({
      error: true,
      message: 'A JSON request body is required.',
      details: {}
    });
  }

  payload = normalizePayload(payload);

  try {
    const result = await predictionService.assessLoanRisk(payload);
    return res.json(result);
  } catch (err: any) {
    const isValidationErr = err.statusCode === 422 || (err.message && (
      err.message.includes('Missing') ||
      err.message.includes('must be') ||
      err.message.includes('Invalid') ||
      err.message.includes('outside acceptable')
    ));
    const statusCode = err.statusCode || (isValidationErr ? 422 : 400);
    return res.status(statusCode).json({
      error: true,
      message: err.message || 'Assessment failed',
      details: err.details || {}
    });
  }
});

// POST /api/simulate — Quick sensitivity simulation
app.post('/api/simulate', (req: Request, res: Response) => {
  let payload = req.body;
  if (!payload || typeof payload !== 'object') {
    return res.status(400).json({
      error: true,
      message: 'A JSON request body is required.',
      details: {}
    });
  }

  payload = normalizePayload(payload);

  try {
    const result = predictLoanRisk(payload);
    return res.json(result);
  } catch (err: any) {
    return res.status(400).json({
      error: true,
      message: err.message || 'Simulation failed',
      details: {}
    });
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
    return res.status(404).json({
      error: true,
      message: `Model '${req.params.modelId}' not found. Supported: logistic-regression, knn, naive-bayes, decision-tree`,
      details: { requested_id: req.params.modelId }
    });
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
    model: 'Logistic Regression'
  });
});

// ─── Error Handlers ───────────────────────────────────────────────────────────

app.use((req: Request, res: Response) => {
  if (req.path.startsWith('/api/')) {
    return res.status(404).json({
      error: true,
      message: 'Resource not found',
      details: { path: req.path }
    });
  }
  res.status(404).render('404');
});

app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  if (err instanceof SyntaxError || err.status === 400 || err.statusCode === 400) {
    if (req.path.startsWith('/api/')) {
      return res.status(400).json({
        error: true,
        message: 'Malformed request. Valid JSON payload required.',
        details: {}
      });
    }
    return res.status(400).send('Bad Request');
  }

  // Only log unexpected internal errors
  console.error('Internal server error:', err);
  if (req.path.startsWith('/api/')) {
    return res.status(500).json({
      error: true,
      message: 'Internal server error occurred',
      details: {}
    });
  }
  res.status(500).render('404');
});

// ─── Server Entry ─────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.PORT || '3000', 10);
const HOST = process.env.HOST || '0.0.0.0';

app.listen(PORT, HOST, () => {
  console.log(`LoanLens server running on http://${HOST}:${PORT}`);
});
