# LoanLens — Loan Default Prediction & Risk Intelligence Platform

LoanLens is an end-to-end Machine Learning web application designed to evaluate loan default risk on 255,347 historical consumer credit records. The platform couples a production-calibrated **L2-Regularized Logistic Regression** model with interactive underwriting simulation, model benchmarking, and exploratory data analysis.

> **Academic Project Notice:** LoanLens is an educational demonstration of credit risk modeling on Kaggle data. Risk scores and model classifications are statistical outputs for educational, evaluative, and simulation purposes only, and do not constitute actual commercial lending decisions or professional financial advice.

---

## 1. Project Overview & Architecture

LoanLens is designed with a modern decoupled architecture:
* **Frontend Web Application**: Express.js (Node.js/TypeScript) serving server-rendered EJS views styled with clean modern CSS, and Chart.js visualization.
* **ML Inference Engine**: Python 3.10 Flask microservice loading pre-trained Scikit-Learn pipelines and artifacts once at startup.
* **Inter-Service Communication**: Node proxy routes (`/api/*`) and client fetch flows calling the Flask backend over internal networking or configured remote URL.

```
[ Browser Client ]
        │
        ▼ (HTTP 3000)
[ Express.js / TypeScript ]  ── (Pages & Static Views)
        │
        ▼ (/api/predict, /api/models, /api/data/insights)
[ Flask Python ML Engine ]   ── (HTTP 5001 / Cloud Port)
        │
   ┌────┴──────────────────────────┐
   │ Loaded Scikit-Learn Artifacts │
   ├───────────────────────────────┤
   │ • preprocessor.pkl            │
   │ • logistic_regression.pkl     │
   │ • knn.pkl                     │
   │ • naive_bayes.pkl             │
   │ • decision_tree.pkl           │
   └───────────────────────────────┘
```

---

## 2. Machine Learning Models

LoanLens evaluates exactly four primary classification algorithms against the 255k dataset, plus one scratch reference model:

| Model ID | Algorithm | Family | ROC-AUC | 5-Fold CV AUC | Default Recall | Selected Status |
|---|---|---|---|---|---|---|
| **logistic-regression** | L2-Regularized Logistic Regression (Balanced) | `LogisticRegression` | **0.7532** | **0.7426 ± 0.006** | **69.95%** | **Selected Production Model** |
| **knn** | Distance-Weighted K-Nearest Neighbors (k=25) | `KNeighborsClassifier` | 0.6927 | 0.6753 ± 0.007 | 0.70% | Benchmark Evaluated |
| **naive-bayes** | Gaussian Naive Bayes (`var_smoothing=1e-5`) | `GaussianNB` | 0.7499 | 0.7396 ± 0.004 | 2.56% | Benchmark Evaluated |
| **decision-tree** | Entropy-Partitioned Decision Tree (`max_depth=6`) | `DecisionTreeClassifier` | 0.7254 | 0.7036 ± 0.006 | 3.02% | Benchmark Evaluated |
| *scratch-logistic-regression* | Batch Gradient Descent Logistic Regression | `Pure NumPy` | 0.7530 | 0.7530 ± 0.003 | 68.40% | Reference Implementation |

### Why Logistic Regression was Selected
In retail lending, false negatives (granting credit to an applicant who defaults) are vastly more costly than false positives. The balanced Logistic Regression model achieves a default recall of **69.95%** (identifying 4,149 out of 5,931 test defaults) and the highest overall discriminative ranking power (**0.7532 ROC-AUC**), while offering transparent log-odds coefficient interpretability.

---

## 3. Dataset & Preprocessing

* **Dataset Source**: Kaggle Banking Lending Dataset (`nikhil1e9/loan-default`)
* **Total Records**: 255,347 clean records (0 duplicates, 0 missing values)
* **Target Distribution**: Non-Default (`0`): 225,694 (88.39%), Default (`1`): 29,653 (11.61%)
* **Features Collected**: 16 raw features (9 continuous, 7 categorical)
* **Preprocessing Pipeline** (`artifacts/preprocessing/preprocessor.pkl`):
  * **Continuous Features (9)**: `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio` normalized via `StandardScaler(with_mean=True, with_std=True)`.
  * **Categorical Features (7)**: `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner` encoded via `OneHotEncoder(drop='first', sparse_output=False)`.
  * **Encoded Feature Dimension**: Exactly 24 standardized numerical columns in consistent sequence.

---

## 4. Risk Score & Transformation Methodology

1. **Raw Model Output**:
   $$P(\text{Default} = 1 \mid \mathbf{X}) = \sigma(\mathbf{w}^T \mathbf{X} + b) \in [0.0000, 1.0000]$$
2. **Institutional Risk Score**:
   $$\text{risk\_score} = \operatorname{round}((1.0 - P(\text{Default})) \times 100) \in [0, 100]$$
   *Higher scores indicate lower default risk and greater creditworthiness.*
3. **Threshold Classification**:
   * **Low Risk**: $P < 0.28$ (Risk Score: 73 – 100) $\to$ Label: `No Default` (`prediction = 0`)
   * **Moderate Risk**: $0.28 \le P < 0.50$ (Risk Score: 51 – 72) $\to$ Label: `No Default` (`prediction = 0`)
   * **High Risk**: $P \ge 0.50$ (Risk Score: 0 – 50) $\to$ Label: `Default` (`prediction = 1`)

---

## 5. API Endpoint Documentation

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service health check returning `{"status": "ok", "model": "Logistic Regression"}` |
| `POST` | `/api/predict` | Runs 16-feature input through preprocessor and Logistic Regression model |
| `GET` | `/api/models` | Lists the 4 primary production models and their metrics |
| `GET` | `/api/models/comparison` | Returns 4-model evaluation table, cross-validation metrics, and ROC curve points |
| `GET` | `/api/models/<modelId>` | Model parameters, confusion matrix, and feature importances/coefficients |
| `GET` | `/api/data/insights` | EDA distributions, correlation matrix, and dataset summary |

### Example Request (`POST /api/predict`)
```json
{
  "Age": 42,
  "Income": 115000,
  "LoanAmount": 25000,
  "CreditScore": 780,
  "MonthsEmployed": 64,
  "NumCreditLines": 3,
  "InterestRate": 6.8,
  "LoanTerm": 36,
  "DTIRatio": 0.22,
  "Education": "Master's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": "Yes",
  "HasDependents": "No",
  "LoanPurpose": "Home",
  "HasCoSigner": "No"
}
```

### Example Response (`200 OK`)
```json
{
  "prediction": 0,
  "label": "No Default",
  "probability": 0.1148,
  "confidence": 0.8852,
  "risk_score": 89,
  "risk_level": "Low",
  "model_used": "Logistic Regression",
  "factors": [
    {
      "feature": "InterestRate",
      "label": "Contract Interest Rate",
      "value": 6.8,
      "benchmark": "10.0%",
      "direction": "down",
      "impact": 25,
      "raw_impact": -0.5023,
      "type": "mitigant",
      "desc": "Contract Interest Rate: -0.5 log-odds (Protective Factor)"
    }
  ],
  "recommendation": {
    "action": "Predicted Default Risk: Low (Model Classification: No Default)",
    "points": [
      "Model predicts strong repayment indicators with estimated default probability below 0.28.",
      "Solid financial fundamentals (healthy DTI, prime credit rating, and stable employment tenure) serve as strong protective factors.",
      "Notice: This prediction is an ML model output for demonstration purposes and does not constitute financial advice."
    ]
  },
  "success": true
}
```

---

## 6. Environment Variables

| Variable | Description | Development Default | Production Recommended |
|---|---|---|---|
| `PORT` | Web server listening port | `3000` | Provided by Cloud Provider (`process.env.PORT`) |
| `HOST` | Web server bind address | `0.0.0.0` | `0.0.0.0` |
| `USE_REMOTE_BACKEND` | Enable Flask inference engine | `true` | `true` |
| `FLASK_BACKEND_URL` | Target URL for Python Flask service | `http://127.0.0.1:5001` | Cloud internal network / deployed Flask URL |
| `FLASK_PORT` | Local port for Flask if separate | `5001` | Provided by Cloud Provider (`PORT`) |
| `ALLOWED_ORIGINS` | Permitted CORS origins for Flask | `*` | Deployed frontend domain (e.g. `https://myapp.run.app`) |

---

## 7. Local Development Setup

### Prerequisites
* Node.js v18+ and npm
* Python 3.10+ with `pip`

### Step 1: Install Dependencies
```bash
# Install Node dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Start Development Server
```bash
npm run dev
```
The application will automatically start the Python Flask backend service in the background and bind the web app to `http://localhost:3000`.

---

## 8. Production Deployment Steps

### Option A: Unified Container Deployment (Google Cloud Run / Docker)
The application is pre-configured to supervise the Flask service as a managed child process inside Node.js:
1. Set environment variables:
   ```bash
   NODE_ENV=production
   PORT=8080
   HOST=0.0.0.0
   USE_REMOTE_BACKEND=true
   FLASK_BACKEND_URL=http://127.0.0.1:5001
   ```
2. Build and start:
   ```bash
   npm run build
   npm start
   ```

### Option B: Decoupled Multi-Service Deployment
If deploying the frontend (Node.js) and ML inference engine (Flask) as separate cloud microservices:
1. **Deploy Flask Backend**:
   * Command: `python3 flask_backend/app.py`
   * Environment: `HOST=0.0.0.0`, `PORT=5001`, `ALLOWED_ORIGINS=https://<your-frontend-domain>`
2. **Deploy Frontend Web App**:
   * Environment: `HOST=0.0.0.0`, `PORT=3000`, `USE_REMOTE_BACKEND=true`, `FLASK_BACKEND_URL=https://<your-flask-service-domain>`
   * Command: `npm start`
