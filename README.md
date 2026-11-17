# HELF: Hybrid Ensemble Learning Framework for UPI Fraud Detection

A real-time fraud detection and risk-scoring system for **UPI (Unified Payments Interface)** transactions. HELF fuses three supervised classifiers and one unsupervised anomaly detector into a single, auditable risk score that drives a three-tier verdict: **Legitimate**, **Suspicious** or **Fraudulent**.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)

---

## Highlights

- **Hybrid design:** Logistic Regression + Random Forest + XGBoost + Isolation Forest in one pipeline
- **Transparent fusion:** a closed-form weighted formula, so every verdict can be recomputed and audited
- **Three-tier verdicts:** automatic approval, analyst escalation or automatic block, instead of a binary flag
- **Strong results:** 98.6% accuracy, 99.1% recall, AUC 0.997 on a 22,108-record held-out test set
- **Zero false negatives at the Fraudulent tier:** every fraud in the test set was either blocked or escalated for review
- **Analyst dashboard:** inspect scores, risk trends and outcomes per transaction
- **Auxiliary CNN modules:** image and QR-code tampering screening

---

## Architecture

```
UPI Transaction Stream
        │
        ▼
Feature Ingestion  ──►  Data Preprocessing  ──►  Ensemble Classification
(transaction, temporal,   (imputation, encoding,    ┌────────────────────┐
 account, device,          scaling, SMOTE)          │ Logistic Regression│
 network features)                                  │ Random Forest      │
                                                    │ XGBoost            │
                                                    │ Isolation Forest   │
                                                    └─────────┬──────────┘
                                                              ▼
                                                   Risk-Score Fusion
                                                              ▼
                                                   Verdict Output
                                        Legitimate │ Suspicious │ Fraudulent
```

Each stage is an independent module, so a component (for example, a retrained XGBoost model) can be swapped without touching the rest of the pipeline.

### Risk-score fusion

```
S = α · (w1·P_LR + w2·P_RF + w3·P_XGB) + (1 − α) · A_IF
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α | 0.70 | Weight of supervised vs. anomaly signal |
| w1 (Logistic Regression) | 0.20 | Supervised weight |
| w2 (Random Forest) | 0.45 | Supervised weight |
| w3 (XGBoost) | 0.35 | Supervised weight |

Weights were fixed on a held-out validation split, so no test data influenced them.

### Verdict thresholds

| Composite score `S` | Verdict | Action |
|---------------------|---------|--------|
| `S < 0.48` | Legitimate | Processed normally |
| `0.48 ≤ S ≤ 0.72` | Suspicious | Routed to a human analyst |
| `S > 0.72` | Fraudulent | Automatically blocked |

---

## Models and Hyperparameters

| Model | Role | Configuration |
|-------|------|---------------|
| Logistic Regression | Interpretable probability baseline | L2 regularization, LBFGS solver |
| Random Forest | Robust supervised classifier | 200 trees |
| XGBoost | Strongest single supervised model | 300 estimators, max depth 6, learning rate 0.05 |
| Isolation Forest | Unsupervised anomaly detector | 150 trees, contamination 0.08 |

Hyperparameters were tuned by cross-validation on the training data using minority-class F1 as the objective.

---

## Features

| Category | Examples |
|----------|----------|
| Transaction | Amount, type (P2P, merchant, bill payment, recharge) |
| Temporal | Hour, day of week, day of month |
| Account | Recent transaction frequency, ratio to average spend, account age |
| Device / channel | Device consistency, VPA verification, session anomalies |
| Network | Account-history score, second-degree account relationships (fraud rings) |

Most influential features (Random Forest importance): transaction amount (~0.183), 24-hour transaction frequency (~0.157), recipient account reputation (~0.141).

---

## Dataset

- **Source:** Kaggle UPI transactions dataset (150,000 records, 12 months, P2P and merchant payments)
- **After cleaning:** 147,382 records
- **Fraud rate:** ~8%
- **Split:** 70% train / 15% validation / 15% test (fraud rate preserved in each)
- **Test set:** 22,108 records (1,816 fraudulent), used only for final evaluation
- **Leakage control:** imputation, scaling and SMOTE are fit on the training split only

> Place the dataset in `data/` (see [Getting Started](#getting-started)). The dataset is not redistributed in this repository.

---

## Results

Performance on the held-out test set (N = 22,108):

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|-------|:-----------:|:-------------:|:----------:|:------:|
| Logistic Regression | 91.3 | 88.6 | 87.2 | 87.9 |
| Random Forest | 96.8 | 95.7 | 94.1 | 94.9 |
| XGBoost | 97.4 | 96.2 | 96.4 | 96.3 |
| Majority-Vote Ensemble | 97.9 | 97.1 | 97.8 | 97.4 |
| **HELF (proposed)** | **98.6** | **97.9** | **99.1** | **98.5** |

**ROC AUC:** HELF 0.997 · Majority vote 0.991 · XGBoost 0.984 · Random Forest 0.979 · Logistic Regression 0.941

**At a fixed 3% false-positive rate**, HELF detects 94.2% of fraud versus 81.7% for XGBoost.

**Verdict distribution on the test set:** 20,227 Legitimate (91.5%), 1,800 Fraudulent (8.1%), 81 Suspicious (0.4%).

---

## Getting Started

> Adjust paths and commands below to match your repository layout.

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Typical dependencies: `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`, `flask` (dashboard), `tensorflow` or `torch` (CNN modules).

### Usage

```bash
# 1. Train the models and fit fusion weights
python train.py

# 2. Evaluate on the held-out test set
python evaluate.py

# 3. Launch the Fraud Analytics Dashboard
python app.py
```

---

## Project Structure

```
.
├── data/                  # Dataset (not included)
├── src/
│   ├── ingestion.py       # Feature ingestion
│   ├── preprocessing.py   # Imputation, encoding, scaling, SMOTE
│   ├── models.py          # LR, RF, XGBoost, Isolation Forest
│   ├── fusion.py          # Risk-score fusion and verdict mapping
│   └── image_module.py    # CNN image / QR tampering screening
├── dashboard/             # Fraud Analytics Dashboard
├── notebooks/             # Experiments and analysis
├── requirements.txt
└── README.md
```

---

## Dashboard

The Fraud Analytics Dashboard lets an analyst inspect any transaction:

- **Prediction:** composite risk score and prediction summary
- **Outcome:** routed result (Approved, Review, Blocked)
- **Risk:** risk-percentage trend over time for a monitored account
- **UPI:** distribution across supported UPI apps
- **QR / image scanning:** flags physical tampering with a confidence score

---

## Interpretability and Compliance

Auditability is a core design goal, motivated by RBI and IT Act requirements for explainable decisions:

- The fusion formula is closed-form, so each model's contribution to a verdict can be recovered after the fact.
- Random Forest and XGBoost expose feature-importance rankings.
- Uncertain cases are escalated to a human instead of defaulting to Legitimate.

---

## Limitations

- The dataset is a public proxy, not live UPI platform data, so production accuracy may differ.
- SMOTE does not capture fraud patterns that emerge after training.
- Isolation Forest uses the same features as the supervised models, so fraud engineered to evade feature-level detection may be missed.
- Continuous retraining and real-time threat-intelligence integration were not evaluated.
- Deepfake-voice fraud and credential harvesting via account aggregators are out of scope.

---

## Roadmap

- [ ] Transformer-based sequence models across related transactions
- [ ] Federated learning for cross-institution collaboration without centralizing data
- [ ] Stronger automated QR-code verification
- [ ] Integration and load testing with real-time production systems

---

## Citation

If you use this work, please cite:

```bibtex
@article{abhilasha2026helf,
  title   = {A Hybrid Ensemble Learning Framework for Real-Time Fraud Detection and Risk Scoring in UPI Transactions},
  author  = {Abhilasha, R. V. and Vijayakarthik, P. and Pol, Basavaraj S.},
  journal = {International Journal of Research in Engineering, Science and Management},
  volume  = {9},
  number  = {8},
  year    = {2026}
}
```

## Authors

R. V. Abhilasha, P. Vijayakarthik, Basavaraj S. Pol
Department of Computer Science and Engineering, R. L. Jalappa Institute of Technology, Kodigehalli, India

Contact: abhilasha884@gmail.com

## License

Distributed under the MIT License. See `LICENSE` for details.
