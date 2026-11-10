# GLP-1 Demand Surge Intelligence

An end-to-end decision-intelligence workflow for predicting whether a Medicaid GLP-1 product will experience a next-quarter prescription-demand surge at the state-product-quarter level.

## Why it matters

Rapid GLP-1 market expansion creates planning pressure across pharmacy networks, manufacturers, payers, and public programs. This project turns longitudinal public utilization data into an auditable early-warning model for demand planning.

## Results

- Built a strict-history analytic sample of **10,272 observations** spanning 2018 Q1 through 2025 Q4.
- Logistic regression reached **79.7% test accuracy** and **0.744 ROC-AUC**.
- XGBoost improved performance to **82.9% accuracy** and **0.795 ROC-AUC**.
- Used chronological train, validation, and locked-test periods to reduce temporal leakage.
- Included calibration, sensitivity, assumption, suppression, and feature-attribution checks.

The logistic model exceeded the prespecified 70% benchmark, but it did not outperform the 81.3% majority-class baseline. That distinction is retained because honest benchmarking matters more than a flattering headline metric.

## Architecture

```text
Public CMS sources
      ↓
Retrieval and validation
      ↓
State-product-quarter feature panel
      ↓
Chronological train / validation / test split
      ↓
Logistic regression + XGBoost benchmark
      ↓
Calibration, sensitivity, explainability, and exported evidence
```

## Portfolio artifacts

- [Rendered analysis notebook](notebook/analysis.html)
- [Portfolio report](reports/glp1-demand-surge-report.pdf)

The public repository includes the rendered notebook and report while intentionally excluding the working `.ipynb` file, restricted source data, local paths, and academic grading material.
## Repository structure

```text
├── data/README.md
├── src/
│   ├── analysis.py
│   └── retrieve_public_data.py
├── tests/test_analysis.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
python src/retrieve_public_data.py
python src/analysis.py
pytest tests/ -v --tb=short
```

## Analytical guardrails

- Aggregate Medicaid utilization supports prediction and association, not causal claims.
- CMS-suppressed values are excluded rather than interpreted as zero utilization.
- The target is imbalanced, so accuracy is interpreted alongside recall, ROC-AUC, log loss, and the majority-class baseline.
- Results are market-planning signals, not clinical or patient-level recommendations.

## Portfolio

More work: [larry-henry-jr.net](https://larry-henry-jr.net)
