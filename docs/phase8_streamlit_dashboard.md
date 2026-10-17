# Phase 8 — Streamlit Enterprise Dashboard

## Overview

RetailPulse AI 2.0 is a synthetic retail analytics,
forecasting, and inventory planning portfolio project.

The Streamlit dashboard contains four pages:

1. Executive Overview
2. Retail Performance
3. Forecast Performance
4. Inventory Planner

## Technology

- Python 3.11
- Streamlit 1.66
- Plotly 6.9
- Pandas
- Enterprise dark CSS theme

## Run Dashboard

Activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

Start Streamlit:

    python -m streamlit run src/dashboard/app.py

Open http://localhost:8501

## Model Governance

The champion model was selected using validation data.
The held-out test was evaluated once.

The dashboard reads saved evaluation artifacts and does not
retrain models, rerun test evaluation, or regenerate predictions.

## Inventory Planning

The saved plan contains 100 store-product recommendations
for December 1, 2025.

Total suggested order units: 4,108.
Series requiring orders: 59.

Open purchase orders are excluded.

## Verification

- Four dashboard pages browser-verified
- Enterprise CSS browser-verified
- Ruff checks passed
- 404 automated tests passed