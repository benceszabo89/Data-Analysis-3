# Assignment 2 — Predicting Fast-Growth Firms

**Course:** ECBS5171 – Data Analysis 3  
**Authors:** Boga Petruska & Bence Szabo  
**Date:** February 2026

## Overview

We predict which firms will experience fast growth (≥20% annualized sales CAGR over 2012–2014) using the Bisnode panel dataset of Czech firms. We compare logit, LASSO, Random Forest, and Gradient Boosting models, then apply the best model to classify firms under an asymmetric loss function.

## Project Structure

```
Assignment 2/
├── analysis/
│   ├── 01_assignment_2_data_prep_v2.ipynb   # Data cleaning, label & feature engineering
│   └── 02_assignment_2_modeling.ipynb       # Model building, classification, industry comparison
├── data/
│   ├── clean/
│   │   └── cs_bisnode_panel.csv             # Raw panel data
│   └── prepped/
│       └── bisnode_firms_prepped.csv        # Analysis-ready dataset
├── outputs/                                 # Saved plots and figures
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Open and run the notebooks manually in order: `analysis/01_assignment_2_data_prep_v2.ipynb` → `analysis/02_assignment_2_modeling.ipynb`.

## Key Results

- **Best model:** LASSO (CV AUC = 0.70, Holdout AUC = 0.68, 42 non-zero coefficients)
- **Loss function:** FP = $1, FN = $5 (missing a fast-grower costs 5× more than a false alarm)
- **Optimal threshold:** 0.171 → Recall = 90%, Precision = 32%
- **Industry comparison:** Model performs better on services (AUC 0.72) than manufacturing (AUC 0.65)

## Data

The Bisnode dataset `cs_bisnode_panel.csv` is located in `data/clean/`.
