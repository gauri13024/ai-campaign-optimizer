# AI-Driven Intelligent Campaign Optimization for Digital Marketing

An end-to-end decision support platform and multi-objective AI optimization engine that maximizes **Customer Lifetime Value (CLV)** and **Conversions** while simultaneously minimizing **Marketing Spend** and **Customer Ad Fatigue / Churn**.

Grounded in:
- *Marketing Science* (Rafieian et al., 2025): **Multiobjective Personalization of Marketing Interventions**.
- *MDPI Mathematics* (Abildaeva et al., 2026): **A Hybrid AI Approach for Strategy Optimization using Machine Learning and Evolutionary Algorithms (NSGA-II / NSGA-III)**.

---

## 📁 Project Architecture & Directory Structure

```text
ai_campaign_optimizer/
│
├── requirements.txt                   # Complete project dependencies
├── run_pipeline.py                    # Master CLI orchestration runner
├── README.md                          # Quick start & documentation
│
├── data/                              # Data Engineering & Touchpoints
│   ├── data_loader.py                 # Kaggle dataset ingestion & RFM feature engineering
│   ├── channel_touchpoint_generator.py # Multi-channel touchpoints & ad wear-out curves
│   └── dataset_survey.py              # Comparative survey (Criteo, Avazu, Bank, Kaggle)
│
├── models/                            # AI Predictive Modeling Suite
│   ├── conversion_model.py            # LightGBM, XGBoost, CatBoost, MLP, Random Forest
│   ├── clv_revenue_model.py           # CLV & expected campaign revenue regressors
│   ├── fatigue_churn_model.py         # Customer fatigue wear-out & churn risk predictor
│   └── model_explainability.py        # Global feature importance & local SHAP explanations
│
├── optimization/                      # Multi-Objective Optimization Engine
│   ├── problem_formulation.py         # Formal MOOP mathematical definition & constraints
│   ├── nsga2_optimizer.py             # Evolutionary NSGA-II Pareto Frontier & Strategy Clusters
│   ├── linear_programming_optimizer.py # Exact Integer Programming (PuLP / Knapsack)
│   ├── contextual_bandit_optimizer.py # Reinforcement Learning / Thompson Sampling Policy
│   └── optimization_benchmark.py      # Head-to-head comparison table across paradigms
│
├── analytics/                         # Marketing Metrics & Attribution
│   ├── marketing_metrics.py           # CTR, CPC, CPA, CAC, ROAS, ROI, CLV Engine
│   └── attribution_models.py          # First-Touch, Last-Touch, Linear, Time-Decay, Shapley
│
├── dashboard/                         # Executive Web Dashboard
│   └── app.py                         # Interactive Streamlit application with What-If Simulator
│
└── docs/                              # Research Documentation & Team Reference
    ├── PROJECT_REPORT.md              # Academic project report
    └── STUDENT_ROLE_GUIDES.md         # Reference guide for team tasks and viva defense
```

---

## ⚡ Quick Start Guide

### 1. Install Dependencies
Open PowerShell or your preferred terminal in the project directory:
```bash
cd "c:\Users\chava\OneDrive\Desktop\ai_campaign_optimizer"
pip install -r requirements.txt
```

### 2. Run the Full Automated Pipeline
To execute data ingestion, AI predictive modeling, multi-objective optimization benchmarks, and attribution analytics:
```bash
python run_pipeline.py
```
*(Optionally specify a custom budget in USD, e.g. `python run_pipeline.py 3500`)*

### 3. Launch the Interactive Decision Dashboard
To launch the interactive executive decision-support system:
```bash
streamlit run dashboard/app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## ⚙️ Core Platform Capability Modules

| Capability Area | Core Focus | Key Modules | System Functionality |
| :--- | :--- | :--- | :--- |
| **Audience & Campaign Strategy** | Audience Selection & Budget Planning | `data_loader.py`, `app.py` | RFM behavioral segmentation, multi-channel budget allocation, What-If simulation. |
| **Marketing Metrics & MTA** | ROAS, CPA, CLV & Attribution | `marketing_metrics.py`, `attribution_models.py` | Unit economics tracking, First/Last/Linear/Time-Decay/Shapley attribution models. |
| **Optimization Algorithms** | Multi-Objective Pareto Frontier | `nsga2_optimizer.py`, `linear_programming_optimizer.py` | NSGA-II evolutionary search, PuLP knapsack solver, Contextual Bandit policies. |
| **Predictive AI Modeling** | Supervised Response & CLV Prediction | `conversion_model.py`, `clv_revenue_model.py` | LightGBM, XGBoost, CatBoost, Neural Networks, SHAP feature attribution. |

