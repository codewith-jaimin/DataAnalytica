# ◆ DataMind Analytics Platform

Multi-agent data analytics platform — Streamlit + Groq AI + scikit-learn.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Setup

Open `app.py` line 14 and replace:
```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```
Get a free key at: https://console.groq.com

## How to use

1. Upload any CSV file via the sidebar
2. Choose a **target variable** (auto-detected, but you can change it)
3. Select one or more **models** from the sidebar checkboxes
4. Click **Run Models**
5. Explore all 5 tabs

## Available models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Regressor

## Tabs

| Tab | Content |
|-----|---------|
| Overview | Agent log, column profile, data preview, stats |
| Visualisations | Heatmap, distributions, scatter, box plots, scale comparison |
| ML Models | Comparison chart, report table, feature importance, actual vs predicted, residuals |
| Insights | AI-generated findings + data summary |
| Ask a Question | One-click example questions + free-text Q&A |
