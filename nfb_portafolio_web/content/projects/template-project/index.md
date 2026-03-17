---
title: "S&P500, Crypto & FX Forecasting Application"
date: 2026-03-17
summary: "Pick a symbol and run price forecasts. Check performance metrics and interact with validation and forecast plots."
authors:
  - me

tags:
  - Forecasting
  - Machine Learning
  - Finance
  - Time Series

featured: false

tech_stack:
  - Python
  - Dash
  - Plotly
  - Prophet
  - statsforecast
  - XGBoost
  - scikit-learn
  - MAPIE
  - yfinance
  - Docker
  - Hugging Face Spaces

links:
  - type: github
    url: https://github.com/nforeroba/fin_fore_app
    label: Code
  - type: live
    url: https://huggingface.co/spaces/nikoniko23/fin_fore_app
    label: Demo

status: "Live"                    
role: "Solo Developer"            
duration: "1 week"                
team_size: 1                      

highlights:
  - "500+ stocks, 100 crypto and 28 FX pairs available"
  - "8 forecasting models with different performance, bias and overfitting metrics"
  - "Deployed on Hugging Face Spaces via Docker"
---

This is a financial asset forecasting application built with Python and Dash. It lets users select S&P500 stocks, top cryptocurrencies or FX pairs, configure a training window and forecast horizon, and run 8 forecasting models simultaneously. The performance of the models can be assessed by means of various metrics in order to choose the best ones for the forecast horizon plot.

## Architecture

This is a Dash app. You can see the structure of the project below:  

```
fin_fore_app/
├── app.py                     # Dash entry point
├── assets/style.css           # Theme, styles
├── src/
│   ├── data/loader.py         # yfinance + dynamic symbol loaders
│   ├── layout/
│   │   ├── components.py      # Header, control panel and symbol information card
│   │   └── plots.py           # Plotly charts (validation and forecast) + metrics table
│   ├── callbacks/
│   │   └── forecast.py        # App callbacks
│   └── models/
│       ├── orchestrator.py    # Central pipeline coordinator
│       ├── statistical.py     # AutoARIMA, AutoETS, Theta
│       ├── prophet_model.py   # Prophet + Prophet+XGBoost errors
│       └── ml_models.py       # ElasticNet, RF, XGBoost + MAPIE
```  

## Forecasting Models

| Model | Family | Description |
|---|---|---|
| AutoARIMA | Statistical | Auto-selects the optimal ARIMA order with drift enabled to capture linear upward trends common in financial series. Uses a season length of 5 to match the 5-day trading week. |
| AutoETS | Statistical | Automatically selects the best Error-Trend-Seasonality combination. Robust for series with clear trend and seasonal patterns. |
| Theta | Statistical | Dynamic Optimized Theta method. Decomposes the series into two lines with different slopes, effective for series with low volatility and smooth trends. |
| Prophet | Additive | Meta's forecasting model. Decomposes the series into trend, seasonality, and holiday effects. Configured with US market holidays and 90% uncertainty intervals. |
| Prophet + XGBoost | Hybrid | Prophet models trend and seasonality; XGBoost fits the in-sample residuals using calendar and cyclic features. Final prediction is the sum of both components. |
| ElasticNet | Lag-Based ML | Regularized linear regression (L1 + L2). Trained on lag features, rolling statistics, and Fourier terms. Confidence intervals via conformal prediction (MAPIE). |
| Random Forest | Lag-Based ML | Ensemble of decision trees. Captures non-linear relationships between lag features and future prices. Confidence intervals via conformal prediction (MAPIE). |
| XGBoost | Lag-Based ML | Gradient boosting on lag features and calendar components. Typically the strongest ML baseline for financial time series. Confidence intervals via conformal prediction (MAPIE). |  





## Section 2

Content for your second major section.

## Conclusion

Summary paragraph wrapping up the main points.

---

**Project Status**: ✅ Live in Production  
**GitHub**: [View Source Code](https://github.com/nforeroba/fin_fore_app)  
**Demo**: [Try it Live](https://huggingface.co/spaces/nikoniko23/fin_fore_app)