## Project Overview
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://prediction-of-the-severity-of-a-car-accident-dm.streamlit.app/) <br>
The **Road Accident Severity Prediction Engine** is an advanced machine learning project developed as a final academic capstone. The system leverages optimized gradient boosting to analyze historical traffic logs, evaluate environmental and structural risk factors, and accurately predict whether a collision will result in a **Slight** or **Major (Serious/Fatal)** outcome. 

By utilizing state-of-the-art interpretability tools and integrating real-time weather data, this framework bridges the gap between predictive modeling and actionable public safety insights, providing a foundation for data-driven traffic management and infrastructure policies.

## Current Pipeline & Architecture
The project has evolved into a fully functional, end-to-end production-ready pipeline implemented in Python:

1. **Data Engineering & Imbalance Correction:** Processed a massive dataset of over 600,000 incident rows, applying aggressive feature reduction from 99 down to 26 highly predictive features. Addressed severe class imbalance by strategically scaling up major/fatal instances (~97K rows added) and injecting custom algorithmic penalties (`class_weight`).
2. **Optimized Predictive Engine:** Implemented a final, tuned **LightGBM Classifier** parameterized via extensive **Optuna** hyperparameter optimization to maximize the overall decision boundary and balance recall.
3. **Live API Integration:** Features a live weather integration layer utilizing the **OpenWeatherMap API**. The system dynamically fetches real-time meteorological conditions (e.g., wind speed, precipitation) for a given city and automatically maps them into the model's operational schema for immediate inference.
4. **Model Explainability (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** via a custom interface to provide human-readable, transparent explanations for individual accident predictions, highlighting the top contributing factors behind each diagnosis.
5. **Academic Visualization Dashboard:** Automatically generates publication-grade visual assets, including a centralized KPI and Model Evaluation Dashboard (Top 5 Feature Importances and Confusion Matrix Heatmap) alongside high-contrast Classification Report matrices tailored for the final graduation poster.

## Model Performance & Key Metrics
The optimized binary model demonstrates high robustness on a stratified validation subset of 180,378 unseen historical incident rows:
* **Overall Model Accuracy:** `73.0%`
* **Precision (Slight / Major):** `0.82 / 0.60`
* **Recall / Sensitivity (Slight / Major):** `0.74 / 0.72`
* **Target Operational Ratio:** Managed at a structurally stable `1 : 1.8` (Major to Slight).

## Tech Stack
* **Core Data Science:** `Python`, `Pandas`, `NumPy`, `Scikit-Learn`
* **Gradient Boosting Machine:** `LightGBM` (Hyperparameter Tuning via `Optuna`)
* **Explainable AI:** `SHAP`
* **API & Integration:** `Requests` (OpenWeatherMap API)
* **Visualization:** `Matplotlib` (GridSpec), `Seaborn`
* **Serialization:** `Joblib` (Model exported as `binary_lightgbm_model.pkl`)
