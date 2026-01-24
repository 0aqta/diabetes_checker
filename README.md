<h1 align="center">🩺 Diabetes Risk Checker – Canada</h1>

<p align="center">
  Educational web app that estimates diabetes risk using an <b>XGBoost model</b> trained on BRFSS big data (2020–2024) processed with <b>Google BigQuery</b> & <b>Cloud Storage</b>, and delivers personalized prevention advice powered by <b>Google Gemini 2.5-flash</b>.
</p>

<p align="center">
  <a href="https://your-diabetes-risk-checker.streamlit.app">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-orange?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Model-XGBoost-green?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/AI-Gemini%202.5--flash-purple?style=for-the-badge&logo=google-gemini&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/Platform-Google%20Cloud-blue?style=for-the-badge&logo=googlecloud&logoColor=white" alt="GCP">
</p>

## ✨ Live Demo

→ **[Diabetes Risk Checker – Try it now!](https://diabeteschecker-j3yy7aexvseumyxuuaudib.streamlit.app/)**  
(Deployed on Streamlit Community Cloud)

## What It Does

1. User enters 14 health-related features (age group, BMI, general health, physical activity, smoking, etc.)
2. XGBoost model predicts probability of diabetes risk
3. Gemini 2.5-flash generates warm, personalized prevention advice with practical tips
4. Includes Canadian health resources (Diabetes Canada helpline, GTA/Toronto programs, etc.)

**Important**: This is **not** medical software — purely for educational awareness.

## Tech Stack & Workflow

- **Data Scale**
  2 million records from BRFSS (Behavioral Risk Factor Surveillance System) 2020–2024
  
- **Data Processing**  
  Merged BRFSS survey data (2020–2024) using **Google BigQuery** (Notebook) + stored/accessed via **Google Cloud Storage buckets**

- **Machine Learning**  
  XGBoost classifier trained locally, saved as pickle (`xgboost_diabetes_model.pkl`)

- **Web App**  
  Streamlit (clean UI, responsive, fast iteration)

- **Personalized AI Advice**  
  Google Gemini 2.5-flash (real-time prompt engineering for empathetic, actionable health guidance)

- **Deployment**  
  Streamlit Community Cloud (free tier) with secrets management for API key

## Acknowledgments

- **Data** 
Behavioral Risk Factor Surveillance System (BRFSS) 2020–2024

- **AI**
Google Gemini 2.5-flash

- **Platform** 
Google Cloud (BigQuery, Cloud Storage), Streamlit

- **Inspiration**
Diabetes Canada awareness initiatives
