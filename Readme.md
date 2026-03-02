# 🛡️ ChurnPredictor AI — Telco Customer Churn Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://anuj192006-genai-project-app-7glevr.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end Machine Learning web application that predicts whether a telecom customer is likely to churn, built with **Logistic Regression**, **Decision Tree**, and deployed via **Streamlit**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Models Used](#-models-used)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [How to Run Locally](#-how-to-run-locally)
- [Features](#-features)
- [Team](#-team)

---

## 🔍 Overview

Customer churn — when a customer stops using a service — is one of the biggest challenges for telecom companies. Retaining an existing customer is **5× cheaper** than acquiring a new one.

This project builds a machine learning pipeline to:
1. **Analyze** customer demographics, service usage, and billing patterns
2. **Predict** the probability of a customer churning
3. **Recommend** personalized retention strategies based on the prediction

---

## 🚀 Live Demo

👉 **[Click here to try the app live on Streamlit Cloud](https://anuj192006-genai-project-app-7glevr.streamlit.app)**

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Telco Customer Churn Dataset (`7 churn.csv`) |
| **Records** | 7,043 customers |
| **Features** | 20 (demographics, services, billing) |
| **Target** | `Churn` — Yes / No |

### Key Features Used:
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Services**: PhoneService, MultipleLines, InternetService, StreamingTV, etc.
- **Billing**: Contract type, PaymentMethod, MonthlyCharges, TotalCharges
- **Account Info**: Tenure (months with the company)

> **Preprocessing**:
> - `TotalCharges` converted to numeric; missing values filled with column mean
> - All categorical columns label-encoded
> - Numerical features standardized with `StandardScaler`

---

## 📁 Project Structure

```
GenAi_Project/
│
├── app.py                    # 🌐 Streamlit web application (main entry point)
├── telco_churn_model.py      # 🧠 Model training script (standalone Python)
├── telco_churn_model.ipynb   # 📓 Jupyter Notebook — exploration & analysis
├── 7 churn.csv               # 📦 Raw dataset
├── requirements.txt          # 📋 Python dependencies
└── Readme.md                 # 📖 This file
```

---

## 🧠 Models Used

### 1. Logistic Regression
- A statistical model that estimates the probability of churn using a linear decision boundary.
- `max_iter=1000`, `random_state=42`
- Best suited for linearly separable data and interpretable predictions.

### 2. Decision Tree Classifier
- A tree-based model that splits data on feature thresholds to make predictions.
- `random_state=42`
- Captures non-linear relationships in customer behavior.

Both models are trained using an **80/20 train-test split** with `StandardScaler` normalization.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | **82%** | **68%** | **58%** | **63%** |
| Decision Tree | 73% | 48% | 51% | 50% |

✅ **Logistic Regression outperforms** Decision Tree across all metrics and is used as the default model in the app.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Framework | scikit-learn |
| Data Processing | pandas, numpy |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Anuj192006/GenAI_Project.git
cd GenAI_Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### 4. (Optional) Run the Training Script
```bash
python telco_churn_model.py
```
This trains the model and saves `telco_churn_model.pkl`.

---

## ✨ Features

- 🎯 **Dual Model Selection** — Switch between Logistic Regression and Decision Tree in the sidebar
- 📊 **Live Metrics** — View Accuracy, Precision, Recall, F1 Score for each model in the sidebar
- 🔥 **Churn Risk Prediction** — Get real-time churn probability with a confidence score
- 💡 **Smart Recommendations** — Personalized retention or upsell strategies based on the result
- 🎨 **Premium Dark UI** — Custom CSS with gradient accents, Google Fonts, and responsive layout
- ☁️ **Self-contained** — No pre-built `.pkl` file needed; model trains from CSV on startup

---

## 👥 Team

This project was built as a collaborative group effort by:

| Name |
|---|
| **Anuj Upadhyay** |
| **Chaitanya Shekhawat** |
| **Shaurya Sharma** |
| **Tanishk Agarwal** |

> B.Tech Students | AI & Machine Learning Project

🔗 [Live App](https://anuj192006-genai-project-app-7glevr.streamlit.app) &nbsp;|&nbsp; [GitHub](https://github.com/Anuj192006/GenAI_Project)

---

> ⭐ If you found this project helpful, please give it a star!
