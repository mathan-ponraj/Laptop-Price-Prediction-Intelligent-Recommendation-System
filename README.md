# 💻 LapPick

> **Find the right laptop with data, not guesswork.**

[![🚀 OPEN LIVE DEMO](https://img.shields.io/badge/🚀%20OPEN%20LIVE%20DEMO-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://laptop-price-and-recommendation-system-icvhjjspmfrmmy2rfmtuaq.streamlit.app/)



## 🎯 Project Overview

**LapPick** is an end-to-end **Python machine learning application** that predicts laptop prices and generates personalized recommendations using **data preprocessing, XGBoost regression, and Streamlit**.

It transforms complex laptop specifications into a simple, data-driven **laptop selection experience**.

## 📸 Visual Preview
![LapPick Preview](images/app-preview.png)

## 💡 The Problem & Core Value
- 🔎 Simplifies laptop discovery across multiple hardware configurations.
- 💰 Helps users evaluate laptops within a defined **budget**.
- 🤖 Combines **price prediction + recommendation ranking** for data-driven decisions.

## ✨ Key Features & User Flow
- 🤖 **XGBoost Price Prediction** — Estimates laptop prices from hardware specifications.
- 🎯 **Personalized Recommendations** — Filters and ranks laptops based on user requirements.
- 📊 **Data Analysis** — Explores pricing, brands, RAM, storage, processors, and OS.
- 📁 **CSV Export** — Saves recommended laptops for further comparison.

**User Flow:**  
`Requirements → Data Processing → Price Prediction → Budget Filtering → Ranking → Recommendations`

## 🛠️ Tech Stack & Architecture Decisions

| Layer | Technology | Purpose |
|---|---|---|
| Language | **Python** | Core development |
| Data Processing | **Pandas, NumPy** | Cleaning & transformation |
| ML Pipeline | **Scikit-learn** | Preprocessing |
| Prediction | **XGBoost** | Price prediction |
| Model Persistence | **Joblib** | Model loading & storage |
| UI | **Streamlit** | Interactive application |
| Visualization | **Matplotlib** | Data analysis |

## 📈 Challenges & Technical Takeaways

**The Obstacle**
- Laptop datasets contained **missing values, duplicates, categorical data, and inconsistent formats**.
- Basic filtering could produce too many irrelevant options.

**The Resolution**
- Built a structured **Scikit-learn preprocessing pipeline**.
- Used **XGBoost** to model nonlinear relationships between hardware specifications and price.
- Combined prediction, filtering, and ranking into a single recommendation workflow.

**Model Performance:** `R² ≈ 0.91`

## ⚙️ Quick Start

```bash
git clone https://github.com/mathan-ponraj/Laptop-Price-And-Recommendation-System.git
cd Laptop-Price-And-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```
