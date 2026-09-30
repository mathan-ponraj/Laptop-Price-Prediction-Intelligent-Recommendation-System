# 💻 LapPick

> **Find the right laptop with data, not guesswork.**

[![🚀 OPEN LIVE DEMO](https://img.shields.io/badge/🚀%20OPEN%20LIVE%20DEMO-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://laptop-price-and-recommendation-system-icvhjjspmfrmmy2rfmtuaq.streamlit.app/)



## Project Overview

LapPick is a machine learning-based laptop recommendation system that helps users find suitable laptops based on their budget, brand, RAM, storage, and storage type requirements.

The project uses an XGBoost regression model to predict laptop ratings and ranks the filtered laptops based on their predicted ratings. The trained model is integrated into a Streamlit application to provide an interactive recommendation experience.

## Skills

- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Machine learning
- Regression modeling
- Model evaluation
- Recommendation system development
- Model deployment

## Technology

Python, Pandas, Scikit-learn, XGBoost, Streamlit, Joblib, Matplotlib, Seaborn

## Workflow

The project includes the following steps:

1. **Data Exploration** — Analyze laptop prices, brands, RAM, storage, ratings, and other available features.
2. **Data Preprocessing** — Handle missing values, cap extreme price values, encode categorical features, and scale numerical features.
3. **Model Training** — Train an XGBoost regression model to predict laptop ratings.
4. **Model Evaluation** — Evaluate the model using RMSE and R² metrics.
5. **Recommendation** — Filter laptops based on user requirements and rank the matching laptops using their predicted ratings.
6. **Deployment** — Integrate the trained model into a Streamlit application with interactive filters and recommendation results.

## Obstacles & Resolutions

- **Different types of laptop features:** Used `ColumnTransformer`, `OneHotEncoder`, and `StandardScaler` to process categorical and numerical features appropriately.
- **Extreme price values:** Applied 1st and 99th percentile price capping to reduce the effect of extreme values during model training.
- **Variation in rating values:** Applied a log transformation to the target variable and converted predictions back to the original scale for evaluation.
- **Keeping preprocessing consistent between training and prediction:** Combined preprocessing and the XGBoost model into a single Scikit-learn pipeline and saved it using Joblib.
- **Matching recommendations with user requirements:** Added filters for budget, brand, RAM, storage, and storage type before ranking the available laptops.
- **Making the model accessible to users:** Built a Streamlit application with interactive filters, top-5 recommendations, error handling, caching, and CSV export.

## Results

The project produces an interactive laptop recommendation application that combines user requirements with machine learning predictions.

Users can filter laptops based on their budget and specifications, view the top 5 matching laptops ranked by predicted rating, and download the recommendations as a CSV file.

The trained preprocessing and XGBoost model pipeline is saved using Joblib and integrated into the Streamlit application.

## Future Improvements

- Add a multi-factor recommendation score using predicted rating, budget fit, and specification matching.
- Add SHAP-based explanations to show why a laptop was recommended.
- Store laptop data in PostgreSQL and use SQL for structured data retrieval.
- Build a FastAPI backend for the recommendation system.
- Add MLflow for experiment and model tracking.
- Add Docker and CI/CD for deployment automation.
- Add an LLM-based assistant that accepts natural-language laptop requirements and converts them into structured recommendations.

## ⚙️ Quick Start

```bash
git clone https://github.com/mathan-ponraj/Laptop-Price-And-Recommendation-System.git
cd Laptop-Price-And-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```
