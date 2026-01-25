# Laptop Recommendation and Rating Prediction System

Live Demo:  
https://laptop-price-and-recommendation-system-icvhjjspmfrmmy2rfmtuaq.streamlit.app/

---

## Overview

This project builds an end-to-end machine learning system that predicts laptop ratings and recommends the best laptops based on user preferences such as budget, RAM, storage, and brand. The system is deployed as an interactive Streamlit web application.

---

## Problem Statement

Laptop buyers face difficulty comparing specifications and identifying high-quality options within a fixed budget.  
This project solves that problem by using machine learning to predict laptop quality and recommend top-performing laptops.

---

## Solution

- Performed data cleaning and exploratory data analysis
- Engineered relevant numerical and categorical features
- Built a preprocessing and modeling pipeline
- Trained an XGBoost regression model
- Deployed the model using Streamlit for real-time interaction

---

## Machine Learning Details

**Target Variable**
- Laptop Rating

**Features**
- Brand  
- Processor Name  
- RAM (GB)  
- Storage  
- Price  
- Operating System  

**Model**
- XGBoost Regressor

**Preprocessing**
- Standard scaling for numerical features
- One-hot encoding for categorical features
- Outlier handling using winsorization
- Log transformation of the target variable

**Evaluation**
- R² Score: ~0.91  
- Low prediction error on unseen data

---

## Application Features

- Budget-based filtering
- Brand and specification selection
- Top five laptop recommendations
- CSV export of recommendations

---


## Repository Structure
```
Laptop_Price_Recommendation_System/
├── README.md # Project documentation
├── data/
│ └── data.csv # Preprocessed dataset
├── notebooks/
│ └── LAPTOP_RECOMMENDATION_MODEL.ipynb # Jupyter Notebook with complete workflow
├── app/
│ └── streamlit_app.py # Streamlit web app script
├── requirements.txt # Required Python libraries
```

---



---

## Key Insights

- Price and RAM strongly influence laptop ratings
- Higher-end processors and brands tend to score better
- Budget-based filtering improves recommendation quality

---

## Limitations

- Dataset does not reflect real-time market prices
- Hardware details such as battery life and GPU are not included
- Recommendations rely on historical data only

---

## Future Improvements

- Add GPU and battery features
- Tune model hyperparameters
- Integrate real-time pricing APIs
- Deploy using Docker and cloud services

---

## Author

Mathan  
LinkedIn: https://www.linkedin.com/in/mathan03/
