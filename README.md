# Laptop Rating Prediction and Recommendation System

Live Demo:  
https://laptop-price-and-recommendation-system-icvhjjspmfrmmy2rfmtuaq.streamlit.app/

## Project Overview

This project builds a machine learning system that predicts laptop ratings based on hardware specifications and recommends suitable laptops according to user-defined budget and preferences. The application is deployed using Streamlit.

## Problem Statement

Customers often struggle to choose laptops due to multiple configuration options across price ranges. The objective of this project is to:

- Predict laptop ratings using structured specification data
- Recommend top-performing laptops within a given budget

## Dataset

The dataset contains structured laptop specification data including:

- Brand  
- Processor  
- RAM  
- Storage  
- Operating System  
- Price  

Target Variable:
- Laptop Rating

Note: The dataset represents historical data and does not use real-time market pricing.

## Data Science Workflow

1. Data Cleaning  
   - Handled missing values  
   - Removed duplicates  
   - Standardized categorical variables  

2. Feature Engineering  
   - Encoded categorical variables  
   - Selected relevant features based on correlation analysis  

3. Model Development  
   - Model: XGBoost Regressor  
   - Evaluation Metric: R² Score  
   - Achieved R² ≈ 0.91 on test data  

## Recommendation Logic

- Filters laptops based on user budget  
- Ranks laptops using predicted ratings  
- Returns Top 5 recommendations  
- Allows CSV export of results  

## Application Features

- Budget-based filtering  
- Brand filtering  
- Top 5 ranked laptops  
- Downloadable CSV output  

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  

## Limitations

- No real-time pricing integration  
- Limited hardware attributes (GPU, battery, display not included)  
- Model is not dynamically retrained  

## Future Improvements

- Add additional hardware features  
- Perform hyperparameter tuning  
- Apply cross-validation  
- Integrate real-time pricing APIs  
- Containerize and deploy to cloud infrastructure  


## Author

Mathan Ponraj  
LinkedIn: https://www.linkedin.com/in/mathan03/
