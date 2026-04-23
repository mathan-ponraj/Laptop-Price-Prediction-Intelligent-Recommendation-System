# Laptop Price Prediction & Intelligent Recommendation System

[![Live Demo]](https://laptop-price-and-recommendation-system-icvhjjspmfrmmy2rfmtuaq.streamlit.app/)

## The Core Challenge
In a crowded electronics market, consumers often face "analysis paralysis" due to the overwhelming variety of hardware configurations. This project was built to bridge the gap between complex technical specs (CPU, RAM, Storage) and user-centric value, providing data-driven price predictions and personalized purchase recommendations.

## Technical Approach

### 1. Data Engineering & Cleaning
Instead of just feeding raw data into a model, I focused on making the data into "machine-learnable":
*   **Feature Standardisation:** Cleaned and unified categorical variables across diverse brands and operating systems.
*   **Handling Sparsity:** Managed missing values and duplicates to ensure the model wouldn't learn from noise.
*   **Encoding Strategy:** Applied strategic encoding to transform hardware specs into numerical formats while preserving feature importance.

### 2. The Model: Why XGBoost?
I chose the **XGBoost Regressor** because of its superior ability to handle tabular data and non-linear relationships between specs and price.
*   **Performance:** Achieved an **R² Score of ~0.91**, indicating that the model captures 91% of the variance in laptop pricing.
*   **Evaluation:** Used standard regression metrics to ensure the model generalizes well to unseen configurations.

### 3. The Recommendation Logic
The system goes beyond simple filtering. It acts as a digital consultant:
*   **Budget-First Filtering:** Instantly narrows down options based on the user's financial constraint.
*   **Ranked Suggestions:** Uses the predicted rating to surface the **Top 5** high-value laptops.
*   **Utility:** Features a CSV export option, allowing users to save their research for offline comparison.

## Tech Stack
*   **Language:** Python (The backbone)
*   **Data Science:** Pandas, NumPy, Scikit-learn
*   **Model:** XGBoost
*   **Deployment:** Streamlit (For a fast, responsive UI)

## Roadmap & Evolution
To evolve this into a production-grade tool, my next steps involve:
*   **MLOps:** Containerizing the application using **Docker** for cloud-agnostic deployment.
*   **Real-time Data:** Integrating Web Scraping or Pricing APIs to move beyond historical datasets.
*   **Deepening Specs:** Including GPU benchmarks and battery life metrics for a more holistic recommendation.

---
**Developed by Mathan Ponraj**  
*CSE Graduate | Data & ML Enthusiast*  
[Connect on LinkedIn](https://www.linkedin.com/in/mathan03/)
