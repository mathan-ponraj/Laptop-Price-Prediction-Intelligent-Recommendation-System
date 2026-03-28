import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuration & Styling ---
st.set_page_config(page_title="Laptop Intelligence System", layout="wide")

# --- Resource Loading with Error Handling ---
@st.cache_resource
def load_assets():
    """Load model and dataset once and cache them for performance."""
    try:
        model_path = "model/model.pkl"
        data_path = "data/data.csv"
        
        if not os.path.exists(model_path) or not os.path.exists(data_path):
            st.error("System Error: Required assets (model/data) are missing.")
            return None, None
            
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        return model, df
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        return None, None

model, df = load_assets()

if df is not None:
    st.title("Enterprise Laptop Recommendation Engine")
    st.markdown("---")

    # --- Sidebar Configuration ---
    st.sidebar.header("Filter Criteria")
    
    budget = st.sidebar.slider("Maximum Budget (INR)", 
                              int(df['Price'].min()), 
                              int(df['Price'].max()), 
                              50000)
    
    brand = st.sidebar.selectbox("Brand Preference", 
                                ['All Brands'] + sorted(df['Brand'].unique().tolist()))
    
    ram = st.sidebar.selectbox("Minimum RAM Requirement (GB)", 
                              sorted(df['RAM_GB'].unique()))
    
    storage = st.sidebar.selectbox("Minimum Storage Capacity (GB)", 
                                  sorted(df['Storage'].unique()))
    
    storage_type = st.sidebar.selectbox("Storage Architecture", 
                                       ['All Types'] + sorted(df['Storage_type'].unique().tolist()))

    # --- Business Logic & Filtering ---
    mask = (df['Price'] <= budget) & (df['RAM_GB'] >= ram) & (df['Storage'] >= storage)
    
    if brand != 'All Brands':
        mask &= (df['Brand'] == brand)
    if storage_type != 'All Types':
        mask &= (df['Storage_type'] == storage_type)
        
    filtered_df = df[mask].copy()

    # --- Prediction & Results ---
    if not filtered_df.empty:
        # Note: Ensure these features match your training script exactly
        features = ['Processor_name', 'OS', 'Brand', 'RAM_GB', 'Storage', 'Price']
        
        try:
            X = filtered_df[features]
            filtered_df['Predicted_Rating'] = model.predict(X)
            
            # Sort by predicted performance/rating
            results = filtered_df.sort_values('Predicted_Rating', ascending=False).head(5)

            st.subheader("Top Ranked Recommendations")
            display_cols = ['Name', 'Brand', 'Processor_name', 'RAM_GB', 'Storage', 'Price', 'Predicted_Rating']
            st.dataframe(results[display_cols], use_container_width=True)

            # Export Functionality
            csv_data = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Recommendations to CSV",
                data=csv_data,
                file_name="laptop_recommendations.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Prediction Engine Error: {str(e)}")
            st.info("Check if features in data.csv match the model's expected input.")
    else:
        st.info("No assets match the current filter criteria. Try expanding your budget or changing preferences.")
else:
    st.warning("Application failed to start. Please verify data paths.")
