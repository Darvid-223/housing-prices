import streamlit as st
import pandas as pd

def predict_sales_price(X_live, features, pipeline):
    '''
    Predict the sales price of a property based on input features and a trained pipeline.
    Parameters:
        X_live (pd.DataFrame): Live input data for prediction.
        features (list): List of features used by the pipeline.
        pipeline (sklearn.pipeline.Pipeline): Trained pipeline for prediction.
    Returns:
        float: Predicted sale price, rounded to 2 decimal places.
    '''
    missing_features = set(features) - set(X_live.columns)
    if missing_features:
        st.error(f"Missing features: {', '.join(missing_features)}")
        return None

    try:
        X_live_sale_price = X_live.filter(features)
        sale_price_prediction = pipeline.predict(X_live_sale_price)
        return float(sale_price_prediction.round(2))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
