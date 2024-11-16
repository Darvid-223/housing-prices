"""
This file and its contents were inspired by the Churnometer Walkthrough Project 2. 
The code has been adapted and extended to analyze housing prices in Ames, Iowa, focusing on 
predictive analytics and insights related to property attributes and sales price.
"""

import streamlit as st


def predict_sales_price(x_live, features, pipeline):
    '''
    Predict the sales price of a property based on input features
    and a trained pipeline.
    Parameters:
        x_live (pd.DataFrame): Live input data for prediction.
        features (list): List of features used by the pipeline.
        pipeline (sklearn.pipeline.Pipeline): Trained pipeline for prediction.
    Returns:
        float: Predicted sale price, rounded to 2 decimal places.
    '''
    missing_features = set(features) - set(x_live.columns)
    if missing_features:
        st.error(f"Missing features: {', '.join(missing_features)}")
        return None

    try:
        x_live_sale_price = x_live.filter(features)
        sale_price_prediction = pipeline.predict(x_live_sale_price)
        return float(sale_price_prediction.round(2))
    except KeyError as key_error:
        st.error(f"Missing or invalid feature: {key_error}")
        return None
    except ValueError as value_error:
        st.error(f"Data type or value issue: {value_error}")
        return None
