'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import streamlit as st
import pandas as pd
from src.data_management import get_cleaned_data, load_model
from src.predictive_analysis_ui import predict_sales_price


def house_price_prediction_page():
    # Load model and related data
    version = 'v1'
    pipeline_path = f"outputs/ml_pipeline/predict_saleprice/{version}"
    
    price_pipeline = load_model(f"{pipeline_path}/best_regressor_pipeline.pkl")
    price_features = pd.read_csv(f"{pipeline_path}/X_train.csv").columns.to_list()
    feature_importance = pd.read_csv(f"{pipeline_path}/feature_importance.csv")['Feature'].tolist()

    # Page header and client information
    st.write("### House Sale Price Prediction Interface")
    st.info(
        f"* The client would like to predict the sale prices for their 4 inherited houses, "
        f"as well as any other properties in Ames, Iowa."
    )
    st.write("---")

    # Predict prices for inherited houses
    st.write("### Inherited Properties Price Prediction")
    st.info(
        f"* Below are the details of the inherited houses and their individual price predictions."
    )
    total_price = predict_inherited_properties(price_pipeline, price_features)
    total_price = "%.2f" % total_price
    st.info(
        f"The total estimated sale price for all inherited properties is: ${total_price}"
    )
    st.write("---")

    # Live price prediction
    st.write("### Live Price Predictor")
    st.info(
        f"* Input the details of a property below to predict its sale price."
    )
    live_data = create_input_widgets()

    if st.button("Run Prediction"):
        predicted_price = predict_sales_price(live_data, price_features, price_pipeline)
        st.info(f"The estimated sale price for the entered property is: ${predicted_price}")


def predict_inherited_properties(pipeline, features):
    inherited_data = get_cleaned_data("inherited")
    total_price = 0

    for idx, row in inherited_data.iterrows():
        property_data = row.to_frame().T
        st.write(property_data)
        predicted_price = predict_sales_price(property_data, features, pipeline)
        predicted_price = "%.2f" % predicted_price
        st.write(f"* Predicted sale price for property {idx + 1}: ${predicted_price}")
        total_price += float(predicted_price)

    return total_price


def create_input_widgets():
    # Load the dataset for default values
    df = get_cleaned_data("default")
    scaling_min, scaling_max = 0.4, 2.0

    # Create input widgets for key features
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    live_data = pd.DataFrame([], index=[0])

    with col1:
        feature = "1stFlrSF"
        live_data[feature] = st.number_input(
            label=f"{feature} (sq ft)",
            min_value=df[feature].min() * scaling_min,
            max_value=df[feature].max() * scaling_max,
            value=df[feature].median()
        )

    with col2:
        feature = "GrLivArea"
        live_data[feature] = st.number_input(
            label=f"{feature} (sq ft)",
            min_value=df[feature].min() * scaling_min,
            max_value=df[feature].max() * scaling_max,
            value=df[feature].median()
        )

    with col3:
        feature = "GarageArea"
        live_data[feature] = st.number_input(
            label=f"{feature} (sq ft)",
            min_value=df[feature].min() * scaling_min,
            max_value=df[feature].max() * scaling_max,
            value=df[feature].median()
        )

    with col4:
        feature = "YearBuilt"
        live_data[feature] = st.number_input(
            label=f"{feature}",
            min_value=int(df[feature].min()),
            max_value=int(df[feature].max()),
            value=int(df[feature].median())
        )

    return live_data
