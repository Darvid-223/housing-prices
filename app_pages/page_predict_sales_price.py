'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_model
from src.pipe_eval import regression_performance, regression_evaluation_plots


def page_predict_sales_price():

    version = 'v1'
    # Load necessary files
    pipeline = load_model(f'outputs/ml_pipeline/predict_saleprice/{version}/best_regressor_pipeline.pkl')
    feature_importance_plot = plt.imread(f'outputs/ml_pipeline/predict_saleprice/{version}/feature_importance.png')
    X_train = pd.read_csv(f'outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv')
    X_test = pd.read_csv(f'outputs/ml_pipeline/predict_saleprice/{version}/X_test.csv')
    y_train = pd.read_csv(f'outputs/ml_pipeline/predict_saleprice/{version}/y_train.csv').values
    y_test = pd.read_csv(f'outputs/ml_pipeline/predict_saleprice/{version}/y_test.csv').values

    # Page header
    st.write("### ML Pipeline: Predict House Sale Price")

    # Display pipeline training summary
    st.info(
        f"* The regression pipeline aims to predict house sale prices accurately. \n"
        f"* The pipeline achieved R² scores of 0.90 on the training set and 0.85 on the test set."
    )

    # Show pipeline structure
    st.write("---")
    st.write("#### ML Pipeline Structure")
    st.write(pipeline)

    # Show feature importance
    st.write("---")
    st.write("### Feature Importance")
    st.write("* Below are the features the model was trained on:")
    st.write(X_train.columns.to_list())
    st.image(feature_importance_plot)

    # Evaluate pipeline performance
    st.write("---")
    st.write("### Pipeline Performance")
    regression_performance(X_train, y_train, X_test, y_test, pipeline)
    regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline)
