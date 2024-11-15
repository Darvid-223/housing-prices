'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import streamlit as st


def predict_sales_price(X_live, features, pipeline):

    # from live data, subset features related to this pipeline
    X_live_sale_price = X_live.filter(features)
    sale_price_prediction = pipeline.predict(X_live_sale_price)

    return float(sale_price_prediction.round(2))