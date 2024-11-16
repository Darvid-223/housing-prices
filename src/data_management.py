"""
This file and its contents were inspired by the Churnometer Walkthrough Project 2. 
The code has been adapted and extended to analyze housing prices in Ames, Iowa, focusing on 
predictive analytics and insights related to property attributes and sales price.
"""

import os
import joblib
import streamlit as st
import pandas as pd


@st.cache_data
def get_raw_housing_data():
    '''
    Load the raw housing data from the specified file path.
    Returns:
        pd.DataFrame: The raw housing dataset.
    '''
    raw_data_path = os.path.join(
        "inputs", "datasets", "raw", "house-price-20211124T154130Z-001", "house-price")
    raw_file = "house_prices_records.csv"
    try:
        housing_data = pd.read_csv(os.path.join(raw_data_path, raw_file))
        return housing_data
    except FileNotFoundError:
        st.error("Raw data file not found. Please check the file path.")
        return pd.DataFrame()

@st.cache_data
def get_cleaned_data(source):
    '''
    Load the cleaned housing data based on the specified source.
    Parameters:
        source (str): Source type ('inherited' or 'default').
    Returns:
        pd.DataFrame: The cleaned housing dataset.
    '''
    try:
        if source == "inherited":
            return pd.read_csv("outputs/datasets/cleaned/clean_inherited_houses.csv")
        return pd.read_csv("outputs/datasets/cleaned/clean_house_price_records.csv")
    except FileNotFoundError:
        st.error("Cleaned data file not found. Please check the file path.")
        return pd.DataFrame()

def load_model(file_path):
    '''
    Load a trained machine learning model from the specified file path.
    Parameters:
        file_path (str): Path to the model file.
    Returns:
        sklearn.pipeline.Pipeline: The loaded ML pipeline.
    '''
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        st.error("Model file not found. Please check the file path.")
        return None
