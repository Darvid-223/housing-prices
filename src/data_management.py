'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import joblib
import streamlit as st
import pandas as pd

@st.cache_data
def get_raw_housing_data():
    raw_data_path = "inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/"
    raw_file = "house_prices_records.csv"
    housing_data = pd.read_csv(raw_data_path + raw_file)
    return housing_data

@st.cache_data
def get_cleaned_data(source):
    if source == "inherited":
        cleaned_data = pd.read_csv("outputs/datasets/cleaned/clean_inherited_houses.csv")
    else:
        cleaned_data = pd.read_csv("outputs/datasets/cleaned/clean_house_price_records.csv")
    return cleaned_data

def load_model(file_path):
    return joblib.load(file_path)
