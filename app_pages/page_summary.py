'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import streamlit as st

def page_summary_body():

    st.write("###Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Dataset**\n"
        f"* This dataset was sourced from **[Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data)** and is used to simulate a real-world predictive analytics project.\n"
        f"* The dataset includes nearly 1.5 thousand entries, each representing a house in Ames, Iowa. Each record provides detailed attributes about the property, such as Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, and Year Built, along with the sale price.\n"
        f"* Houses in this dataset were built between 1872 and 2010, allowing for an analysis of how various features influence housing prices over a long time span."
    )


    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Darvid-223/housing-prices).")
    

    # Project "Business Requirements" section adapted for House Price Prediction README file
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The customer wants to understand the key factors that influence house prices in Ames, Iowa. "
        f"To achieve this, we will conduct data visualization and a correlation analysis to identify the "
        f"most impactful variables on sale price. This involves creating visualizations such as scatter plots, "
        f"bar charts, and heatmaps to illustrate the relationship between features like house size, year built, "
        f"lot size, and their correlation with sale price.\n"
        f"* 2 - The customer needs a machine learning model to accurately predict the sales prices of her inherited properties. "
        f"We will develop a regression model that estimates the sale price of houses based on various characteristics, "
        f"such as lot area, number of rooms, year built, and garage area. Additionally, we will implement a dashboard "
        f"that allows the customer to input property details to receive price predictions and explore insights from the analysis "
        f"and model."
    )

        