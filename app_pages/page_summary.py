"""
This file and its contents were inspired by the Churnometer Walkthrough Project 2. 
The code has been adapted and extended to analyze housing prices in Ames, Iowa, 
focusing on predictive analytics and insights related to property attributes and sales price.
"""

import streamlit as st


def page_summary_body():
    """
    Renders the summary page in the Streamlit app, providing an overview of the dataset,
    project documentation, and business requirements.
    """
    # Dataset content information
    st.info(
        "**Project Dataset**\n"
        "* This dataset was sourced from "
        "**[Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data)** "
        "and is used to simulate a real-world predictive analytics project.\n"
        "* The dataset includes nearly 1.5 thousand entries, each representing a house "
        "in Ames, Iowa. Each record provides"
        "detailed attributes about the property, such as Floor Area, Basement, Garage, "
        "Kitchen, Lot, Porch, Wood Deck, and Year Built, along with the sale price.\n"
        "* Houses in this dataset were built between 1872 and 2010, allowing for an "
        "analysis of how various features influence housing prices over a long time span."
    )

    st.write("---")

    # Link to README file
    st.write(
        "* For additional information, please visit and **read** the "
        "[Project README file](https://github.com/Darvid-223/housing-prices)."
    )

    # Business requirements
    st.success(
        "The project has 2 business requirements:\n"
        "* 1 - The customer wants to understand the key factors that influence house "
        "prices in Ames, Iowa. To achieve this, we will conduct data visualization and a"
        "correlation analysis to identify the most impactful variables on sale price. "
        "This involves creating visualizations such as scatter plots, bar charts, and heatmaps "
        "to illustrate the relationship between features like house size, year built, lot size, "
        "and their correlation with sale price.\n"
        "* 2 - The customer needs a machine learning model to accurately predict the sales "
        "prices of her inherited properties. We will develop a regression model that "
        "estimates the sale price of houses based on various characteristics, such as lot area, "
        "number of rooms, year built, and garage area. Additionally, we will implement a dashboard "
        "that allows the customer to input property details to receive price predictions and "
        "explore insights from the analysis and model."
    )
