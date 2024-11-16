"""
This file and its contents were inspired by the Churnometer Walkthrough Project 2. 
The code has been adapted and extended to analyze housing prices in Ames, Iowa, focusing on 
predictive analytics and insights related to property attributes and sales price.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from src.data_management import get_raw_housing_data

sns.set_style("whitegrid")


def page_sales_price_study():
    """
    Renders the sales price study page in the Streamlit app.
    Displays correlations between sale price and key house attributes.
    """
    # Load data
    df = get_raw_housing_data()

    # Correlation variable list and descriptions
    corr_var_list = [
        'YearBuilt', 'GarageArea', 'GrLivArea',
        '1stFlrSF', 'OverallQual', 'TotalBsmtSF'
    ]
    variable_descriptions = {
        'YearBuilt': 'Year the house was built',
        'GarageArea': 'Garage area in square feet',
        'GrLivArea': 'Above ground living area in square feet',
        '1stFlrSF': 'First floor area in square feet',
        'OverallQual': 'Overall material and finish quality',
        'TotalBsmtSF': 'Total basement area in square feet',
    }

    # Format variable descriptions
    formatted_corr_var_list = [
        f"{var} ({variable_descriptions.get(var, 'No description available')})"
        for var in corr_var_list
    ]
    corr_var_str = ", ".join(formatted_corr_var_list)

    # Page header
    st.write("### Sales Price Study")
    st.info(
        "The client wants to know how the sale price correlates to certain house attributes. "
        "By understanding these correlations, the client aims to identify which features "
        "have the most significant impact on property value."
    )

    # Inspect data
    if st.checkbox("Inspect House Price Data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
            "Below are the first 10 rows."
        )
        st.write(df.head(10))

    st.write("---")

    # Correlation study summary
    st.write(
        "A correlation study was conducted to better understand how the variables are "
        f"correlated to the sale price. The most correlated variables are: **{corr_var_str}**"
    )

    st.info(
        "* Garage Area: Indicates that the size of the garage is a strong factor in determining "
        "a home’s value, likely due to the added utility and storage space it provides.\n"
        "* Above Ground Living Area (GrLivArea): Larger living areas above ground are "
        "highly valued, emphasizing the importance of spacious, functional living space.\n"
        "* Overall Quality (OverallQual): High-quality construction and materials are strongly w"
        "associated with higher prices, reflecting buyer preference for well-built properties.\n"
        "* Total Basement Area (TotalBsmtSF): A larger basement area contributes to home value, "
        "potentially due to its flexibility for additional living or storage space.\n"
        "* First Floor Area (1stFlrSF): The size of the first floor is a key factor, as a larger "
        "main floor can improve layout and accessibility.\n"
        "* Year Built: Newer properties generally sell for more, as modern construction standards "
        "and newer materials are appealing to buyers."
    )

    # EDA of Correlated Variable List
    df_eda = df.filter(corr_var_list + ['SalePrice'])

    # Individual plots per variable
    if st.checkbox("Variable correlation with Sale Price"):
        variable_correlation_to_sale_price(df_eda, corr_var_list)


def variable_correlation_to_sale_price(df_eda, corr_var_list):
    """
    Generates and displays correlation plots between variables and sale price.
    """
    target_var = 'SalePrice'
    for col in corr_var_list:
        plot_numerical(df_eda, col, target_var)
        st.write("\n\n")


def plot_numerical(df, col, target_var):
    """
    Creates and displays a regression plot for a numerical variable and the target variable.
    """
    fig, _ = plt.subplots(figsize=(15, 8))
    sns.regplot(data=df, x=col, y=target_var)
    plt.title(f"{col}", fontsize=20)
    st.pyplot(fig)
