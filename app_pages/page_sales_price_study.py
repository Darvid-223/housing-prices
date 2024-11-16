'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import streamlit as st
from src.data_management import get_raw_housing_data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def page_sales_price_study():
    # load data
    df = get_raw_housing_data()

    # hard copied from churned customer study notebook from Churnometer Walkthrough Project 2
    corr_var_list = ['YearBuilt', 'GarageArea', 'GrLivArea', '1stFlrSF', 'OverallQual', 'TotalBsmtSF']
    variable_descriptions = {
        'YearBuilt': 'Year the house was built',
        'GarageArea': 'Garage area in square feet',
        'GrLivArea': 'Above ground living area in square feet',
        '1stFlrSF': 'First floor area in square feet',
        'OverallQual': 'Overall material and finish quality',
        'TotalBsmtSF': 'Total basement area in square feet',
    }

    # Format the correlation variable list with descriptions
    formatted_corr_var_list = [
        f"{var} ({variable_descriptions.get(var, 'No description available')})"
        for var in corr_var_list
    ]
    corr_var_str = ", ".join(formatted_corr_var_list)

    st.write("### Sales Price Study")
    st.info(
        f"The client wants to know how the sale price correlates "
        f"to certain house attributes. By understanding these correlations, "
        f"the client aims to identify which features have the most "
        f"significant impact on property value."
    )

    # inspect data
    if st.checkbox("Inspect House Price Data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows."
        )
        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to the sale price. \n"
        f"The most correlated variables are: **{corr_var_str}**"
    )


    # Based on "sales_price_study" notebook
    st.info(
        f"* The following are the variables isolated in the correlation study:\n"
        f"* Garage Area: Indicates that the size of the garage is a strong factor "
        f"in determining a home’s value, likely due to the added utility and storage space it provides.\n"
        f"* Above Ground Living Area (GrLivArea): Larger living areas above ground are highly valued, "
        f"emphasizing the importance of spacious, functional living space.\n"
        f"* Overall Quality (OverallQual): High-quality construction and materials are strongly associated "
        f"with higher prices, reflecting buyer preference for well-built properties.\n"
        f"* Total Basement Area (TotalBsmtSF): A larger basement area contributes to home value, "
        f"potentially due to its flexibility for additional living or storage space.\n"
        f"* First Floor Area (1stFlrSF): The size of the first floor is a key factor, "
        f"as a larger main floor can improve layout and accessibility.\n"
        f"* Year Built: Newer properties generally sell for more, as modern construction standards "
        f"and newer materials are appealing to buyers."
    )


    # EDA of Correlated Variable List
    df_eda = df.filter(corr_var_list + ['SalePrice'])

    # Individual plots per variable
    if st.checkbox("Variable correlation Sale Price"):
        variable_correlation_to_sale_price(df_eda, corr_var_list)


def variable_correlation_to_sale_price(df_eda, corr_var_list):
    # function based on the "sale_price_study" notebook
    target_var = 'SalePrice'
    for col in corr_var_list:
        plot_numerical(df_eda, col, target_var)
        st.write("\n\n")


def plot_numerical(df, col, target_var):
    # function based on the "sale_price_study" notebook
    fig, axes = plt.subplots(figsize=(15, 8))
    sns.regplot(data=df, x=col, y=target_var)
    plt.title(f"{col}", fontsize=20)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()