'''
This file and its contents were inspired by the Churnometer Walkthrough Project 2. 
The code has been adapted and extended to analyze housing prices in Ames, Iowa, focusing on 
predictive analytics and insights related to property attributes and sales price.
'''

import streamlit as st

def page_hypothesis():
    '''
    Display the hypothesis and their validation for the project.
    '''
    st.write("### Project Hypothesis and Validation")

    st.success(
        "* Lot Size and Property Size Correlate with Price: Our first hypothesis is that both "
        "property size and lot size have a strong correlation with the sales price. Larger "
        "properties and lots are likely to be more valuable due to their potential use and "
        "space.\n\n"

        "* Kitchen Quality Correlates with Price: We hypothesize that houses with higher "
        "kitchen quality (rated as Excellent or Good) will have a higher sales price compared "
        "to houses with lower kitchen quality, as kitchen quality is often a major factor in "
        "buyer preference.\n\n"

        "* Year Built Correlates with Price: We believe that houses built more recently will "
        "have a higher sales price compared to older houses. Newer construction is expected "
        "to have modern standards and less need for renovations, making them more attractive "
        "to buyers.\n\n"

        "* Garage Area Correlates with Price: Houses with larger garage areas are likely to "
        "have higher sales prices, as a larger garage provides more space for parking and storage, "
        "which can increase the attractiveness of the property."
    )
