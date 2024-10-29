'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import streamlit as st

def page_hypothesis():

    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* Lot Size and Property Size Correlate with Price: Our first hypothesis is that both "
        f"property size and lot size have a strong correlation with the sales price. Larger "
        f"properties and lots are likely to be more valuable due to their potential use and "
        f"space.\n\n"
        
        f"* Kitchen Quality Correlates with Price: We hypothesize that houses with higher "
        f"kitchen quality (rated as Excellent or Good) will have a higher sales price compared "
        f"to houses with lower kitchen quality, as kitchen quality is often a major factor in "
        f"buyer preference.\n\n"

        f"* Year Built Correlates with Price: We believe that houses built more recently will "
        f"have a higher sales price compared to older houses. Newer construction is expected "
        f"to have modern standards and less need for renovations, making them more attractive "
        f"to buyers."

        f"* Garage Area Correlates with Price: Houses with larger garage areas are likely to "
        f"have higher sales prices, as a larger garage provides more space for parking and storage, "
        f"which can increase the attractiveness of the property."
    )

