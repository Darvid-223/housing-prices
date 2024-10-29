'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import streamlit as st
from app_pages.multi_page import MultiPage


from app_pages.page_summary import page_summary_body
from app_pages.page_hypothesis import page_hypothesis
# from app_pages.predict_sales_price import predict_sales_price




app = MultiPage(app_name= "Housing Prices") # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Hypothesis", page_hypothesis)
#app.add_page("House price prediction", predict_sales_price)


app.run()  # Run the app