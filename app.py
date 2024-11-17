'''
This file and its contents were inspired by the Churnometer Walkthrough Project 2. 
The code has been adapted and extended to analyze housing prices in Ames, Iowa, focusing on 
predictive analytics and insights related to property attributes and sales price.
'''

from app_pages.multi_page import MultiPage

# Load page scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_hypothesis import page_hypothesis
from app_pages.page_sales_price_study import page_sales_price_study
from app_pages.page_price_predictor import house_price_prediction_page
from app_pages.page_predict_sales_price import page_predict_sales_price

# Create an instance of the app
app = MultiPage(app_name="Housing Prices")

# Add pages to the app
app.add_page("Project Summary", page_summary_body)  # Summary of the project
app.add_page("Hypothesis", page_hypothesis)  # Hypothesis page
app.add_page("House Price Study", page_sales_price_study)  # House price correlation study
app.add_page("House Price Predictor", house_price_prediction_page)  # Price prediction page
app.add_page("Model Performance & Insights", page_predict_sales_price)  # evaluating pipeline performance

# Run the app
app.run()
