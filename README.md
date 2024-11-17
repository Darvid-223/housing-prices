# Housing Prices

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Business Requirement 1: Data Visualization and Correlation

- Lydia wants to understand the key factors that influence house prices in Ames, Iowa.
- To achieve this, we will conduct data visualization and a correlation analysis to identify which variables have the strongest impact on the sale price.
- This requirement involves creating visualizations such as scatter plots, bar charts, and heatmaps that show the relationship between attributes like house size, year built, and lot size, and their correlation with sales price.

## Business Requirement 2: Price Prediction Using Machine Learning

- Lydia needs a machine learning model to accurately predict the sales prices of her inherited properties.
- We will use a regression model to estimate the sale price of the houses based on their features, such as lot area, number of rooms, year built, and garage area.
- Additionally, we will create a dashboard that allows Lydia to easily input property characteristics and receive price predictions, as well as explore the insights obtained from the analysis and model.

## Hypothesis and how to validate?

1. Lot Size and Property Size Correlate with Price: Our first hypothesis is that both property size and lot size have a strong correlation with the sales price. Larger properties and lots are likely to be more valuable due to their potential use and space.

2. Kitchen Quality Correlates with Price: We hypothesize that houses with higher kitchen quality (rated as Excellent or Good) will have a higher sales price compared to houses with lower kitchen quality, as kitchen quality is often a major factor in buyer preference.

3. Year Built Correlates with Price: We believe that houses built more recently will have a higher sales price compared to older houses. Newer construction is expected to have modern standards and less need for renovations, making them more attractive to buyers.

4. We hypothesize that houses with larger garage areas are likely to have higher sales prices. A larger garage provides additional space for parking and storage, which can increase the overall attractiveness and value of the property.

## The Rationale to Map the Business Requirements to the Data Visualisations and ML Tasks

This project addresses two key business requirements, and the following rationale maps each requirement to the corresponding data visualizations and machine learning tasks implemented:

### Business Requirement 1: Data Visualization and Correlation
- **Goal:** Lydia wants to understand the key factors that influence house prices in Ames, Iowa.
- **Approach:**
  - Conduct exploratory data analysis (EDA) to identify features strongly correlated with sale price.
  - Use **scatter plots** and **regression plots** to visualize relationships between numerical features (e.g., `GarageArea`, `GrLivArea`) and the sale price.
  - Generate a **heatmap** to display the overall correlation matrix, highlighting features with the strongest positive or negative correlations to the sale price.
  - Summarize findings in the **Sales Price Study** page.

### Business Requirement 2: Price Prediction Using Machine Learning
- **Goal:** Lydia needs a regression model to predict house prices for her inherited properties and other custom inputs.
- **Approach:**
  - Train a machine learning regression pipeline on cleaned and preprocessed housing data.
  - Evaluate the pipeline using metrics such as **R² score** and **RMSE** to ensure its performance is reliable.
  - Provide interactive predictions through the **Price Predictor** page, allowing users to input custom property details and receive estimated sale prices.
  - Display **feature importance** from the machine learning model to enhance interpretability and show which features contributed most to the predictions.


## ML Business Case

* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

## Dashboard Design

In this project, I did not create wireframes for the dashboard design. Instead, I used the [Churnometer](https://github.com/Code-Institute-Solutions/churnometer) walkthrough project by Code Institute as a template for structuring the dashboard.

The Churnometer project provided a solid foundation with its user-friendly design and functional layout built using Streamlit. Streamlit is a standard library for creating interactive dashboards, and its use ensures a standardized and professional approach to designing web applications. By adapting the structure of the Churnometer project, I was able to maintain a clean, intuitive interface while focusing on implementing business-specific requirements and features for the Heritage Housing Issues project. This approach allowed me to deliver a functional and aesthetically pleasing dashboard efficiently.

The Streamlit dashboard consists of multiple pages, each designed to fulfill specific business requirements and provide a seamless user experience. Below is a detailed description of each page:

### 1. Hypothesis Page (`page_hypothesis.py`)
This page presents the key hypotheses of the project and their expected impact on house prices. 
- **Content:** 
  - Four hypotheses are displayed, focusing on variables like lot size, kitchen quality, year built, and garage area.
  - Information is presented using Streamlit's `st.success` to emphasize insights clearly.

### 2. Predict Sales Price Page (`page_predict_sales_price.py`)
This page provides an in-depth look into the machine learning pipeline used for house price prediction.
- **Content:** 
  - Visualizations of the pipeline's structure and feature importance.
  - Performance metrics (e.g., R² score, RMSE) for the training and test datasets.
  - Scatter plots comparing actual vs. predicted values for evaluation.
- **Interactive Features:**
  - Dynamic visualizations of feature importance and model predictions.

### 3. Sales Price Study Page (`page_sales_price_study.py`)
This page explores correlations between house attributes and sale prices to address the first business requirement.
- **Content:**
  - A correlation study for key variables like `GarageArea`, `GrLivArea`, `OverallQual`, etc.
  - Detailed insights about how these features influence house prices.
  - Regression plots showing the relationship between each variable and sale price.
- **Interactive Features:**
  - Checkboxes to inspect raw data and visualize specific correlations.

### 4. Summary Page (`page_summary.py`)
This page provides an overview of the project and its objectives, including dataset details and business requirements.
- **Content:**
  - Dataset description, including its source, key attributes, and time span.
  - Links to additional project documentation, such as the README file.
  - A summary of the two business requirements addressed in the project.

### 5. Predict Price for Live Data (`page_price_predictor.py`)
This page enables the user to input custom house data and predict its sale price using the regression model.
- **Content:**
  - Widgets for user input on key features such as `YearBuilt`, `GarageArea`, and `GrLivArea`.
  - Dynamic table displaying the live input data in real-time.
  - Predictions for both custom input and the client's inherited properties.
- **Interactive Features:**
  - Input widgets for live data.
  - A button to trigger the prediction and display the results.

### 6. Multi-Page Management (`multi_page.py`)
This file orchestrates the navigation and structure of the dashboard.
- **Content:**
  - A sidebar with a menu to navigate between pages.
  - Dynamic rendering of the selected page content.

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

* The App live link is: <https://iowa-house-price-prediction-5717aa87801c.herokuapp.com//>
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

### Acknowledgements

- Special thanks to my friend Lucas Behrendt, whose feedback and tips from his experience in the same course were immensely helpful.
- Special thanks to [Udemy's 100 Days of Code: The Complete Python Pro Bootcamp for 2023](https://www.udemy.com/course/100-days-of-code/) for providing comprehensive lessons on Python and object-oriented programming, which significantly contributed to the development of this project.
This project was developed with the assistance of OpenAI's ChatGPT in the following areas:
- **Code Validation**: ChatGPT helped validate the syntax and logic of the code.
- **Spelling and Grammar Checks**: Assisted in checking and correcting spelling and grammar in the documentation and code comments.
- **Translations**: Provided translations for multilingual support in the documentation.
- **Coding Advice**: Offered suggestions and advice on coding practices and problem-solving approaches.
- **Real-Time Troubleshooting**: Supported real-time debugging and troubleshooting during the development process.
- **Code Comments and Docstrings**: Helped in crafting clear and concise comments and docstrings to improve code readability and maintainability.