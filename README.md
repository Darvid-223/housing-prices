# Housing Prices

## Introduction & Overview

This project, **Housing Prices**, is designed to analyze housing data from Ames, Iowa, and provide insights into the key factors that influence house prices. The goal is to assist the client, Lydia, in making informed decisions about selling her inherited properties by leveraging data visualizations and predictive analytics.

### Purpose
The primary purpose of this project is to:
1. Understand how house attributes, such as size, quality, and location, correlate with sale prices.
2. Develop a machine learning model to accurately predict the sale prices of properties based on their attributes.

### Dataset Content

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

#### Key Features of the Dataset:
- **Numerical Attributes:**
  - `GrLivArea`: Above-ground living area in square feet.
  - `GarageArea`: Size of the garage in square feet.
  - `TotalBsmtSF`: Total basement area in square feet.
  - `YearBuilt`: Year the house was constructed.

- **Categorical Attributes:**
  - `OverallQual`: Overall material and finish quality (rated from 1 to 10).
  - `KitchenQual`: Kitchen quality (rated as Excellent, Good, Typical/Average, or Fair).
  - `GarageFinish`: Interior finish of the garage.

- **Target Variable:**
  - `SalePrice`: The price at which the house was sold, ranging from $34,900 to $755,000.

By analyzing and processing these features, the project aims to generate actionable insights and provide reliable price predictions for both Lydia's inherited properties and any custom property input through the dashboard.

---

## Business Requirements

### Business Requirement 1: Data Visualization and Correlation
Lydia wants to understand the key factors that influence house prices in Ames, Iowa. 
- **Objective:** Conduct data visualizations and correlation analysis to identify which variables have the strongest impact on sales price.
- **Deliverables:**
  - Create visualizations such as scatter plots, regression plots, and heatmaps to illustrate the relationship between house attributes (e.g., lot size, house size, year built) and sale price.
  - Summarize findings on the **Sales Price Study** page of the dashboard.

### Business Requirement 2: Price Prediction Using Machine Learning
Lydia needs a machine learning model to accurately predict the sales prices of her inherited properties and other houses in Ames, Iowa.
- **Objective:** Develop a regression model to predict house sale prices based on property features.
- **Deliverables:**
  - Build an interactive dashboard page, **Price Predictor**, where users can input house attributes and receive price predictions.
  - Display insights from the model, such as **feature importance**, to highlight the most impactful variables on house prices.
  - Evaluate model performance using metrics like **R² score** and **RMSE**.

---

## Hypotheses and Validation

1. **Lot Size and Property Size Correlate with Price:** Larger properties and lots are likely to have higher sales prices due to their potential use and space. 
   - **Validation:** Correlation analysis and scatter plots of lot size and property size against sale price.

2. **Kitchen Quality Correlates with Price:** Houses with higher kitchen quality (rated Excellent or Good) are expected to have higher sale prices as kitchen quality is often a major factor for buyers.
   - **Validation:** Analysis of kitchen quality categories and their average sale prices.

3. **Year Built Correlates with Price:** Newer houses are expected to have higher sale prices due to modern standards and reduced renovation needs.
   - **Validation:** Regression plots of year built against sale price.

4. **Garage Area Correlates with Price:** Larger garages provide more utility, making the property more desirable and increasing its value.
   - **Validation:** Scatter plots and regression analysis of garage area vs. sale price.

---

## Rationale for Mapping Business Requirements to Tasks

### Business Requirement 1: Data Visualization and Correlation
- **Goal:** Identify key factors influencing house prices.
- **Tasks:**
  - Conduct **Exploratory Data Analysis (EDA)** to understand feature distributions and relationships.
  - Use visualizations:
    - **Scatter plots** and **regression plots** to examine relationships between numerical features (e.g., `GarageArea`, `GrLivArea`) and sale price.
    - **Heatmaps** to highlight correlations between variables and identify strong predictors of sale price.
  - Summarize findings on the **Sales Price Study** dashboard page.

### Business Requirement 2: Price Prediction Using Machine Learning
- **Goal:** Build a regression model to estimate house prices.
- **Tasks:**
  - Train a **regression pipeline** using cleaned and preprocessed housing data.
  - Evaluate the pipeline with metrics like **R² score** and **RMSE** to ensure reliability.
  - Develop an interactive **Price Predictor** dashboard page:
    - Users input house details to receive live predictions.
    - Provide model-generated insights, such as **feature importance**, to enhance interpretability.

---

## ML Business Case

### Predicting Sale Prices
- **Problem Statement:** Lydia needs to estimate the sales prices of her inherited properties and other houses in Ames, Iowa.
- **Approach:** 
  - Train a supervised regression model to predict the numerical sale price based on house attributes.
  - Use the model to deliver accurate price predictions via an interactive dashboard.
- **Success Criteria:**
  - Achieve an **R² score** of at least 0.75 on both training and test datasets.
  - Provide actionable insights through the **Price Predictor** dashboard to help Lydia make informed decisions.

---

## CRISP-DM Methodology

The project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology, a widely accepted framework for implementing data science projects. This methodology ensures a structured and systematic approach to understanding, analyzing, and deploying machine learning solutions.

### 1. Business Understanding
The project began with understanding the client's needs:
- **Objective:** To predict house prices in Ames, Iowa, based on their attributes and provide actionable insights.
- **Business Requirements:**
  1. Conduct a correlation study to identify how house attributes affect sale prices.
  2. Develop a machine learning model to predict the sale price of specific properties.

### 2. Data Understanding
- The dataset, sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data), contains 1,460 rows and detailed house attributes such as size, year built, and overall quality.
- Initial exploratory data analysis (EDA) included:
  - Identifying missing values.
  - Understanding feature distributions.
  - Assessing potential relationships between features and the target variable (sale price).

### 3. Data Preparation
- **Data Cleaning:** Handled missing values using techniques such as median imputation and mode replacement for categorical variables.
- **Feature Engineering:**
  - Created new features, such as interaction terms between significant predictors.
  - Transformed skewed variables using log transformations to improve model performance.
  - Scaled numerical features to ensure consistency.
- **Feature Selection:** Selected the most impactful features for the model using correlation and feature importance analysis.

### 4. Modeling
- A regression model pipeline was developed using **scikit-learn**.
- Key steps included:
  - Splitting the dataset into training and testing sets.
  - Building a regression pipeline incorporating feature preprocessing and hyperparameter tuning.
  - Optimizing the model using grid search to find the best hyperparameters.

### 5. Evaluation
- Model performance was evaluated using metrics such as:
  - **R² Score:** Indicates how well the model explains variance in the sale price.
  - **Root Mean Squared Error (RMSE):** Measures prediction error magnitude.
- Achieved R² scores:
  - Training set: **0.90**
  - Test set: **0.85**
- Visualizations such as scatter plots of predicted vs. actual values were used to validate predictions.

### 6. Deployment
- The final model and its insights were deployed using **Streamlit** to create an interactive dashboard.
- Users can:
  - Input custom property data to predict sale prices.
  - Visualize key insights through correlation plots and feature importance charts.
- The application is hosted on **Heroku**, making it accessible for stakeholders.

This structured approach ensured alignment between the project objectives, data insights, and the final predictive tool, delivering a comprehensive solution for the client's needs.

---

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

---

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

### Deployment

#### 1. **Deployment on Heroku**

The project was deployed to Heroku, a cloud platform that simplifies application deployment. Below are the steps followed for deployment:

1. **Prepare Project Files**:
   - Ensure the following files are included in your project:
     - `requirements.txt` - Lists all dependencies for the project.
     - `Procfile` - Specifies how Heroku should run your application (e.g., `web: streamlit run app.py`).
     - `runtime.txt` - Specifies the Python version used (e.g., `python-3.10.12`).

2. **Create a Heroku Account and App**:
   - Log in to [Heroku](https://heroku.com).
   - Create a new app and provide it with a unique name.

3. **Connect Heroku to GitHub**:
   - Navigate to the **Deploy** tab in the Heroku dashboard.
   - Select "GitHub" as the deployment method and link your GitHub account.
   - Search for and select your project repository.

4. **Deploy the App**:
   - Choose the branch to deploy (usually `main`) and click "Deploy Branch."
   - Wait for the process to complete and click "Open App" to view the deployed application.

#### 2. **Running Locally**

If you want to run the application locally instead of using Heroku:

1. **Clone the Repository**:
   - Clone the project from GitHub:
     ```bash
     git clone <repository-url>
     ```

2. **Install Dependencies**:
   - Navigate to the project folder and activate a virtual environment:
     ```bash
     python -m venv env
     source env/bin/activate  # On Windows: .\env\Scripts\activate
     ```
   - Install dependencies from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Application**:
   - Start the Streamlit application:
     ```bash
     streamlit run app.py
     ```

4. **Open in Browser**:
   - Open the local URL generated, typically `http://localhost:8501`.

---

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

---

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

---

## Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

---

## Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

---

## Acknowledgements

- Special thanks to my friend Lucas Behrendt, whose feedback and tips from his experience in the same course were immensely helpful.
- Special thanks to [Udemy's 100 Days of Code: The Complete Python Pro Bootcamp for 2023](https://www.udemy.com/course/100-days-of-code/) for providing comprehensive lessons on Python and object-oriented programming, which significantly contributed to the development of this project.
This project was developed with the assistance of OpenAI's ChatGPT in the following areas:
- **Code Validation**: ChatGPT helped validate the syntax and logic of the code.
- **Spelling and Grammar Checks**: Assisted in checking and correcting spelling and grammar in the documentation and code comments.
- **Translations**: Provided translations for multilingual support in the documentation.
- **Coding Advice**: Offered suggestions and advice on coding practices and problem-solving approaches.
- **Real-Time Troubleshooting**: Supported real-time debugging and troubleshooting during the development process.
- **Code Comments and Docstrings**: Helped in crafting clear and concise comments and docstrings to improve code readability and maintainability.