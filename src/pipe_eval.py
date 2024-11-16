'''
This file and its contents were inspired by the Churnometer Walkthrough Project 2. 
The code has been adapted and extended to analyze housing prices in Ames, Iowa, focusing on 
predictive analytics and insights related to property attributes and sales price.
'''

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set_style('whitegrid')


def regression_performance(x_train, y_train, x_test, y_test, model_pipeline):
    """
    Display regression performance metrics for train and test datasets.
    """
    st.write("### Regression Model Performance")

    # Train set evaluation
    st.info("Train Set Performance")
    regression_evaluation(x_train, y_train, model_pipeline)

    # Test set evaluation
    st.info("Test Set Performance")
    regression_evaluation(x_test, y_test, model_pipeline)


def regression_evaluation(features, target, model_pipeline):
    """
    Compute and display performance metrics for a given dataset and pipeline.
    """
    predictions = model_pipeline.predict(features)
    st.write(f"R2 Score: **{r2_score(target, predictions):.3f}**")
    st.write(f"Mean Absolute Error (MAE): **{mean_absolute_error(target, predictions):.3f}**")
    st.write(f"Mean Squared Error (MSE): **{mean_squared_error(target, predictions):.3f}**")
    st.write(f"Root Mean Squared Error (RMSE): **{np.sqrt(mean_squared_error(
        target, predictions)):.3f}**")


def regression_evaluation_plots(data, model_pipeline, alpha_scatter=0.5):
    """
    Generate and display scatter plots comparing actual vs predicted values for train and test sets.
    """
    train_predictions = model_pipeline.predict(data['x_train'])
    test_predictions = model_pipeline.predict(data['x_test'])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Train set plot
    sns.scatterplot(
        x=data['y_train'].flatten(), y=train_predictions, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=data['y_train'].flatten(), y=data['y_train'].flatten(), color='red', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("Train Set")

    # Test set plot
    sns.scatterplot(x=data['y_test'].flatten(), y=test_predictions, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=data['y_test'].flatten(), y=data['y_test'].flatten(), color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Test Set")

    st.pyplot(fig)
