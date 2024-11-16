'''
This file and its contents were inspired by and adapted from the Churnometer Walkthrough Project 2.
'''

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set_style('whitegrid')


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    Display regression performance metrics for train and test datasets.
    """
    st.write("### Regression Model Performance")
    
    # Train set evaluation
    st.info("Train Set Performance")
    regression_evaluation(X_train, y_train, pipeline)
    
    # Test set evaluation
    st.info("Test Set Performance")
    regression_evaluation(X_test, y_test, pipeline)


def regression_evaluation(X, y, pipeline):
    """
    Compute and display performance metrics for a given dataset and pipeline.
    """
    predictions = pipeline.predict(X)
    st.write(f"R2 Score: **{r2_score(y, predictions):.3f}**")
    st.write(f"Mean Absolute Error (MAE): **{mean_absolute_error(y, predictions):.3f}**")
    st.write(f"Mean Squared Error (MSE): **{mean_squared_error(y, predictions):.3f}**")
    st.write(f"Root Mean Squared Error (RMSE): **{np.sqrt(mean_squared_error(y, predictions)):.3f}**")


def regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5):
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Train set plot
    sns.scatterplot(x=y_train.flatten(), y=train_predictions, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train.flatten(), y=y_train.flatten(), color='red', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("Train Set")

    # Test set plot
    sns.scatterplot(x=y_test.flatten(), y=test_predictions, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test.flatten(), y=y_test.flatten(), color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Test Set")

    st.pyplot(fig)

