'''
This file and its contents were inspired by the Churnometer Walkthrough Project 2.
The code has been adapted and extended to analyze housing prices in Ames, Iowa, focusing on
predictive analytics and insights related to property attributes and sales price.
'''

import streamlit as st

class MultiPage:
    '''
    Class to generate multiple Streamlit pages using an object-oriented approach.
    '''

    def __init__(self, app_name) -> None:
        '''
        Initialize the MultiPage class with the application name.
        Parameters:
            app_name (str): The name of the application.
        '''
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🖥️"
        )

    def add_page(self, title, func) -> None:
        '''
        Add a new page to the application.
        Parameters:
            title (str): The title of the page.
            func (callable): The function to render the page content.
        '''
        self.pages.append({"title": title, "function": func})

    def run(self):
        '''
        Run the application and display the selected page.
        '''
        st.title(self.app_name)
        page = st.sidebar.radio(
            'Menu',
            self.pages,
            format_func=lambda page: page['title']
        )
        page['function']()
