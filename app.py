import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.title('Moon Real Estate')
st.markdown('Welcome to rocket moon real estate 2021')

st.header('Load Data')


# read data
@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    # convert date
    data['date'] = pd.to_datetime(data['date'])

    return data


# Load data

data = get_data('kc_house_data.csv')


# Filter bedrooms
bedrooms = st.sidebar.multiselect(
    'Number of Bedrooms',
    data['bedrooms'].unique()

)
