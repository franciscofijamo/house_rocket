import geopandas
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px

st.set_page_config(layout='wide')


# read data
@st.cache(allow_output_mutation=True)  # ler na memory
def get_data(path):
    data = pd.read_csv(path)
    # convert date
    data['date'] = pd.to_datetime(data['date'])

    return data


@st.cache(allow_output_mutation=True)  # ler na memory
def get_geofile(url):
    """

    :param url:
    :return:
    """
    geofile = geopandas.read_file(url)
    return geofile


# Load data
data = get_data('kc_house_data.csv')

# get geofile
# url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
url = 'Zip_Codes .geojson'
geofile = get_geofile(url)

# filter zipcode
# Feature 1

data['price_m2'] = data['price'] / data['sqft_lot']

# filter columns
f_attributes = st.sidebar.multiselect('Select columns', data.columns)
# st.write(f_attributes)

# filter zipcode
f_zipcode = st.sidebar.multiselect(
    'Select Zipcode',
    data['zipcode'].unique())
# st.write(f_zipcode)


# Apply filters
st.title('Data Overview')

if (f_zipcode != []) & (f_attributes != []):
    data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
elif (f_zipcode != []) & (f_attributes == []):
    # return only rows
    data = data.loc[data['zipcode'].isin(f_zipcode), :]
elif (f_zipcode == []) & (f_attributes != []):
    # return all  rows, and selected columns
    data = data.loc[:, f_attributes]
else:
    # return the same data
    data = data.copy()

# Average metrics
# How much IDs have per zip code, count
df1 = data[['id', 'zipcode', ]].groupby('zipcode').count().reset_index()
# mean price
df2 = data[['price', 'zipcode', ]].groupby('zipcode').mean().reset_index()
df3 = data[['sqft_living', 'zipcode', ]].groupby('zipcode').mean().reset_index()
df4 = data[['price_m2', 'zipcode', ]].groupby('zipcode').mean().reset_index()

# merge data


m1 = pd.merge(df1, df2, on='zipcode', how='inner')
m2 = pd.merge(m1, df3, on='zipcode', how='inner')
df = pd.merge(m1, df4, on='zipcode', how='inner')

# rename columns
df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'SQRT_LIVING', 'PRICE m/2']

# Descriptive statistic

num_attributes = df.select_dtypes(include=['int64', 'float64'])
mean = pd.DataFrame(num_attributes.apply(np.mean))
median = pd.DataFrame(num_attributes.apply(np.median))
std = pd.DataFrame(num_attributes.apply(np.std))

max = pd.DataFrame(num_attributes.apply(np.max))
min = pd.DataFrame(num_attributes.apply(np.min))

df1 = pd.concat([min, max, mean, median, std, ], axis=1).reset_index()
df1.columns = ['Columns', 'Min', 'Max', 'Mean', 'Median', 'Std']

# main
# COLUMN GRID CONFIG

st.dataframe(data)

c1, c2 = st.beta_columns((2, 1))

c1.header('Descriptive Statistics')
c1.dataframe(df1)

c2.header('Resumes')
c2.dataframe(df)

# MAPS
st.title("Region overview")
c3, c4 = st.beta_columns((1, 1))
c3.header('Portfolio Density')

# data sample to plot in the map = 2000
df = data.sample(20)

# Base Map
density_map = folium.Map(location=[data['lat'].mean(),
                                   data['long'].mean()],
                         default_zoom_start=15)
# make cluster
marker_cluster = MarkerCluster().add_to(density_map)
for name, row in df.iterrows():
    folium.Marker([row['lat'], row['long']],
                  popup='Sold UDS{0} on {1}. Features{2}sft {3} bedrooms,{4} bathrooms, year built {5}'.format(
                      row['price'],
                      row['date'],
                      row['sqft_living'],
                      row['bedrooms'],
                      row['bathrooms'],
                      row['yr_built'])).add_to(marker_cluster)
with c3:
    folium_static(density_map, width=1500)

# REGION PRICE
c4.header('Price Density')

# create filters
df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()

df.columns = [['ZIP', 'PRICE']]

df = df.sample(20)
# Base Map
region_price_map = folium.Map(location=[data['lat'].mean(),
                                        data['long'].mean()],
                              default_zoom_start=15)

# creating regions
region_price_map.choropleth(data=df,
                            geo_data=geofile,  # receiv a geofile arquive with limit of ZIPCODE on the map
                            columns=['ZIP', 'PRICE'],
                            key_on='feature.properties.ZIP',
                            fill_color='Y10rRd',
                            fill_opacity=0.6,
                            lan_opacity=0.2,
                            legend_name='AVG PRICE')
with c4:
    folium_static(region_price_map)
