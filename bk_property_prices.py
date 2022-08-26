import streamlit as st
import pickle
import geopandas as gpd
import json
import pandas as pd
import altair as alt
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import e

st.set_page_config(layout='wide')

header = st.container()
bk_map = st.container()
directory = st.container()
eda = st.container()
model = st.container()


@st.cache()
def load_data():
    with open('bk_data_clean', 'rb') as f:
        df = pickle.load(f)
    return df

@st.cache()
def load_map():
    df = gpd.read_file('brooklyn.geojson')
    df = df.rename(columns={'modzcta': 'zipcode'})
    return df

@st.cache()
def load_model():
    with open('lr_model', 'rb') as f:
        lr = pickle.load(f)
    return lr


# function to apply filters
def filter_data(neighborhood_selection=None, hometype_selection=None, bathroom_selection=None, bedroom_selection=None, livingarea_selection=0):

    convert_dict = {
                    'bathrooms': int,
                    'bedrooms': int
                    }

    df_final = df
    if neighborhood_selection==None or neighborhood_selection=='All Neighborhoods':
        df_final = df_final
    else:
        df_final = df_final[df_final['neighborhood'] == neighborhood_selection]

    if hometype_selection==None or hometype_selection=='All Home Types':
        df_final = df_final
    else:
        df_final = df_final[df_final['home_type'] == hometype_selection]

    if bathroom_selection==None or bathroom_selection=='Any':
        df_final = df_final
    else:
        df_final = df_final[df_final['bathrooms'] == bathroom_selection]

    if bedroom_selection==None or bedroom_selection=='Any':
        df_final = df_final
    else:
        df_final = df_final[df_final['bedrooms'] == bedroom_selection]

    df_final = df_final[df_final['living_area'] >= livingarea_selection]

    df_final = df_final.astype(convert_dict)

    return df_final


# function to spit out model results
@st.cache()
def print_results(inputs, intercept, coef):
    result = 0
    for key in inputs:
        result += inputs[key] * coef[key]
    result += intercept

    return e**result


with header:
   st.title('Brooklyn, NY Properties For Sale')
   st.write('Property data pulled from real estate websites such as Zillow, StreetEasy, Redfin (as of August 2022)')


# combine geojson file with dataframe with aggregated metrics
bkmap = load_map()
bkmap = bkmap.rename(columns={'name':'neighborhood'})


# filter sidebar
st.sidebar.header("Filter Options")

df = load_data()

# batch input widgets / filters
with st.sidebar.form(key='columns_in_form'):

   neighborhood_list = df["neighborhood"].unique().tolist()
   neighborhood_list.sort()
   neighborhood_list.insert(0, "All Neighborhoods")
   neighborhood_selection = st.selectbox("Neighborhood", neighborhood_list, index=0)

   hometype_list = df["home_type"].unique().tolist()
   hometype_list.sort()
   hometype_list.insert(0, "All Home Types")
   hometype_selection = st.selectbox("Home Type", hometype_list, index=0)

   bathroom_selection = st.selectbox('Number of Bathrooms', options=['Any', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
   bedroom_selection = st.selectbox('Number of Bedrooms', options=['Any', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

   livingarea_selection = st.number_input('Minimum Living Area in Square Feet', min_value=300)

   submitted = st.form_submit_button('Apply Filters')


with bk_map:
    df_agg = filter_data(neighborhood_selection, hometype_selection, bathroom_selection, bedroom_selection,
                               livingarea_selection)[['neighborhood', 'zipcode', 'price']].groupby(['neighborhood','zipcode']).agg(['count', 'min', 'max', 'mean', 'median'])['price'].reset_index()

    df_agg2 = bkmap.merge(df_agg, on='zipcode')

    convert_dict = {
                    'mean': int
                    }

    df_agg2 = df_agg2.astype(convert_dict)

    with st.expander(label='Map of Ongoing Sales (based on filters)', expanded=True):
        st.markdown(
            """
        <style>
        .streamlit-expanderHeader {
            font-size: 26px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )


        brooklyn_lat = 40.650002
        brooklyn_long = -73.949997
        m = folium.Map(location=[brooklyn_lat, brooklyn_long], zoom_start=11, tiles="Stamen Toner")

        folium.GeoJson(df_agg2, name='Brooklyn').add_to(m)

        try:
            folium.Choropleth(
                geo_data=df_agg2,
                name='choropleth',
                data=df_agg2,
                columns=['neighborhood', 'count'],
                key_on='feature.properties.neighborhood',
                fill_color="YlGn",
                fill_opacity=0.7,
                line_opacity=0.2
            ).add_to(m)

            folium.LayerControl().add_to(m)

            style_function = lambda x: {'fillColor': '#ffffff',
                                        'color': '#000000',
                                        'fillOpacity': 0.1,
                                        'weight': 0.1}
            highlight_function = lambda x: {'fillColor': '#000000',
                                            'color': '#000000',
                                            'fillOpacity': 0.1,
                                            'weight': 0.1}

            feature = folium.features.GeoJson(
                data=df_agg2,
                style_function=style_function,
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['neighborhood', 'count', 'min', 'max', 'mean', 'median'],
                    aliases=['Neighborhood', 'Ongoing Sales', 'Min Sales Price',' Max Sales Price', 'Average Sales Price', 'Median Sales Price'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            )

            feature.add_to(m)
        except:
            st.write('No properties fit filtered values')

        folium_static(m)


with directory:

    with st.expander(label='List of Ongoing Sales (based on filters)', expanded=True):
        st.markdown(
            """
        <style>
        .streamlit-expanderHeader {
            font-size: 26px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        column_list = ['address','city','state','zipcode','neighborhood','home_type','bathrooms','bedrooms','living_area','price','price_per_sqft']
        df_directory = filter_data(neighborhood_selection, hometype_selection, bathroom_selection, bedroom_selection, livingarea_selection)[column_list]
        df_directory = df_directory.sort_values('price', ascending=True).reset_index(drop=True)

        st.write(df_directory)


with eda:

    with st.expander(label='Exploratory Data Analysis', expanded=True):
        st.markdown(
            """
        <style>
        .streamlit-expanderHeader {
            font-size: 26px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = plt.figure(figsize=(15, 8))

            plt.title("Bathrooms vs. Sales Prices")
            ax = sns.regplot(data=df, x="bathrooms", y="price", line_kws={"color": "red"})
            ax.ticklabel_format(style='plain')
            st.pyplot(fig)
            plt.clf()

            plt.title('Square Footage vs Sales Price')
            ax = sns.regplot(data=df, x="living_area", y="price", line_kws={"color": "red"})
            ax.ticklabel_format(style='plain')
            st.pyplot(fig)
            plt.clf()

        with col2:
            fig = plt.figure(figsize=(15, 8))

            plt.title("Bedrooms vs. Sales Prices")
            ax = sns.regplot(data=df, x="bedrooms", y="price", line_kws={"color": "red"})
            ax.ticklabel_format(style='plain')
            st.pyplot(fig)
            plt.clf()

            plt.title("Correlation Matrix")
            corr = df[['price', 'living_area', 'bedrooms', 'bathrooms']].corr()
            ax = sns.heatmap(corr, cmap="Blues", annot=True)
            st.pyplot(fig)
            plt.clf()

        col1, col2 = st.columns(2)

        with col1:

            plt.title("Boxplot of Sales Prices and Neighborhood")
            plt.ticklabel_format(style='plain', axis='y')
            ax = sns.boxplot(x='neighborhood', y='price', data=df, palette='Accent');
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            st.pyplot(fig)
            plt.clf()

        with col2:
            plt.title("Boxplot of Sales Prices and Home Type")
            plt.ticklabel_format(style='plain', axis='y')
            ax = sns.boxplot(x='home_type', y='price', data=df, palette='Accent');
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            st.pyplot(fig)
            plt.clf()

        st.markdown('**EDA Observations (No surprises here!)**')
        """
        * Living area, number of bedrooms, and number of bathrooms is positively correlated with sales price.
        * Living area has the highest correlation to sales price, followed by number of bathrooms and number of bedrooms.
        * There is also high correlation between the predictor variables which suggests there can be concerns of multicollinearity in the regression model.
        * Boxplot visuals show how sales prices vary by neighborhood and home type.
        """



with model:

    df_model = df[['bathrooms', 'bedrooms', 'living_area', 'home_type', 'neighborhood']]

    with st.expander(label='Predicting Housing Prices Using Regression', expanded=True):
        st.markdown(
            """
        <style>
        .streamlit-expanderHeader {
            font-size: 26px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )


        with st.form(key='input_form'):
            neighborhood_input = st.selectbox('Neighborhood',options=np.sort(df['neighborhood'].unique()).tolist())
            bathroom_input = st.selectbox('Bathroom', options=[1,2,3,4,5,6,7,8,9,10])
            bedroom_input = st.selectbox('Bedroom', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            hometype_input = st.selectbox('Home Type', options=df['home_type'].unique().tolist())
            livingarea_input = st.number_input('Living Area in Square Feet', min_value=300)
            submitted_inputs = st.form_submit_button(label='Submit Inputs')

        input_dict = {
            'bathrooms':bathroom_input,
            'bedrooms':bedroom_input,
            'living_area': livingarea_input,
            'home_type':[hometype_input],
            'neighborhood': [neighborhood_input]
        }

        df_input = pd.DataFrame(input_dict)
        df_userinput = pd.concat([df_model, df_input], ignore_index=True)

        X = pd.get_dummies(df_userinput, drop_first=True)
        X = X.iloc[-1:].reset_index(drop=True)
        user_input = dict(X.iloc[0])

        lr = load_model()
        intercept = lr.intercept_
        coef_dictionary = dict(zip(X, lr.coef_))


        estimated_price = "${:,.0f}".format(print_results(user_input, intercept, coef_dictionary))
        st.write("Estimated sales price based on inputs: "+estimated_price)


        """
        
        Other important features missing from the model include distance from train station, expenses such as property taxes, maintenance fees, and year of construction to name a few.  
        """
