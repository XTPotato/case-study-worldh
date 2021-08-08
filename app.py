#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
#from PIL import Image

def r(x, y):
    return np.mean(((x - np.mean(x)) / np.std(x)) * ((y - np.mean(y)) / np.std(y)))

#def abline(slope, intercept):
#    """Plot a line from slope and intercept"""
#    axes = plt.gca()
#    x_vals = np.array(axes.get_xlim())
#    y_vals = intercept + slope * x_vals
#    plt.plot(x_vals, y_vals, color='black')

st.sidebar.subheader('Table of contents')
st.sidebar.write('1. ','<a href=#case-study-on-the-correlation-between-income-and-life-expectancy>Introduction</a>', unsafe_allow_html=True)
st.sidebar.write('2. ','<a href=#data-cleaning>Data cleaning</a>', unsafe_allow_html=True)
st.sidebar.write('3. ','<a href=#exploratory-data-analysis>Exploratory data analysis</a>', unsafe_allow_html=True)
st.sidebar.write('4. ','<a href=#performing-linear-regression-analysis>Linear regression analysis</a>', unsafe_allow_html=True)
st.sidebar.write('5. ','<a href=#conclusion>Conclusion</a>', unsafe_allow_html=True)
st.sidebar.write('6. ','<a href=#additional-resources>Additional resources</a>', unsafe_allow_html=True)
#st.sidebar.write('2. ','<a href=></a>', unsafe_allow_html=True)

st.title('Case study on the correlation between income and life expectancy')

st.header('Goal of this case study')
st.subheader('The goal of this case study is to investigate and quantize the relationship between Log GDP per capita and life expectancy, these two are supposed to be closely related and so I will be verifying their relationship. ')

st.header('Definition of Log GDP per capita and Life expectancy at birth')
st.subheader('Log GDP per capita is the log of (Total GDP / Population), it is used to measure the increase of GDP rather than the value of GDP, thus making countries more comparable. Life expectancy at birth means the average number of years a newborn will be expected to live if mortality patterns at the time of its birth will remain constant in the future.')

st.header('Source of the data used for this project:')
st.subheader('Kaggle Dataset: https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021')

st.header('Data cleaning')
st.write('This is the original csv, it has 1948 rows of information for each country each year, and 11 columns including the country, year, and statistics.')
happy = pd.read_csv('world-happiness-report.csv')
st.dataframe(happy)

st.write('Dropping the unused columns because I am only going to investigate the life expectancy and log GDP per capita')
dropped = happy.drop(happy.columns[[2, 4, 6, 7, 8, 9, 10]], axis = 1).copy()
st.dataframe(dropped)

st.write('GROUPBY operation on the countries in order to reduce the amount of rows and make countries comparable to each other, the year column in this case becomes meaningless because I took the mean of each country. ')
grouped = dropped.groupby('Country name').mean().reset_index()
st.dataframe(grouped)

st.header('Exploratory data analysis')
st.write('This is a histogram of the two variables I will be comparing. A histogram displays countinuous data and shows their distribution relative to each other, the higher the bar, the more amount of data that falls within the range defined by the width of the bar on the X axis. It is roughly normal so a linear regression model applies. ')
exfig1 = px.histogram(grouped, 'Log GDP per capita', 'Healthy life expectancy at birth', color_discrete_sequence=px.colors.qualitative.T10, nbins=22)
st.plotly_chart(exfig1)

st.header('Performing Linear Regression Analysis')
st.subheader('Code for computing the regression line and residual plot')
st.write('This code computes all the needed values to plot a linear regression line')

x = grouped['Log GDP per capita']
y = grouped['Healthy life expectancy at birth']

correlation = r(x, y)
gradient = correlation * np.std(y) / np.std(x)
intercept = np.mean(y) - gradient * np.mean(x)
fitted = gradient * x + intercept
residual = y - fitted
grouped['residuals'] = residual

code1 = '''def r(x, y):
    return np.mean(((x - np.mean(x)) / np.std(x)) * ((y - np.mean(y)) / np.std(y)))
x = grouped['Log GDP per capita']
y = grouped['Healthy life expectancy at birth']

correlation = r(x, y)
gradient = correlation * np.std(y) / np.std(x)
intercept = np.mean(y) - gradient * np.mean(x)
fitted = gradient * x + intercept
residual = y - fitted'''
st.code(code1, language="python")

st.subheader('Regression line drawn')
st.write('This is the linear regression line, it shows the most possible Y value given the X value, which are life expectancy and log GDP per capita in this case. ')
fig1 = px.scatter(grouped, 'Log GDP per capita', 'Healthy life expectancy at birth', trendline='ols', hover_name='Country name')
st.plotly_chart(fig1)

st.subheader('Residual plot drawn')
st.write('A residual plot is the difference of each point from the regression line, it is used to verify if the relationship is linear. As long as the points are evenly spread then it is linear, which in this case it’s good enough. ')
fig2 = px.scatter(grouped, 'Log GDP per capita', 'residuals')
st.plotly_chart(fig2)

def predictLE(gdp):
    return gradient * gdp + intercept

st.header('Interactive slider for prediction of life expectancy based on Log GDP per capita')
st.subheader('The code for the interactive slider')
st.write('Here is a simple way to utilize the linear regression model I’ve made to predict the life expectancy of countries, given the log GDP per capita')
code2 = '''def predictLE(gdp):
    return gradient * gdp + intercept'''
st.code(code2, language='python')

sliderinput = st.slider("Slider to predict the life expectancy in years", min_value=7.0, max_value=11.5, step=0.01)
slideroutput = predictLE(sliderinput).round(2)
st.write('The slider input of log GDP per capita on the X axis is', sliderinput)
st.write('The predicted life expectency of the Y axis is', slideroutput, 'years')

st.header('Conclusion')
st.write('The code works very well and so does the prediction model. However, I don’t think all of the variance can be modelled with linear regression, as can seen from R squared being only 0.73, which means only 0.73 of the variance in the data can be explained using linear regression. From a realistic standpoint, our income increases faster than our medical capabilities, so the increase of life expectancy should start to flat out at high GDP rates. This is evident in the residual plot seen by lack of points in the top right. A logarithmic model might be more suitable, but it also might not because it is only the top that flats out. Overall, I think this gives a lot of insight to the distance between societies and countries in our world. ')

st.header('Additional resources')
st.write('Here are some tools that utilize all the columns from the original table. Use the drop down selection boxes to select the statistics to compare. ')
st.subheader('Scatter plot of 2 statistics to visualize their relationship(Choosing the same two will result in a diagonal line)')

allgrouped = happy.groupby('Country name').mean().reset_index()

graphsize = 600

col1, col2 = st.beta_columns([3, 4])
with col1:
    option1 = st.selectbox('X axis',['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])
    option2 = st.selectbox('Y axis',['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])

fig3 = px.scatter(allgrouped, option1, option2, color_discrete_sequence=px.colors.qualitative.T10, hover_name='Country name', width=graphsize, height=graphsize)

with col2:
    st.plotly_chart(fig3)
    
st.subheader('Yearly progression box plot of statistics given by the original table of the entire world')
st.write('A box plot shows the rough distribution of the statistic, the two ends of the whiskers are the maximum and the minimum, the middle line is the median and the top and bottom of the box are the 75th percentile and the 25th percentile, hover over each box to display the exact numbers.')
st.write('There is a significant lack of information for the year 2005, hence the huge distance between 2005 and 2006.')

col3, col4 = st.beta_columns([3, 4])
with col3:
    option3 = st.selectbox('Statistic to compare', ['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices', 'Generosity', 'Perceptions of corruption'])

fig4 = px.box(happy, 'year', option3, color_discrete_sequence=px.colors.qualitative.T10, width=graphsize, height=graphsize)

with col4:
    st.plotly_chart(fig4)

st.subheader('Box plot of countries and the statistic to compare, sorted by their median')
st.write('A box plot shows the rough distribution of the statistic, the two ends of the whiskers are the maximum and the minimum, the middle line is the median and the top and bottom of the box are the 75th percentile and the 25th percentile, hover over each box to display the exact numbers.')

col5, col6 = st.beta_columns([3, 4])

with col5:
    option4 = st.selectbox('Statistic to compare', ['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices', 'Generosity', 'Perceptions of corruption'], key='aaa')
    numberinput1 = st.number_input('Number of countries to show', min_value=1, max_value=100, value=15)
sortedseries = happy.groupby('Country name').median().reset_index().sort_values(option4, ascending=False)[['Country name']]
indexed = sortedseries[:numberinput1].reset_index()['index']
pivoted = happy.pivot(index='Country name', columns='year', values=option4).reset_index().iloc[np.array(indexed)]
fig5 = px.box(pivoted, pivoted.columns[1:], 'Country name', height=numberinput1*30+150, width=graphsize, color_discrete_sequence=px.colors.qualitative.T10, labels={'value':str(option4)})

with col6:
    st.plotly_chart(fig5)

st.write('The code used for the box plot above')
code3 = '''sortedseries = happy.groupby('Country name').median().reset_index().sort_values(option4, ascending=False)[['Country name']]
indexed = sortedseries[:numberinput1].reset_index()['index']
pivoted = happy.pivot(index='Country name', columns='year', values=option4).reset_index().iloc[np.array(indexed)]
fig5 = px.box(pivoted, pivoted.columns[1:], 'Country name', height=numberinput1*30+150, width=graphsize)'''
st.code(code3, language='python')
st.write('Through this case study, I experienced how to use data analysis to explore the relationships between variables in our world, and how the world can be quantified into numbers, as each feature of the world combines together to form our world. When I perform a linear regression analysis, I am only able to visualize a correlation between two variables, but the causal relationships are ignored because statistics cannot prove cause and effect. The two variables I was comparing in the case study definitely did not have a direct causal relationship, instead there must be several similar causes to both of them. Furthermore, the real world is still much more complex than if it were quantified into numbers, after all, there is only a limited type of numbers to quantify into. Although this case study is simpler than the real world because of the previously mentioned points, it still gives us certain amounts of insight to our world. Therefore, this case study still proves to be worth some value in terms of statistics, it reveals a simple correlation between income and life expectancy. ')
# color_discrete_sequence=px.colors.qualitative.T10, width=graphsize, height=graphsize
#'Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth','Freedom to make life choices', 'Generosity', 'Perceptions of corruption'

    # slider4 = st.text_input('Top amount')
    # print(slider4)
    # o = slider4
    # success = False
    # try:
    #     value = slider4
    #     int(value)
    #     success = True
    #     p = int(value)
    # except:
    #     st.error('oof')
