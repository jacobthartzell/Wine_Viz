#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:57:51 2023

@author: jhartzel1
"""


import pandas as pd
import numpy as np
import nltk as nk
import re
import string 
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

#Load in datasets and concat different 

formal = pd.read_csv(r'/Users/jhartzel1/Documents/Documents/IAA_Masters_of_Analytics/Side_Project/Formalized_Wine_Ratings/wine_2.csv')


#Clean data

# column_in_df = column you want values removed from
# cutoff = character cutoff to remove
def remove_long(column_in_df, cutoff) :
    for i in range(len(column_in_df)) :
        if len(column_in_df[i]) > cutoff :
            column_in_df[i] = np.nan
        else :
            i = i + 1

def get_country(s) :
    return s.rsplit(',', 1)[-1]
            

# Replace percentages and money signs so that columns can be labeled float
formal['alcohol'] = formal['alcohol'].str.rstrip('%')
formal['alcohol'] = formal['alcohol'].astype(float)
formal['price'] = formal['price'].str.lstrip('$')
formal['price'] = formal['price'].astype(str)

# remove strings longer than 7 values 

remove_long(formal['price'], 7)

formal['price'] = formal['price'].astype(float)

#Remove values with a dollar sign from appelation column
formal['appellation'] = np.where(formal['appellation'].str.contains('\$'), np.nan, formal['appellation'])
formal['appellation'] = np.where(formal['appellation'].str.contains('\Buy Now'), np.nan, formal['appellation'])



#Create a column for country
formal['appellation'] = formal['appellation'].astype(str)
#formal['country'] = formal['appellation'].apply(get_country)

#Check to make sure there aren't "Buy Now" Values
#print(formal['country'].unique())

#Clean the 'appellation' column up. Get the number of commas, split the text on that number of commas, and get the region
#Also, if appellation has 3 commas, then it probably has a state
def split_string(row):
    num_commas = row['appellation'].count(',')
    
    if num_commas >= 3:
        return pd.Series(row['appellation'].split(', ', 3), index=['subregion', 'region', 'state', 'country'])
    elif num_commas == 2:
        return pd.Series(row['appellation'].split(', ', 2), index=['subregion', 'region', 'country'])
    elif num_commas == 1:
        return pd.Series(row['appellation'].split(', ', 1), index=['region', 'country'])
    else:
        return pd.Series({'country': row['appellation']})

formal['country'] = np.nan    
formal['state'] = np.nan
formal['region'] = np.nan
formal['subregion'] = np.nan

# Apply the function to the DataFrame
formal[['country', 'region', 'state', 'subregion']] = formal.apply(split_string, axis=1)

us_states = ['Alabama',
            'Alaska',
            'Arizona',
            'Arkansas',
            'California',
            'Colorado',
            'Connecticut',
            'Delaware',
            'Florida',
            'Georgia',
            'Hawaii',
            'Idaho',
            'Illinois',
            'Indiana',
            'Iowa',
            'Kansas',
            'Kentucky',
            'Louisiana',
            'Maine',
            'Maryland',
            'Massachusetts',
            'Michigan',
            'Minnesota',
            'Mississippi',
            'Missouri',
            'Montana',
            'Nebraska',
            'Nevada',
            'New Hampshire',
            'New Jersey',
            'New Mexico',
            'New York',
            'North Carolina',
            'North Dakota',
            'Ohio',
            'Oklahoma',
            'Oregon',
            'Pennsylvania',
            'Rhode Island',
            'South Carolina',
            'South Dakota',
            'Tennessee',
            'Texas',
            'Utah',
            'Vermont',
            'Virginia',
            'Washington',
            'West Virginia',
            'Wisconsin',
            'Wyoming']

for i in range(len(formal['region'])):
    if formal['region'][i] in us_states: 
        formal['state'][i] = formal['region'][i]
        
#Check for years present in the string, save them in column 'year'
formal['year'] = formal['wine'].apply(lambda x: re.search(r'\b\d{4}\b', str(x)).group() if re.search(r'\b\d{4}\b', str(x)) else None)
formal['rating_out_of_100'] = (formal['rating'] - 80) * 5
formal['Value Index'] = formal['rating_out_of_100']/formal['price']
formal_sorted = formal.sort_values(by = 'Value Index', ascending = False)        

#Identify some outliers
#Make a scatterplot to look at both price and rating to get an understanding of their 
plt.scatter(formal['rating'],formal['price'], color='blue')
# Show the plot
plt.show()

#Identify and handle missing values


#Utilize studentized residuals to identify outliers (>3)
formal = formal.dropna(subset = ['price'])
X = formal['price']
y = formal[['rating_out_of_100']]
X = sm.add_constant(X)
model = sm.OLS(y, X.astype(float)).fit()

influence = OLSInfluence(model)
studentized_residuals = influence.resid_studentized_internal
outliers = np.abs(studentized_residuals) > 3
formal['outlier'] = outliers
sum(formal['outlier'])


# Plot the studentized residuals
plt.scatter(formal['price'], studentized_residuals, c=outliers, cmap='viridis')
plt.axhline(y=2, color='r', linestyle='--', label='Threshold')
plt.axhline(y=-2, color='r', linestyle='--')
plt.xlabel('X')
plt.ylabel('Studentized Residuals')
plt.title('Scatter Plot of Studentized Residuals')
plt.legend()
plt.show()

#Remove those that are outliers (marked true)
formal_no_na = formal[~formal['outlier']]
formal_no_na = formal_no_na.drop(columns = ['outlier'])
formal_no_na['price'].max()

#Try a new index, one where the rating is exponentiated by 1.5
formal_no_na['rating_20'] = formal['rating'] - 80
formal_no_na['Value_rating_heavy_index'] = (formal_no_na['rating_20']**2)/formal_no_na['price']

#Now, get an idea of which regions (countries) are most mentioned in the table
country_count = formal_no_na['country'].value_counts()
plt.pie(country_count, labels = country_count.index, autopct='%1.1f%%', startangle=90, counterclock = False)

#Country 'New York, US' identified, replace for country = US, state = "New York"
formal_no_na.loc[formal_no_na['country'] == 'New York, US', 'state'] = 'New York'
formal_no_na['country'] = formal_no_na['country'].str.replace('New York, US', 'US')


#Save formal_no_na as csv file
#formal_no_na.to_csv('/Users/jhartzel1/Documents/Documents/IAA_Masters_of_Analytics/Side_Project/formal_no_na.csv', index = False)

