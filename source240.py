#Import necessary libraries

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
#Load the dataset
                              

file_path = 'D:/Project 240/new code/sentiment_analysis_dataset .csv'
dataset = pd.read_csv(file_path, encoding='ISO-8859-1')

#Initialize VADER Sentiment Analyzer

vader_analyzer = SentimentIntensityAnalyzer()
#Function to calculate sentiment using VADER

def vader_sentiment(review):
    return vader_analyzer.polarity_scores(review)['compound']
#Function to calculate sentiment using TextBlob

def textblob_sentiment(review):
    return TextBlob(review).sentiment.polarity
#Apply VADER and TextBlob sentiment analysis on the 'Review Text'

dataset['VADER Sentiment'] = dataset['Review Text'].apply(vader_sentiment)
dataset['TextBlob Sentiment'] = dataset['Review Text'].apply(textblob_sentiment)
#Define ACL thresholds for categorizing sentiment

def acl_category(sentiment_score):
    if sentiment_score > 0.05:
        return 'Positive'
    elif sentiment_score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'
#Apply ACL to categorize the VADER sentiment scores

dataset['Sentiment Category'] = dataset['VADER Sentiment'].apply(acl_category)
#'Sentiment Category' to filter or analyze specific categories
#For example, filtering positive reviews

positive_reviews = dataset[dataset['Sentiment Category'] == 'Positive']
#Aggregate sentiment by year

yearly_sentiment = dataset.groupby('Year').agg({'VADER Sentiment': 'mean', 'TextBlob Sentiment': 'mean'}).reset_index()

#Plot the trend over the years

plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='VADER Sentiment', data=yearly_sentiment, marker='o', label='VADER Sentiment')
sns.lineplot(x='Year', y='TextBlob Sentiment', data=yearly_sentiment, marker='o', label='TextBlob Sentiment')
plt.title('Average Sentiment of Product Reviews Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Sentiment')
plt.legend()
plt.show()
#Gender distribution in the dataset

plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', data=dataset)
plt.title('Distribution of Reviews by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
#State distribution in the dataset

plt.figure(figsize=(12, 8))
sns.countplot(y='state', data=dataset, order=dataset['state'].value_counts().index)
plt.title('Distribution of Reviews by State')
plt.xlabel('Count')
plt.ylabel('State')
plt.show()

#Create age groups

bins = [18, 25, 35, 45, 55, 65, 75]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74']
dataset['Age Group'] = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
#Age group distribution

plt.figure(figsize=(10, 6))
sns.countplot(x='Age Group', data=dataset, order=dataset['Age Group'].value_counts().index)
plt.title('Distribution of Reviews by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()
#Calculate average sentiment for each product

product_sentiment = dataset.groupby('Product  Name').agg({'VADER Sentiment': 'mean', 'TextBlob Sentiment': 'mean'}).reset_index()
#Top 10 most positively rated products

top_positive_products = product_sentiment.sort_values(by='VADER Sentiment', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='VADER Sentiment', y='Product  Name', data=top_positive_products)
plt.title('Top 10 Most Positively Rated Products')
plt.xlabel('Average VADER Sentiment')
plt.ylabel('Product Name')
plt.show()
#Save the enhanced dataset with sentiment scores and categories

dataset.to_csv('enhanced_sentiment_analysis_dataset_with_acl.csv', index=False)
# Streamlit App
import streamlit as st
import pandas as pd


st.title("Sentiment Analysis Dashboard")

st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    dataset = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    st.write("Dataset Preview", dataset.head())

    # Apply sentiment analysis
    dataset['VADER Sentiment'] = dataset['Review Text'].apply(vader_sentiment)
    dataset['TextBlob Sentiment'] = dataset['Review Text'].apply(textblob_sentiment)
    dataset['Sentiment Category'] = dataset['VADER Sentiment'].apply(acl_category)

    st.write("Sentiment Analysis Results", dataset.head())

    # Plotting
    st.header("Average Sentiment Over the Years")
    yearly_sentiment = dataset.groupby('Year').agg({'VADER Sentiment': 'mean', 'TextBlob Sentiment': 'mean'}).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Year', y='VADER Sentiment', data=yearly_sentiment, marker='o', label='VADER Sentiment', ax=ax)
    sns.lineplot(x='Year', y='TextBlob Sentiment', data=yearly_sentiment, marker='o', label='TextBlob Sentiment', ax=ax)
    plt.title('Average Sentiment of Product Reviews Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Average Sentiment')
    plt.legend()
    st.pyplot(fig)

    st.header("Distribution of Reviews by Gender")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Gender', data=dataset, ax=ax)
    plt.title('Distribution of Reviews by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    st.pyplot(fig)

    st.header("Distribution of Reviews by State")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(y='state', data=dataset, order=dataset['state'].value_counts().index, ax=ax)
    plt.title('Distribution of Reviews by State')
    plt.xlabel('Count')
    plt.ylabel('State')
    st.pyplot(fig)

    st.header("Distribution of Reviews by Age Group")
    bins = [18, 25, 35, 45, 55, 65, 75]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74']
    dataset['Age Group'] = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Age Group', data=dataset, order=dataset['Age Group'].value_counts().index, ax=ax)
    plt.title('Distribution of Reviews by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    st.pyplot(fig)

    st.header("Top 10 Most Positively Rated Products")
    product_sentiment = dataset.groupby('Product  Name').agg({'VADER Sentiment': 'mean', 'TextBlob Sentiment': 'mean'}).reset_index()
    top_positive_products = product_sentiment.sort_values(by='VADER Sentiment', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='VADER Sentiment', y='Product  Name', data=top_positive_products, ax=ax)
    plt.title('Top 10 Most Positively Rated Products')
    plt.xlabel('Average VADER Sentiment')
    plt.ylabel('Product Name')
    st.pyplot(fig)

    # Save enhanced dataset
    st.sidebar.header("Save Enhanced Dataset")
    if st.sidebar.button("Save Dataset"):
        output = StringIO()
        dataset.to_csv(output, index=False)
        st.download_button(label="Download Enhanced Dataset", data=output.getvalue(), file_name='enhanced_sentiment_analysis_dataset_with_acl.csv', mime='text/csv')
