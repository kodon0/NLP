#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:52:04 2020

@author: kieranodonnell
"""


# Sentiment analysis in Crude Oil Webpage

# General url = https://oilprice.com/

# Import libraries

import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Start with empty lists for urls, headlines and text
url_list = []
headlines = []
date_time = []
news_text = []

# Set n in range for number for web pages

for i in range (1,3): # Numerically listed pages makes this simple -> generates first 10 apges of data
    url = 'https://oilprice.com/Energy/Crude-Oil/Page-{}.html'.format(i)
    request = requests.get(url)
    soup = BeautifulSoup(request.text,"html.parser")
    for links in soup.find_all('div', {'class': 'categoryArticle'}): # Location of article in html = categoryClass
        for info in links.find_all('a'):  # Tag A has href
            if info.get('href') not in url_list: # Avoid duplicate hrefs to go into url_list
                url_list.append(info.get('href'))
                
                

for www in url_list:
    # Extract headlines from url itself after last '/'
    headlines.append(www.split('/')[-1].replace('-',' ')) # replace dash with space
    request = requests.get(www)
    soup = BeautifulSoup(request.text,"html.parser")
    
    # Get dates/times for articles
    # This relies on the website being properly filled out -> one article had no date...
    # Could also omit date if required...
    for dates in soup.find_all('span', {'class': 'article_byline'}): # Location of dates in html under span
        date_time.append(dates.text.split('-')[-1]) # Extract append the text with it split at dashes
        
    # Find and store text -> within <p> tags in html
    temp = [] # Getting more info than needed so temp with be a temporary list
    for news in soup.find_all('p'):
        temp.append(news.text)
        
    # Selecting only useful text
    # "By", starting from the bottom should give the last sentence in a useful paragraph
    # Looping from bottom means it is more likely to work
    for last_sentence in reversed(temp):
        if last_sentence.split(' ')[0]=='By' and last_sentence.split(' ')[-1]=='Oilprice.com':
            break
        elif last_sentence.split(' ')[0]=='By':
            break
    
    # Skip where 'More Info' is found, and join preceding text all the way to last sentence
    joined_text = ' '.join(temp[temp.index("More Info")+1: temp.index(last_sentence)])
    news_text.append(joined_text) # This is the useful text body
    
    
# Save everything into a DF
news_df = pd.DataFrame({'Headline': headlines, 'Date':date_time, 'Text':news_text})


# Use Vader or other to analyze sentiment

analyser = SentimentIntensityAnalyzer()

# Define a function to generate scores - vader
def comp_score_vader(text):
   return analyser.polarity_scores(text)["compound"]   
  
news_df["sentiment-vader"] = news_df["Text"].apply(comp_score_vader)

# Define a function to generate scores - textblob
def comp_score_textblob(text):
    return TextBlob(text).sentiment

news_df["polarity/subjectivty"] = news_df["Text"].apply(comp_score_textblob)

