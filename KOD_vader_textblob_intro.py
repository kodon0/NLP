#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:30:42 2020

@author: kieranodonnell
"""


# VADER Intro

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Run each line individually and notice output scores

analyser = SentimentIntensityAnalyzer()
analyser.polarity_scores("This is a good course")
analyser.polarity_scores("This is an awesome course") # degree modifier
analyser.polarity_scores("The instructor is so cool")
analyser.polarity_scores("The instructor is so cool!!") # exclaimataion changes score
analyser.polarity_scores("The instructor is so AWESOME!!") # Capitalization changes score
analyser.polarity_scores("Machine learning makes me :)") #emoticons
analyser.polarity_scores("His antics had me ROFL")
analyser.polarity_scores("The movie SUX") #Slangs


# Textblob intro

from textblob import TextBlob

# Textblob classifier is based on movie reviews, so may or may not be good for finance etc

TextBlob("His").sentiment
TextBlob("remarkable").sentiment
TextBlob("work").sentiment
TextBlob("ethic").sentiment
TextBlob("impressed").sentiment
TextBlob("me").sentiment
TextBlob("His remarkable work ethic impressed me").sentiment
