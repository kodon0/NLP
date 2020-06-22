#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:05:24 2020

@author: kieranodonnell
"""


# ML (Naive Bayes) classification on Oil Prices

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
#import pickle

df = pd.read_csv("OilPrice.com_news_articles.csv", encoding = "ISO-8859-1") # Training data

# Split data 

X = df.iloc[:,1]

# Tokenize
vectorizer = CountVectorizer(stop_words = 'english')
X_vect = vectorizer.fit_transform(X.values.astype('U'))
print(X_vect)
print(vectorizer.vocabulary_)

#pickle.dump(vectorizer, open("vectorizer_crude_oil", "wb")) # Save for future use
X_vect = X_vect.todense() # Convert sparse to dense matrix

# Transform data using Tf-Idf

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_vect)
X_tfidf = X_tfidf.todense()

# Using Naive Bayes
X_train = X_tfidf[:,:]
y_train = df.iloc[:,2]

classifier = GaussianNB().fit(X_train, y_train)
#pickle.dump(classifier, open("nb_classifier_crude_oil", "wb")) #Saving classifier for future use

# Make test set - new df
 # This file can be generated as before by selecting an unseen group of text
df_predictions = pd.read_csv("OilPrice.com_news_articles_test.csv", encoding = "ISO-8859-1")

X_test = df_predictions.iloc[:,1]
X_test_vect = vectorizer.transform(X_test.values.astype('U'))
X_test_vect = X_test_vect.todense()

# Tfidf on test set
X_tfidf_test = tfidf.fit_transform(X_test_vect)
X_tfidf_test = X_tfidf_test.todense()

# Make predicitions

y_pred = classifier.predict(X_tfidf_test)

print(y_pred)