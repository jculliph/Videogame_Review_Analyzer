#!/usr/bin/env python
# coding: utf-8
# author: Joshua Cullipher

# This first section is used to read the dataset into a pandas dataframe and do some basic data cleaning and
# preparation.
import pandas as pd
import gzip
import json
import nltk
import numpy as np
import random
from sklearn.cluster import AffinityPropagation
from difflib import SequenceMatcher
import plotly.express as px
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


def parse(path):
    g = gzip.open(path, 'rb')

    for l in g:
        yield json.loads(l)

    
def getDF(path):
    i = 0
    df = {}

    for d in parse(path):
        df[i] = d
        i += 1

    return pd.DataFrame.from_dict(df, orient='index')


def skip_unwanted(pos_tuple):
    word, tag = pos_tuple

    if not word.isalpha() or word in unwanted:
        return False

    if tag.startswith("NN"):
        return False

    return True


# This loads the dataset into a pandas dataframe (Change path as needed)
df = getDF(r'C:\Users\Joshua\Downloads\Video_Games_5.json.gz')

nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# This reads the reviews into categories and filters words based on their rating. 3 star ratings are considered neutral
# and ignored.
positive_reviews = df.loc[df["overall"] >= 4, "reviewText"].to_string()
negative_reviews = df.loc[df["overall"] < 3, "reviewText"].to_string()

positive_reviews_token = nltk.word_tokenize(positive_reviews)
positive_pos = nltk.pos_tag(positive_reviews_token) 

negative_reviews_token = nltk.word_tokenize(negative_reviews)
negative_pos = nltk.pos_tag(negative_reviews_token)

unwanted = nltk.corpus.stopwords.words("english")

positive_words = [word for word, tag in filter(
    skip_unwanted,
    positive_pos
)]

negative_words = [word for word, tag in filter(
    skip_unwanted,
    negative_pos
)]


# The second section uses affinity propagation, a form of cluster analysis, to gain information on the dataset.
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def word_affinity(text):
    words = text
    sample_size = min(1000, len(words)//10)
    words = random.sample(words, sample_size)
    words = np.asarray(words) 
    word_similarity = -1*np.array([[similarity(w1, w2) for w1 in words] for w2 in words])

    affprop = AffinityPropagation(max_iter=1000, preference=0, affinity="precomputed", random_state=0)
    affprop.fit_predict(word_similarity)
    
    word_clusters = list()
    for cluster_id in np.unique(affprop.labels_):
        cluster = words[affprop.cluster_centers_indices_[cluster_id]]
        word_clusters.append(cluster.lower())

    return word_clusters, affprop


def graph_clusters(clustering_object, word_clusters):
    data_points = clustering_object.cluster_centers_indices_
    cluster_dict = {'cluster_names': word_clusters, 'x': data_points, 'y': data_points}
    data_frame = pd.DataFrame(data=cluster_dict)
    data_frame['size'] = data_frame.apply(lambda row: len((clustering_object.labels_ == row.x).nonzero()[0]) + 1,
                                          axis=1)
    fig = px.scatter(data_frame, x='x', y='y', size='size', text='cluster_names')

    return fig


def graph_frequency(freq_data):
    words = [word for word, count in freq_data]
    count = [count for word, count in freq_data]
    freq_dict = {'words': words, 'count': count}
    data_frame = pd.DataFrame(data=freq_dict)
    fig = px.bar(data_frame, x='words', y='count')

    return fig


def graph_pie_chart(pie_dict):
    data_frame = pd.DataFrame(data=pie_dict)
    fig = px.pie(data_frame, values='values', names='categories')

    return fig


# This section uses Affinity Propagation, a form of clustering, to group the word lists
positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]
    
positive_top_words = {word for word, count in positive_fd.most_common(1000)}
negative_top_words = {word for word, count in negative_fd.most_common(1000)}

positive_word_clusters, positive_affinity_propagation = word_affinity(positive_top_words)
negative_word_clusters, negative_affinity_propagation = word_affinity(negative_top_words)

fig_1 = graph_clusters(positive_affinity_propagation, positive_word_clusters)
fig_2 = graph_clusters(negative_affinity_propagation, negative_word_clusters)
fig_3 = graph_frequency(positive_fd.most_common(10))
fig_4 = graph_frequency(negative_fd.most_common(10))
pie_dict = {'categories': ["positive words", "negative words"], 'values': [len(positive_word_clusters),
                                                                           len(negative_word_clusters)]}
fig_5 = graph_pie_chart(pie_dict)

if not os.path.exists("images"):
    os.mkdir("images")

fig_1.write_image("images/fig1.png")
fig_2.write_image("images/fig2.png")
fig_3.write_image("images/fig3.png")
fig_4.write_image("images/fig4.png")
fig_5.write_image("images/fig5.png")


# This section prepares the information gained from affinity propagation for use in a machine learning algorithm.
# It also performs feature extraction on the dataset.
def extract_features(text):
    text = str(text)
    features = list()
    features_list = list()
    positive_score_words = 0
    negative_score_words = 0
    
    for word in nltk.word_tokenize(text):
        if word.lower() in positive_word_clusters:
            positive_score_words += 1
        if word.lower() in negative_word_clusters:
            negative_score_words += 1
            
    features.append((positive_score_words, negative_score_words))

    for item in features:
        for thing in item:
            features_list.append(thing)
        
    return features_list


# This section prepares the data gained from Affinity Propagation to be used by a machine learning algorithm
features = [extract_features(text) for text in df["reviewText"]]
ratings = [ratings for ratings in df["overall"]]


# This section uses the dataset features and information from affinity propagation to train a logistic regressor to
# make predictions with data. This section splits the data into training and testing sets and uses logistic regression
# to process the data
X_train, X_test, y_train, y_test = train_test_split(features, ratings, train_size=0.7, random_state=0)
regr = LogisticRegression().fit(X_train, y_train)

regr.predict(X_test)
accuracy = "{:.2%}".format(regr.score(X_test, y_test))

print(f"Model predictive accuracy is: {accuracy}")

# This section saves the created items above using pickling.
with open('model.pkl', 'wb') as model_file:
    pickle.dump(regr, model_file)

with open('positive_word_clusters.pkl', 'wb') as positive_file:
    pickle.dump(positive_word_clusters, positive_file)

with open('negative_word_clusters.pkl', 'wb') as negative_file:
    pickle.dump(negative_word_clusters, negative_file)
