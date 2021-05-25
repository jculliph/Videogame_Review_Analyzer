#!/usr/bin/env python
# coding: utf-8
# author: Joshua Cullipher
import nltk
from flask import Flask, render_template, request
import pickle

nltk.download('punkt')

app = Flask(__name__, static_folder='')


@app.route('/')
def home():
    return render_template('index.html')


def extract_features(text):
    text = str(text)
    features = list()
    positive_score_words = 0
    negative_score_words = 0

    with app.open_resource('positive_word_clusters.pkl', 'rb') as positive_file:
        positive_word_clusters = pickle.load(positive_file)
    with app.open_resource('negative_word_clusters.pkl', 'rb') as negative_file:
        negative_word_clusters = pickle.load(negative_file)

    for word in nltk.word_tokenize(text):
        if word.lower() in positive_word_clusters:
            positive_score_words += 1
        if word.lower() in negative_word_clusters:
            negative_score_words += 1

    features.append((positive_score_words, negative_score_words))

    return features


def predictor(text_to_predict):
    text = extract_features(text_to_predict)

    with app.open_resource('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    predicted_result = model.predict_proba(text)

    return predicted_result


@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        predicted_result_array = predictor(list(input_data.values())[0])
        predicted_result = predicted_result_array.tolist()[0]
        prediction = ""

    for item in predicted_result:
        prediction += "The predicted probability of {number} star(s) is: {prob:.2%}.\n".format(
                                                                                number=predicted_result.index(item)+1,
                                                                                prob=item)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run()