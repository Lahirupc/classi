#!flask/bin/python
# @Date: 2017-09-19
# @Last Modified by: Lahiru Karunaratne
# @Last Modified Date: 2017-09-25

import pandas
import sklearn.datasets
import sklearn.metrics
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import numpy as np
from flask import Flask, request, jsonify, g


app = Flask(__name__)
(clf, x, y) = (0, 0, 0)


@app.route('/load_csv', methods=['post'])
def load_csv():
    global x, y
    # extracting the values form the POST JSON body
    # JSON body format: {"path":"document path", "x_column":"2", "y_column":"5"}
    path = request.json['path']
    x_column = request.json['x_column']
    y_column = request.json['y_column']

    # loading the training data
    try:
        g.df = pandas.read_csv(path)
    except:
        return jsonify({'message': 'Loading failed!'})

    # extracting data from the df
    array = g.df.values
    x = array[:, x_column] #2
    y = array[:, y_column] #5
    g.df = None
    return jsonify({'message': 'Successfully loaded!'})


@app.route('/select_model', methods=['post'])
def select_model():
    global clf
    # JSON body format: {"model":"svm","n_neighbors":"0.3-only for knn","weights":"uniform-only for knn"}
    # extracting the values form the POST JSON body
    modelarg = request.json['model']
    n_neighbors = request.json['n_neighbors']
    weights = request.json['weights']

    # selecting the relevant model
    if modelarg == 'svm':
        clf = sklearn.svm.LinearSVC()
    elif modelarg == 'mnb':
        clf = sklearn.naive_bayes.MultinomialNB()
    elif modelarg == 'knn':
        if not (n_neighbors.isdigit()):
            n_neighbors = 11
        elif not (weights == 'uniform' or weights == 'distance'):
            weights = 'uniform'
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    # making the return string
    rstr = 'Model selected: ' + modelarg
    if modelarg == 'knn':
        rstr = rstr + '  (' + n_neighbors + ' ' + weights + ')'

    return jsonify({'message': rstr})


@app.route('/train_model')
def train_model():
    # GET request, no variables
    global clf, x, y
    # calculating TF-IDF
    g.t = calculateTFIDF(x)
    # training the model with sample data
    try:
        clf.fit(g.t, y)
    except:
        return jsonify({'message': 'Training failed!'})
    else:
        return jsonify({'message': 'Successfully trained!'})


@app.route('/evaluate_model', methods=['post'])
def evaluate_model():
    global clf, x, y
    # extracting the values form the POST JSON body
    # JSON body format: {"test_size":"0.3","confusion":"false"}
    test_size = request.json['test_size']
    confusion = request.json['confusion']

    # Loading the labels in the y column
    labels = np.unique(y)
    labels.sort()

    # calculating TF-IDF
    g.t = calculateTFIDF(x)

    # training the model with sample data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(g.t, y, test_size=test_size)
    clf.fit(x_train, y_train)

    y_predicted = clf.predict(x_test)

    if not confusion:
        return sklearn.metrics.classification_report(y_test, y_predicted, target_names=labels)
    else:
        return np.array_str(sklearn.metrics.confusion_matrix(y_test, y_predicted))


@app.route('/classify_document', methods=['post'])
def classify_document():
    # extracting the values form the POST JSON body
    # JSON body format: {"data":"Document Text Here"}
    doc = request.json['data']

    # appending test data to x
    g.c = np.append(x, [doc], axis=0)

    # calculating TF-IDF
    g.c = calculateTFIDF(g.c)

    # predicting the category for the document
    category = clf.predict(g.c[-1])
    return jsonify({'category': category[0]})


def calculateTFIDF(x):
    # calculate the BOW representation
    word_counts = bagOfWords(x)
    # calculating TF-IDF
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    return tf_transformer.transform(word_counts)


def bagOfWords(files_data):
    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    return count_vector.fit_transform(files_data)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
