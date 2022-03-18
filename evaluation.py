from ast import Mult
import os
from random import Random
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess import Preprocessor
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

models = []
sk_models = []
names = []
MODELS_DIR = "models/"
DIR = "data/"
N = 10000

# Retrieve models
files = os.listdir(MODELS_DIR)
for model in files:
    m = tf.keras.models.load_model(f'{MODELS_DIR}{model}')
    models.append((m, model))

# Create random test set from data
p = Preprocessor(DIR, N)
x_test, y_test = p.get_test_set()

# for sklearn models
x_train, y_train = p.get_train_set()
# p.plot_test_set()

# Go through models and evaluate
for model in models:
    print(f"{model[1]}")
    loss, acc = model[0].evaluate(np.array(x_test), np.array(y_test), verbose=2)
    print(f"accuracy:\t{100 * acc}%")
    print(f"loss:\t{100*loss}%")

# Testing out sklearn models
x_train = np.reshape(x_train, (len(x_train), 500))
y_train = np.reshape(y_train, len(y_train))
x_test = np.reshape(x_test, (len(x_test), 500))
y_test = np.reshape(y_test, len(y_test))

mnb = MultinomialNB().fit(x_train, y_train)
print("Naive Bayes")
print(f"train score: {mnb.score(x_train, y_train)}\ttest score: {mnb.score(x_test, y_test)}")
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
print("Logistic Regression")
print(f"train score: {lr.score(x_train, y_train)}\ttest score: {lr.score(x_test, y_test)}")
knn = KNeighborsClassifier(algorithm='brute', n_jobs=-1)
knn.fit(x_train, y_train)
print("K-Nearest Neighbours")
print(f"train score: {knn.score(x_train, y_train)}\ttest score: {knn.score(x_test, y_test)}")
svm = LinearSVC(C=0.0001)
svm.fit(x_train, y_train)
print("Linear Support Vector Machine")
print(f"train score: {svm.score(x_train, y_train)}\ttest score: {svm.score(x_test, y_test)}")
rf = RandomForestClassifier(n_estimators=30, max_depth=9)
rf.fit(x_train, y_train)
print("Random Forest")
print(f"train score: {rf.score(x_train, y_train)}\ttest score: {rf.score(x_test, y_test)}")