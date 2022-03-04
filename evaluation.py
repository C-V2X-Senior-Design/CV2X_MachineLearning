import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess import Preprocessor
import numpy as np

models = []
names = []
MODELS_DIR = "models/"
DIR = "data/"

# Retrieve models
files = os.listdir(MODELS_DIR)
for model in files:
    m = tf.keras.models.load_model(f'{MODELS_DIR}{model}')
    models.append((m, model))

# Create random test set from data
p = Preprocessor(DIR, N=100000)
x_test, y_test = p.get_test_set()
p.plot_test_set()

# Go through models and evaluate
for model in models:
    print(f"{model[1]}")
    loss, acc = model[0].evaluate(np.array(x_test), np.array(y_test), verbose=2)
    print(f"accuracy:\t{100 * acc}%")
    print(f"loss:\t{100*loss}%")