import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# TODO read data and preprocess
VALUES_DIR = "data/values/" # reads from values folder
LABELS_DIR = "data/labels/"
N = 500
decoded_data = []
labels = []
resource_pools = []

dir = os.listdir(VALUES_DIR)
fname = dir[0]
print(f"opening {fname}")

# TODO split the serialization by \n to avoid going through it again
encoded_data = open(f"{VALUES_DIR}{fname}").read().splitlines()
dir = os.listdir(LABELS_DIR)
fname = dir[0]
print(f"opening {fname}")
labels = open(f"{LABELS_DIR}{fname}").read().splitlines()

for i in range(0, len(encoded_data), N):
    temp = []
    for j in range(i, i+N - 1):
        temp.append(int(float(encoded_data[j])))
    resource_pools.append(temp)

# TODO check if labels match
print(len(resource_pools))
print(len(labels))

# test model with MNIST standard
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(500, 10)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()