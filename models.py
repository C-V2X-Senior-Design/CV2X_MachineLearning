import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# TODO attach this to a bigger models library

class SimpleMNISTModel:
    model = None
    def __init__(self, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(0.001)):
        # creates the model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(50, 10)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    def summary(self):
        if self.model != None:
            self.model.summary()

    # TODO add train and evaluate functions
    def train(self, x_train, y_train, epoch=5):
        if self.model == None:
            return
        
        self.model.fit(x_train, y_train, epochs=epoch)
    
    def test(self, x_test, y_test):
        if self.model == None:
            return
        
        self.model.evaluate(x_test, y_test, verbose=2)