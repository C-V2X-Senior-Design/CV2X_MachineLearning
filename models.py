import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

class Model:
    model = None
    name = ""

    # input size - 2D (X, Y)
    X = 10
    Y = 50

    def summary(self):
        if self.model != None:
            self.model.summary()

    def train(self, x_train, y_train, epoch=10):
        if self.model == None:
            return
        
        self.model.fit(x_train, y_train, epochs=epoch)
    
    def test(self, x_test, y_test):
        if self.model == None:
            return
        
        self.model.evaluate(x_test, y_test, verbose=2)
    
    def save(self):
        if self.model == None:
            return

        self.model.save(f"models/{self.name}")

class SimpleMNISTModel(Model):
    def __init__(self, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(0.001)):
        self.name = "SimpleMNISTModel"
        # creates the model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.X, self.Y)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

class SimpleMNISTModelv2(Model):
    def __init__(self, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(0.001)):
        self.name = "SimpleMNISTModelv2"
        # creates the model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.X, self.Y)),
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(2)
        ])

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

class SimpleMNISTModelv3(Model):
    def __init__(self, loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(0.001)):
        self.name = "SimpleMNISTModelv3"
        # creates the model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.X, self.Y)),
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(2)
        ])

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )

class ImprovedCNNModel(Model):
    def __init__(self, loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(0.001)):
        self.name = "ImprovedCNNModel"
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.X, self.Y)),
            tf.keras.layers.Dense(256, activation='sigmoid'),
            # tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dense(128, activation='sigmoid'),
            # tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dense(64, activation='sigmoid'),
            tf.keras.layers.Dense(2)
        ])

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )