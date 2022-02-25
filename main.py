# Main program file for running simulator and different models
import os
from random import randint
import sys
import numpy as np
from simulator import ResourcePoolSim
from preprocess import Preprocessor
from models import SimpleMNISTModel
from tqdm import tqdm
import tensorflow as tf

DIR = "data/"
N = 100000

x_train = [] # resource pool training
y_train = [] # training labels
x_test = [] # resource pool test
y_test = [] # testing labels

# Simulate data
sim = ResourcePoolSim(5, 10, 10)
for i in tqdm(range(N)):
    sim.generateGrid(jamType=randint(0,1), RBGAlloc=randint(0,1))
    # sim.generateGrid(jamType=randint(0, 1), RBGAlloc=0)
sim.writeGridToFile()

# Preprocess simulated data
preprocessor = Preprocessor(DIR, N)
x_train, y_train = preprocessor.get_train_set() # get training data
x_test, y_test = preprocessor.get_test_set() # get testing data
preprocessor.plot_train_set()

# Run and train model on preprocessed data
MNISTmodel = SimpleMNISTModel()
MNISTmodel.summary() # confirm that model has been created

# x_train = tf.expand_dims(x_train, 0)
# print(np.shape(x_train[0])) # get shape
# print(np.shape(y_train[0]))
MNISTmodel.train(x_train=np.array(x_train), y_train=np.array(y_train), epoch=1000)

# test model on testing data
MNISTmodel.test(x_test=np.array(x_test), y_test=np.array(y_test))

# save model for later use
MNISTmodel.save()