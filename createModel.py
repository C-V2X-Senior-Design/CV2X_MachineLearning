# Main program file for running simulator and different models
import os
from random import randint
import sys
import numpy as np
from simulator import ResourcePoolSim
from preprocess import Preprocessor
import models as m
from tqdm import tqdm
import tensorflow as tf

DIR = "data/"
N = 10000

models = []

x_train = [] # resource pool training
y_train = [] # training labels
x_test = [] # resource pool test
y_test = [] # testing labels

# Simulate data
sim = ResourcePoolSim(N, 5, 10, 10)
for i in tqdm(range(N)):
    sim.generateGrid(jamType=randint(0,1), RBGAlloc=1)
    # sim.generateGrid(jamType=randint(0, 1), RBGAlloc=0)
sim.writeGridToFile()

# Preprocess simulated data
preprocessor = Preprocessor(DIR, N)
x_train, y_train = preprocessor.get_train_set() # get training data
x_test, y_test = preprocessor.get_test_set() # get testing data
# preprocessor.plot_train_set()

# Run and train model on preprocessed data
MNISTmodel = m.SimpleMNISTModel()
MNISTmodel.summary() # confirm that model has been created
models.append(MNISTmodel)
MNISTmodel2 = m.SimpleMNISTModelv2()
MNISTmodel2.summary()
models.append(MNISTmodel2)
MNISTmodel3 = m.SimpleMNISTModelv3()
MNISTmodel3.summary()
models.append(MNISTmodel3)
CNNmodel = m.ImprovedCNNModel()
CNNmodel.summary()
models.append(CNNmodel)

# x_train = tf.expand_dims(x_train, 0)
# print(np.shape(x_train[0])) # get shape
# print(np.shape(y_train[0]))
for m in tqdm(models):
    print(f"Running model {m.name}")
    m.train(x_train=np.array(x_train), y_train=np.array(y_train), epoch=int(N/100))
    m.test(x_test=np.array(x_test), y_test=np.array(y_test))
    m.save()