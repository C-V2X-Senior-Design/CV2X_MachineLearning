# Main program file for running simulator and different models
import os
from random import randint
import sys
import numpy as np
from simulator import ResourcePoolSim
from preprocess import Preprocessor
from models import SimpleMNISTModel
from tqdm import tqdm

DIR = "data/"
N = 1000

x_train = [] # resource pool training
y_train = [] # training labels
x_test = [] # resource pool test
y_test = [] # testing labels

# Simulate data
sim = ResourcePoolSim(5, 10, 10)
for i in tqdm(range(N)):
    sim.generateGrid(jamType=randint(0,1), RBGAlloc=randint(0,1))
sim.writeGridToFile()

# Preprocess simulated data
preprocessor = Preprocessor(DIR)
x_train, y_train = preprocessor.get_train_set() # get training data
x_test, y_test = preprocessor.get_test_set() # get testing data

# Run and train model on preprocessed data
MNISTmodel = SimpleMNISTModel()
MNISTmodel.summary() # confirm that model has been created
MNISTmodel.train(x_train=x_train, y_train=y_train)

# test model on testing data
MNISTmodel.test(x_test=x_test, y_test=y_test)
