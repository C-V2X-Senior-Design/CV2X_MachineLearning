import resource
from urllib import response
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

encoded_data = open(f"{VALUES_DIR}{fname}").read().splitlines()

print(len(encoded_data))
for i in range(0, len(encoded_data), N):
    temp = []
    for j in range(i, i+N - 1):
        temp.append(int(float(encoded_data[j])))
    decoded_data.append(temp)

print(len(decoded_data))
for rp in decoded_data:
    frames = []
    # print(len(rp))
    for i in range(0, len(rp), 100):
        frames.append(rp[i:i+100-1])
    print(len(frames))
    resource_pools.append(frames)

# TODO better preprocessing
print(len(resource_pools))