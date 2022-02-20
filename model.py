import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# TODO read data and preprocess
DIR = "data/" # reads from data folder
if len(os.listdir(DIR)) == 0:
    print("No Data Detected!")
else:
    files = os.listdir(DIR)
    input = files[len(os.listdir(DIR)) - 1] # get latest entry
    df = pd.read_csv(f"{DIR}{input}")
    print(df)

# NOTE each index is a resource pool with 5 frames (or len(resource_pool))
# pass this in model as X amount of frames and specify SUBCHANNELS and SUBFRAMES from data file only


# TODO create model and categorized between jammed and not jammed linearly