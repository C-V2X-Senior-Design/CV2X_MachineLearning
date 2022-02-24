from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class Preprocessor:
	resource_pools = []
	labels = []
	def __init__(self, dir):
		N = 500
		value_folder = f"{dir}values/"
		label_folder = f"{dir}labels/"

		# Get serialized resource pools
		val = os.listdir(value_folder)
		fname = val[0]
		print(f"Opening {fname}")
		data = open(f"{value_folder}{fname}").read().splitlines()
		for i in tqdm(range(0, len(data), N)):
			temp = []
			for j in range(i, i+N - 1):
				temp.append(int(float(data[j])))
			self.resource_pools.append(temp)

		# Get labels for each resource pool
		lbl = os.listdir(label_folder)
		fname = lbl[0]
		print(f"Opening {fname}")
		self.labels = open(f"{label_folder}{fname}").read().splitlines()

		self.split_train_test()
		print("Finished preprocessing")
	
	def split_train_test(self):
		# 80/20 split for training and testing data
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.resource_pools, self.labels, test_size=0.20, random_state=42)
	
	def get_train_set(self):
		return self.x_train, self.y_train
	
	def get_test_set(self):
		return self.x_test, self.y_test