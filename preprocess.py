from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
from PIL import Image

class Preprocessor:
	resource_pools = []
	labels = []
	def __init__(self, dir, N):
		value_folder = f"{dir}values/"
		label_folder = f"{dir}labels/"

		# Get serialized resource pools
		val = os.listdir(value_folder)
		fname = val[0]
		print(f"Opening {fname}")
		data = open(f"{value_folder}{fname}").read().splitlines()

		count = 0
		i = 0
		temp = []
		pbar = tqdm(total=len(data))
		while count < len(data):
			if i == int(len(data) / N):
				self.resource_pools.append(temp)
				i = 0 
				temp = []
			
			temp.append(int(float(data[count])))
			count += 1
			i += 1
			pbar.update(1)
		self.resource_pools.append(temp) # add last temp array

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
		# reshape x_train
		for i in range(len(self.x_train)):
			self.x_train[i] = np.reshape(self.x_train[i], [10, 50])
			# self.x_train[i] = tf.expand_dims(self.x_train[i])
		return self.x_train, self.y_train
	
	def get_test_set(self):
		return self.x_test, self.y_test

	def plot_train_set(self):
		fig = plt.figure(figsize=(10, 7))
		row = 2
		col = 2
		j = 1
		for i in range(0, len(self.x_train), int(len(self.x_train) / 4)):
			# img = Image.fromarray(np.uint8(self.x_train[i]  * 255), 'L')
			# img.show()
			fig.add_subplot(row, col, j)
			plt.imshow(self.x_train[i], cmap="plasma")
			plt.axis('on')
			plt.title(f"Signal {i}")
			plt.xlabel("SubFrames")
			plt.ylabel("SubChannels")
			j += 1
		plt.show()