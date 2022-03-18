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
		fname = val[len(val) - 1]
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
		pbar.update(count)
		self.resource_pools.append(temp) # add last temp array

		# Get labels for each resource pool
		lbl = os.listdir(label_folder)
		print(f"Opening {fname[:-4]}_labels.txt")
		self.labels = open(f"{label_folder}{fname[:-4]}_labels.txt").read().splitlines()

		self.split_train_test()
		print("Finished preprocessing")
	
	def split_train_test(self):
		# 80/20 split for training and testing data
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.resource_pools, self.labels, test_size=0.20, random_state=42)
	
	def get_train_set(self):
		# reshape x_train
		for i in range(len(self.x_train)):
			self.x_train[i] = np.reshape(self.x_train[i], [10, 50])
			self.y_train[i] = tf.strings.to_number(self.y_train[i])
		self.y_train = tf.expand_dims(self.y_train, 1) # adding the batch size to match tf input
		return self.x_train, self.y_train
	
	def get_test_set(self):
		for i in range(len(self.x_test)):
			self.x_test[i] = np.reshape(self.x_test[i], [10, 50])
			self.y_test[i] = tf.strings.to_number(self.y_test[i])
		self.y_test = tf.expand_dims(self.y_test, 1) # adding the batch size to match tf input
		return self.x_test, self.y_test

	def plot_train_set(self):
		fig = plt.figure(figsize=(8, 8))
		row = 2
		col = 2
		j = 1
		for i in range(0, len(self.x_train), int(len(self.x_train) / 4)):
			# img = Image.fromarray(np.uint8(self.x_train[i]  * 255), 'L')
			# img.show()
			fig.add_subplot(row, col, j)
			plt.imshow(self.x_train[i], cmap="plasma")
			plt.axis('on')
			plt.title(f"Resource Pool {i}")
			plt.xlabel("SubFrames")
			plt.ylabel("SubChannels")
			plt.colorbar()
			plt.clim(0, 1)
			j+=1
		plt.show()
	
	def plot_test_set(self):
		fig = plt.figure(figsize=(8, 8))
		row = 2
		col = 2
		j = 1
		for i in range(0, len(self.x_test), int(len(self.x_test) / 4)):
			# img = Image.fromarray(np.uint8(self.x_train[i]  * 255), 'L')
			# img.show()
			fig.add_subplot(row, col, j)
			plt.imshow(self.x_test[i], cmap="plasma")
			plt.axis('on')
			plt.title(f"Resource Pool {i}")
			plt.xlabel("SubFrames")
			plt.ylabel("SubChannels")
			plt.colorbar()
			plt.clim(0, 1)
			j+=1
		plt.show()