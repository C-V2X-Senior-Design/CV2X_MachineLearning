import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

filename = "testfile.txt"
dim = 10
a = np.zeros((dim, dim))
with open(filename) as f:
	testcontents = f.readlines()
	for i in range(dim):
		for j in range(dim):
			a[i][j] = testcontents[i][j]

print(a)
plt.figure()
plt.imshow(a, cmap="Greys")
plt.colorbar()
plt.grid(False)
plt.show()