import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

i = 1
dim = 10
a = np.zeros((dim, dim))
dir = 'psuedo_data/'
N = 1000
while tqdm((N > 0)):
	for r in range(dim):
		for c in range(dim):
			a[r][c] = random.randint(0, 1)
	fn = f'{dir}pseudo_data_{i}'
	np.savetxt(fn, a)
	i += 1
	N -= 1

print("Done")