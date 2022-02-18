import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

i = 1
dim = 10
a = np.zeros((dim, dim))
dir = 'pseudo_data/'
N = 1000
for i in tqdm(range(N)):
	for r in range(dim):
		for c in range(dim):
			a[r][c] = random.randint(0, 1)
	fn = f'{dir}pseudo_data_{i}'
	np.savetxt(fn, a)

print("Done!")