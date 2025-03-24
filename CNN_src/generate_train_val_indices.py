import numpy as np

N = 41

train_ratio = -0.2

train_indices = []
test_indices = []

for i in range(N):
    if np.random.random() <= train_ratio:
        train_indices.append(i)
    else:
        test_indices.append(i)

np.save('train_indices.npy',train_indices)
np.save('test_indices.npy',test_indices)
