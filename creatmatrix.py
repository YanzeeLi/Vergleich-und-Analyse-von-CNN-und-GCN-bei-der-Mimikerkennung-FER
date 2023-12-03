'''
This file is for generating adjacency and degree matrices from input images.
Neighbors in the four directions of up, down, left, and right of pixels in the image are considered adjacent.
The resulting adjacency and degree matrices include self-loops.
A_ = A + I
'''

import numpy as np
import sys

# initialization matrix
A = np.zeros((12544, 12544), dtype=int)
D = np.zeros((12544, 12544), dtype=int)
I = np.ones((12544,), dtype=int)

np.set_printoptions(threshold=sys.maxsize)

# build the adjacency matrix
for i in range(12544):

    A[i][i] = 1
    if i % 112 != 0:
        A[i][i-1] = 1

    if (i + 1) % 112 != 0:
        A[i][i + 1] = 1

    if (i + 112) < 12544:
        A[i][i + 112] = 1

    if (i - 112) >= 0:
        A[i][i - 112] = 1

# save the adjacency matrix
np.save("A.npy", A)

# calculation degree
N = A.dot(I)

# build degree matrix
for j in range(12544):
    D[j][j] = N[j]

# save the degree matrix
np.save("D.npy", D)
