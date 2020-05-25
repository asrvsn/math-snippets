import numpy as np
from scipy.linalg import expm
from numpy.linalg import matrix_power, solve

# continuous time markov chain
Q = [
	[-9, 2, 7, 0, 0, 0, 0],
	[2, -9, 0, 0, 0, 7, 0],
	[0, 0, -1, 1, 0, 0, 0],
	[0, 0, 0, -1, 1, 0, 0],
	[0, 0, 1, 0, -1, 0, 0],
	[0, 0, 0, 0, 0, -2, 2],
	[0, 0, 0, 0, 0, 2, -2]
]

# Q = [
# 	[-1, 1, 0],
# 	[0, -1, 1],
# 	[1, 0, -1]
# ]

# Q = [
# 	[-2, 2],
# 	[2, -2]
# ]

Q = np.array(Q)
# heat kernel
H = expm(Q)

x = np.array([1, 0, 0, 0, 0, 0, 0])

Ht = matrix_power(H, 100)

print(H)
print(x@Ht)
# print(H.sum(1)) # should be ones

# solve stationary dist
pi = solve(Q.T, np.zeros(Q.shape[0]))
print(pi / pi.sum())