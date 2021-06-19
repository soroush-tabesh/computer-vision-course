import scipy.optimize as opt
import scipy.linalg as lin
import numpy as np

A = np.array([[1, -2, 3], [4, 5, -6], [7, 8, 9], [-10, 11, 12]])
b = np.array([15, 16, 10, 18])
# %%
opt.lsq_linear(A, b, bounds=(np.zeros(3), np.ones(3))).x
# %%
lin.lstsq(A, b)[0]

# %%
A = np.loadtxt('./A.txt', delimiter=',')
b = np.ones(50) * 1.747845484
p = opt.lsq_linear(A, b, bounds=(np.zeros(70), np.ones(70))).x
np.linalg.norm(A @ p - b)
