import numpy as np

from chebyquad import chebyquad as func
from GenericLineSearchMethod import GLSM

tol = 1e-4  # tolerance for the line search method

x0 = np.array([0.33, 0.66])
# x0 = np.array([0.2, 0.4, 0.6])
# x0 = np.array([0.2, 0.4, 0.6, 0.8])
# x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
# x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
path = GLSM(x0, func, tol)

print(path)
