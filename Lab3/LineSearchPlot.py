"""
This plots the iterates of the generic line search method for a given
function on the contour plot of the function
"""

import numpy as np
import matplotlib.pyplot as plt

# from himmelblau import himmelblau
from rosenbrock import rosenbrock
from ex13func import ex13
from ex21func import ex21
from ex46func import ex46
from SourceLocalisation import sourceloc

from GenericLineSearchMethod import GLSM

# ------------------ parameters for the method --------------------------

# function can be "Rosenbrock", "Ex21", "Ex46", "Himmelblau", "SourceLoc"
function = "Ex21"
# function = "Rosenbrock"
# function = "SourceLoc"
# function = "Ex13"

tol = 1e-4  # tolerance for the line search method
n = 100  # number of points for the contour discretization
nl = 50  # number of levels for the contour plot

# ------------------ end parameters for the method --------------------------
if function == "Rosenbrock":
    func = rosenbrock
    x0 = np.array([0, 0])  # starting point (Rosenbrock)
    # after this come limits for the contour plot
    lmt_xlo, lmt_xup = -0.1, 1.1  # these are for Rosenbrock
    lmt_ylo, lmt_yup = -0.1, 1.1
elif function == "Ex13":
    func = ex13
    x0 = np.array([0.5, 1])  # starting point (Ex1.3) - try this point first
    # x0 = np.array([0.8, 0])  # starting point (Ex1.3) - for the second test
    # x0 = np.array([-1.2, 0.5])  # starting point (Ex1.3) - for the third test
    lmt_xlo, lmt_xup = -2.0, 2.0  # these are for Ex1.3 (covering both min)
    lmt_ylo, lmt_yup = -1.1, 1.1
elif function == "Ex21":
    func = ex21
    # x0 = np.array([1, 1])  # starting point (Ex2.1)
    x0 = np.array([0.5, 1])  # starting point (Ex2.1)
    # lmt_xlo, lmt_xup = -0.6, 1 # these are for Ex2.1 (up to nearest min)
    # lmt_ylo, lmt_yup = -0.4, 1.2
    lmt_xlo, lmt_xup = -1.2, 1  # these are for Ex2.1 (covering both min)
    lmt_ylo, lmt_yup = -1.2, 1.2
elif function == "Ex46":
    func = ex46
    x0 = np.array([2, 2])  # starting point (Ex4.6)
    lmt_xlo, lmt_xup = -1.2, 2.1  # these are for Ex4.6
    lmt_ylo, lmt_yup = -1.2, 2.1
elif function == "SourceLoc":
    func = sourceloc
    x0 = np.array([2, 2])  # starting point SourceLoc
    x0 = np.array([0, 0])  # starting point SourceLoc
    x0 = np.array([5, 2])  # starting point SourceLoc
    lmt_xlo, lmt_xup = -7.0, 15.0  # these are for SourceLoc
    lmt_ylo, lmt_yup = -15.0, 7.0
elif function == "Himmelblau":
    raise ValueError("Himmelblau not supported yet")
    # func = himmelblau
    x0 = [1, 2]  # starting point
    lmt_xlo, lmt_xup = -1.2, 2.1  # these are for Ex4.6
    lmt_ylo, lmt_yup = -1.2, 2.1
else:
    raise ValueError("Function code not recognized")


# - - - - - - - - - This is the code for the contour plot - - - - - - -
plt.ion()
x_list = np.linspace(lmt_xlo, lmt_xup, n)
y_list = np.linspace(lmt_ylo, lmt_yup, n)
X, Y = np.meshgrid(x_list, y_list)

Z = func(0, X, Y)

# - - uncomment this to get contours with logarithmic progression
# lg_min = np.log((Z.min()))
# lg_max = np.log((Z.max()))
# lvls = np.exp(np.linspace(lg_min, lg_max, 40))

fig, ax = plt.subplots(1, 1)
cp = ax.contour(X, Y, Z, nl)
# cp = ax.contour(X, Y, Z, lvls)  # for logarithmic progression
# uncomment below for labels on contours
# ax.clabel(cp, inline=True, fmt='%1.0f', fontsize=10)

# - - - - Calls the generic line search method and plots the path - - - -
path = GLSM(x0, func, tol)
# ln = ax.plot(path[:,0], path[:,1],'x-')
ln = ax.plot(path[:, 0], path[:, 1])

# print(path)
# n_list, _ = path.shape
# print(n_list)
# x_star = path[n_list-1]
# print(x_star)
# nrm = np.linalg.norm(path, axis=1)
# print(nrm)
# print(nrm[1:n_list]/nrm[0:n_list-1])

plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")

input("Press Enter to continue...")
