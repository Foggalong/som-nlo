#
# This plots the iterates of the genertic line search method for a given
# function on the contour plot of the function
#

import numpy as np
import matplotlib.pyplot as plt
import sys

#from himmelblau import himmelblau 
from rosenbrock import rosenbrock 
from ex21func import ex21
from ex46func import ex46

from GenericLineSearchMethod import GLSM

# ------------------ parameters for the method --------------------------

# function can be "Rosenbrock", "Ex21", "Ex46", "Himmelblau"
function = "Ex21"
#function = "Rosenbrock"

tol = 1e-4 # tolerance for the line search method
n = 100  # number of points for the coutour discretization
nl = 50 # number of levels for the contour plot

# ------------------ end parameters for the method --------------------------
if function=="Rosenbrock":
    func = rosenbrock
    x0 = np.array([0, 0])  # starting point (Rosenbrock)
    # after this come limits for the contour plot
    lmt_xlo, lmt_xup = -0.1, 1.1 # these are for Rosenbrock
    lmt_ylo, lmt_yup = -0.1, 1.1
elif function=="Ex21":
    func = ex21
    #x0 = np.array([1, 1])  # starting point (Ex2.1)
    x0 = np.array([0.5, 1])  # starting point (Ex2.1)
    #lmt_xlo, lmt_xup = -0.6, 1 # these are for Ex2.1 (up to nearest min)
    #lmt_ylo, lmt_yup = -0.4, 1.2
    lmt_xlo, lmt_xup = -1.2, 1 # these are for Ex2.1 (covering both min)
    lmt_ylo, lmt_yup = -1.2, 1.2
elif function=="Ex46":
    func = ex46
    x0 = np.array([2, 2])  # starting point (Ex4.6)
    lmt_xlo, lmt_xup = -1.2, 2.1 # these are for Ex4.6
    lmt_ylo, lmt_yup = -1.2, 2.1
elif function=="Himmelblau":
    print("Himmelblau not supported yet")
    sys.exit()
    #func = himmelblau
    x0 = [1, 2]  # starting point
    lmt_xlo, lmt_xup = -1.2, 2.1 # these are for Ex4.6
    lmt_ylo, lmt_yup = -1.2, 2.1
else:
    print("Function code not recognized")
    sys.exit()
            


# - - - - - - - - - This is the code for the contour plot - - - - - - - 
plt.ion()
xlist = np.linspace(lmt_xlo, lmt_xup, n)
ylist = np.linspace(lmt_ylo, lmt_yup, n)
X, Y = np.meshgrid(xlist, ylist)

Z = func(0, X, Y)

# - - uncomment this to get contours with logarithmic progression
#lgmin = np.log((Z.min()))
#lgmax = np.log((Z.max()))
#lvls = np.exp(np.linspace(lgmin, lgmax, 40))

fig,ax=plt.subplots(1,1)
cp = ax.contour(X, Y, Z, nl)
#cp = ax.contour(X, Y, Z, lvls)   # for logarithmic progression

# - - - - Calls the generic line search method and plots the path - - - -
path = GLSM(x0, func, tol)
ln = ax.plot(path[:,0], path[:,1])
#ln = ax.plot(path[:,0], path[:,1],'x-')

#plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")

input("Press Enter to continue...")
