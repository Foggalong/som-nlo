import numpy as np
import sys

from nls import nls
#from chebyquad import chebyquad

from GenericLineSearchMethod import GLSM

# ------------------ parameters for the method --------------------------
# function can be "chebyquad", "nls" 
function = "nls"
if function=="chebyquad":
    func = chebyquad

    #x0 = np.array([0.33, 0.66])
    #x0 = np.array([0.2, 0.4, 0.6])
    #x0 = np.array([0.2, 0.4, 0.6, 0.8])
    x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    #x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
elif function=="nls":
    func = nls
    #x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
else:
    raise ValueError("Function code not recognized")
            

    
tol = 1e-4 # tolerance for the line search method
# ------------------ end parameters for the method --------------------------

# call the generic Line Search Method
path = GLSM(x0, func, tol)

print(path)
