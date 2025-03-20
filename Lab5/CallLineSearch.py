import numpy as np
import sys

#from nls import nls
#from chebyquad import chebyquad

from L2PenaltyClass import L2Penalty as L2PC

from GenericLineSearchMethod import GLSM
#from GenericTrustRegionMethod import GTRM

# ------------------ parameters for the method --------------------------
# function can be "chebyquad", "nls" 
function = "l2pen"

if function=="chebyquad":
    #func = chebyquad

    #x0 = np.array([0.33, 0.66])
    #x0 = np.array([0.2, 0.4, 0.6])
    #x0 = np.array([0.2, 0.4, 0.6, 0.8])
    x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    #x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
elif function=="nls":
    #func = nls
    #x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
elif function=="l2pen":
    l2penClass = L2PC("ph1sm.nl")
    l2penClass.set_fact(5.0)
    func = l2penClass.l2pen
    x0 = 0.3*np.ones(l2penClass.get_nvar())
else:
    print("Function code not recognized")
    sys.exit()
            

    
tol = 1e-4 # tolerance for the line search method
# ------------------ end parameters for the method --------------------------


# call the generic Line Search Method
path = GLSM(x0, func, tol)
#path = GTRM(x0, func, tol)

print("Path to solution:")
print(path)
sz, _ = path.shape
xsol = np.array(path[sz-1])
print("Solution (xsol) = ")
print(xsol)
#for i in range(nlp.nvar):
#    print("%f %f %f"%(nlp.bl[i], xsol[i], nlp.bu[i]))
