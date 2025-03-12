import numpy as np

from nls import nls
from chebyquad import chebyquad
from GenericTrustRegionMethod import GTRM

# ------------------ parameters for the method --------------------------
# this parameter can be "chebyquad", "nls"
function_name = "nls"

if function_name == "chebyquad":
    func = chebyquad

    # x0 = np.array([0.33, 0.66])
    # x0 = np.array([0.2, 0.4, 0.6])
    # x0 = np.array([0.2, 0.4, 0.6, 0.8])
    x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    # x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
elif function_name == "nls":
    func = nls
    # x0 = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
else:
    raise ValueError("Function code not recognized")

tol = 1e-4  # tolerance for the line search method
# ------------------ end parameters for the method --------------------------

# call the generic Trust Region Method
path = GTRM(x0, func, tol)

print(path)
