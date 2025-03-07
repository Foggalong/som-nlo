"""
 Quasi-Newton Direction:
     calculates the next direction for Quasi-Newton methods

 - It uses a variant that updates H (by default)

 - it needs to be passed the previous H (and also yk and deltak)
 
 - It supports SR1, DFP and BFGS

 - it can be linked with the generic line search method and
   different line searches
 
 Called as dk, Hkp =  CGDQuasiNewton(Hk, xk, xkp, gk, gkp), where
   - Hk is old (inverse) Hessian approximation
   - xkp, xk are the previous two iterates
   - gkp, gk are the previous two gradients
 Is returns
   - dkp: next search direction dkp = -Hkp gkp
   - Hkp: the new Hessian approximation
"""

#update = "SR1"
#update = "DFP"
update = "BFGS"

import numpy as np

def  QNDirection(Hk, xkp, xk, gkp, gk):
    print("Called QNDirection")

    delta = xkp - xk
    y = gkp - gk

    # check that the QN-curvature condition is satisfied
    # (this really has to be the case by the Wolfe conditions which 
    #  are enforced by the Bisection Linesearch).
    yd = y.dot(delta)
    if yd<=0:
        print("QN-curvature condition not satisfied: STOP!")   
        raise Exception("QN curvature condition failed")

    #Hkp = Hk.copy();

    if update == "SR1":
        # The symmetric SR1 update (H version)
        # H+ = H + (delta-H*y)*(delta-H*y)'/((delta-H*y)'y)
        Hy = Hk.dot(y)
        dH = np.outer(delta-Hy, delta-Hy)
        fac = y.dot(delta-Hy)
        dH = dH/fac
        Hkp = Hk + dH
    elif update =="DFP":
        # The DFP update
        # H+ = H + dd^T/d^Ty - Hy(Hy)'/y'Hy
        Hy = Hk.dot(y)
        Hkp = Hk + np.outer(delta,delta)/yd - np.outer(Hy,Hy)/(Hy.dot(y))
    elif update == "BFGS":
        # The BFGS update
        # H+ = H + (1+y'Hy/y'd)*(dd')/y'd - (dy'H' + Hyd')/y'd 
        Hy = Hk.dot(y)  # remember this to make calculations more efficient

        Hkp = Hk + (1+y.dot(Hy)/yd)*(np.outer(delta, delta))/yd  \
             - (np.outer(delta, Hy)+np.outer(Hy, delta))/yd
    else:
        raise ValueError("did not recognise update")

    print("QN inverse Hessian approx is:")
    print(Hkp)
    
    dkp = -Hkp.dot(gkp)

    return dkp, Hkp
