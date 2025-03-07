"""
 Conjugate Gradient Direction:
     calculates the next conjugate gradient search direction

 - it needs to be passed the previous gk and pk
 
 - there are variants to calculate bk based on Fletcher Reeves or Polack-Riv

 - it can be linked with the generic line search method and
   different line searches
 
 Called as pkp =  CGDirection(gkp, gk, pk), where
   - gkp is the gradient at x_{k+1} (calculated outside this method)
   - gk is the gradient at x_k
   - pk is the previous search direction

"""

import numpy as np

def  CGDirection(gkp, gk, pk):
    print("Called CGDirection")

    # in first iteration return the steepest descent direction
    if gk is None:
        print("CG return SD direction")
        return -gkp


    # this is Fletcher Reeves
    bkp = np.dot(gkp, gkp)/np.dot(gk, gk)
    # this is Polak-Riviere
    bkp = np.dot(gkp - gk, gkp)/np.dot(gk, gk)

    pkp = -gkp + bkp*pk

    # restart in case PR does not give descent direction
    if np.dot(pkp, gkp)>0:
        print("CG: not a descent direction")
        pkp = -gkp
        
    print("pkp = ")
    print(pkp)
    return pkp
