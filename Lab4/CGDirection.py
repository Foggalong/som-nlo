import numpy as np

# method = "FR"
method = "PR"
fixPR = "false"
# fixPR = "true"


def CGDirection(gkp, gk, pk):
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

    print("Called CGDirection")

    # in first iteration return the steepest descent direction
    if gk is None:
        print("CG return SD direction")
        return -gkp

    if method == 'FR':
        # this is Fletcher Reeves
        bkp = np.dot(gkp, gkp)/np.dot(gk, gk)
    elif method == 'PR':
        # this is Polak-Riviere
        bkp = np.dot(gkp - gk, gkp)/np.dot(gk, gk)
    else:
        raise ValueError("Method code not recognized")

    pkp = -gkp + bkp*pk

    # restart in case PR does not give descent direction
    if fixPR == 'true':
        if np.dot(pkp, gkp) > 0:
            print("CG: not a descent direction")
            pkp = -gkp

    # print("pkp = ")
    # print(pkp)
    return pkp
