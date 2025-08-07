import numpy as np


def CGDirection(gkp, gk, pk, output=True):
    """
    Calculates the next conjugate gradient search direction

    - it needs to be passed the previous gk and pk

    - there are variants to calculate bk based on Fletcher Reeves or Polak-Riv

    - it can be linked with the generic line search method and
      different line searches

    Called as pkp =  CGDirection(gkp, gk, pk), where
      - gkp is the gradient at x_{k+1} (calculated outside this method)
      - gk is the gradient at x_k
      - pk is the previous search direction
    """

    if output:
        print("Called CGDirection")

    # in first iteration return the steepest descent direction
    if gk is None:
        if output:
            print("CG return SD direction")
        return -gkp

    # this is Fletcher Reeves
    bkp = np.dot(gkp, gkp)/np.dot(gk, gk)
    # this is Polak-Riviere
    # bkp = np.dot(gkp - gk, gkp)/np.dot(gk, gk)
    # if output:
    #     print(f"Bk = {bkp}")

    pkp = -gkp + bkp*pk

    if output:
        print(f"pkp = \n{pkp}")
    return pkp
