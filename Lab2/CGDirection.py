import numpy as np


def CGDirection(gkp, gk, pk, output=True):
    """
    Calculates the next conjugate gradient search direction using
    Fletcher Reeves, where:
    - gkp is the gradient at x_{k+1} (calculated outside this method)
    - gk is the gradient at x_k
    - pk is the previous search direction

    Takes an optional boolean `output` (default `True`) which controls
    whether to print progress.
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
    pkp = -gkp + bkp*pk

    if output:
        print(f"pkp = \n{pkp}")

    return pkp
