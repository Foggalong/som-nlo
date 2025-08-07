import numpy as np


def CGDirection(gkp, gk, pk, method="FR", fixPR=False, output=True):
    """
    Calculates the next conjugate gradient search direction, where:
    - gkp is the gradient at x_{k+1} (calculated outside this method)
    - gk is the gradient at x_k
    - pk is the previous search direction

    Takes an optional string `method` which determines whether bk is
    found using Fletcher Reeves (`FR`, default) or Polak-Riviere (`PR`).

    Takes an optional boolean `fixPR` (default `False`) which restarts
    in case PR does not give a descent direction.

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
    if fixPR and np.dot(pkp, gkp) > 0:
        if output:
            print("CG: not a descent direction")
        pkp = -gkp

    return pkp
