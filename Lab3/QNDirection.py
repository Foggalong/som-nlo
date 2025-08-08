import numpy as np


def QNDirection(Hk, xkp, xk, gkp, gk, update="SR1", output=True):
    """
    Calculates the next direction for Quasi-Newton methods, where
    - Hk is old (inverse) Hessian approximation
    - xkp, xk are the previous two iterates
    - gkp, gk are the previous two gradients

    It returns
    - dkp: next search direction dkp = -Hkp gkp
    - Hkp: the new Hessian approximation

    Takes an optional string `update` to control the method for updating H. It
    supports `SR1` (default), `DFP` and `BFGS`.

    Takes an optional boolean `output` (default `True`) which controls
    whether to print progress.
    """

    if output:
        print("Called QNDirection")

    delta = xkp - xk
    y = gkp - gk

    # Double check that the QN-curvature condition is satisfied. This should
    # be the case by the Wolfe conditions, which are enforced by the Bisection
    # Linesearch.
    yd = y.dot(delta)
    if yd <= 0:
        raise ValueError("QN-curvature condition not satisfied!")

    # This is required by all three update methods, so save it to a variable
    # to make calculations more efficient.
    Hy = Hk.dot(y)
    Hkp = Hk.copy()

    if update == "SR1":
        # The symmetric SR1 update equation is (H version)
        # H+ = H + (delta-H*y)*(delta-H*y)'/((delta-H*y)'y)
        Hkp += np.outer(delta-Hy, delta-Hy)/y.dot(delta-Hy)

    elif update == "DFP":
        # The DFP update equation is
        # H+ = H + dd^T/d^Ty - Hy(Hy)'/y'Hy
        Hkp += np.outer(delta, delta)/yd - np.outer(Hy, Hy)/(Hy.dot(y))

    elif update == "BFGS":
        # The BFGS update equation is
        # H+ = H + (1+y'Hy/y'd)*(dd')/y'd - (dy'H' + Hyd')/y'd
        Hkp += (1 + y.dot(Hy)/yd)*(np.outer(delta, delta))/yd - \
               (np.outer(delta, Hy) + np.outer(Hy, delta))/yd

    else:
        raise ValueError("Update method not recognised")

    if output:
        print(f"QN inverse Hessian approx is:\n{Hkp}")

    dkp = -Hkp.dot(gkp)

    return dkp, Hkp
