import numpy as np
from numpy import linalg as LA   # we need this module for the norm

from LineSearchExact import LineSearchExact as LSExact
from LineSearchBacktrack import LineSearchBacktrack as LSBacktrack
from LineSearchBisection import LineSearchBisection as LSBisection
from CGDirection import CGDirection

# ------------------ parameters for the method --------------------------

# direction can be:
# - 'CD' for coordinate descent,
# - 'SD' for steepest descent,
# - 'CG' for conjugate gradient.
direction = "SD"

# line search can be 'Exact', 'Armijo', 'Wolfe', '1overk', 'fullstep'
linesearch = "Exact"

# set line search method parameters
c1 = 0.1  # for Armijo/Backtracking linesearch
c2 = 0.5  # for Bisection linesearch
tol = 1e-7  # tolerance for exact linesearch
max_iter = 100  # iteration limit
do_CG_restarts = False  # restart CG method every n iterations

# ------------------ end parameters for the method --------------------------


def GLSM(x0, func, eps, output=1):
    """
    Generic Line Search Method: applies a generic line search method to
    optimize a given function

    Can choose between different search directions (coordinate descent,
    steepest descent, conjugate gradients) and line search methods
    (backtracking, bisection, exact and pre-determined)

    will return the path taken by the method (so that it can be plotted later)

    Called as path = GLSM(x0, function, tol);

    where
        x0 = starting point
        function = the function to be minimized
                [assumes that the function can be called as
                function(0, x), function(1, x), function(2, x)]
        tol = the convergence tolerance: iterate until |nabla f(x)| < tol

    possible calling sequence is

    import numpy as np
    from rosenbrock import rosenbrock
    from GenericLineSearchMethod import GLSM
    path = GLSM(x0, rosenbrock, 1e-4)

    output = 1  # printing from method: 0=quiet, 1=path, 2=diagnostic
    """

    if not isinstance(x0, np.ndarray):
        raise TypeError("The argument x0 must be a np.array")

    n = x0.size
    # this is to keep a list of iterates
    iterates = []

    xk = x0
    fk = func(0, xk)
    gk = func(1, xk)
    iterates.append(np.array(xk))
    tot_n_eval = 1

    if output:
        if output == 2:
            print(f"Initial g0 = \n{gk}")
        print(f"f = {fk:8.5g}, |g| = {LA.norm(gk):8.5g}")

    # set these for Conjugate Gradients
    dk_prev = None
    gk_prev = None

    iteration = 0
    while (LA.norm(gk) >= eps and iteration < max_iter):
        iteration += 1

        # - - - - - - - - - find search directions - - - - - - - - - -

        if direction == 'CD':
            # coordinate descent
            # i-th coordinate direction
            dk = np.eye(1, n, (iteration-1) % n).flatten()
            if (dk.dot(gk) > 0):  # make sure it is pointing downwards
                dk = -dk
        elif direction == 'SD':
            # steepest descent
            dk = -gk
        elif direction == 'CG':
            # conjugate gradient method
            # CG restarts
            if do_CG_restarts and iteration % n == 1:
                dk_prev = None
                gk_prev = None
            dk = CGDirection(gk, gk_prev, dk_prev)
        elif direction == 'Newton':
            # Newton Method
            Hk = func(2, xk)
            dk = - np.linalg.solve(Hk, gk)
        else:
            raise ValueError("Direction code not recognized")

        # check for descent direction
        if dk.dot(gk) >= 0:
            if output:
                print(f"dk = \n{dk}")
            raise ValueError("search direction is not a descent direction: STOP")
            dk = -gk

        # - - - - - - - - - -  do a line search - - - - - - - - - -
        if output == 2:
            print(f"dk = \n{dk}")

        # counter for number of function evaluations in line search method
        n_eval = 0

        if linesearch == "Exact":
            # Exact line search:
            alpha, n_eval = LSExact(xk, dk, tol, func, ret_n_eval=True)
        elif linesearch == "Armijo":
            # Backtracking Armijo linesearch:
            alpha, n_eval = LSBacktrack(xk, dk, c1, func, ret_n_eval=True)
        elif linesearch == "Wolfe":
            # Bisection Wolfe linesearch:
            alpha, n_eval = LSBisection(xk, dk, c1, c2, func, ret_n_eval=True)
        elif linesearch == "1overk":
            # predetermined stepsize
            alpha = 1./(LA.norm(dk)*iteration)
        elif linesearch == "fullstep":
            alpha = 1
        else:
            raise ValueError("Linesearch code not recognized")

        tot_n_eval += n_eval
        if output:
            print(f"Line search took {n_eval} function evaluation")

        # take step
        xk += alpha*dk

        # remember last dk and gk for conjugate gradients
        gk_prev = gk
        dk_prev = dk

        fk = func(0, xk)
        gk = func(1, xk)
        tot_n_eval += 1

        iterates.append(np.array(xk))

        if output:
            print(f"it = {iteration:3d}: f = {fk:8.5g}, |g| = {LA.norm(gk):8.5g}")
            if output == 2:
                print(f"xk =\n{xk}")
                print(f"gk =\n{gk}")

    if output:
        print(f"GLSM took total of {tot_n_eval} function evaluations")

    return np.array(iterates)
