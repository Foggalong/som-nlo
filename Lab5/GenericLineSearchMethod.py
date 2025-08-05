import numpy as np
from numpy import linalg as LA   # We need this module for the norm

# from LineSearchExact2 import LineSearchExact as LSExact
from LineSearchBacktrack import LineSearchBacktrack as LSBacktrack
from LineSearchBisection import LineSearchBisection as LSBisection
# from CGDirection import CGDirection    # conjugate gradients
from QNDirection import QNDirection    # Quasi-Newton
# from Cholesky import Cholesky as Chol

# ------------------ parameters for the method --------------------------

# direction can be 'CD', 'SD', 'CG' for coordinate descent, steepest descent
#                  'Newton', QN'    for Newton and Quasi-Newton
# direction = "SD"
# direction = "Newton"
# conjugate gradient
# direction = "CG"
direction = "QN"

# line search can be 'Exact', 'Armijo', 'Wolfe', '1overk', 'fullstep'
# linesearch = "Armijo"
linesearch = "Wolfe"
# linesearch = "Exact"
# linesearch = "fullstep"

# -------------- set line search method parameters ------------------
c1 = 0.1  # for Armijo/Backtracking linesearch
c2 = 0.5  # for Bisection linesearch
tol = 1e-7 # tolerance for exact linesearch
max_iterations = 3000  # iterationsation limit
do_CG_restarts = False  # restart CG method every n iterationsations
do_Newton_InertiaCorr = True  # add tau*I to the Newton Hessian for pos def
minEV = 1e-4

out = 1  # printing from method: 0=quiet, 1=path, 2=diagnostic

# ------------------ end parameters for the method --------------------------


def  GLSM(x0, func, eps):
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
    tol = the convergence tolerance: iterate until || nabla f(x) || < tol

    possible calling sequence is

    import numpy as np
    from rosenbrock import rosenbrock
    from GenericLineSearchMethod import GLSM
    path = GLSM(x0, rosenbrock, 1e-4)
    """

    if (not isinstance(x0, np.ndarray)):
        print("The argument x0 must be a np.array")
        return

    n = x0.size
    # this is to keep a list of iterates
    iterate_list = []

    xk = x0
    fk = func(0, xk)
    gk = func(1, xk)
    iterate_list.append(np.array(xk))
    tot_n_eval = 1

    if out == 2:
        print("Initial x0 = ")
        print(xk)
        print("Initial f0 = ")
        print(fk)
        print("Initial g0 = ")
        print(gk)

    if out >= 1:
        print("Start generic line search method: ", end="")
        print("Line search = %s, Direction = %s" % (linesearch, direction))

    if out >= 1:
        print("it = %d: f = % 8.5g, |g| = % 8.5g" % (0, fk, LA.norm(gk)))

    # set these for Conjugate Gradients
    dk_prev = None
    gk_prev = None

    # initial Hessian approximation for QN
    Hk = np.identity(n)

    iterations = 0
    while (LA.norm(gk) >= eps and iterations < max_iterations):
        iterations = iterations + 1

        # - - - - - - - - - find search directions - - - - - - - - - -

        if direction == 'CD':
            # coordinate descent
            # i-th coordinate direction
            dk = np.eye(1, n, (iterations-1) % n).flatten()
            if (dk.dot(gk) > 0):  # make sure it is pointing downwards
                dk = -dk
        elif direction == 'SD':
            # steepest descent
            dk = -gk
        elif direction == 'CG':
            # conjugate gradient method
            # CG restarts
            if do_CG_restarts and iterations % n == 1:
                dk_prev = None
                gk_prev = None
            dk = CGDirection(gk, gk_prev, dk_prev)
        elif direction == 'Newton':
            # Newton Method
            Hk = func(2, xk)
            # print(Hk)
            # this down here is the inertia correction
            v, w = LA.eig(Hk)
            print("EV of Newton:")
            print(v)
            l_min = min(v)
            if (do_Newton_InertiaCorr and l_min < minEV):
                # val, L = Chol(Hk)
                # print((val, L))

                tau = -l_min+minEV
                Hk = Hk + tau*np.eye(n)
                print("Inertia correction: %f" % tau)
                #    print(Hk)
                v, w = LA.eig(Hk)
                print(v)

            dk = - np.linalg.solve(Hk, gk)
            # check for too large direction
            if LA.norm(dk)>1e10:
                print("Newton direction has size: %f" % LA.norm(dk))
                print("Stop!")
                raise Exception("Direction too large")
        elif direction == 'QN':
            if iterations == 1:  # in the first iteration use the initial H0
                dk = -Hk.dot(gk)
            else:
                dk, Hkp = QNDirection(Hk, xk, xk_prev, gk, gk_prev)
                Hk = Hkp
        else:
            raise ValueError("Direction code not recognized")

        # check for descent direction
        if dk.dot(gk) >= 0:
            print("search direction is not a descent direction: STOP")
            print(dk)
            raise Exception("Not descent direction")
            dk = -gk

        # - - - - - - - - - -  do a line search - - - - - - - - - -
        if out > 1:
            print("dk = ")
            print(dk)

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
            alpha = 1./(LA.norm(dk)*iterations)
        elif linesearch == "fullstep":
            alpha = 1
        else:
            raise ValueError("Linesearch code not recognized")

        tot_n_eval += n_eval
        if out >= 1:
            print("Line search took "+str(n_eval)+" function evaluation")

        # remember last xk for conjugate gradients and QN
        xk_prev = xk

        # take step
        xk = xk + alpha*dk

        # remember last dk and gk for conjugate gradients and QN
        gk_prev = gk
        dk_prev = dk

        fk = func(0, xk)
        gk = func(1, xk)
        tot_n_eval = tot_n_eval + 1

        iterate_list.append(np.array(xk))

        if out >= 1:
            print("it=% 3d: f = % 8.5g, |g| = % 8.5g" %(iterations, fk, LA.norm(gk)));
        if out > 1:
            print("xk=")
            print(xk)
            print("gk=")
            print(gk)

    if out >= 2:
        print("Hessian at sol:")
        print(func(2, xk))

    print("GLSM took total of "+str(tot_n_eval)+" function evaluations")
    return np.array(iterate_list)
