import numpy as np
from numpy import linalg as LA   # We need this module for the norm

from solveTRLevenbergMarquardt import solveTRLM
from solveTRL2Cauchy import solveTRL2Cauchy

# ------------------ parameters for the method --------------------------

# solver can be LM (Levenberg-Marquardt)
subsolver = "LM"
# subsolver = "DL"

# set parameters for the TR method
r0 = 1  # initial TR size
del_up = 0.75  # upper threshold for ratio: above this expand TR
del_lo = 0.25  # lower threshold for ratio: below this reduce TR
fact_inc = 2  # factor by which to increase TR
fact_dec = 0.25  # factor by which to decrease TR

iterations = 100  # iteration limit

out = 1  # printing from method: 0=quiet, 1=path, 2=diagnostic

# ------------------ end parameters for the method --------------------------


def GTRM(x0, func, eps):
    """
    Generic Trust Region Method: applies a generic trust region method to
    optimize a given function

    Can choose between different ways to solve the TR sub-problem:
    - L_1 exact (by a QP solver)
    - L_2 exact (by Levenberg Marquardt)
    - L_2 approximate (by dog-leg)

    will return the path taken by the method (so that it can be plotted later)

    Called as path = GTRM(x0, function, tol) where:

        x0 = starting point

        function = the function to be minimized
            [assumes that the function can be called as
            function(0, x), function(1, x), function(2, x)]

        tol = the convergence tolerance: iterates until || nabla f(x)|| < tol

    possible calling sequence is

        import numpy as np
        from rosenbrock import rosenbrock
        from GenericTrustRegionMethod import GTRM
        path = GTRM(x0, rosenbrock, 1e-4)
    """

    if (not isinstance(x0, np.ndarray)):
        print("The argument x0 must be a np.array")
        return

    # n = x0.size  # BUG variable unused
    # this is to keep a list of iterates
    iterations_list = []

    xk = x0
    fk = func(0, xk)
    gk = func(1, xk)
    Hk = func(2, xk)
    iterations_list.append(np.array(xk))
    tot_f_eval = 1
    tot_g_eval = 1
    tot_solves = 0

    if out >= 3:
        print(f"Initial x0=\n{xk}")
        print(f"Initial f(x0)=\n{fk}")
        print(f"Initial g(x0)=\n{gk}")

    if out >= 1:
        print(f"it=  0: f = {fk:8.5g}, |g| = {LA.norm(gk):8.3g}")

    rho = r0
    iterations = 0
    while (LA.norm(gk) >= eps and iterations < iterations):
        iterations += 1

        # - - - - - - - - - solve TR subproblem - - - - - - - - - -
        # define the TR model: m(d) = 0.5*d*H*d + b'*d
        B = Hk
        g = gk

        if subsolver == "LM":
            dk, n_solves = solveTRLM(B, g, rho, 0.05, ret_n_eval=True)
        elif subsolver == "DL":
            dk, n_solves = solveTRL2Cauchy(B, g, rho, ret_n_eval=True)
        else:
            raise ValueError("Direction code not recognized")

        if out >= 2:
            print(f"TR subproblem takes {n_solves:d} solves")
        tot_solves += n_solves

        if out >= 3:
            print(f"dk = {dk}")

        # - - - - - - - - - -  do the TR logic - - - - - - - - - - -

        xkp = xk+dk
        fkp = func(0, xkp)
        tot_f_eval +=  1

        # predicted decrease = m(0) - m(dk) =
        pred_dec = -(0.5*np.dot(dk, np.dot(B, dk)) + np.dot(g, dk))
        # actual decrease = f(xk) - f(xkp)
        actual_dec = fk - fkp
        if out >= 2:
            print(f"  pred dec = {pred_dec:8.5f}")
            print(f"  actu dec = {actual_dec:8.5f}")

        # delta to predicted/actual
        delta = actual_dec/pred_dec
        if out >= 2:
            print(f"  -> delta = {delta:8.5f}")

        # do the actual trust region logic
        if delta > del_up:
            # accept step and increase TR radius
            xkn = xkp
            tot_g_eval += 1
            if np.abs(LA.norm(dk)-rho)/rho < 0.1:
                rho = fact_inc*rho
            if out >= 2:
                print(f"  ->accept step and increase rho = {rho}")
        elif delta > del_lo:
            # accept step but leave TR radius as it was
            xkn = xkp
            tot_g_eval += 1
            if out >= 2:
                print("  ->accept step and leave rho unchanged")
        else:
            # reject step and decrease TR radius
            xkn = xk
            rho = fact_dec*rho
            if out >= 2:
                print(f"  ->reject step and decrease rho = {rho}")

        # take whatever step was decided
        xk = xkn

        fk = func(0, xk)
        gk = func(1, xk)
        Hk = func(2, xk)

        iterations_list.append(np.array(xk))

        if out >= 1:
            print(f"it={iterations:3d}: f = {fk:8.5g}, |g| = {LA.norm(gk):8.3g}")
            print(f", rho={rho:8.3g}, n_solves = {n_solves:3d}")
        if out >= 3:
            print(f"xk={xk}")
            print(f"gk={gk}")

    print(f"GTRM took total of {tot_f_eval} function evaluations")
    print(f"GTRM took total of {tot_g_eval} gradient evaluations")
    print(f"GTRM took total of {tot_solves} solves in TR subproblem")
    return np.array(iterations_list)
