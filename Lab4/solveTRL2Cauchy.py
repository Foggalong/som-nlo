import numpy as np
from numpy import linalg as LA

out = 2        # level pf printing from the method (0 or 1)
max_iter = 20  # max trials in the lambda-adjustment loop


def solveTRL2Cauchy(B, g, rho, ret_n_eval=False):
    """
    Solve the L2 Trust Region subproblem with the Cauchy-Point method
    (and potentially dog-leg improvement)

        min 0.5*d'*Q*d + b'*d  subject to ||d||_2 <= rho

    by finding
        (1) the steepest descent direction dS
        (2) the point that minimizes q(x) within the TR along the dS direction
            -> the Cauchy point
        (3) the unconstrained minimizer of q (if pos def)
        (4) the minimizer along the dog-leg path
    """

    if out >= 1:
        print(f"Called solveTRL2Cauchy with rho={rho:8.3g}")
        if out >= 2:
            print(B)
            v, w = np.linalg.eig(B)
            print(v)
            print(g)

    n = g.size

    n_eval = 0

    # (1) get the steepest descent direction
    dS = - g

    # (2) find the Cauchy point

    tau = 1.0
    gBg = np.dot(g, np.dot(B, g))
    if gBg > 0:
        stepsize = LA.norm(g)**3/(rho*gBg)
        if stepsize < 1:
            tau = stepsize

    if out >= 1:
        print(f"tau = {tau:f}")
    dC = tau*rho/LA.norm(g)*dS
    mdC = 0.5*np.dot(dC, np.dot(B, dC)) + np.dot(g, dC)
    if out >= 1:
        print(f"|dC| = {LA.norm(dC):f}")
        print(f"mdC = {mdC:f}")

    # Dog-leg method:

    # if the step is already to the TR boundary then the second leg of the
    # dog-leg is fully outside the TR, so don't need to consider

    d = dC

    if tau < 1:
        if out >= 1:
            print("Attempt dog-leg")
        # (3) find the unconstrained minimizer

        try:
            dU = LA.solve(B, -g)
        except LA.LinAlgError:
            if out >= 1:
                print("Cannot solve for dU")
            dU = np.zeros(n)

        n_eval += 1
        mdU = 0.5*np.dot(dU, np.dot(B, dU)) + np.dot(g, dU)

        if out >= 1:
            print(f"|dU| = {LA.norm(dU):f}")
            print(f"mdU = {mdU:f}")

        # only consider the second leg of the dog-leg path if
        # the end point is outside the TR and leads to improvement
        if (LA.norm(dU) <= rho and mdU < mdC):
            # no dog-leg, but return dU
            d = dU
            if out >= 1:
                print("return dU")

        if (LA.norm(dU) > rho and mdU < mdC):
            # (4) find the minimizer along the dog-leg path
            if out >= 1:
                print(f"dC = {dC}")
                print(f"dU = {dU}")

            # solve (dC + lam*(dU-dC))'* (dC + lam*(dU-dC)) = rho^2
            # => dC'*dC + 2*lam*(dU-dC)'*dC + lam^2(dU-dC)'*(dU-dC) = rho^2
            # => A*lam^2 + B*lam + C = 0,
            #        A = (dU-dC)'*(dU-dC), B = 2*(dU-dC)'*dC, C = dC'*dC-rho^2

            A = np.dot(dU-dC, dU-dC)
            B = 2*np.dot(dU-dC, dC)
            C = np.dot(dC, dC) - rho**2

            # transfer to x^2 + p*x + q = 0
            p = B/A
            q = C/A

            # pq-formula
            x1 = -p/2 + np.sqrt(p**2/4 - q)
            x2 = -p/2 - np.sqrt(p**2/4 - q)

            lam = x1
            if lam < 0:
                lam = x2

            dDL = dC + lam*(dU-dC)
            mdDL = 0.5*np.dot(dDL, np.dot(B, dDL)) + np.dot(g, dDL)

            if out >= 1:
                print(f"|dDL| = {LA.norm(dDL):f}")
                print(f"mdDL = {mdDL:f}")

            if np.abs(LA.norm(dDL)-rho) > 1e-4:
                print("dDL not on TR boundary")
                raise Exception("Not on TR boundary")

            if mdDL < mdC:
                d = dDL
                if out >= 1:
                    print("return dDL")

    if ret_n_eval:
        return d, n_eval

    return d
