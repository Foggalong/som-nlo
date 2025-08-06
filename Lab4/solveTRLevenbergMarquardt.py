import numpy as np
from numpy import linalg as LA

out = 0        # level pf printing from the method (0 or 1)
max_iterations = 20  # max trials in the lambda-adjustment loop


def solveTRLM(B, g, rho, tol, ret_n_eval=False):
    """
    Solve the L2 Trust Region subproblem with the Levenberg-Marquard method

        min 0.5*d'*Q*d + b'*d  subject to ||d||_2 <= rho

    by finding lam such that

        - dk = -(B+lam I)^{-1} b

        - (B+lam I) positive semidefinite

        - lam = 0, or ||d_k||_2 = rho
    """

    if out >= 1:
        print(f"Called solveTRLM with rho={rho:8.3g}")
        print(B)
        print(g)

    n_eval = 0
    norm_dk = np.infty

    n = g.size

    lam_lo = 0
    lam_up = np.infty

    lam = 0

    # isPosDef = False
    # while isPosDef == False:
    #     try:
    #         L = LA.cholesky(B+lam*eye(n))
    #         isPosDef = True
    #     except:
    #         lam += 1

    # if we get here we should have that B+lam*I is pos def

    # check if lam=0 works
    isPosDef = True
    try:
        L = LA.cholesky(B)
        n_eval += 1
        if out >= 1:
            print("  initial lam=0 results in pd matrix")
    except:
        isPosDef = False
        if out >= 1:
            print("  initial lam=0 not pd")

    if not isPosDef:
        lam_lo = lam
    else:
        dk = LA.solve(L, -g)
        dk = LA.solve(L.transpose(), dk)
        # dk = LA.solve(B, -g)
        norm_dk = LA.norm(dk)
        if out >= 1:
            print(f"  lam=0 => pd and |dk|= {norm_dk}")

        if norm_dk <= rho:
            if out >= 1:
                print("  initial dk = -B^{-1}g is solution")

            if ret_n_eval:
                return dk, n_eval
            return dk
        else:
            lam_lo = lam

    # if we get here then lam=0 was rejected either because not pos def
    # or step size too large

    # start the loop to find the optimal lam
    iterations = 0
    # new_lam = 0

    while (np.abs(norm_dk-rho)/rho > tol) and (iterations < max_iterations):
        iterations += 1
        if lam_up > 1e10:
            lam = max(2*lam, 1)
            if out >= 1:
                print(f"  increase lam to {lam}")
        else:
            lam = 0.5*(lam_lo+lam_up)
            if out >= 1:
                print(f"  new lam = {lam}")

        # if new_lam>0 and new_lam>=lam_lo and new_lam<=lam_up:
        #     lam = new_lam
            # print(f"  new lam(2) = {lam}")
        # try to do Cholesky factors
        if out >= 2:
            print("B+lam*I = ")
            print(B+lam*np.eye(n))
        isPosDef = True
        try:
            L = LA.cholesky(B+lam*np.eye(n))
        except:
            isPosDef = False
        n_eval += 1

        if not isPosDef:
            if out >= 1:
                print("  B+lam*I is not pd")
            lam_lo = lam

        else:
            if out >= 1:
                print("  B+lam*I is pd")
            # dk = LA.solve(B+lam*np.eye(n), -g)
            dk = LA.solve(L, -g)
            dk = LA.solve(L.transpose(), dk)
            qk = LA.solve(L, dk)  # BUG unused variable
            norm_dk = LA.norm(dk)
            if out >= 1:
                print(f"  lam>0 => pd and |dk|={norm_dk}")

            # new_lam = lam + np.dot(dk, dk)/np.dot(qk,qk)*(norm_dk-rho)/rho

            if norm_dk <= rho:
                lam_up = lam
            else:
                lam_lo = lam
            # print(f"new_lam = {new_lam:f}")
            # print(f"[llo, lup] = [{lam_lo:f}, {lam_up:f}]")

    if ret_n_eval:
        return dk, n_eval

    return dk
