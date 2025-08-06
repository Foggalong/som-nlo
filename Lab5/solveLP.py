import numpy as np
from scipy.optimize import linprog


def solveLP(c, A, bl, bu, cl, cu):
    """
    this is a convenient wrapper method that allows to solve an LP problem
    defined as

        min c'*x subject to bl <= A*x <= bu,
                            cl <=  x  <=cu

    which is more convenient for SLP/SQP type methods. It transforms the
    problem into the form accepted by scipy.optimize.linprog and then passes
    it to linprog to be solved. linprog solves

        min c'*x subject to A_ub*x <= b_ub, A_eq*x = b_eq, cl <= x <= cu

    This method identifies (and keeps) the equality constraints, whereas
    the inequality (one or two sided) are tranformed from

        bl <= Ax <= bu     to     Ax - s = 0, bl <= s <= bu

    Return values
    - x: solution (in the given form, not the linprog form, i.e. without s)
    - status: from linprog (0=solved, 2=infeasible)
    """

    out = 0

    n = cl.size
    m = bl.size

    # 1) ------ change the inequality constraints to equalities -----

    if out >= 2:
        print("solve LP called with the following LP:")
        print(f"c = \n{c}")
        print(f"A = \n{A}")
        print(f"bl = \n{bl}")
        print(f"bu = \n{bu}")
        print(f"cl = \n{cl}")
        print(f"cu = \n{cu}")

    # count number of inequality constraints in A

    n_ineq = 0
    for j in range(m):
        if np.abs(bu[j]-bl[j])>1e-8:
            n_ineq += 1

    # set up a matrix S for slacks (Ax - Ss = 0). Also vectors to hold
    # - bounds for s: scl, scu
    # = the new vector b (old bounds for orig equality c/s and 0 for others)
    S = np.zeros((n_ineq, m))
    newb = bl.copy()
    scl = np.zeros(n_ineq)
    scu = np.zeros(n_ineq)
    cnt = 0
    # loop through all constraints and set up S, scl, scu, b
    for j in range(m):
        if np.abs(bu[j]-bl[j]) > 1e-8:
            S[cnt, j] = 1.0
            scl[cnt] = bl[j]
            scu[cnt] = bu[j]
            newb[j] = 0.0
            cnt += 1

    # now we can rewrite bl <= A*x <= bu, cl<=x<=cu as
    #    [A -S]*[x;s] = newb, scl<=s<=scu, cl<=x<=cu

    # define augmented c, A to pass to linprog
    caug = np.append(c, np.zeros(n_ineq))
    AAug = np.block([A, -S.transpose()])
    # print(A.shape)
    # print(S.shape)

    # linprog wants bounds as a list of tuples. Can this be done quicker?
    bounds = []
    for i in range(n):
        bounds.append((cl[i], cu[i]))
    for i in range(n_ineq):
        bounds.append((scl[i], scu[i]))

    # call linprog
    res = linprog(caug, A_eq=AAug, b_eq = newb, bounds = bounds)

    if out >= 1:
        print(res.message)
    x = res.x[0:n]

    if out >= 1:
        print(f"Linprog return solution:\n{x}")

    if res.status > 0:
        if out >= 2:

            for i in range(n):
                print(f"cl, x, cu[{i:d}]: {cl[i]:f} {x[i]:f} {cu[i]:f}")

            for j in range(m):
                print(f"bl, bu[{j:d}]: {bl[j]:f} {bu[j]:f}")

    return x, res.status

# Linprog does not return dual variables (Lagrange multipliers)
# -> we can work them out by hand (I guess we can always solve the dual)

# c - A'*lam - mu = 0
# to work them out set lamda/mu=0 for basic variables (not at bounds)
# the rest should then just give a square linear system of equations

# code below was a (at the moment) abandoned attempt

# ax = np.dot(A, x)
# for i in range(m):
    # print(f"bl, Ax, bu[{i:d}]: {bl[i]:f} {ax[i]:f} {bu[i]:f}")

# print(AAug.shape)
# print(caug.shape)
# # find number of res.x solution at bounds
# natbnd = 0
# active = []
# for i in range(n):
#     if np.abs(res.x[i]-cl[i])<1e-6 or np.abs(res.x[i]-cu[i])<1e-6:
#         natbnd += 1
#         active.append(True)
#     else:
#         active.append(False)
# for i in range(n_ineq):
#     if np.abs(res.x[n+i]-scl[i])<1e-6 or np.abs(res.x[n+i]-scu[i])<1e-6:
#         natbnd += 1
#         active.append(True)
#     else:
#         active.append(False)
# print(f"At bound = {natbnd:d}")
# print(active)
