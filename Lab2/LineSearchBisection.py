def LineSearchBisection(xk, d, c1, c2, func, ret_n_eval=False):
    """
    Bisection Linesearch from xk in direction d with parameters c1 and c2

    Called as alpha =  LineSearchBisection(xk, d, c1, c2, function);

    assumes that xk and d are of type numpy.array

    possible calling sequence is

    import numpy as np
    from rosenbrock import rosenbrock
    from LineSearchBisection import LineSearchBisection
    alpha = LineSearchBisection(np.array([-1,1]), np.array([1, -1]), 0.1, 0.9, rosenbrock)
    """

    # initial trial stepsize
    alpha = 1

    # initial interval
    alpha_l = 0
    alpha_u = float("inf")

    # require output? (values 0 or 1)
    out = 1

    n_eval = 0
    fk = func(0, xk)  # initial value
    gk = func(1, xk).dot(d)  # initial slope
    n_eval = n_eval + 1

    if out == 1:
        print("Interval= % 8.5f  % 8.5f" % (alpha_l, alpha_u))

    fk1 = func(0, xk+alpha*d)         # value at new trial point
    gk1 = func(1, xk+alpha*d).dot(d)  # slope at new trial point
    n_eval = n_eval + 1

    # found is an indicator that is set to 1 once both conditions are satisfied
    found = 0

    # start loop
    while (found == 0):

        # remember old alpha (only for progress report)
        alpha_old = alpha

        # test Armijo condition
        if (fk1 > fk + c1*alpha*gk):

            alpha_u = alpha
            alpha = 0.5*(alpha_l + alpha_u)

            if (out == 1):
                print("alpha = % f does not satisfy Armijo" % (alpha_old))
                print("New Interval % f % f" % (alpha_l, alpha_u))

        # test curvature condition
        elif (gk1 < c2*gk):

            alpha_l = alpha
            if (alpha_u > 1e10):
                alpha = 2*alpha_l
            else:
                alpha = 0.5*(alpha_l+alpha_u)

            if (out == 1):
                print("alpha = % f does not satisfy curvature" % (alpha_old))
                print("New Interval % f % f" % (alpha_l, alpha_u))

        else:
            found = 1

        if (out == 1):
            print("return alpha = % f" % (alpha))

        fk1 = func(0, xk+alpha*d)         # value at new trial point
        gk1 = func(1, xk+alpha*d).dot(d)  # slope at new trial point
        n_eval = n_eval + 1

    # end of loop

    if ret_n_eval:
        return alpha, n_eval

    return alpha
