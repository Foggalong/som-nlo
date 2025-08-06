def LineSearchBacktrack(xk, d, c1, func, ret_n_eval=False):
    """
    Backtracking Armijo Linesearch from xk in direction d with parameter c1

    Called as alpha =  LineSearchBacktrack(xk, d, c1, function);

    assumes that xk and d are of type numpy.array

    possible calling sequence is

    import numpy as np
    from rosenbrock import rosenbrock
    from LineSearchBacktrack import LineSearchBacktrack
    alpha = LineSearchBacktrack(np.array([-1,1]), np.array([1, -1]), 0.9, rosenbrock)
    """

    # parameters to be used in the line search
    tau = 0.5
    alpha0 = 1

    # require output? (values 0 or 1)
    out = 1

    n_eval = 0
    f0 = func(0, xk)  # initial value
    g0 = func(1, xk).dot(d)  # initial slope
    n_eval = n_eval + 1

    if out > 1:
        print(f"f0 = {f0}")
        print(f"g0 = {g0}")

    alpha = alpha0

    # evaluate function value at xk+alpha*d
    f1 = func(0, xk+alpha*d)
    n_eval + n_eval + 1

    if out == 1:
        print(f"al= {alpha:8.5f}, reduction= {f0-f1:8.5f}, required= {-c1*alpha*g0:8.5f}")

    # start loop (if not enough reduction)
    while (f0-f1 < -c1*alpha*g0):
        # reduce alpha and evaluate function at new point
        alpha = alpha*tau
        f1 = func(0, xk+alpha*d)
        n_eval = n_eval + 1

        # report progress
        if out == 1:
            print(f"al= {alpha:8.5f}, reduction= {f0-f1:8.5f}, required= {-c1*alpha*g0:8.5f}")

    if out == 1:
        print(f"return al = {alpha:8.5f}")

    if ret_n_eval:
        return alpha, n_eval

    return alpha
