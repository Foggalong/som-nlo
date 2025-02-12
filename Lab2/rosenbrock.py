import numpy as np


def rosenbrock(ord, x, y=None):
    """
    This implements Rosenbrock function

        f(x1,x2) = 100*(x2 - x1*x1)**2 + (1-x1)**2

    This is called as
        f = FuncRosenbrock(0, x);   - for the function value f(x) at x
        g = FuncRosenbrock(1, x);   - for the gradient value nabla f(x) at x
        H = FuncRosenbrock(2, x);   - for the Hessian value nabla^2 f(x) at x
    """

    # check how many arguments are passed
    if y is None:
        # ensure that this works if x is a list, 1-d np.array, 2-d np.array
        # (in row or column orientation). we convert whatever it is into a
        # 2-d np.array (column) make sure that x is a column vector
        x = np.atleast_2d(x)  # convert to a 2d array if it was 1d
        shpx = np.shape(x)
        if shpx[1] > shpx[0]:
            x = np.transpose(x)

        # and get the components (need two indices since it is a 2-d array)
        x1 = x[0][0]
        x2 = x[1][0]
    else:
        # in this case x and y should be np.arrays of the same size
        # and we want to evaluate the function for every point in the array
        if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)):
            raise ValueError("Arguments x and y must be of type np.array")
        if (x.shape != y.shape):
            raise ValueError("Arguments x and y must have the same shape")
        # if we get here we know that x, y are np.arrays of same shape
        if ord > 0:
            raise ValueError("If x and y are arrays can only evaluate "
                             "function not gradient or hessian")

        x1 = x
        x2 = y

    # - - - - - evaluate the function and its derivatives

    if ord == 0:  # function value
        return 100*(x2 - x1*x1)**2 + (1-x1)**2
    elif ord == 1:  # gradient value
        return np.array([
            -400*(x2-x1**2)*x1-2*(1-x1),
            200*(x2-x1**2)
        ])
    elif ord == 2:  # hessian value
        return np.array([
            [800*x1**2+400*(x1**2-x2)+2, -400*x1],
            [-400*x1,   200]
        ])
    else:
        raise ValueError("first argument must be 0, 1 or 2.")
