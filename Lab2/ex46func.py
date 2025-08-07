import numpy as np


def ex46(ord, x, y=None):
    """
    This implements the Rosenbrock function from Example 4.6

        f(x1,x2) = sqrt(1+x1^2) + sqrt(1+x2^2)

    This is called as
        f = ex46(0, x);   - to get the function value f(x) at x
        g = ex46(1, x);   - to get the gradient value nabla f(x) at x
        H = ex46(2, x);   - to get the Hessian value nabla^2 f(x) at x
    """

    # check how many arguments are passed
    if y is None:
        # ensure that this works if x is a list, 1-d np.array, 2-d np.array
        # (in row or column orientation). we convert whatever it is into a
        # 2-d np.array (column) make sure that x is a column vector
        x = np.atleast_2d(x)  # convert to a 2d array if it was 1d
        shape_x = np.shape(x)
        if shape_x[1] > shape_x[0]:
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
        return np.sqrt(1+x1*x1) + np.sqrt(1+x2*x2)
    elif ord == 1:  # gradient value
        # (1+x^2)^(1/2) => f'(x) = 1/2(1+x^2)^{-1/2)*2x = x*(1+x^2)^(-1/2)
        return np.array([
            x1*(1+x1*x1)**(-1/2),
            x2*(1+x2*x2)**(-1/2)
        ])
    elif ord == 2:  # hessian value
        # f''(x) = (1+x^2)^(-3/2)
        return np.array([
            [(1+x1*x1)**(-3/2), 0],
            [0,  (1+x2*x2)**(-3/2)]
        ])
    else:
        raise ValueError("first argument must be 0, 1 or 2.")
