import numpy as np


def simplequad(ord, x, y=None, *, Q=np.array([[1, 0], [0, 1]]), b=np.array([0, 0])):
    """
    This implements a simple quadratic

        q(x) = b'*x + 1/2*x'*Q*x

    This is called as
        f = simplequad(0, x, Q, b);  - for the function value f(x) at x
        g = simplequad(1, x, Q, b);  - for the gradient value nabla f(x) at x
        H = simplequad(2, x, Q, b);  - for the Hessian value nabla^2 f(x) at x

    Q and b are optional (arguments that default to Q=I and b=0 if not given)
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
        return 0.5*Q[0][0]*x1*x1 + Q[0][1]*x1*x2 + 0.5*Q[1][1]*x2*x2 + b[0]*x1 + b[1]*x2
    elif ord == 1:  # gradient value
        return np.array([
            Q[0][0]*x1 + Q[0][1]*x2 + b[0],
            Q[0][1]*x1 + Q[1][1]*x2 + b[1]
        ])
    elif ord == 2:  # hessian value
        return Q
    else:
        raise ValueError("first argument must be 0, 1 or 2.")
