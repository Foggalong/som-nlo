import numpy as np


def ex13(ord: int, x, y=None):
    """
    Function for evaluating Example 1.3 from the lecture, which is also
    Example 0.1 from Roos, Terlaky, and de Klerk:

    f(x,y) =  x^2(4-2.1*x^2+1/3*x^4) + xy + y^2(-4+4y^2)

    df/dx = 2x*(4-2.1*x^2+1/3*x^4) + x^2*(4/3x^3-4.2*x) + y
          = 2/3x^5-4.2x^3+8x +4/3x^5 - 4.2x^3 + y
          = 2x^5 - 8.4x^3 + 8x + y
    df/dy = x + y^2(8y) + 2y(4y^2-4)
          = 8y^3 + 8y^3 -8 y + x = 16y^3 - 8y + x

    d2f/dx2 =10*x^4 - 25.2x^2 + 8
    d2f/dxdy = 1
    d2f/dy2 = 48y^2 - 8

    This is called as
       f = ex13(0, x);   - to get the function value f(x) at x
       g = ex13(1, x);   - to get the gradient value nabla f(x) at x
       H = ex13(2, x);   - to get the Hessian value nabla^2 f(x) at x
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
        return 1/3*x1**6 - 2.1*x1**4 + 4*x1**2 + x1*x2 + 4*x2**4 - 4*x2**2
    elif ord == 1:  # gradient
        # gradient of |x-loc(i)|^2 is 2*(x-loc(i))
        return np.array([
            2*x1**5 - 8.4*x1**3 + 8*x1 + x2,  # = dx
            x1 + 16*x2**3 - 8*x2              # = dy
        ])
    elif ord == 2:  # Hessian
        return np.array([
            [10*x1**4 - 25.2*x1**2 + 8, 1],
            [1, 48*x2**2 - 8]
        ])
    else:
        raise ValueError("first argument must be 0, 1 or 2.")
