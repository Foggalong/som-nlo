import numpy as np

# problem data (at the moment hardwired)
s = np.array([[6, 4], [0, -10], [5, -3], [1, -4], [3, -3]])
r = np.array([8.0051, 13.0112, 9.1138, 7.7924, 8.0210])


def sourceloc(ord, x, y=None):
    """
    This implements the objective of the Source Localisation problem from
    Antoniou/Lu, Ch 9.5, pp275

        f(x) = 1/2*sum_{i=1}^m (|x - s_i|_2 - r_i)^2

    This is called as:
        f = sourceloc(0, x);   - to get the function value f(x) at x
        g = sourceloc(1, x);   - to get the gradient value nabla f(x) at x
        H = sourceloc(2, x);   - to get the Hessian value nabla^2 f(x) at x
    """

    if y is None:
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
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            raise ValueError("Arguments x and y must be of type np.array")
        if not (x.shape == y.shape):
            raise ValueError("Arguments x and y must have the same shape")
        # if we get here we know that x, y are np.arrays of same shape
        if ord > 0:
            raise ValueError("If x and y are arrays can only evaluate "
                             "function not gradient or hessian")

        x1 = x
        x2 = y

    m, n = s.shape

    if ord == 0:
        val = 0
        for i in range(m):
            norm = np.sqrt((s[i][0]-x1)**2 + (s[i][1]-x2)**2)
            val += (norm - r[i])**2
        return 0.5*val
    elif ord == 1:
        # gradient value is required
        val = np.array([0, 0])
        for i in range(m):
            norm = np.sqrt((s[i][0]-x1)**2 + (s[i][1]-x2)**2)
            fact = 1-r[i]/norm
            val += fact*np.array([x1-s[i][0], x2-s[i][1]])
        return val
    elif ord == 2:
        # hessian value is required
        H = np.array([[0, 0], [0, 0]])
        tau = m
        for i in range(m):
            norm = np.sqrt((s[i][0]-x1)**2 + (s[i][1]-x2)**2)
            fact = r[i]/(norm**3)
            tau = tau - r[i]/norm
            H += fact*np.array([
                [(x1-s[i][0])**2, (x1-s[i][0])*(x2-s[i][1])],
                [(x1-s[i][0])*(x2-s[i][1]), (x2-s[i][1])**2]])

        H += tau*np.array([[1, 0], [0, 1]])
        return H
    else:
        raise ValueError("first argument must be 0, 1 or 2.")
