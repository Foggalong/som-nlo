from doubleexpdecay import ded as rfunc
import numpy as np

# -------- parameters -------------
use_GN = False
# use_GN = True
# ---------------------------------


def nls(ord, x):
    """
    This implements a nonlinear least squares optimization to fit the
    model function phi(t; x) (where t is the fitting variable and x are
    the model parameters) to the data (t_j, p_j):

        f(x) = sum_{j=1}^J r_j(x)^2, where r_j(x) = phi(t_j, x) - p_j

    we have

        nabla f(x)   = J(x)'*r(x)
        nabla^2 f(x) = J'(x)'*J(x) + sum_{j=1}^J r_j(x) nabla^2 r_j(x)

    where

               (nabla r_1(x)')
        J(x) = (     :        )
               (nabla r_J(x)')

    Unlike the other methods that accept three argument, this only accepts
    x as an np.array().

    This is called as:

        f = nls(0, x);   - to get the function value f(x) at x
        g = nls(1, x);   - to get the gradient value nabla f(x) at x
        H = nls(2, x);   - to get the Hessian value nabla^2 f(x) at x
    """

    x = x.flatten()  # in case it was passed as 2-d array.
    # n = x.size  # BUG unused variable

    # print(f"nls called with x = \n{x}")

    H = []

    # if we just want the function value
    if ord == 0:
        r = rfunc(0, x)  # this is a J-vector of r_j(x)
        return 0.5*np.dot(r, r)
    elif ord == 1:
        r = rfunc(0, x)
        J = rfunc(1, x)  # This is a matrix with nabla r_j(x) as rows
        return np.dot(J.transpose(), r)
    elif ord == 2:
        r = rfunc(0, x)
        J = rfunc(1, x)
        ddr = rfunc(2, x)  # This is a list with nabla^2 r_j(x) as elements

        nJ = r.size
        H = np.dot(J.transpose(), J)
        if (not use_GN):
            for j in range(nJ):
                H += r[j]*ddr[j]
        return H
