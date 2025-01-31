"""
% This implements a simple quadratic 
%
%   q(x) = b'*x + 1/2*x'*Q*x 
%
% This is called as 
%    f = simplequad(0, x, Q, b);  - to get the function value f(x) at x
%    g = simplequad(1, x, Q, b);  - to get the gradient value \nabla f(x) at x
%    H = simplequad(2, x, Q, b);  - to get the Hessian value \nabla^2 f(x) at x
%
% Q and b are optional (arguments that default to Q=I and b=0 if not given
%
"""
import numpy as np
#rosenbrock function code
def simplequad(ord, x, y=None, *, Q=None, b=None):
    if y is None:
        x = np.atleast_2d(x) # convert to a 2d array if it was 1d
        shpx = np.shape(x)
        #x
        #shpx
        if shpx[1]>shpx[0]:
            x = np.transpose(x)

        # and get the components (need two indices since it is a 2-d array)
        x1 = x[0][0]
        x2 = x[1][0]
    else:
        # in this case x and y should be np.arrays of the same size
        # and we want to evaluate the function for every point in the array
        if not(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            print("If two arguments are passed they must be np.arrays")
            raise ValueError("Arguments x and y must be of type np.array")
        if not(x.shape==y.shape):
            print("Arguments for x and y must have the same shape");
            raise ValueError("Arguments x and y must have the same shape")
        # if we get here we know that x, y are np.arrays of same shape
        if ord>0:
            print("If x and y are arrays can only evaluate function not gradient or hessian")
            raise ValueError("If x and y are arrays can only evaluate function not gradient or hessian")

        x1 = x
        x2 = y

    if Q is None:
        Q = np.array([[1, 0],[0, 1]]);

    if b is None:
        b = np.array([0, 0]);
        
    if ord == 0:
        return 0.5*Q[0][0]*x1*x1+Q[0][1]*x1*x2+0.5*Q[1][1]*x2*x2+b[0]*x1+b[1]*x2
    elif ord == 1:
        # gradient value is required
  
        val = np.array([ 
            Q[0][0]*x1 + Q[0][1]*x2 + b[0],
            Q[0][1]*x1 + Q[1][1]*x2 + b[1]
            ])
        return val
    elif ord == 2:
        return Q
    else:
        print("first argument must be 0, 1 or 2.")
        
