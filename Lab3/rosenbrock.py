"""
% This implements Rosenbrocks function
%
%   f(x1,x2) = 100*(x2 - x1*x1)**2 + (1-x1)**2
%
% This is called as 
%    f = FuncRosenbrock(0, x);   - to get the function value f(x) at x
%    g = FuncRosenbrock(1, x);   - to get the gradient value \nabla f(x) at x
%    H = FuncRosenbrock(2, x);   - to get the Hessian value \nabla^2 f(x) at x
%
"""
import numpy as np
#rosenbrock function code
def rosenbrock(ord, x, y=None):
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
        
    if ord == 0:
        return 100*(x2 - x1*x1)**2 + (1-x1)**2
    elif ord == 1:
        # gradient value is required
  
        val = np.array([ 
            -400*(x2-x1**2)*x1-2*(1-x1),
            200*(x2-x1**2)
            ])
        return val
    elif ord == 2:
        # hessian value is required

        H = np.array([[800*x1**2+400*(x1**2-x2)+2, -400*x1],
             [-400*x1,   200]])

        return H
    else:
        print("first argument must be 0, 1 or 2.")
        
