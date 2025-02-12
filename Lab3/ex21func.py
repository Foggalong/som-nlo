"""
% This is the Example 2.1 from the lecture
%
%  f(x,y) = x^4 + 2x^3 + 2x^2 + y^2 -2xy
% 
% df/dx = 4x^3 + 6x^2 + 4x - 2y
% df/dy = 2y - 2x
%
% d2f/dx2  = 12x^2 + 12x + 4
% d2f/dxdy = -2
% d2f/dy2  = 2
%
% This is called as 
%    f = ex21(0, x);   - to get the function value f(x) at x
%    g = ex21(1, x);   - to get the gradient value \nabla f(x) at x
%    H = ex21(2, x);   - to get the Hessian value \nabla^2 f(x) at x
%
"""
import numpy as np

# ex31 function code
def ex21(ord, x, y=None):
    if y is None:
        #print("argument y was not passed")
        #if isinstance(x, list):
        #    print("x is a list")
        #else:
        #    print("x is not a list")
            
        #if isinstance(x, np.ndarray):
        #    print("x is a numpy array")
        #else:
        #    print("x is not a numpy array")
    

        # the next bit of code is to ensure that this works if x is a
        # list, 1-d np.array, 2-d np.array (in row or column orientation)
        # -> we convert whatever it is into a 2-d np.array (column)
        # make sure that x is a column vector
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
        # return the function value
        f = x1**4 + 2*x1**3 + 2*x1**2 + x2**2 - 2*x1*x2
        return f
    elif ord == 1:
        # gradient value is required

        # gradient of \|x-loc(i)\|^2 is 2*(x-loc(i))
        dx = 4*x1**3 + 6*x1**2 + 4*x1 - 2*x2
        dy = 2*x2 - 2*x1
        val = np.array([dx, dy])
        return val
    elif ord == 2:
        # hessian value is required
        val = np.array([
            [12*x1**2 + 12*x1 + 4, -2],
            [-2, 2]
            ])
        return val
    else:
        print("first argument must be 0, 1 or 2.")
        
