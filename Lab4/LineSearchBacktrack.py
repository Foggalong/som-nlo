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

import numpy as np

def  LineSearchBacktrack(xk, d, c1, func, ret_neval=False):
    
    # parameters to be used in the line search
    tau = 0.5
    alpha0 = 1

    # require output? (values 0 or 1)
    out = 1

    neval = 0
    f0 = func(0, xk)    # initial value
    g0 = func(1, xk).dot(d) # initial slope
    neval = neval + 1

    if out>1:
        print("f0 = "+str(f0))
        print("g0 = "+str(g0))
    

    alpha = alpha0

    # evaluate function value at xk+alpha*d
    f1 = func(0, xk+alpha*d)
    neval + neval + 1
    
    if out==1:
        print("al= % 8.5f, reduction= % 8.5f, required= % 8.5f" %(alpha, f0-f1, (-c1*alpha*g0)))
        
        

    # start loop (if not enough reduction)
    while (f0-f1 < -c1*alpha*g0):

        # reduce alpha and evaluate function at new point
        alpha = alpha*tau
        f1 = func(0, xk+alpha*d)
        neval = neval + 1
    
        # report progress
        if out==1: 
            print("al= % 8.5f, reduction= % 8.5f, required= % 8.5f" %(alpha, f0-f1, (-c1*alpha*g0)))

    if out==1:
        print("return al = %8.5f" %alpha)
              
    if ret_neval==True:
        return alpha, neval

    return alpha
            
