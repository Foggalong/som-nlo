"""
 Exact Linesearch from xk in direction d with tolerance eps

 Called as alpha =  LineSearchExact(xk, d, tol, function);

 assumes that xk and d are of type numpy.array

 possible calling sequence is

 import numpy as np
 from rosenbrock import rosenbrock
 from LineSearchExact import LineSearchExact
 alpha = LineSearchExact(np.array([-1,1]), np.array([1, -1]), 0.01, rosenbrock)

"""

import numpy as np
import sys   # for the sys.exit() function
from numpy import linalg as LA

out = 0   # level pf printing from the method (0 or 1)

def  LineSearchExact(xk, d, tol, func, ret_neval=False):
    
    # The exact line search will try to obtain a point alpha at which the
    # directional derivative f'(x_k+alpha d, d) is zero
    # ->  \nabla f(x_k + alpha_k d)'* d = 0

    # this will do a bisection type algorithm to find the exact minimizer
    # - step 1: find an interval [al, au] such that
    #             f(x+al) <= f(x), f(x+au) < f(x)  (guaranteed decrease)
    #           and
    #              d'*\nabla f(x+al*d)<0,  d'*\nabla f(x+au*d)>0
    #
    # - step 2: then do a bisection linesearch to find the solution to
    #              d'*\nabla f(x+au*d)=0
    #           to within the specified tolerance

    #print(xk)
    #print(func(1, xk))
    #print(d)
    
    neval = 0 #count number of function evaluations

    nd = LA.norm(d)
    d = d/nd # make sure the direction vector has length=1

    fl = func(0, xk)    # initial value
    f00 = fl # remember the function value at alpha = 0
    gl = func(1, xk).dot(d) # initial slope
    neval = neval + 1
    # The f and gg values at alpha = 0 are the first values at al
    # => label them as fl, gl
    
    # - - - - - - - - step 1 - - - - - - -
    # For al, al=0 will do, but we can update to something better if we find
    # it. But we need an au with f(x+au*d) < f(x), g(x+au*d) > 0 

    al = 0
    au = float("inf")
    alpha = 1

    # evaluate f and g at the next trial point alpha => fn(ew), gn(ew)
    fn = func(0, xk+alpha*d)  # function value
    gn = func(1, xk+alpha*d).dot(d) # slope
    neval = neval + 1
    #print(func(1, xk+au*d))
    #print(d)
    #print(gn)

    # do a bisection type line search for step 1
    found = 0  # loop until we find an au with fu<f00 and gu>0
    while (found==0):

        # if fn<f00 and gn>0 we are done
        # if fn>f00 we need to get to smaller values
        if out==1:
            print("alpha = "+str(alpha))
        if fn<f00:
            if gn>0:
                if out==1:
                    print("f1<f0 and g1>0: end of phase 1")
                fu = fn
                gu = gn
                au = alpha
                found = 1
            else:   #gn<0
                # should try larger values of alpha
                # but the currently found alpha is a good al
                if out==1:
                    print("still gn<0, try larger alpha and al = alpha")
                al = alpha
                fl = fn
                gl = gn
                if au > 1e10:    # if au is still at Inf cannot take midpoint
                    alpha = 2*alpha
                else:
                    # we have aready decreased au (to some old alpha), but
                    # g(alpha)<0 and f(alpha)<0
                    # -> need to increase alpha
                    al = alpha
                    alpha = 0.5*(al+au)
                    if out==1:
                        print("f(alpha)<0 and g(alpha)<0 -> increase alpha")
        else: # in case we have stepped too far with alpha and get increase
            if out==1:
                print("fn>f0 -> try smaller alpha, reduce au = alpha")
            # in this case fn>f0:
            # -> should try smaller values of alpha
            au = alpha
            alpha = 0.5*(al+au)

        fn = func(0, xk+alpha*d)
        gn = func(1, xk+alpha*d).dot(d) 
        neval = neval + 1

    # we should now be at a position where the exact alpha is between al and au
    # report progress
    if out==1: 
        print("after step 1: al = % 8.5f, au= % 8.5f" %(al, au))
        print("f0 = "+str(f00)+", f(al) = "+str(f0)+", f(au) = "+str(f1))
        print("g(al) = "+str(g0)+", g(au) = "+str(g1))

    # - - - - - - step 2 - - - - - - -
    # At this point we have an interval al, au with
    # - f0 = f(xk+al*d) <= f00, f1 = f(xk+au*d) <= f00, 
    # - g0 = f'(xk+al*d)'*d <0, g1 = f'(xk+au*d)'*d >0 

    am = al # just in case after phase 1 the al/fl/gl already satisfies tol
    gn = gl
    while abs(gn)>tol:
        # by default just take the mid point between al and au
        am = 0.5*(al+au)
        
        # get f and g an the new trial point
        fn = func(0, xk+am*d)
        gn = func(1, xk+am*d).dot(d)
        neval = neval + 1
        if out==1: 
            print("am= % 8.5f, slope = % 8.5f" %(am, gn))

        if gn>0:
            au = am
            fu = fn
            gu = gn
        else:
            al = am
            fl = fn
            gl = gn
        if out==1: 
            print("new interval: al = % 8.5f, au= % 8.5f" %(al, au))

    if fn>f00:
        print("We do not have decrease, this should not happen!")
        sys.exit()
        
    # If we get here the slope at g1 should be below tol and am is the exact
    # line search value

    if ret_neval==True:
        return am/nd, neval

    return am/nd


