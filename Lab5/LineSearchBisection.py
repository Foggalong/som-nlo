"""
 Bisection Linesearch from xk in direction d with parameters c1 and c2

 Called as alpha =  LineSearchBisection(xk, d, c1, c2, function);

 assumes that xk and d are of type numpy.array

 possible calling sequence is

 import numpy as np
 from rosenbrock import rosenbrock
 from LineSearchBisection import LineSearchBisection
 alpha = LineSearchBisection(np.array([-1,1]), np.array([1, -1]), 0.1, 0.9, rosenbrock)

"""
import numpy as np
import sys
#from LSPlot import LSPlot

max_alpha = 10000

# require output? (values 0 or 1)
out = 0

def  LineSearchBisection(xk, d, c1, c2, func, ret_neval=False):

    # initial trial stepsize
    alpha = 1

    #print(xk)
    #print(d)
    #LSPlot(xk, d, c1, func, alpha)


    # initial interval
    alphal = 0
    alphau = np.infty


    neval = 0
    fk = func(0, xk)    # initial value
    gk = func(1, xk).dot(d) # initial slope
    neval = neval + 1
    
    if out==1:
        print("Interval= % 8.5f  % 8.5f" %(alphal, alphau))


    fk1 = func(0, xk+alpha*d)         # value at new trial point
    gk1 = func(1, xk+alpha*d).dot(d)  # slope at new trial point
    neval = neval + 1

    # found is an indicator that is set to 1 once both conditions are satisfied
    found = 0

    # start loop
    while (found==0):
   
        # remember old alpha (only for progress report)
        alpha_old = alpha

        # test Armijo condition
        if (fk1 > fk + c1*alpha*gk):

            alphau = alpha
            alpha = 0.5*(alphal + alphau)

            if (out==1):
                print("alpha = % f does not satisfy Armijo" %(alpha_old))
                print("New Interval % f % f" %(alphal, alphau))

  
        # test curvature condition
        elif (gk1<c2*gk):

            alphal = alpha
            if (alphau > 1e10):
                alpha = 2*alphal
                if alpha>max_alpha:
                    print("Bisection line search found unbounded direction")
                    print("STOP")
                    sys.exit(1)
            else:
                alpha = 0.5*(alphal+alphau)
      

            if (out==1):
                print("alpha = % f does not satisfy curvature condition" %(alpha_old))
                print("New Interval % f % f" %(alphal, alphau))
                
                
        else:
            found = 1

        if (out==1):
            print("return alpha = % f" %(alpha))


        fk1 = func(0, xk+alpha*d)        # value at new trial point
        gk1 = func(1, xk+alpha*d).dot(d) # slope at new trial point
        neval = neval + 1

    #end of loop

    if ret_neval==True:
        return alpha, neval

    return alpha
