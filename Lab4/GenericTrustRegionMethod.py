"""
 Generic Trust Region Method: applies a generic trust region method to
 optimize a given function

 Can choose between different ways to solve the TR subproblem:
  - L_1 exact (by a QP solver)
  - L_2 exact (by Levenberg Marquardt)
  - L_2 approximate (by dog-leg)

 will return the path taken by the method (so that it can be plotted later)

 Called as path = GTRM(x0, function, tol);

 where
   x0 = starting point
   function = the function to be minimized
          [assumes that the function can be called as
           function(0, x), function(1, x), function(2, x)]
   tol = the convergence tolerance: iterate until || nabla f(x)|| < tol

 possible calling sequence is

   import numpy as np
   from rosenbrock import rosenbrock 
   from GenericTrustRegionMethod import GTRM
   path = GTRM(x0, rosenbrock, 1e-4)

"""
# ------------------ parameters for the method --------------------------

# solver can be LM (Levenberg-Marquardt)
#
subsolver = "LM"
#subsolver = "DL"

# set parameters for the TR method

r0 = 1 # inital TR size
del_up = 0.75  # upper threshhold for ratio: above this expand TR
del_lo = 0.25 # lower threshold for ratio: below this reduce TR
fact_inc = 2 # factor by which to increase TR
fact_dec = 0.25 # factor by which to decrease TR

max_iter = 100  # iteration limit

out = 1 # printing from method: 0=quiet, 1=path, 2=diagnostic

# ------------------ end parameters for the method --------------------------
import numpy as np
from numpy import linalg as LA   # We need this module for the norm
import sys

from solveTRLevenbergMarquardt import solveTRLM
from solveTRL2Cauchy import solveTRL2Cauchy

def  GTRM(x0, func, eps):
    if (not isinstance(x0, np.ndarray)):
        print("The argument x0 must be a np.array")
        return

    n = x0.size
    # this is to keep a list of iterates
    list = []

    xk = x0
    fk = func(0, xk)
    gk = func(1, xk)
    Hk = func(2, xk)
    list.append(np.array(xk))
    tot_feval = 1
    tot_geval = 1
    tot_solves = 0
    
    if out>=3:
        print("Initial x0=")
        print(xk)
        print("Initial f(x0)=")
        print(fk)
        print("Initial g(x0)=")
        print(gk)


    if out>=1:
        print("it=  0: f = % 8.5g, |g| = % 8.3g" %(fk, LA.norm(gk)))

    rho = r0
    iter = 0
    while (LA.norm(gk)>=eps and iter<max_iter):
        iter = iter + 1

        # - - - - - - - - - solve TR subproblem - - - - - - - - - - 
        # define the TR model: m(d) = 0.5*d*H*d + b'*d
        B = Hk
        g = gk

        if subsolver == "LM":
            dk, nsolves = solveTRLM(B, g, rho, 0.05, ret_neval=True)
        elif subsolver == "DL":
            dk, nsolves = solveTRL2Cauchy(B, g, rho, ret_neval=True)
        else:
            raise ValueError("Direction code not recognized")

            
        if out>=2:
            print("TR subproblem takes %d solves"%(nsolves))
        tot_solves = tot_solves + nsolves

        if out>=3:
            print("dk = ")
            print(dk)

        # - - - - - - - - - -  do the TR logic - - - - - - - - - - -

        xkp = xk+dk
        fkp = func(0, xkp)
        tot_feval = tot_feval + 1

        # predicted decrease = m(0) - m(dk) = 
        pred_dec = -(0.5*np.dot(dk, np.dot(B, dk)) + np.dot(g, dk) )
        # actual decrease = f(xk) - f(xkp)
        actual_dec = fk - fkp
        if out>=2:
            print("  pred dec = %8.5f"%pred_dec)
            print("  actu dec = %8.5f"%actual_dec)

        # delta to predicted/actual
        delta = actual_dec/pred_dec
        if out>=2:
            print("  -> delta = %8.5f"%delta)

        # do the actual trust region logic
        if delta>del_up:
            # accept step and increase TR radius
            xkn = xkp
            tot_geval = tot_geval + 1
            if np.abs(LA.norm(dk)-rho)/rho<0.1:
                rho = fact_inc*rho
            if out>=2:
                print("  ->accept step and increase rho = ",end="")
                print(rho)
        elif delta>del_lo:
            # accept step but leave TR radius as it was
            xkn = xkp            
            tot_geval = tot_geval + 1
            if out>=2:
                print("  ->accept step and leave rho unchanged")
        else:
            # reject step and decrease TR radius
            xkn = xk
            rho = fact_dec*rho
            if out>=2:
                print("  ->reject step and decrease rho = ",end="")
                print(rho)

        # take whatever step was decided
        xk = xkn

        fk = func(0, xk)
        gk = func(1, xk)
        Hk = func(2, xk)
        
        list.append(np.array(xk))

        if out>=1:
            print("it=% 3d: f = % 8.5g, |g| = %8.3g" %(iter, fk, LA.norm(gk)),end="");
            print(", rho=%8.3g, nsolves = %3d"%(rho, nsolves))
        if out>=3:
            print("xk=")
            print(xk)
            print("gk=")
            print(gk)


    print("GTRM took total of "+str(tot_feval)+" function evaluations")    
    print("GTRM took total of "+str(tot_geval)+" gradient evaluations")    
    print("GTRM took total of "+str(tot_solves)+" solves in TR subproblem")
    return np.array(list)
