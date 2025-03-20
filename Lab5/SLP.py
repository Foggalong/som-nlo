import numpy as np
import numpy.linalg as LA
import sys

from solveLP import solveLP 
import NonlinearProblem as NLP
import SLP_util as util


# ------------------ parameters for the method --------------------------
# problem can be "poolhav1", "ex915", "ex915mod"

problem = "poolhav1"
#problem = "ex915"
#problem = "ex915mod"


# Set (initial) trust region radius
#rho = 0.5
#rho = 1  
rho = 10 

do_TR_logic = False

max_iter = 50

tol = 1e-6 # tolerance for the line search method

out = 1
# ------------------ end parameters for the method --------------------------
if problem=="poolhav1":
    #nlfile = "ph1sc.nl"
    nlfile = "ph1sm.nl"
elif problem=="ex915":
    nlfile = "F915b.nl"
elif problem=="ex915mod":
    nlfile = "F915.nl"
else:
    print("Problem not recognized")
    sys.exit(1)
    

nlp = NLP.NonlinearProblem(nlfile)

# - - - - set starting point - - - -
# for a random starting point between lower and upper bound
#x0 = nlp.bl + np.random.rand(nlp.nvar)*(nlp.bu - nlp.bl)

#x0 = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
x0 = 0.3*np.ones(nlp.nvar)

if out>=2:
    print("x0 =")
    print(x0)


# make sure that the starting point is within the bounds of the problem
# if not -> adjust
for i in range(nlp.nvar):
    if x0[i]>nlp.bu[i]:
        x0[i] = nlp.bu[i]
    if x0[i]<nlp.bl[i]:
        x0[i] = nlp.bl[i]

xk = x0

# - - - - - - - - - start of main SLP iteration - - - - - - - - -

# iterate until maximal nuber of iteration reached
for iter in range(max_iter):
    if out>=2:
        print("================ iter %d ================="%(iter))
        print(xk)

    # evaluate f(x), nabla f(x), c(x), nabla c(x) at current point x=xk
    f = nlp.obj(xk)
    c = nlp.cons(xk)
    df = nlp.grad(xk)          # gradient of objective
    Jk = nlp.jac(xk).transpose()  # Jacobian of constraints

    # and evaluate total constraint violation
    h = util.eval_cviol(nlp, c)        
    
    if out>=1:
        print("SLP it %3d: f = %12.6f, |g(x)+|+|h(x)| = %12.5g"%(iter+1, f, h))
    if out>=2:
        print(xk)

    # The NLP is min f(x) subject to cl <= c(x) <= cu, bl <= x <= bu

    # The LP to solve at each iteration is
    # min (\nabla f(xk))'*d s.t.  cl - c(xk) <= (nabla c(xk))'*d <= cu - c(xk) 
    #                                 bl -xk <= d  <= bu - xk

    # get adjusted bounds for LP subproblem
    cl = nlp.cl-c
    cu = nlp.cu-c
    bl = nlp.bl-xk
    bu = nlp.bu-xk

    # impose Trust Region bounds:  -rho <= d_i <= rho
    bu = np.minimum(bu, rho) 
    bl = np.maximum(bl, -rho) 

    # solve the LP subproblem for step d
    d, stat = solveLP(df, Jk, cl, cu, bl, bu) 

    # test if LP was solved 
    if stat!=0:
        print("LP is infeasible. call Phase 1")
        sys.exit(1)
        #xkp = SLP_phase1(xk, rho)

    # test if step is small enough => then stop iteration
    if LA.norm(d)<tol:
        break

    xkp = xk+d

    # this is only done if we switch on the TR logic
    if do_TR_logic:
        if util.is_improvement(nlp, xk, xkp):
            rho = 2.*rho
            xk = xkp
        else:
            rho = rho/4.

        if out>=1:
            print("new rho = %f"%(rho))
    else:
        xk = xkp


print("SLP loop terminated after %d iters."%(iter+1))
if iter==max_iter-1:
    print("Maximal number of iterations reached.")
print("Solution/final point is:")
print("f = %f"%(f))
print("x* = ")
print(xk)



