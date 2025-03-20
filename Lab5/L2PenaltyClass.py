"""
% This implements an L1 Penalty function for a constrained NLP
%  
%   min f(x), s.t. bl <= c(x) <= bu
%
% L1pen(x) = f(x)
%
% This is called as 
%    f = l1pen(0, x);   - to get the function value L1pen(x) at x
%    g = l1pen(1, x);   - to get the gradient value \nabla L1pen(x) at x
%    H = l1pen(2, x);   - to get the Hessian value \nabla^2 L1pen(x) at x
%
"""
import numpy as np

import NonlinearProblem as NLP

#nlp = NLP.NonlinearProblem("poolhav1sc.nl")
#nlp = NLP.NonlinearProblem("ph1sc.nl")
#nlp = NLP.NonlinearProblem("ph1sm.nl")
#nlp = NLP.NonlinearProblem("ph1sc.nl","EqBnd")
nlp = None

default_fact = 50.0

out = 0

class L2Penalty:

    def __init__(self, nlfile):
        global nlp
        self.fact = default_fact 
        nlp = NLP.NonlinearProblem(nlfile)


    def set_fact(self, fact):
        self.fact = fact
        
    def get_nvar(self):
        return nlp.nvar

    def get_nlp_bl(self):
        return nlp.bl

    def get_nlp_bu(self):
        return nlp.bu

    def l2pen(self, ord, x):
        x = x.flatten()  # in case it was passed as 2-d array. 
        n = x.size

        fact = self.fact
        H=[]
    
        # if we just want the function value
        if ord == 0:
            # L2pen(x) = nlp.f(x)
            # + 0.5*sum_i(max(c_i(x)-bl_i, 0)^2) + 0.5*sum_i(max(bu_i-c_i(x), 0)^2)
            
            f = nlp.obj(x)
            c = nlp.cons(x)
            for i in range(nlp.ncon):
                if c[i]<nlp.cl[i]:
                    f = f + 0.5*fact*(nlp.cl[i] - c[i])**2
                if c[i]>nlp.cu[i]:
                    f = f + 0.5*fact*(c[i] - nlp.cu[i])**2

            for i in range(nlp.nvar):
                if x[i]<nlp.bl[i]:
                    f = f + 0.5*fact*(nlp.bl[i] - x[i])**2
                if x[i]>nlp.bu[i]:
                    f = f + 0.5*fact*(x[i] - nlp.bu[i])**2

            return f
        elif ord==1:
            # gradient of L2 penalty: f(x) + 0.5*c h(x)'*h(x)
            #  nabla f(x) + c*h(x)*nabla h(x)
        
            df = nlp.grad(x)
            c = nlp.cons(x)
            jac = nlp.jac(x)
            if out>=1:
                print("x = ")
                print(x)
                print("df = ")
                print(df)
                print("c = ")
                print(c)
                print("jac = ")
                print(jac)
        
        
            for i in  range(nlp.ncon):
                if c[i]<nlp.cl[i]:
                    df = df - fact*(nlp.cl[i]-c[i])*jac[:,i]
                if c[i]>nlp.cu[i]:
                    df = df + fact*(c[i]-nlp.cu[i])*jac[:,i]

            for i in  range(nlp.nvar):
                if x[i]<nlp.bl[i]:
                    df[i] = df[i] - fact*(nlp.bl[i]-x[i])
                if x[i]>nlp.bu[i]:
                    df[i] = df[i] + fact*(x[i]-nlp.bu[i])

            if out>=1:
                print("df = ")
                print(df)

            return df
        elif ord ==2:
            # Hessian of L2 penalty: f(x) + 0.5*c h(x)'*h(x)
            #  grad = nabla f(x) + c*h(x)*nabla h(x)
            #  hess = nabla^2 f(x) + c*h(x)\nabla^2 h(x)+c*\nabla h(x)'*\nabla h(x)

            lam = np.zeros(nlp.ncon)
            c = nlp.cons(x)
            H0 = nlp.hess(x, lam)
            J0 = nlp.jac(x)

            #lam = fact*c
            H = H0.copy()

            for i in range(nlp.ncon):
                lam[i] = 1.0
                Hi = nlp.hess(x, lam) - H0
                if c[i]<nlp.cl[i]:
                    H = H + fact*((nlp.cl[i]-c[i])*Hi + np.outer(J0[:,i],J0[:,i]))
                if c[i]>nlp.cu[i]:
                    H = H + fact*((c[i]-nlp.cu[i])*Hi + np.outer(J0[:,i],J0[:,i]))
                lam[i] = 0.0

            for i in range(nlp.nvar):
                if x[i]<nlp.bl[i]:
                    H[i,i] = H[i,i] + 1.0*fact
                if x[i]>nlp.bu[i]:
                    H[i,i] = H[i,i] + 1.0*fact

        
            return H
        else:
            print("first argument must be 0, 1 or 2.")
        

