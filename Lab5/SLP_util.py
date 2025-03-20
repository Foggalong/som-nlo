import numpy as np

import NonlinearProblem as NLP

out = 1
# --------------------------- eval_cviol() -----------------------
def eval_cviol(nlp, c):
    h = 0.0
    for j in range(nlp.ncon):
        if c[j]>nlp.cu[j]:
            h = h + (c[j]-nlp.cu[j])
        if c[j]<nlp.cl[j]:
            h = h + (nlp.cl[j]- c[j])
    return h


# --------------------------- is_improvement -----------------------

gamma = 1    # gamma to use for the merit function

def is_improvement(nlp, xk, xkp):

    # evaluate objective function value and constraint violation at
    # the current point and the potential new point
    
    f = nlp.obj(xk)
    c = nlp.cons(xk)
    h = eval_cviol(nlp, c)        

    fp = nlp.obj(xkp)
    cp = nlp.cons(xkp)
    hp = eval_cviol(nlp, cp)        

    # work out value of the merit function at both the current and new point
    m = f+gamma*h
    mp = fp+gamma*hp

    if out>=1:
        print("merit function before/after: %f, %f"%(f+gamma*h, fp+gamma*hp)) 

    if (fp+gamma*hp<f+gamma*h):
        if out>=1:
            print("improvement")
        return True
    else:
        if out>=1:
            print("no improvement")
        return False



