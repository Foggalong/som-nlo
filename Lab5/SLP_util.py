import numpy as np

import NonlinearProblem as NLP


def eval_constraint_violation(nlp, c):
    h = 0.0
    for j in range(nlp.ncon):
        if c[j] > nlp.cu[j]:
            h += (c[j] - nlp.cu[j])
        if c[j] < nlp.cl[j]:
            h += (nlp.cl[j] - c[j])
    return h


gamma = 1    # gamma to use for the merit function


def is_improvement(nlp, xk, xkp, output=True):
    # evaluate objective function value and constraint violation at
    # the current point and the potential new point

    f = nlp.obj(xk)
    c = nlp.cons(xk)
    h = eval_constraint_violation(nlp, c)

    fp = nlp.obj(xkp)
    cp = nlp.cons(xkp)
    hp = eval_constraint_violation(nlp, cp)

    # work out value of the merit function at both the current and new point
    m = f+gamma*h
    mp = fp+gamma*hp

    if output:
        print(f"merit function before/after: {f+gamma*h:f}, {fp+gamma*hp:f}")

    if (fp+gamma*hp < f+gamma*h):
        if output:
            print("improvement")
        return True
    else:
        if output:
            print("no improvement")
        return False
