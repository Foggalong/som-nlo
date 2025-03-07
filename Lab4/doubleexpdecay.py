"""
% This the residual function for the double exponential decay
%
%    phi(t, x=(r0, c1, c2, l1, l2)) = r0 + c1*exp(-l1*t) + c2*exp(-l2*t)
%
%    r_j(x) = phi(tj, x) - pj
%    
%
% This is called as 
%    r = ded(0, x);   - to get r = (r_1, ..., r_J)^T as a vector
%    J = ded(1, x);   - to get J = (nabla r_1(x)'; ... nabla r_J(x)') 
%    ddr = ded(2, x); - to get a list containing the nabla^2 r_j(x)
%
"""

import numpy as np
import math
import sys

# data
dat = np.array([
    [0.00000,     0.20000,     0.40000,     0.60000,     0.80000,     1.00000,
     1.20000,     1.40000,     1.60000,     1.80000,     2.00000,     2.20000,
     2.40000,     2.60000,     2.80000,     3.00000,     3.20000,     3.40000,
     3.60000,     3.80000,     4.00000,     4.20000,     4.40000,     4.60000,
     4.80000,     5.00000,     5.20000,     5.40000,     5.60000,     5.80000,
     6.00000,     6.20000,     6.40000,     6.60000,     6.80000,     7.00000,
     7.20000,     7.40000,     7.60000,     7.80000,     8.00000,     8.20000,
     8.40000,     8.60000,     8.80000,     9.00000,     9.20000,     9.40000,
     9.60000,     9.80000,    10.00000], 
    [9.5767,   8.4259,   8.0427,   6.9757,   6.5746,   6.2393,   5.5388,   5.6526,
     5.3516,   5.0689,   5.0862,   5.0781,   4.8370,   4.8209,   4.4617,   3.9884,
     4.7276,   4.1673,   3.9957,   4.0792,   4.0149,   4.0809,   3.6640,   3.8568,
     4.2736,   3.9693,   3.7559,   3.7391,   3.7753,   3.3620,   3.8535,   3.5764,
     3.6145,   3.7718,   3.5052,   3.3325,   3.3819,   3.8914,   3.4982,   3.4990,
     3.5908,   3.6194,   3.3982,   3.3793,   3.5217,   3.4442,   3.4576,   3.6838,
     3.5241,   3.6742,   3.6619]])
dat = dat.transpose()
# PS: the data was generated with the following parameters (plus errors) 
# r0 = 3.5
# c1 = 2
# c2 = 4
# l1 = 2
# l2 = 0.5

# unlike the other methods that accept three argument, this only
# accepts x as a np.array()
def ded(ord, x):
    x = x.flatten()  # in case it was passed as 2-d array. 
    n = x.size
    J, _ = dat.shape

    r0 = x[0]
    c1 = x[1]
    c2 = x[2]
    l1 = x[3]
    l2 = x[4]
    #print("params = ")
    #print([r0, c1, c2, l1, l2])
    
    # if we just want the function value
    if ord == 0:
        r = np.zeros(J)
        for j in range(J):
            dj = dat[j]
            tj = dj[0]
            pj = dj[1]
            try:
                r[j] = r0 + c1*math.exp(-l1*tj) + c2*math.exp(-l2*tj) - pj
            except OverflowError:
                print("Overflow in evaluation of NLS residual for")
                print("l1 = %g, l2 = %g" % (l1, l2))
                print("Cannot evaluate exp() - function")
                raise Exception("Overflow")
        return r
    elif ord==1:
        Jac = np.zeros((J,5))
        for j in range(J):
            dj = dat[j]
            tj = dj[0]
            pj = dj[1]

            Jac[j][0] = 1
            Jac[j][1] = math.exp(-l1*tj)
            Jac[j][2] = math.exp(-l2*tj)
            Jac[j][3] = -tj*c1*math.exp(-l1*tj)
            Jac[j][4] = -tj*c2*math.exp(-l2*tj)
            
        return Jac

    elif ord ==2 :
        ddr = []
        for j in range(J):
            dj = dat[j]
            tj = dj[0]
            pj = dj[1]
            Hj = np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, -tj*math.exp(-l1*tj), 0],
                [0, 0, 0, 0, -tj*math.exp(-l2*tj)],
                [0, -tj*math.exp(-l1*tj), 0, tj*tj*c1*math.exp(-l1*tj), 0],
                [0, 0, -tj*math.exp(-l2*tj), 0, tj*tj*c2*math.exp(-l2*tj)]])
            ddr.append(Hj)
        return ddr
        
