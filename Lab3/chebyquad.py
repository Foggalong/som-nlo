"""
% This implements the chebyquad testfunction (eg. Fletcher 1965)
% Detailed description in More, Garbow, Hillstrom 
%
%       f(x) = \sum_{i=1}^n [f_i(x)]^2
%     where
%         f_i(x) = 1/n*(sum_{j=1}^n T_i(x_j)) - \int_0^1 T_i(x) dx 
%     here T_i(x) is the ith chebychev polynomial shifted to [0,1]
%    
%     we have \int_0^1 T_i(x) dx = 0,            i odd
%                                = -1/(i^2 - 1), i even
%    
%     T_i(x) can be obtained from the recurrence
%       T_0(x) = 1
%       T_1(x) = x   (likely -1+2*x on [0,1]
%       T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
%
% This is called as 
%    f = chebyquad(0, x);   - to get the function value f(x) at x
%    g = chebyquad(1, x);   - to get the gradient value \nabla f(x) at x
%    H = chebyquad(2, x);   - to get the Hessian value \nabla^2 f(x) at x
%
"""
import numpy as np

# problem data (at the moment hardwired)

# unlike the other methods that accept three argument, this only
# accepts x as a np.array()
def chebyquad(ord, x):
    x = x.flatten()  # in case it was passed as 2-d array. 
    n = x.size

    H=[]
    
    # if we just want the function value
    if ord == 0:
        # The chebyquad function is defined by
        #    f(x) = \sum_{i=1}^n (f_i(x))^2,
        # where
        #    f_i(x) = 1/n*(sum_{j=1}^n T_i(x_j)) - \int_0^1 T_i(x) dx 
        # here T_i(x) is the ith chebychev polynomial shifted to [0,1]
        #
        # we have \int_0^1 T_i(x) dx = 0,            i odd
        #                            = -1/(i^2 - 1), i even
        #
        # T_i(x) can be obtained from the recurrence (x in [-1,1])
        #   T_0(x) = 1
        #   T_1(x) = x   (likely -1+2*x on [0,1]
        #   T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
        #
        # or for y in [0,1]: x = 2y-1 and then evaluate T_k(x)
        #
        fvec = get_fvec(x)
        return np.sum(fvec*fvec)
    elif ord==1:
        fvec = get_fvec(x)
        J = np.zeros((n, n))
        tk = 1./n
        for j in  range(n):
            temp1 = 1
            temp2 = 2*x[j] - 1
            temp = 2*temp2
            temp3 = 0
            temp4 = 2
            for k in range(n):
                J[k][j] = tk*temp4
                ti = 4*temp2 + temp*temp4 - temp3
                temp3 = temp4
                temp4 = ti
                ti = temp*temp2 - temp1
                temp1 = temp2
                temp2 = ti
        # each column (row?) of J is now the gradient of one of the f_k
        # seems this is J[k][.]
        # f(x) = sum_k f_k(x)^2
        # => d/dx_i f(x) = \sum_k 2f_k(x) d/dxi f(k)
        g = np.zeros(n)
        for k in range(n):
            g = g + 2*fvec[k]*J[k]
        return g
    elif ord ==2 :
        H = np.zeros((n,n))
        hesd = np.zeros(n)  # these are the diagonal elements of the Hessian
        hesl = np.zeros(round(n*(n+1)/2))  # L elements by row
        gvec = np.zeros(n)

        # ----- first get the fvec 
        fvec = get_fvec(x)

        # ---------------------
        d1 = 1./n
        d2 = 2*d1
        im = 0
        for j in range(n):
            hesd[j] = 4*d1
            H[j][j] = 4*d1
            t1 = 1
            t2 = 2*x[j] - 1
            t = 2*t2
            s1 = 0
            s2 = 2
            p1 = 0
            p2 = 0
            gvec[0] = s2
            for i in range(1,n):
                th = 4*t2 + t*s2 - s1
                s1 = s2
                s2 = th
                th = t*t2 - t1
                t1 = t2
                t2 = th
                th = 8*s1 + t*p2 - p1
                p1 = p2
                p2 = th
                gvec[i] = s2
                hesd[j] = hesd[j] + fvec[i]*th + d1*s2**2
                H[j][j] = H[j][j] + fvec[i]*th + d1*s2**2

            hesd[j] = d2*hesd[j]
            H[j][j] = d2*H[j][j]
            for k in range(j):
                im = im + 1
                hesl[im] = 0
                H[k][j] = 0  #??
                H[j][k] = 0  #??
                tt1 = 1
                tt2 = 2*x[k] - 1
                tt = 2*tt2
                ss1 = 0
                ss2 = 2
                for i in range(n):
                    hesl[im] = hesl[im] + ss2*gvec[i]
                    H[k][j] = H[k][j] + ss2*gvec[i]
                    H[j][k] = H[j][k] + ss2*gvec[i]
                    tth = 4*tt2 + tt*ss2 - ss1
                    ss1 = ss2
                    ss2 = tth
                    tth = tt*tt2 - tt1
                    tt1 = tt2
                    tt2 = tth

                hesl[im] = d2*d1*hesl[im]
                H[k][j] = d2*d1*H[k][j]
                H[j][k] = d2*d1*H[j][k]
                
        #H = hesdl2H (hesd, hesl)

        return H
    else:
        print("first argument must be 0, 1 or 2.")
        

def get_fvec(x):
    n = x.size
    fvec = np.zeros(n)   # vector to build up fi(x)
    for j in range(n):    # for all values x_j
        temp1 = 1            # T0 (or T_{n-1})
        temp2 = 2*x[j] - 1   # T1 (or T_n or y)
        temp = 2*temp2       # 2*T_n(x)
        for i in range(n):
            fvec[i] = fvec[i] + temp2
            ti = temp*temp2 - temp1   # T_{n+1} = 
            temp1 = temp2
            temp2 = ti
            
    tk = 1./n
    iev = -1  # i is odd
    for k in range(n):
        fvec[k] = tk*fvec[k]
        if (iev > 0):  # if i is even
            fvec[k] = fvec[k] + 1./((k+1)**2 - 1)
        iev = -iev

    return fvec

def hesdl2H(hesd, hesl):
    # Reconstitute the Hessian matrix H from hesd and hesl
    # hesd : a column matrix, the diagonal part of the Hessian
    # hesl : a column matrix, the lower part terms, row by row.

    
    H = np.diag(hesd)
    n = hesd.size
    kc = 1
    k1 = 1
    k2 = 1
    #for i = 2:n
    for i in range(1,n):
        H[i][1:i-1] = hesl[k1:k2]
        H[1:i-1][i] = hesl[k1:k2]
        kc = kc + 1
        k1 = k2 + 1
        k2 = k1 + kc - 1

    return H

