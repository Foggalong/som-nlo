"""
 This method calculates a sequence approximating pi by the partial sums of
    
 4 - 4/3 + 4/5 - 4/7 + 4/9 - ...

 parameters:
     n: number of terms to return
"""
def pi_iter1_arctan(n):
    list = []
    xs = 0
    for i in range(n):
        xs = xs + ((-1)**i)*4./(2*i+1)
        list.append(xs)
        #print(xs)

    return list

# --------------------------------------------------------------------
"""
 This method calculates a sequence approximating pi by using Archimedes
 recurrence:

     a0 = 2*sqrt(3)
     b0 = 3

     a_{n+1} = 2*an*bn/(an+bn)
     b_{n+1} = sqrt(a_{n+1}*bn)
     
 parameters:
     n: number of terms to return
"""

import math

def pi_iter2_archim(n):
    list = []

    an = 2*math.sqrt(3)
    bn = 3
    list.append(an)

    for i in range(n-1):
        an = 2*an*bn/(an+bn)
        bn = math.sqrt(an*bn)    

        list.append(an)

    return list

# -------------------------------------------------------------------------
"""
 This method calculates a sequence approximating pi by using Borwein's
 recurrence:

     a0 = sqrt(2)
     b0 = 0
     p0 = 2*sqrt(2)
     
     a_{n+1} = (sqrt(an)+1/sqrt(an))/2
     b_{n+1} = (1+bn)*sqrt(an)/(an+bn)
     p_{n+1} = (1+an)*pn*bn/(1+bn)
     
 parameters:
     n: number of terms to return
"""
import math

def pi_iter3_borwein(n):
    list = []

    an = math.sqrt(2)
    bn = 0
    pn = 2+math.sqrt(2)
    list.append(pn)
    
    for i in range(n-1):
        san = math.sqrt(an)
        anp = (san+1./san)/2
        bnp = (1+bn)*san/(an+bn)
        an = anp
        bn = bnp
        pn = (1+an)*pn*bn/(1+bn)

        list.append(pn)

    return list

