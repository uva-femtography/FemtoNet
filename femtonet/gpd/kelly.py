import math
import numpy as np

def kelly(t):

    a1_ep = -0.24
    a1_mp = 0.12
    a1_mn = 2.33

    b1_ep = 10.98
    b1_mp = 10.97
    b1_mn = 14.72

    b2_ep = 12.82
    b2_mp = 18.86
    b2_mn = 24.20

    b3_ep = 0.12
    b3_mp = 6.55
    b3_mn = 84.1

    ea1ep = 0.12
    ea1mp = 0.04
    ea1mn = 1.40

    eb1ep = 0.19
    eb1mp = 0.11
    eb1mn = 1.70

    eb2ep = 1.10
    eb2mp = 0.28
    eb2mn = 9.80

    eb3ep = 6.80
    eb3mp = 1.20
    eb3mn = 41.0

    aa = 1.70
    bb = 3.30
    eaa = 0.04
    ebb = 0.32

    amp = 0.93828
    amup = 2.790
    amun = -1.910

    tau = t/(4*amp**2)

    gep = (1+a1_ep*tau)/(1+b1_ep*tau+b2_ep*tau**2+b3_ep*tau**3)
    gmp = amup*(1+a1_mp*tau)/(1+b1_mp*tau+b2_mp*tau**2+b3_mp*tau**3)
    gmn = amun*(1+a1_mn*tau)/(1+b1_mn*tau+b2_mn*tau**2+b3_mn*tau**3)
    gen = aa*tau/(1+tau*bb)/(1+t/0.71)**2

    e1 = tau/(1+b1_ep*tau+b2_ep*tau**2+b3_ep*tau**3)
    e2 = (1+a1_ep*tau)/(1+b1_ep*tau+b2_ep*tau**2+b3_ep*tau**3)**2*tau
    e3 = (1+a1_ep*tau)/(1+b1_ep*tau+b2_ep*tau**2+b3_ep*tau**3)**2*tau**2
    e4 = (1+a1_ep*tau)/(1+b1_ep*tau+b2_ep*tau**2+b3_ep*tau**3)**2*tau**3
    
    egep = math.sqrt((e1*ea1ep)**2+(e2*eb1ep)**2+(e3*eb2ep)**2+(e4*eb3ep)**2)

    e1m = tau/(1+b1_mp*tau+b2_mp*tau**2+b3_mp*tau**3)
    e2m = (1+a1_mp*tau)/(1+b1_mp*tau+b2_mp*tau**2+b3_mp*tau**3)**2*tau
    e3m = (1+a1_mp*tau)/(1+b1_mp*tau+b2_mp*tau**2+b3_mp*tau**3)**2*tau**2
    e4m = (1+a1_mp*tau)/(1+b1_mp*tau+b2_mp*tau**2+b3_mp*tau**3)**2*tau**3

    egmp = math.sqrt((e1m*ea1mp)**2+(e2m*eb1mp)**2+(e3m*eb2mp)**2+(e4m*eb3mp)**2)
    
    e1n = tau/(1+b1_mn*tau+b2_mn*tau**2+b3_mn*tau**3)
    e2n = (1+a1_mn*tau)/(1+b1_mn*tau+b2_mn*tau**2+b3_mn*tau**3)**2*tau
    e3n = (1+a1_mn*tau)/(1+b1_mn*tau+b2_mn*tau**2+b3_mn*tau**3)**2*tau**2
    e4n = (1+a1_mn*tau)/(1+b1_mn*tau+b2_mn*tau**2+b3_mn*tau**3)**2*tau**3

    egmn = math.sqrt((e1n*ea1mn)**2+(e2n*eb1mn)**2+(e3n*eb2mn)**2+(e4n*eb3mn)**2)
    e1t = tau/(1+tau*bb)/(1+t/0.71)**2
    e2t = aa*tau**2/(1+tau*bb)**2/(1+t/0.71)**2
    egen = math.sqrt((e1t*eaa)**2 + (e2t*ebb)**2)

    f1p = (tau*gmp+gep)/(1+tau)
    f2p = (gmp-gep)/(1+tau)
    f1n = (tau*gmn + gen)/(1+tau)
    f2n = (gmn-gen)/(1+tau)

    ef1p = math.sqrt((tau*egmp)**2+(egep)**2)/(1+tau)
    ef2p = math.sqrt((egmp)**2+(egep)**2)/(1+tau)
    ef1n = math.sqrt((tau*egmn)**2+(egen)**2)/(1+tau)
    ef2n = math.sqrt((egmn)**2+(egen)**2)/(1+tau)
    
    return [gep,gen,gmp,gmn,f1p,f2p,f1n,f2n,egep,egen,egmp,egmn]
