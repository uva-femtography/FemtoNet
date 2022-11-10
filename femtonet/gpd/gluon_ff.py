def gffA(t):
    
    alpha_h = 0.58
    hmg = 1.13
    gffa = alpha_h/pow(1+t/(hmg*hmg),2)
    return gffa

def gffB(t):

    alpha_e = 0.0978
    emg = -2.5579
    gffb = alpha_e/pow(1+t/(emg*emg),2)
    return gffb

def gffD(t):

    alpha_d = -10
    dmg = 0.48
    gffd = alpha_d/pow(1+t/(dmg*dmg),2)
    return gffd
