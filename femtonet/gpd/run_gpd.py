import gpd
import math
import numpy as np

xarr = np.zeros(491)
for i in range(len(xarr)):
    if i <= 290.:
        yyy = math.log(1.e+4)*(330.-i+1)/330.
        xv = math.exp(-yyy)
    else:
        xstart = math.exp(-math.log(1.e+4)*41./330.)
        xv = xstart + (i-290.)*(1.-xstart)/201.
    xarr[i] = xv


M = 0.93828    
for zeta in xarr:
    if zeta <= 0.6:
        tmin = -zeta*zeta*M*M/(1-zeta)
        for t in np.arange(tmin-0.03,-2.03,-0.03):
            Q2 = 1.820
            gpd_hu = gpd.GPD("hu",zeta,t,Q2)
            gpd_hu.set_params()
            gpd_hu.calc_gpd()
            print(zeta," ",t," ",gpd_hu.calc_cff("re"))
            
