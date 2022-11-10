import numpy as np
import math
import matplotlib.pyplot as plt
from kelly import kelly
from fut import fut
from gauss import dinter,dgaus1
from gluon_ff import gffA, gffB, gffD

class GPD:

    M = 0.93828

    def __init__(self,name,zeta,t,Q2):
        self.name = name
        self.zeta = zeta
        self.t = t
        self.Q2 = Q2
        
        xarr = np.zeros(491)
        for i in range(len(xarr)):
            if i <= 290.:
                yyy = math.log(1.e+4)*(330.-i+1)/330.
                xv = math.exp(-yyy)
            else:
                xstart = math.exp(-math.log(1.e+4)*41./330.)
                xv = xstart + (i-290.)*(1.-xstart)/201.
            xarr[i] = xv
        self.xarr = xarr

        gpd_minus_arr = np.zeros(491)
        gpd_plus_arr = np.zeros(491)
        gpd_arr = np.zeros(491)
        dglap_arr = np.zeros(491)

        self.gpd_minus_arr = gpd_minus_arr
        self.gpd_plus_arr = gpd_plus_arr
        self.gpd_arr = gpd_arr
        self.dglap_arr = dglap_arr

        self.m = 0
        self.Mx = 0
        self.Ml = 0
        self.alpha = 0
        self.alphap = 0
        self.p = 0
        self.N = 0

        f1p = kelly(-self.t)[4]
        f2p = kelly(-self.t)[5]
        f1n = kelly(-self.t)[6]
        f2n = kelly(-self.t)[7]

        gluon_a20 = gffA(-self.t)
        gluon_b20 = gffB(-self.t)
        gluon_c20 = gffD(-self.t)

        xi = self.zeta/(2.-self.zeta)
        
        self.ff = 0
        
        if self.name == "hu":
            self.ff = 2*f1p + f1n
        elif self.name == "hd":
            self.ff = 2*f1n + f1p
        elif self.name == "eu":
            self.ff = 2*f2p + f2n
        elif self.name == "ed":
            self.ff = 2*f2n + f2p
        elif self.name == "hg":
            self.ff = gluon_a20 + xi**2*gluon_c20
        elif self.name == "eg":
            self.ff = gluon_b20 - xi**2*gluon_c20

    def set_params(self):
        if self.name == "hu":
            self.m = 0.420
            self.Mx = 0.604
            self.Ml = 1.018
            self.alpha = 0.210
            self.alphap = 2.448
            self.p = 0.620
            self.N = 2.043
        elif self.name == "hd":
            self.m = 0.275
            self.Mx = 0.913
            self.Ml = 0.860
            self.alpha = 0.0317
            self.alphap = 2.209
            self.p = 0.658
            self.N = 1.570
        elif self.name == "eu":
            self.m = 0.420
            self.Mx = 0.604
            self.Ml = 1.018
            self.alpha = 0.210
            self.alphap = 2.811
            self.p = 0.863
            self.N = 1.803
        elif self.name == "ed":
            self.m = 0.275
            self.Mx = 0.913
            self.Ml = 0.860
            self.alpha = 0.0317
            self.alphap = 1.362
            self.p = 1.115
            self.N = -2.780
        elif self.name == "hg":
            self.m = 0
            self.Mx = 1.12
            self.Ml = 1.045
            self.alpha = 0.005
            self.N = 1.525
            self.alphap= 0.275
            self.p = 0.17
        elif self.name == "eg":
            self.m = 0
            self.Mx = 1.12
            self.Ml = 1.10
            self.alpha = 0.053
            self.N = 3.97
            self.alphap = 0.45
            self.p = -0.2
            

    def calc_gpd_dglap(self):

        Xp = (self.xarr - self.zeta)/(1.-self.zeta)
        XX = (self.xarr - self.zeta)/(1.-self.xarr)
        MMp = Xp*GPD.M**2 - XX*self.Mx**2 - self.Ml**2
        MM = self.xarr*GPD.M**2 - (self.xarr/(1.-self.xarr))\
            *self.Mx**2 - self.Ml**2
        t0 = -self.zeta**2*GPD.M**2/(1.-self.zeta)
        dT = math.sqrt((t0 - self.t)*(1.-self.zeta))

        omx = (1.-self.xarr)
        omx3 = omx**3
        zeta2 = self.zeta**2
        omz2 = (1.-self.zeta)**2
        mas = self.m + GPD.M*self.xarr
        masp = self.m + GPD.M*Xp
        omxp = (1.-self.xarr)/(1.-self.zeta)
        xmz = self.xarr - self.zeta
        omx2 = omx**2

        for kT in np.arange(0.,5.,0.1):
            D0 = (1.-self.xarr)*MM-kT*kT
            D1 = (1.-Xp)*MMp-kT**2 - dT**2*(1-Xp)**2
            D2 = (1.-Xp)*kT*dT
            reg1 = pow(self.xarr,-self.alpha)
            regp = -self.alphap*pow((1.-self.xarr),self.p)
            reg2 = pow(self.xarr,regp*self.t)
            kt2 = kT**2
            kmm = self.xarr*(1.-self.xarr)*GPD.M**2 - self.xarr*self.Mx**2\
                -(1.-self.xarr)*self.Ml**2 - kt2
            kmm2 = kmm**2
            aaa = Xp*omx*GPD.M**2 - (self.xarr-self.zeta)*self.Mx**2\
                -(1.-self.zeta)*kt2 - omxp*omx*dT**2 - omx*self.Ml**2
            bbb = -2*kT*dT
            

            if self.name == "hu" or self.name == "hd":
                self.dglap_arr += -self.N*2*np.pi*(1.-self.zeta/2)\
                    *reg1*reg2*(omx3/omz2)*kT*(((mas*masp+kt2)*D1-2*D2*D2)\
                    /(pow((D1*D1-4*D2*D2),1.5)*D0*D0))
            elif self.name == "eu" or self.name == "ed":
                self.dglap_arr += self.N*2*np.pi*(1-self.zeta/2)\
                    *reg1*reg2*((omx3)/(1-self.zeta))*kT*((2*GPD.M\
                    *((-2*GPD.M)*(self.xarr-Xp)*kt2-mas*(1.-Xp)*D1))\
                    /(pow((D1*D1-4*D2*D2),1.5)*D0*D0))
            elif self.name == "hg":
                Hg1 = -2*np.pi*self.N*reg1*reg2*kT*(1.-self.xarr)\
                    *(1.-self.xarr)*(self.xarr*xmz*(omx*GPD.M-self.Mx)\
                    *(omxp*GPD.M-self.Mx))*(1/(kmm**2))*(aaa/(pow(aaa*aaa\
                    -omx2*bbb*bbb,1.5)))
                Hg2 = -2*np.pi*self.N*reg1*reg2*kT*(1.-self.xarr)\
                    *(1.-self.xarr)*(1.-self.zeta+omx2)\
                    *kT*kT*(1./(kmm**2))*(aaa/(pow(aaa*aaa-omx2*bbb*bbb,1.5)))
                Hg3 = 2*np.pi*self.N*reg1*reg2*kT*(1.-self.xarr)\
                    *(1.-self.xarr)*(1.-self.zeta+omx2)\
                    *kT*dT*omxp*(1./(kmm**2))*(bbb/(pow(aaa*aaa\
                    -omx2*bbb*bbb,1.5)))
                self.dglap_arr += Hg1 + Hg2 + Hg3
                

        self.dglap_arr = self.dglap_arr*0.1
        return self.dglap_arr

    def fix_gpdh(self):
        new_dglap = 0
        if self.name == "hu":
            name = "eu"
        elif self.name == "hd":
            name = "ed"
        gpd_e = GPD(name,self.zeta,self.t,self.Q2)
        gpd_e.set_params()
        new_dglap = gpd_e.calc_gpd_dglap()
        self.dglap_arr += self.zeta**2/(4*(1.-self.zeta))*new_dglap
        return self.dglap_arr

    def calc_gpd_erbl(self,sym):
        xxsum = np.zeros(48)
        intsum = np.zeros(48)
        erbl_arr = np.zeros(491)
        
        for i in range(48):
            xxsum[i] = dinter(self.zeta,1,48,i)
            norm = 1/(1.-self.zeta/2.)
            if self.name == "hg":
                intsum[i] = fut(self.xarr,self.dglap_arr,xxsum[i])*norm
            else:
                intsum[i] = fut(self.xarr,self.dglap_arr,xxsum[i])
        dglap_int = dgaus1(48,self.zeta,1,intsum)

        if self.zeta != 0:
            gpdzeta = fut(self.xarr,self.dglap_arr,self.zeta)
            if sym == "minus":
                Serbl = (1.-(self.zeta/2.))*(self.ff\
                    - ((dglap_int)/(1.-(self.zeta/2))))
                aminus = (6/(self.zeta**3))*(self.zeta*gpdzeta-2*Serbl)
                erbl_arr += aminus*self.xarr**2\
                    - aminus*self.zeta*self.xarr + gpdzeta
            elif sym == "plus":
                aplus = 2000
                c = (2*gpdzeta+0.5*aplus*self.zeta**3)/self.zeta
                d = -gpdzeta
                erbl_arr += aplus*self.xarr**3 \
                    - 1.5*aplus*self.zeta*self.xarr**2 + c*self.xarr + d

        return erbl_arr

    def calc_gpd(self):
        self.calc_gpd_dglap()
        if self.name == "hu" or self.name == "hd":
            self.fix_gpdh()
        if self.name == "hg":
            plus_erbl = np.zeros(491)
            minus_erbl = self.calc_gpd_erbl("minus")
        else:
            plus_erbl = self.calc_gpd_erbl("plus")
            minus_erbl = self.calc_gpd_erbl("minus")
        for i in range(len(self.dglap_arr)):
            if self.xarr[i] < self.zeta:
                self.dglap_arr[i] = 0
            else:
                plus_erbl[i] = 0
                minus_erbl[i] = 0
        self.gpd_plus_arr = self.dglap_arr + plus_erbl
        self.gpd_minus_arr = self.dglap_arr + minus_erbl
        self.gpd_arr = 0.5*(self.gpd_plus_arr + self.gpd_minus_arr)

    def int_gpd(self,a,b):
        xxsum = np.zeros(48)
        intsum = np.zeros(48)

        for i in range(48):
            xxsum[i] = dinter(a,b,48,i)
            intsum[i] = fut(self.xarr,self.gpd_minus_arr,xxsum[i])

        gpd_int = dgaus1(48,a,b,intsum)

        return gpd_int

    def ff_gpd(self):
        return self.ff

    def to_ret(self,sym):
        if sym == "plus":
            to_ret = self.gpd_plus_arr
        elif sym == "minus":
            to_ret = self.gpd_minus_arr
        else:
            to_ret = self.gpd_arr

        return to_ret

    def calc_moment(self,n):
        xxsum = np.zeros(48)
        intsum = np.zeros(48)
        norm = 1./(1.-self.zeta/2.)
        if n % 2 == 0:
            to_int = self.gpd_minus_arr
        else:
            to_int = self.gpd_plus_arr
        for i in range(48):
            xxsum[i] = dinter(zeta/2,1,48,i)
            intsum[i] = fut(self.xarr,to_int\
                    *(((self.xarr-self.zeta/2.)\
                    /(1.-self.zeta/2.))**n)*(norm**(n+1)),xxsum[i])
        gpd_int = dgaus1(48,zeta/2,1,intsum)
        return gpd_int

    def calc_cff(self,reim):
        if self.zeta != 0:
            if reim == "re":
                xx = np.zeros(48)
                xx2 = np.zeros(48)
                erb = np.zeros(48)
                erb2 = np.zeros(48)
                erbz = np.zeros(48)
                erb2z = np.zeros(48)
                gpdzeta = fut(self.xarr,self.gpd_plus_arr,self.zeta)
                for i in range(48):

                    xx[i] = dinter(self.zeta/2,self.zeta,48,i)
                    xx2[i] = dinter(self.zeta,1.,48,i)

                    #zeta/2 -> zeta
                    erb[i] = (fut(self.xarr,self.gpd_plus_arr,xx[i]) \
                              - gpdzeta)/(xx[i]-self.zeta)
                    erb2[i] = fut(self.xarr,self.gpd_plus_arr,xx[i])/xx[i]

                    #zeta -> 1
                    erbz[i] = (fut(self.xarr,self.gpd_plus_arr,xx2[i]) \
                               - gpdzeta)/(xx2[i]-self.zeta)
                    erb2z[i] = fut(self.xarr,self.gpd_plus_arr,xx2[i])/xx2[i]
                    
                perb = dgaus1(48,self.zeta/2.,self.zeta,erb)
                perbz = dgaus1(48,self.zeta,1.,erbz)
                perb2 = dgaus1(48,self.zeta/2.,self.zeta,erb2)
                perbz2 = dgaus1(48,self.zeta,1,erb2z)
                plog = gpdzeta*np.log((1.-self.zeta)/(self.zeta/2))

                recff = perb + perb2 + plog + perbz + perbz2

                return recff
            
            elif reim == "im":
                imcff = np.pi*fut(self.xarr,self.gpd_plus_arr,self.zeta)
                return imcff
