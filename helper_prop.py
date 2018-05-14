import time
import numpy as np
import cmath
from helper_Print import Print
from helper_ndot import ndot
from helper_cc import CCEnergy
from helper_cc import CCHbar
from helper_cc import CCLambda
from helper_cc import CCDensity

bold        = '\033[1m'
underline   = '\033[4m'
red         = '\033[31m'
green       = '\033[92m'
yellow      = '\033[93m'
blue        = '\033[94m'
purple      = '\033[95m'
cyan        = '\033[96m'
end         = '\033[0m'
colors      = [red,green,yellow,blue,purple,cyan]

class RK4(object):
    def __init__(self,ccsd,Lambda,density,options,memory=2):
        Print(yellow+"\n\nStarting time propagation with 4th order Runge Kutta...\n"+end)

        # Start timer
        time_init = time.time()

        self.options = options

        F  = ccsd.F.copy()
        t1 = ccsd.t1.copy()
        t2 = ccsd.t2.copy()
        l1 = Lambda.l1.copy()
        l2 = Lambda.l2.copy()

        self.mu = density.mu

        h = options['timestep']
        N = options['number of steps']
        T = options['timelength']
        t = 0.0
        Print(blue+'{:>6s}{:8.4f}'.format('t =',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('mu (Real):',float(self.mu[2]),'mu (Imag):',float(self.mu[2].imag))+end)

        for i in range(N):
            t += h
            t1 = t1 + self.ft1(t,h,ccsd,F,t1,t2,self.Vt)
            t2 = t2 + self.ft2(t,h,ccsd,F,t1,t2,self.Vt)
            hbar = CCHbar(ccsd,F+self.Vt(t),t1,t2)
            l1 = l1 + self.fl1(t,h,F,t1,t2,hbar,Lambda,l1,l2,self.Vt)
            l2 = l2 + self.fl2(t,h,F,t1,t2,hbar,Lambda,l1,l2,self.Vt)

            D = density.compute_ccsd_density(t1,t2,l1,l2)
            dipole = density.compute_ccsd_dipole(D)
            Print(blue+'{:>6s}{:8.4f}'.format('t =',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('mu (Real):',float(dipole[2].real),'mu (Imag):',float(dipole[2].imag))+end)

            if time.time() > T:
                Print(yellow+'\nEnd of propagation reached: time = %s seconds' %T+end)
                break

        Print(yellow+'\n End of propagation reached: steps = %s' %N+end)
        Print(yellow+'\t time elapsed: time = %.1f seconds' %(time.time()-time_init)+end)

    def Vt(self,t):
        # Field Parameters
        A = self.options['field amplitude']
        w = self.options['field frequency']

        # Select z-direction dipole moment
        mu = self.mu[2]
        V = -A*mu*np.exp(1j*w*2*np.pi*t)
        return V

    def ft1(self,t,h,ccsd,F,t1,t2,Vt):
        k1 = ccsd.update_t1(t1          ,t2,F+Vt(t))
        k2 = ccsd.update_t1(t1+h*k1/2.0 ,t2,F+Vt(t+h/2.0))
        k3 = ccsd.update_t1(t1+h*k2/2.0 ,t2,F+Vt(t+h/2.0))
        k4 = ccsd.update_t1(t1+h*k3     ,t2,F+Vt(t+h))
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def ft2(self,t,h,ccsd,F,t1,t2,Vt):
        k1 = ccsd.update_t2(t1,t2               ,F+Vt(t))
        k2 = ccsd.update_t2(t1,t2 + h*k1/2.0    ,F+Vt(t+h/2.0))
        k3 = ccsd.update_t2(t1,t2 + h*k2/2.0    ,F+Vt(t+h/2.0))
        k4 = ccsd.update_t2(t1,t2 + h*k3        ,F+Vt(t+h))
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def fl1(self,t,h,F,t1,t2,hbar,Lambda,l1,l2,Vt):
        k1 = Lambda.update_l1(hbar,t1,t2,l1          ,l2,F+Vt(t))
        k2 = Lambda.update_l1(hbar,t1,t2,l1+h*k1/2.0 ,l2,F+Vt(t+h/2.0))
        k3 = Lambda.update_l1(hbar,t1,t2,l1+h*k2/2.0 ,l2,F+Vt(t+h/2.0))
        k4 = Lambda.update_l1(hbar,t1,t2,l1+h*k3     ,l2,F+Vt(t+h))
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def fl2(self,t,h,F,t1,t2,hbar,Lambda,l1,l2,Vt):
        k1 = Lambda.update_l2(hbar,t1,t2,l1,l2               ,F+Vt(t))
        k2 = Lambda.update_l2(hbar,t1,t2,l1,l2 + h*k1/2.0    ,F+Vt(t+h/2.0))
        k3 = Lambda.update_l2(hbar,t1,t2,l1,l2 + h*k2/2.0    ,F+Vt(t+h/2.0))
        k4 = Lambda.update_l2(hbar,t1,t2,l1,l2 + h*k3        ,F+Vt(t+h))
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0
