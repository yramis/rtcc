import time
import json
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
        Print(yellow+"\nStarting time propagation with 4th order Runge Kutta...\n"+end)

        # Start timer
        time_init = time.time()

        self.options = options

        F  = ccsd.F.copy()
        t1 = ccsd.t1.copy()
        t2 = ccsd.t2.copy()
        l1 = Lambda.l1.copy()
        l2 = Lambda.l2.copy()

        self.mu = density.mu
        D = density.compute_ccsd_density(t1,t2,l1,l2)
        np.set_printoptions(precision=5,suppress=True)
        self.check_trace(D,ccsd.n_e)

        data = {}
        data['parameters'] = options
        data['time']    = []
        data['t1 (real part)']      = []
        data['t1 (imag part)']      = []
        data['t2 (real part)']      = []
        data['t2 (imag part)']      = []
        data['l1 (real part)']      = []
        data['l1 (imag part)']      = []
        data['l2 (real part)']      = []
        data['l2 (imag part)']      = []
        data['dipole (real part)']  = []
        data['dipole (imag part)']  = []

        h = options['timestep']
        N = options['number of steps']
        T = options['timelength']
        t = 0.0
        Print(blue+'{:>6s}{:8.4f}'.format('t =',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('mu (Real):',self.mu[2],'mu (Imag):',self.mu[2].imag)+end)

        for i in range(N):
            t += h
            t1 = t1 + self.ft1(ccsd,t,h,F,self.Vt,t1,t2)
            t2 = t2 + self.ft2(ccsd,t,h,F,self.Vt,t1,t2)
            hbar = CCHbar(ccsd,F+self.Vt(t),t1,t2)
            l1 = l1 + self.fl1(hbar,Lambda,t,h,F,self.Vt,t1,t2,l1,l2)
            l2 = l2 + self.fl2(hbar,Lambda,t,h,F,self.Vt,t1,t2,l1,l2)

            D = density.compute_ccsd_density(t1,t2,l1,l2)
            self.check_trace(D,ccsd.n_e)
            dipole = density.compute_ccsd_dipole(D)

            data['time'].append(t)
            data['t1 (real part)'].append(t1.real.tolist())
            data['t1 (imag part)'].append(t1.imag.tolist())
            data['t2 (real part)'].append(t2.real.tolist())
            data['t2 (imag part)'].append(t2.imag.tolist())
            data['l1 (real part)'].append(l1.real.tolist())
            data['l1 (imag part)'].append(l1.imag.tolist())
            data['l2 (real part)'].append(l2.real.tolist())
            data['l2 (imag part)'].append(l2.imag.tolist())
            data['dipole (real part)'].append(dipole[2].real)
            data['dipole (imag part)'].append(dipole[2].imag)

            Print(blue+'{:>6s}{:8.4f}'.format('t =',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('mu (Real):',dipole[2].real,'mu (Imag):',dipole[2].imag)+end)

            if t>0 and round(t/h)%1000 == 0:      #write checkpoint
                init = time.time()
                with open('data.json','w') as outfile:
                    json.dump(data,outfile,indent=2)
                Print(yellow+'\n\t checkpoint: saving data to json in %.1f seconds\n' %(time.time()-init)+end)

            if time.time() > T:
                Print(yellow+'\nEnd of propagation reached: time = %s seconds' %T+end)
                break

        Print(yellow+'\n End of propagation reached: steps = %s' %N+end)
        Print(yellow+'\t time elapsed: time = %.1f seconds' %(time.time()-time_init)+end)

        with open('data.json','w') as outfile:
            json.dump(data,outfile,indent=2)

    def Vt(self,t):
        # Field Parameters
        A = self.options['field amplitude']
        w = self.options['field frequency']

        # Select z-direction dipole moment
        mu = self.mu[2]
        V = -A*mu*np.exp(1j*w*np.pi*t)
        if t<=0.5/w:
            #return V
            return 0.0 + 0j
        else:
            return 0.0 + 0j

    def ft1(self,ccsd,t,h,F,Vt,t1,t2):
        k1 = ccsd.update_t1(F + Vt(t),          t1,             t2)
        k2 = ccsd.update_t1(F + Vt(t+h/2.0),    t1 + h*k1/2.0,  t2)
        k3 = ccsd.update_t1(F + Vt(t+h/2.0),    t1 + h*k2/2.0,  t2)
        k4 = ccsd.update_t1(F + Vt(t+h),        t1 + h*k3,      t2)
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def ft2(self,ccsd,t,h,F,Vt,t1,t2):
        k1 = ccsd.update_t2(F + Vt(t),          t1, t2)
        k2 = ccsd.update_t2(F + Vt(t+h/2.0),    t1, t2 + h*k1/2.0)
        k3 = ccsd.update_t2(F + Vt(t+h/2.0),    t1, t2 + h*k2/2.0)
        k4 = ccsd.update_t2(F + Vt(t+h),        t1, t2 + h*k3)
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def fl1(self,hbar,Lambda,t,h,F,Vt,t1,t2,l1,l2):
        k1 = Lambda.update_l1(hbar, F + Vt(t),          t1,t2,  l1,             l2)
        k2 = Lambda.update_l1(hbar, F + Vt(t+h/2.0),    t1,t2,  l1 + h*k1/2.0,  l2)
        k3 = Lambda.update_l1(hbar, F + Vt(t+h/2.0),    t1,t2,  l1 + h*k2/2.0,  l2)
        k4 = Lambda.update_l1(hbar, F + Vt(t+h),        t1,t2,  l1 + h*k3,      l2)
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def fl2(self,hbar,Lambda,t,h,F,Vt,t1,t2,l1,l2):
        k1 = Lambda.update_l2(hbar, F + Vt(t),          t1,t2,l1,   l2)
        k2 = Lambda.update_l2(hbar, F + Vt(t+h/2.0),    t1,t2,l1,   l2 + h*k1/2.0)
        k3 = Lambda.update_l2(hbar, F + Vt(t+h/2.0),    t1,t2,l1,   l2 + h*k2/2.0)
        k4 = Lambda.update_l2(hbar, F + Vt(t+h),        t1,t2,l1,   l2 + h*k3)
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def check_trace(self,M,t):
        trace = np.trace(M).real
        if trace-t>1e-14: 
            Print(red+'Warning: Trace of density matrix deviated from expected value'+end)
            print(trace)
        return
