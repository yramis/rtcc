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
from opt_einsum import contract

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
        np.set_printoptions(precision=8,suppress=True)

        F  = ccsd.F.copy()
        t1 = ccsd.t1.copy()
        t2 = ccsd.t2.copy()
        l1 = Lambda.l1.copy()
        l2 = Lambda.l2.copy()

        t1_0 = t1.copy()
        t2_0 = t2.copy()
        l1_0 = l1.copy()
        l2_0 = l2.copy()

        # TESTS
        print('testing\n')
        print('t1 from CCSD', t1.real)
        #r_t1 = ccsd.update_t1(F,ccsd.t1,ccsd.t2)
        #print('t1 residual', r_t1.real)
        r_t1 = ccsd.residual_t1(F,ccsd.t1,ccsd.t2)
        print('t1 residual 2', r_t1.real)

        print('\nt2', t2.real)
        r_t2 = ccsd.update_t2(F,t1,t2)
        print('t2 residual', r_t2.real)

        print('\ntest update convergence')
        t1 = t1*0
        t2 = t2*0
        converged = False
        while not converged:
            r_t1 = ccsd.update_t1(F,t1,t2).real
            r_t2 = ccsd.update_t2(F,t1,t2).real
            t1 += r_t1*ccsd.Dia
            t2 += r_t2*ccsd.Dijab
            if np.linalg.norm(r_t1) < 1e-10 and np.linalg.norm(r_t2) < 1e-10:
                converged = True
        print(t1.real)
        print(t2.real)
#
#        print('\ntest imaginary time propagation T')
#        t1 = t1*0
#        t2 = t2*0
#        t = 0.0
#        h = 0.1
#        converged = False
#        while not converged:
#            r_t1 = - self.ft1(ccsd,t,h,F,self.zero,t1,t2)
#            r_t2 = - self.ft2(ccsd,t,h,F,self.zero,t1,t2)
#            t1 += r_t1
#            t2 += r_t2
#            t += h
#            if np.linalg.norm(r_t1) < 1e-10 and np.linalg.norm(r_t2) < 1e-10:
#                converged = True
#        print(t1.real)
#        print(t2.real)
#        print('energy', ccsd.compute_ccsd_energy())
#        print()
#
#        print('\nLambda')
#        hbar = CCHbar(ccsd,ccsd.F,ccsd.t1,ccsd.t2)
#        print('l1', np.transpose(l1.real))
#        r_l1 = Lambda.update_l1(hbar,t1,t2,l1,l2)
#        print('l1 residual', np.transpose(r_l1.real))
#
#        print('l2', np.transpose(l2.real))
#        r_l2 = Lambda.update_l2(hbar,t1,t2,l1,l2)
#        print('l2 residual', np.transpose(r_l2.real))
#
#        print('\ntest update convergence L')
#        l1 = l1*0
#        l2 = l2*0
#        converged = False
#        while not converged:
#            r_l1 = Lambda.update_l1(hbar,ccsd.t1,ccsd.t2,l1,l2).real
#            r_l2 = Lambda.update_l2(hbar,ccsd.t1,ccsd.t2,l1,l2).real
#            l1 += r_l1*Lambda.Dia
#            tmp = r_l2
#            tmp+= r_l2.swapaxes(0,1).swapaxes(2,3)
#            l2 += tmp*Lambda.Dijab
#            if np.linalg.norm(r_l1) < 1e-10 and np.linalg.norm(r_l2) < 1e-10:
#                converged = True
#        print(np.transpose(l1.real))
#        print(np.transpose(l2.real))
#        e = ndot('abij,ijab->',l2,ccsd.TEI[ccsd.o,ccsd.o,ccsd.v,ccsd.v],prefactor=0.5).real
#
#        print('\ntest imaginary time propagation L')
#        l1 = l1*0
#        l2 = l2*0
#        t = 0.0
#        h = 0.1
#        Lambda.hbar1 = CCHbar(ccsd,F,t1,t2)
#        Lambda.hbar2 = CCHbar(ccsd,F,t1,t2)
#        Lambda.hbar3 = CCHbar(ccsd,F,t1,t2)
#        converged = False
#        while not converged:
#            r_l1 = - self.fl1(Lambda,t,h,ccsd.t1,ccsd.t2,l1,l2)
#            r_l2 = - self.fl2(Lambda,t,h,ccsd.t1,ccsd.t2,l1,l2)
#            l1 += r_l1
#            l2 += r_l2
#            t += h
#            if np.linalg.norm(r_l1) < 1e-10 and np.linalg.norm(r_l2) < 1e-10:
#                converged = True
#        print(np.transpose(l1.real))
#        print(np.transpose(l2.real))
#        e = ndot('abij,ijab->',l2,ccsd.TEI[ccsd.o,ccsd.o,ccsd.v,ccsd.v],prefactor=0.5).real
#        print('pseudoenergy', e)
#        print()

        return
        # build the electric dipole operator (this is where the electric field orientation is set)
        ints = ccsd.mints.ao_dipole()
        self.mu = contract('ui,uv,vj->ij',ccsd.npC,np.asarray(ints[2]),ccsd.npC)
        
        D       = density.compute_ccsd_density(t1,t2,l1,l2)
        dipole  = density.compute_ccsd_dipole(D)
        D0      = D.copy()
        self.check_trace(D,ccsd.n_e)

        energy  = ccsd.compute_ccsd_energy()
        e0      = energy.copy()

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
        data['energy (real)']       = []
        data['energy (imag)']       = []

        data['time'].append(0)
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
        data['energy (real)'].append(energy.real)
        data['energy (imag)'].append(energy.imag)

        h = options['timestep']
        N = options['number of steps']
        T = options['timelength']
        t = 0.0
        Print(blue+'{:>6s}{:8.4f}'.format('t =',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('mu (Real):',dipole[2].real,'mu (Imag):',dipole[2].imag)+end)

        for i in range(N):
            #t1 = t1 + 1j*self.ft1(ccsd,t,h,F,self.Vt,t1,t2)
            #t2 = t2 + 1j*self.ft2(ccsd,t,h,F,self.Vt,t1,t2)
            t1 = t1 - 1j*ccsd.update_t1(F+self.Vt(t),t1,t2)
            t2 = t2 - 1j*ccsd.update_t2(F+self.Vt(t),t1,t2)
            #Lambda.hbar1 = CCHbar(ccsd,F+self.Vt(t)        ,t1,t2)
            #Lambda.hbar2 = CCHbar(ccsd,F+self.Vt(t+h/2.0)  ,t1,t2)
            #Lambda.hbar3 = CCHbar(ccsd,F+self.Vt(t+h)      ,t1,t2)
            #l1 = l1 - 1j*self.fl1(Lambda,t,h,t1,t2,l1,l2)
            #l2 = l2 - 1j*self.fl2(Lambda,t,h,t1,t2,l1,l2)
            t += h

            #D = density.compute_ccsd_density(t1,t2,l1,l2)
            #self.check_trace(D,ccsd.n_e)
            #dipole = density.compute_ccsd_dipole(D)
            energy = self.compute_ccsd_energy(ccsd)

            data['time'].append(t)
            data['t1 (real part)'].append(t1.real.tolist())
            data['t1 (imag part)'].append(t1.imag.tolist())
            data['t2 (real part)'].append(t2.real.tolist())
            data['t2 (imag part)'].append(t2.imag.tolist())
            #data['l1 (real part)'].append(l1.real.tolist())
            #data['l1 (imag part)'].append(l1.imag.tolist())
            #data['l2 (real part)'].append(l2.real.tolist())
            #data['l2 (imag part)'].append(l2.imag.tolist())
            #data['dipole (real part)'].append(dipole[2].real)
            #data['dipole (imag part)'].append(dipole[2].imag)
            data['energy (real)'].append(energy.real)
            data['energy (imag)'].append(energy.imag)

            #Print(blue+'{:>6s}{:8.4f}'.format('t =',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('mu (Real):',dipole[2].real,'mu (Imag):',dipole[2].imag)+end)
            Print(blue+'{:>6s}{:8.4f}'.format('t =',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('e (Real):',energy.real,'e (Imag):',energy.imag)+end)

            if time.time() > T:
                Print(yellow+'\nEnd of propagation reached: time = %s seconds' %T+end)
                break

            elif t>0 and round(t/h)%1000 == 0:                                                # write checkpoint
                init = time.time()
                with open('data.json','w') as outfile:
                    json.dump(data,outfile,indent=2)
                Print(yellow+'\n\t checkpoint: saving data to json in %.1f seconds\n' %(time.time()-init)+end)

            elif round(time.time()-time_init)>1 and round(time.time()-time_init)%3600 == 0:   # write checkpoint
                init = time.time()
                with open('data.json','w') as outfile:
                    json.dump(data,outfile,indent=2)
                Print(yellow+'\n\t checkpoint: saving data to json in %.1f seconds\n' %(time.time()-init)+end)

        Print(yellow+'\n End of propagation reached: steps = %s' %N+end)
        Print(yellow+'\t time elapsed: time = %.1f seconds' %(time.time()-time_init)+end)

        with open('data.json','w') as outfile:
            json.dump(data,outfile,indent=2)

    def Vt(self,t):
        # Field Parameters
        A = self.options['field amplitude']
        w = self.options['field frequency']

        V = -A*self.mu*np.exp(-(t-3)**2/(2*w**2))

        return V

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

    def fl1(self,Lambda,t,h,t1,t2,l1,l2):
        k1 = Lambda.update_l1(Lambda.hbar1, t1, t2, l1,             l2)
        k2 = Lambda.update_l1(Lambda.hbar2, t1, t2, l1 + h*k1/2.0,  l2)
        k3 = Lambda.update_l1(Lambda.hbar2, t1, t2, l1 + h*k2/2.0,  l2)
        k4 = Lambda.update_l1(Lambda.hbar3, t1, t2, l1 + h*k3,      l2)
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def fl2(self,Lambda,t,h,t1,t2,l1,l2):
        k1 = Lambda.update_l2(Lambda.hbar1, t1, t2, l1, l2)
        k2 = Lambda.update_l2(Lambda.hbar2, t1, t2, l1, l2 + h*k1/2.0)
        k3 = Lambda.update_l2(Lambda.hbar2, t1, t2, l1, l2 + h*k2/2.0)
        k4 = Lambda.update_l2(Lambda.hbar3, t1, t2, l1, l2 + h*k3)
        return (k1 + 2*k2 + 2*k3 + k4)*h/6.0

    def check_trace(self,M,t):
        trace = np.trace(M).real
        if trace-t>1e-14: 
            Print(red+'Warning: Trace of density matrix deviated from expected value'+end)
            print(trace)
        return

    def compute_ccsd_energy(self,ccsd):
        o = ccsd.o
        v = ccsd.v
        e = ndot('ijab,abij->',ccsd.build_tau(),ccsd.V[v,v,o,o])
        return e
    
    def zero(self,t):
        return 0
