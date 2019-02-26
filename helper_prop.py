import time
import json
import numpy as np
import cmath
import psi4
from helper_Print import Print
from helper_ndot import ndot
from helper_cc import CCEnergy
from helper_cc import CCLambda
from helper_cc import CCProperties
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
    def __init__(self,ccsd,Lambda,prop,options,memory=2):
        Print(yellow+"\nStarting time propagation with 4th order Runge Kutta...\n"+end)

        # Start timer
        time_init = time.time()

        # read CCSD data
        t1 = ccsd.t1.copy()
        t2 = ccsd.t2.copy()
        l1 = Lambda.l1.copy()
        l2 = Lambda.l2.copy()
        F  = ccsd.F.copy()
        self.F = F  # this is to be implicitly passed downstream to check for spin integration
        e_conv = psi4.core.get_option('CCENERGY','E_CONVERGENCE')
        energy = ccsd.compute_corr_energy()
        dipole = prop.compute_ccsd_dipole()

        # build the electric dipole operator (this is where the electric field orientation is set)
        ints = ccsd.mints.ao_dipole()
        self.mu = contract('ui,uv,vj->ij',ccsd.npC,np.asarray(ints[2]),ccsd.npC)
        
        # read options & prepare data output
        self.options = options
        np.set_printoptions(precision=8,linewidth=200,suppress=True)


        # TESTS
        test = False
        if test:
            print('\ntest update T convergence')
            t1 = t1*0
            t2 = t2*0
            converged = False
            counter = 0
            while not converged:
                r_t1 = ccsd.residual_t1(F,t1,t2)
                r_t2 = ccsd.residual_t2(F,t1,t2)
                t1 += r_t1*ccsd.Dia
                t2 += r_t2*ccsd.Dijab
                if np.linalg.norm(r_t1) < 1e-14 and np.linalg.norm(r_t2) < 1e-14:
                    converged = True
                counter += 1
            if np.allclose(t1-ccsd.t1,0*t1): print('t1 has converged to the CCSD result in %s steps' %counter)
            if np.allclose(t2-ccsd.t2,0*t2): print('t2 has converged to the CCSD result in %s steps' %counter)

            print('\ntest update Lambda convergence')
            l1 = l1*0
            l2 = l2*0
            converged = False
            counter = 0
            while not converged:
                r_l1 = Lambda.residual_l1(ccsd, F,t1,t2,l1,l2)
                r_l2 = Lambda.residual_l2(ccsd, F,t1,t2,l1,l2)
                l1 += r_l1*Lambda.Dia
                l2 += r_l2*Lambda.Dijab
                if np.linalg.norm(r_l1) < 1e-14 and np.linalg.norm(r_l2) < 1e-14:
                    converged = True
                counter += 1
            if np.allclose(l1-Lambda.l1,0*l1): print('l1 has converged to the CCSD result in %s steps' %counter)
            if np.allclose(l2-Lambda.l2,0*l2): print('l2 has converged to the CCSD result in %s steps' %counter)

            print('\ntest imaginary time propagation T')
            t1 = t1*0
            t2 = t2*0
            t = 0.0
            h = 0.01
            converged = False
            counter = 0
            while not converged:
                dt1, dt2 = self.prop_t(ccsd,t,h,F,self.zero,t1,t2)
                t1 += dt1 * (-1.0j)
                t2 += dt2 * (-1.0j)
                t += h
                if np.linalg.norm(dt1) < 1e-8 and np.linalg.norm(dt2) < 1e-8:
                    converged = True
                counter += 1
            if np.allclose(t1-ccsd.t1,0*t1,1e-6): 
                print('t1 has converged to the CCSD result in %s steps' %counter)
            else:
                print('after %s steps, t1 differs from the CCSD result by ' %counter, np.linalg.norm(t1-ccsd.t1))
            if np.allclose(t2-ccsd.t2,0*t2,1e-6): 
                print('t2 has converged to the CCSD result in %s steps' %counter)
            else:
                print('after %s steps, t2 differs from the CCSD result by ' %counter, np.linalg.norm(t2-ccsd.t2))

            print('\ntest imaginary time propagation Lambda')
            l1 = l1*0
            l2 = l2*0
            t = 0.0
            h = 0.01
            converged = False
            counter = 0
            while not converged:
                dl1, dl2 = self.prop_l(ccsd,Lambda, t,h,F,self.zero,t1,t2,l1,l2)
                l1 += dl1 * (1.0j)
                l2 += dl2 * (1.0j)
                t += h
                if np.linalg.norm(dl1) < 1e-8 and np.linalg.norm(dl2) < 1e-8:
                    converged = True
                counter += 1
            if np.allclose(l1-Lambda.l1,0*l1,1e-6): 
                print('l1 has converged to the CCSD result in %s steps' %counter)
            else:
                print('after %s steps, l1 differs from the CCSD result by ' %counter, np.linalg.norm(l1-Lambda.l1))
            if np.allclose(l2-Lambda.l2,0*l2,1e-6): 
                print('l2 has converged to the CCSD result in %s steps' %counter)
            else:
                print('after %s steps, l2 differs from the CCSD result by ' %counter, np.linalg.norm(l2-Lambda.l2))

            raise SystemExit


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
        Print(blue+'{:>6s}{:8.4f}'.format('t = ',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('mu (Real):',dipole[2].real,'mu (Imag):',dipole[2].imag)+end)
        #Print(blue+'{:>6s}{:8.4f}'.format('t = ',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('e (Real):',energy.real,'e (Imag):',energy.imag)+end)

        for i in range(N):
            dt1, dt2 = self.prop_t(ccsd, t,h,F,self.Vt,t1,t2)
            dl1, dl2 = self.prop_l(ccsd,Lambda, t,h,F,self.Vt,t1,t2,l1,l2)
            dt1 = np.around(dt1,decimals=-int(np.log10(e_conv)))
            dt2 = np.around(dt2,decimals=-int(np.log10(e_conv)))
            dl1 = np.around(dl1,decimals=-int(np.log10(e_conv)))
            dl2 = np.around(dl2,decimals=-int(np.log10(e_conv)))
            t1 += dt1
            t2 += dt2
            l1 += dl1
            l2 += dl2

            t += h

            energy = ccsd.compute_corr_energy(F,t1,t2)
            dipole = prop.compute_ccsd_dipole(t1,t2,l1,l2)

            if abs(energy.real) > 1000:
                Print(red+'\nThe propagation is unstable, restart with a smaller time step'+end)
                raise SystemExit

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
            data['energy (real)'].append(energy.real)
            data['energy (imag)'].append(energy.imag)

            Print(blue+'{:>6s}{:8.4f}'.format('t =',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('mu (Real):',dipole[2].real,'mu (Imag):',dipole[2].imag)+end)
            #Print(blue+'{:>6s}{:8.4f}'.format('t = ',t)+cyan+'\t{:>15s}{:10.5f}{:>15s}{:12.3E}' .format('e (Real):',energy.real,'e (Imag):',energy.imag)+end)

            if time.time() > T:
                Print(yellow+'\nEnd of propagation reached: time = %s seconds' %T+end)
                break

            elif t>0 and round(t/h)%1000 == 0:                                                # write checkpoint
                init = time.time()
                #with open('data.json','w') as outfile:
                #    json.dump(data,outfile,indent=2)
                Print(yellow+'\n\t checkpoint: saving data to json in %.1f seconds\n' %(time.time()-init)+end)

            elif round(time.time()-time_init)>1 and round(time.time()-time_init)%3600 == 0:   # write checkpoint
                init = time.time()
                #with open('data.json','w') as outfile:
                #    json.dump(data,outfile,indent=2)
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
        
        # Round-about way of checking whether input is spin-adapted
        if V.shape[0] != self.F.shape[0]:
            # Tile for alpha/beta spin
            V           = np.repeat(V,2,axis=0)
            V           = np.repeat(V,2,axis=1)
            spin_ind    = np.arange(V.shape[0], dtype=np.int) % 2
            V           *= (spin_ind.reshape(-1, 1) == spin_ind)

        return V

    
    def prop_t(self,ccsd, t,h,F,Vt,t1,t2):
        # stage 1
        self.k1_t1 = ccsd.residual_t1(F + Vt(t),        t1,                    t2                       )
        self.k1_t2 = ccsd.residual_t2(F + Vt(t),        t1,                    t2                       )
        # stage 2
        self.k2_t1 = ccsd.residual_t1(F + Vt(t+h/2.0),  t1 + h*self.k1_t1/2.0,  t2 + h*self.k1_t2/2.0   )
        self.k2_t2 = ccsd.residual_t2(F + Vt(t+h/2.0),  t1 + h*self.k1_t1/2.0,  t2 + h*self.k1_t2/2.0   )
        # stage 3
        self.k3_t1 = ccsd.residual_t1(F + Vt(t+h/2.0),  t1 + h*self.k2_t1/2.0,  t2 + h*self.k2_t2/2.0   )
        self.k3_t2 = ccsd.residual_t2(F + Vt(t+h/2.0),  t1 + h*self.k2_t1/2.0,  t2 + h*self.k2_t2/2.0   )
        # stage 4
        self.k4_t1 = ccsd.residual_t1(F + Vt(t+h),      t1 + h*self.k3_t1,      t2 + h*self.k3_t2       )
        self.k4_t2 = ccsd.residual_t2(F + Vt(t+h),      t1 + h*self.k3_t1,      t2 + h*self.k3_t2       )
        #
        dt1 = (h/6.0) * (self.k1_t1 + 2*self.k2_t1 + 2*self.k3_t1 + self.k4_t1) * (-1.0j)
        dt2 = (h/6.0) * (self.k1_t2 + 2*self.k2_t2 + 2*self.k3_t2 + self.k4_t2) * (-1.0j)
        return dt1, dt2

    def prop_l(self,ccsd,Lambda, t,h,F,Vt,t1,t2,l1,l2):
        # get stages from t prop
        k1_t1 = self.k1_t1
        k2_t1 = self.k2_t1
        k3_t1 = self.k3_t1
        k4_t1 = self.k4_t1
        k1_t2 = self.k1_t2
        k2_t2 = self.k2_t2
        k3_t2 = self.k3_t2
        k4_t2 = self.k4_t2
        # stage 1
        k1_l1 = Lambda.residual_l1(ccsd, F + Vt(t),         t1,                 t2,                 l1,                 l2              )
        k1_l2 = Lambda.residual_l2(ccsd, F + Vt(t),         t1,                 t2,                 l1,                 l2              )
        # stage 2
        k2_l1 = Lambda.residual_l1(ccsd, F + Vt(t+h/2.0),   t1 + h*k1_t1/2.0,   t2 + h*k1_t2/2.0,   l1 + h*k1_l1/2.0,   l2 + h*k1_l2/2.0)
        k2_l2 = Lambda.residual_l2(ccsd, F + Vt(t+h/2.0),   t1 + h*k1_t1/2.0,   t2 + h*k1_t2/2.0,   l1 + h*k1_l1/2.0,   l2 + h*k1_l2/2.0)
        # stage 3
        k3_l1 = Lambda.residual_l1(ccsd, F + Vt(t+h/2.0),   t1 + h*k2_t1/2.0,   t2 + h*k2_t2/2.0,   l1 + h*k2_l1/2.0,   l2 + h*k2_l2/2.0)
        k3_l2 = Lambda.residual_l2(ccsd, F + Vt(t+h/2.0),   t1 + h*k2_t1/2.0,   t2 + h*k2_t2/2.0,   l1 + h*k2_l1/2.0,   l2 + h*k2_l2/2.0)
        # stage 4
        k4_l1 = Lambda.residual_l1(ccsd, F + Vt(t+h),       t1 + h*k3_t1,       t2 + h*k3_t2,       l1 + h*k3_l1,       l2 + h*k3_l2    )
        k4_l2 = Lambda.residual_l2(ccsd, F + Vt(t+h),       t1 + h*k3_t1,       t2 + h*k3_t2,       l1 + h*k3_l1,       l2 + h*k3_l2    )
        #
        dl1 = (h/6.0) * (k1_l1 + 2*k2_l1 + 2*k3_l1 + k4_l1) * (1.0j)
        dl2 = (h/6.0) * (k1_l2 + 2*k2_l2 + 2*k3_l2 + k4_l2) * (1.0j)
        return dl1, dl2

    def check_trace(self,M,t):
        trace = np.trace(M).real
        if trace-t>1e-14: 
            Print(red+'Warning: Trace of density matrix deviated from expected value'+end)
            print(trace)
        return

    def zero(self,t):
        return 0
