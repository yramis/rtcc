import time,sys
import numpy as np
import cmath
from opt_einsum import contract
from helper_Print import Print
from helper_ndot import ndot
from helper_diis import helper_diis
from helper_local import localize_occupied
import psi4

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

class CCEnergy(object):
    def __init__(self,mol,memory=2):
        Print(yellow+"\nInitializing CCSD object..."+end)

##------------------------------------------------------
##  SCF Procedures
##------------------------------------------------------
##  E_X :   Total energy of method X (e.g., E_ccsd = ccsd total energy)
##  e_x :   Method X specific energy (e.g., e_ccsd = ccsd correlation energy)
##
##  P   :   AO Density matrix
##  F_ao:   AO fock matrix
##
##  F   :   Canonical MO fock matrix

        # Start timer
        time_init = time.time()

        # Read molecule data
        psi4.core.set_active_molecule(mol)
        N_atom  = mol.natom()
        N_e     = int(sum(mol.Z(A) for A in range(N_atom))-mol.molecular_charge())
        self.n_occ   = int(N_e / 2) # can also be read as self.wfn.doccpi()[0] after an scf instance

        self.e_nuc = mol.nuclear_repulsion_energy()


        self.e_scf,self.wfn = psi4.energy('scf',return_wfn=True)    # This makes psi4 run the scf calculation
        Print(blue+'The SCF energy is'+end)
        Print(cyan+'\t%s\n'%self.e_scf+end)
        
        self.memory = memory
        self.nmo    = self.wfn.nmo()
        self.n_virt = self.nmo - self.n_occ
        # Make slices
        self.o = slice(self.n_occ)
        self.v = slice(self.n_occ,self.nmo)

        # Read SCF data
        self.mints  = psi4.core.MintsHelper(self.wfn.basisset())
        self.TEI_ao = np.asarray(self.mints.ao_eri())
        self.S_ao   = np.asarray(self.mints.ao_overlap())
        self.pot    = np.asarray(self.mints.ao_potential())
        self.kin    = np.asarray(self.mints.ao_kinetic())
        self.H      = self.pot + self.kin

        self.C      = self.wfn.Ca_subset('AO','ALL')
        self.npC    = np.asarray(self.C)
        #localize_occupied(self)

        # Build AO density and fock matrix
        self.P = self.build_P()
        self.F = self.build_F()

        # check scf energy matches psi4 result
        e_scf_plugin = ndot('vu,uv->',self.P,self.H+self.F,prefactor=0.5)
        if not abs(e_scf_plugin+self.e_nuc-self.e_scf)<1e-7:
            Print(red+"Warning! There is a mismatch in the scf energy")
            Print("\tthis could be due to Density-Fitting - switch SCF type to direct or PK"+end)
            Print("the psi4 scf energy is %s" %self.e_scf)
            Print("the plugin scf energy is %s" %(e_scf_plugin+self.e_nuc))
            raise Exception

        # Transform to MO basis
        Print(yellow+"\n..Starting AO -> MO transformation..."+end)

        ERI_size = self.nmo * 128e-9
        memory_footPrint = ERI_size * 5
        if memory_footPrint > self.memory:
            psi.clean()
            Print(red+"Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB."
                                                % (memory_footPrint, self.memory)+end)
            raise Exception

        self.F_ao   = self.F.copy()
        self.F      = contract('up,uv,vq->pq',self.npC,self.F,self.npC)
        self.TEI    = np.asarray(self.mints.mo_eri(self.C,self.C,self.C,self.C))
        self.TEI    = self.TEI.swapaxes(1,2)                # change indexing i,a,j,b -> i,j,a,b
        # Two Electron Integrals are stored as (left out,right out | left in,right in)
        Print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_size, self.nmo))

        # Antisymmetrize the TEI
        tmp         = self.TEI.copy()
        self.VS     = tmp
        self.VS    -= tmp.swapaxes(2,3)
        self.V      = self.VS + self.TEI

        # Build denominators
        eps         = np.diag(self.F)
        self.Dia    = 1/(eps[self.o].reshape(-1,1) - eps[self.v])
        self.Dijab  = 1/(eps[self.o].reshape(-1,1,1,1) + eps[self.o].reshape(-1,1,1) - eps[self.v].reshape(-1,1) - eps[self.v])

        # Build MBPT(2) initial guess
        Print(yellow+"\n..Building CCSD initial guess from MBPT(2) amplitudes...")

        self.t1 = np.zeros((self.n_occ,self.n_virt))                    # t1 (ia)   <- 0
        self.t2 = self.TEI[self.o,self.o,self.v,self.v] * self.Dijab    # t2 (iJaB) <- (ia|JB) * D(iJaB)
        self.t2_aa = self.t2 - self.t2.swapaxes(0,1)

        Print(yellow+"\n..Initialized CCSD in %.3f seconds." %(time.time() - time_init)+end)

    def build_P(self):  # Build AO density
        o = self.o
        C = self.npC
        P = ndot('ui,vi->uv',C[:,o],C[:,o],prefactor=2)
        return P

    def build_F(self):  # Build AO fock matrix
        F   = self.H.copy()
        F  += ndot('ls,uvsl->uv',self.P,self.TEI_ao)
        F  -= ndot('ls,ulsv->uv',self.P,self.TEI_ao,prefactor=0.5)
        return F

##------------------------------------------------------
##  CCSD Procedures
##------------------------------------------------------
##  Notation used in this code for the coupled cluster equations
##
##  i,j -> target occupied indeces
##  a,b -> target virtual indeces
##
##  k,l -> implicit (summed-over) occupied indeces,   target index for intermediates
##  c,d -> implicit (summed-over) virtual indeces,    target index for intermediates
##
##  m,l -> implicit occupied indeces (in intermediates)
##  e,f -> implicit occupied indeces (in intermediates)
##
##
##  F,G: effective 1 particle intermediates
##  W,Tau: effective 2 particle intermediates
##      Fx (ex: F1, F2) x refers to the number of tilde
##
##  In tensors, indeces are read from left to right, and bottom to top. <ab||ij> -> V(i,j,a,b)
##      I apologize for the inconvenience that will inevitably ensue, I just like it better this way
##  As much as possible, TEI indeces are reorganized so that the virtual ones are flushed right, for contraction efficiency
##
##  t2      -> alpha/beta slice of T2 array     -> t2 (iJaB)
##  t2_aa   -> alpha/alpha slice of T2 array    -> t2_aa (ijab) = t2 (iJaB) - t2 (iJbA) [RHF case]
##
##  All equations are spin integrated

    # Compute the effective two-particle excitation operators tau and tau tilde
    # Tau is used in the T2 amplitude equations and in the 2 particle intermediates W
    # Tau tilde is used in the 1 particle intermediates F

    def build_tau(self):
        tau = self.t2.copy()
        tau = tau + ndot('ia,jb->ijab',self.t1,self.t1)
        return tau      # Tau (alpha/beta)

    def build_tau_tilde(self):
        tau = self.t2.copy()
        tau = tau + ndot('ia,jb->ijab',self.t1,self.t1,prefactor=0.5)
        return tau  # Tau tilde (alpha/beta)


    def build_Foo(self):
        o = self.o
        v = self.v
        
        Foo  = self.F[o,o].copy()
        Foo += ndot('ke,ei->ik',self.t1,self.F[v,o],prefactor=0.5)
        Foo += ndot('me,kmie->ik',self.t1,self.V[o,o,o,v])
        Foo += ndot('imef,kmef->ik',self.build_tau_tilde(),self.TEI[o,o,v,v],prefactor=2)
        Foo += ndot('imef,kmfe->ik',self.build_tau_tilde(),self.TEI[o,o,v,v],prefactor=-1)
        return Foo      # Fmi

    def build_Fvv(self):
        o = self.o
        v = self.v

        Fvv  = self.F[v,v].copy()
        Fvv -= ndot('mc,am->ca',self.t1,self.F[v,o],prefactor=0.5)
        Fvv += ndot('me,maec->ca',self.t1,self.V[o,v,v,v])
        Fvv -= ndot('mnae,mnce->ca',self.build_tau_tilde(),self.TEI[o,o,v,v],prefactor=2)
        Fvv -= ndot('mnae,mnec->ca',self.build_tau_tilde(),self.TEI[o,o,v,v],prefactor=-1)
        return Fvv      # Fae

    def build_Fvo(self):
        o = self.o
        v = self.v

        Fvo  = self.F[v,o].copy()
        Fvo += ndot('me,kmce->ck',self.t1,self.V[o,o,v,v])
        return Fvo      # Fme


    def build_Woooo(self):
        o = self.o
        v = self.v

        W  = self.TEI[o,o,o,o].copy()   # V(ijkl)
        W  = W + ndot('ijef,klef->ijkl',self.build_tau(),self.TEI[o,o,v,v])
        
        tmp = ndot('je,klie->ijkl',self.t1,self.TEI[o,o,o,v])
        W += tmp + tmp.swapaxes(0,1).swapaxes(2,3)
        return W           # Wmnij

    def build_Z(self):      # Computes parts of Tau*Wvvvv to avoid computing Wvvvv
        o = self.o
        v = self.v

        Z = ndot('ijcd,ladc->ijal',self.build_tau(),self.TEI[o,v,v,v])
        return Z

    def build_Wvoov(self,string):
        o = self.o
        v = self.v
        tmp = 0.5*self.t2_aa + ndot('je,mb->jmeb',self.t1,self.t1)
        if string == 'ab':
            W  = self.TEI[v,o,o,v].copy()    # V(cjkb)
            W  = W + ndot('mjeb,kmce->cjkb',self.t2,self.VS[o,o,v,v],prefactor=0.5)
            W  = W + ndot('je,kbce->cjkb',self.t1,self.TEI[o,v,v,v])
            W -= ndot('mb,mkjc->cjkb',self.t1,self.TEI[o,o,o,v])
            W -= ndot('jmeb,kmce->cjkb',tmp,self.TEI[o,o,v,v])
            return W
        elif string == 'aa':
            W  = self.VS[v,o,o,v].copy()
            W  = W + ndot('mjeb,kmce->cjkb',self.t2,self.TEI[o,o,v,v],prefactor=0.5)
            W  = W + ndot('je,kbce->cjkb',self.t1,self.VS[o,v,v,v])
            W -= ndot('mb,mkjc->cjkb',self.t1,self.VS[o,o,o,v])
            W -= ndot('jmeb,kmce->cjkb',tmp,self.VS[o,o,v,v])
            return W        # Wmbej
        else:
            Print(red+"build_Wvoov: string %s must be ab or aa." %string+end)
            raise Exception

    def build_Wvovo(self):
        o = self.o
        v = self.v

        W  = self.TEI[v,o,v,o].copy()       # V(cjak)
        W  = W + ndot('je,kaec->cjak',self.t1,self.TEI[o,v,v,v])
        W -= ndot('ma,kmjc->cjak',self.t1,self.TEI[o,o,o,v])
        tmp = 0.5*self.t2 + ndot('ma,je->mjae',self.t1,self.t1)
        W -= ndot('mjae,mkce->cjak',tmp,self.TEI[o,o,v,v])
        return W        # Wmbje

    # Update amplitudes
    def update(self):
        o = self.o
        v = self.v

        Foo = self.build_Foo()
        Fvv = self.build_Fvv()
        Fvo = self.build_Fvo()

        t1_new  = self.F[o,v].copy()
        t1_new += ndot('ic,ca->ia',self.t1,Fvv)
        t1_new -= ndot('ka,ik->ia',self.t1,Foo)
        TS      = self.t2 + self.t2_aa
        t1_new += ndot('ikac,ck->ia',TS,Fvo)

        t1_new += ndot('ld,ladi->ia',self.t1,self.TEI[o,v,v,o],prefactor=2)
        t1_new += ndot('ld,laid->ia',self.t1,self.TEI[o,v,o,v],prefactor=-1)

        t1_new += ndot('ikcd,kadc->ia',self.t2,self.TEI[o,v,v,v],prefactor=2)
        t1_new += ndot('ikdc,kadc->ia',self.t2,self.TEI[o,v,v,v],prefactor=-1)

        t1_new -= ndot('klac,klic->ia',self.t2_aa,self.VS[o,o,o,v],prefactor=0.5)
        t1_new -= ndot('klac,klic->ia',self.t2,self.TEI[o,o,o,v])
        
        tau     = self.build_tau()
        tau1    = self.build_tau_tilde()
        Woooo   = self.build_Woooo()
        Z       = self.build_Z()
        Wvoov_ab= self.build_Wvoov('ab')
        Wvoov_aa= self.build_Wvoov('aa')
        Wvovo   = self.build_Wvovo()

        t2_new  = self.TEI[o,o,v,v].copy()

        t2_new += ndot('klab,ijkl->ijab',tau,Woooo)

        t2_new += ndot('ijcd,cdab->ijab',tau,self.TEI[v,v,v,v])
        t2_new -= ndot('lb,ijal->ijab',self.t1,Z)
        t2_new -= ndot('lb,ijal->jiba',self.t1,Z)

        tmp     = ndot('ikac,cjkb->ijab',self.t2,Wvoov_aa)
        tmp    += ndot('ikac,cjkb->ijab',self.t2_aa,Wvoov_ab)
        tmp    -= ndot('ikcb,cjak->ijab',self.t2,Wvovo)

        tmp    += ndot('ic,jcba->ijab',self.t1,self.TEI[o,v,v,v])

        tmp    -= ndot('ka,ijkb->ijab',self.t1,self.TEI[o,o,o,v])
        tmp1    = Fvv-0.5*ndot('kb,ck->cb',self.t1,Fvo)
        tmp    += ndot('ijac,cb->ijab',self.t2,tmp1)
        tmp1    = Foo+0.5*ndot('jc,ck->jk',self.t1,Fvo)
        tmp    -= ndot('ikab,jk->ijab',self.t2,tmp1)
        t2_new += tmp + tmp.swapaxes(0,1).swapaxes(2,3)

        t2_new -= contract('ic,kb,jcka->ijab',self.t1,self.t1,self.TEI[o,v,o,v])
        t2_new -= contract('ic,ka,cjkb->ijab',self.t1,self.t1,self.TEI[v,o,o,v])
        t2_new -= contract('jc,ka,ickb->ijab',self.t1,self.t1,self.TEI[o,v,o,v])
        t2_new -= contract('jc,kb,cika->ijab',self.t1,self.t1,self.TEI[v,o,o,v])

        self.t1 += t1_new*self.Dia
        self.t2 += t2_new*self.Dijab

        return

    # Split amplitude update into T1 and T2 functions (used for the time-propagation)
    def update_t1(self,t1,t2,F):
        o = self.o
        v = self.v

        # Save current t1,t2,F
        t1_saved = self.t1.copy()
        t2_saved = self.t2.copy()
        F_saved  = self.F.copy()

        # overwrite with input
        self.t1 = t1.copy()
        self.t2 = t2.copy()
        self.F  = F.copy()
        self.t2_aa = t2 - t2.swapaxes(2,3)

        # Compute 1 particle intermediates
        Foo = self.build_Foo()
        Fvv = self.build_Fvv()
        Fvo = self.build_Fvo()

        # Solve T1 equation
        t1_new  = self.F[o,v].copy()
        t1_new += ndot('ic,ca->ia',self.t1,Fvv)
        t1_new -= ndot('ka,ik->ia',self.t1,Foo)
        TS      = self.t2 + self.t2_aa
        t1_new += ndot('ikac,ck->ia',TS,Fvo)

        t1_new += ndot('ld,ladi->ia',self.t1,self.TEI[o,v,v,o],prefactor=2)
        t1_new += ndot('ld,laid->ia',self.t1,self.TEI[o,v,o,v],prefactor=-1)

        t1_new += ndot('ikcd,kadc->ia',self.t2,self.TEI[o,v,v,v],prefactor=2)
        t1_new += ndot('ikdc,kadc->ia',self.t2,self.TEI[o,v,v,v],prefactor=-1)

        t1_new -= ndot('klac,klic->ia',self.t2_aa,self.VS[o,o,o,v],prefactor=0.5)
        t1_new -= ndot('klac,klic->ia',self.t2,self.TEI[o,o,o,v])
        
        t1 = self.t1 + t1_new*self.Dia

        # restore saved values
        #self.t1 = t1_saved.copy()
        #self.t2 = t2_saved.copy()
        #self.F  = F_saved.copy()
        #del t1_saved
        #del t2_saved
        #del F_saved

        return t1

    def update_t2(self,t1,t2,F):
        o = self.o
        v = self.v

        # Save current t1,t2,F
        t1_saved = self.t1.copy()
        t2_saved = self.t2.copy()
        F_saved  = self.F.copy()

        # overwrite with input
        self.t1 = t1.copy()
        self.t2 = t2.copy()
        self.F  = F.copy()
        self.t2_aa = t2 - t2.swapaxes(2,3)

        # Compute 1 particle intermediates
        Foo = self.build_Foo()
        Fvv = self.build_Fvv()
        Fvo = self.build_Fvo()

        # Compute 2 particle intermediates
        tau     = self.build_tau()
        tau1    = self.build_tau_tilde()
        Woooo   = self.build_Woooo()
        Z       = self.build_Z()
        Wvoov_ab= self.build_Wvoov('ab')
        Wvoov_aa= self.build_Wvoov('aa')
        Wvovo   = self.build_Wvovo()

        # Solve T2 equation
        t2_new  = self.TEI[o,o,v,v].copy()
        t2_new  = t2_new + ndot('klab,ijkl->ijab',tau,Woooo)
        t2_new += ndot('ijcd,cdab->ijab',tau,self.TEI[v,v,v,v])
        t2_new -= ndot('lb,ijal->ijab',self.t1,Z)
        t2_new -= ndot('lb,ijal->jiba',self.t1,Z)

        tmp     = ndot('ikac,cjkb->ijab',self.t2,Wvoov_aa)
        tmp    += ndot('ikac,cjkb->ijab',self.t2_aa,Wvoov_ab)
        tmp    -= ndot('ikcb,cjak->ijab',self.t2,Wvovo)
        tmp    += ndot('ic,jcba->ijab',self.t1,self.TEI[o,v,v,v])
        tmp    -= ndot('ka,ijkb->ijab',self.t1,self.TEI[o,o,o,v])
        tmp1    = Fvv-0.5*ndot('kb,ck->cb',self.t1,Fvo)
        tmp    += ndot('ijac,cb->ijab',self.t2,tmp1)
        tmp1    = Foo+0.5*ndot('jc,ck->jk',self.t1,Fvo)
        tmp    -= ndot('ikab,jk->ijab',self.t2,tmp1)
        t2_new += tmp + tmp.swapaxes(0,1).swapaxes(2,3)

        t2_new -= contract('ic,kb,jcka->ijab',self.t1,self.t1,self.TEI[o,v,o,v])
        t2_new -= contract('ic,ka,cjkb->ijab',self.t1,self.t1,self.TEI[v,o,o,v])
        t2_new -= contract('jc,ka,ickb->ijab',self.t1,self.t1,self.TEI[o,v,o,v])
        t2_new -= contract('jc,kb,cika->ijab',self.t1,self.t1,self.TEI[v,o,o,v])

        t2 = self.t2 + t2_new*self.Dijab

        # restore saved values
        #self.t1 = t1_saved.copy()
        #self.t2 = t2_saved.copy()
        #self.F  = F_saved.copy()
        #del t1_saved
        #del t2_saved
        #del F_saved

        return t2

    def compute_ccsd_energy(self):
        o = self.o
        v = self.v
        e = ndot('ijab,abij->',self.build_tau(),self.V[v,v,o,o])
        self.e_ccsd = e
        self.E_ccsd = e + self.e_scf
        return e
    
    def compute_ccsd(self,maxiter=50,max_diis=8,start_diis=1):
        ccsd_tstart = time.time()
        
        self.e_mp2 = self.compute_ccsd_energy()

        Print('\n\t  Summary of iterative solution of the CC equations')
        Print('\t------------------------------------------------------')
        Print('\t\t\tCorrelation\t      RMS')
        Print('\t Iteration\tEnergy\t\t     error')
        Print('\t------------------------------------------------------')
        Print('\t{:4d}{:26.15f}{:>25s}' .format(0,self.e_ccsd,'MBPT(2)'))

        e_conv = psi4.core.get_option('CCENERGY','E_CONVERGENCE')

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.t1, self.t2, max_diis)

        # Iterate
        for iter in range(1,maxiter+1):
            t1_old = self.t1.copy()
            t2_old = self.t2.copy()
            e_old = self.e_ccsd

            self.update()
            e_ccsd = self.compute_ccsd_energy()
            rms = e_ccsd - e_old
            Print('\t{:4d}{:26.15f}{:15.5E}   DIIS={:d}' .format(iter,e_ccsd,rms,diis_object.diis_size))

            # Check convergence
            if abs(rms)<e_conv:
                Print('\t------------------------------------------------------')

                Print(yellow+"\n..The CCSD equations have converged in %.3f seconds" %(time.time()-ccsd_tstart)+end)
                Print(blue+'The ccsd correlation energy is'+end)
                Print(cyan+'\t%s \n' %e_ccsd+end)

                return

            #  Add the new error vector
            diis_object.add_error_vector(self.t1, self.t2)

            if iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1, self.t2)

            self.t2_aa = self.t2 - self.t2.swapaxes(0,1)
            # End CCSD iterations
    # End CCEnergy class


class CCHbar(object):
    def __init__(self,ccsd,F=[],t1=[],t2=[],memory=2):  # ccsd input must be ccsd = CCEnergy(mol, memory=x)
        if F==[]:
            Print(yellow+"\nInitializing Hbar object..."+end)

        # Start timer
        time_init = time.time()

        # Read relevant data from ccsd class
        self.n_occ  = ccsd.n_occ
        self.n_virt = ccsd.n_virt
        self.o      = ccsd.o
        self.v      = ccsd.v

        self.TEI    = ccsd.TEI
        if F==[]:
            self.F  = ccsd.F
        else:
            self.F  = F
        if t1==[]:
            self.t1 = ccsd.t1
        else:
            self.t1 = t1
        if t2==[]:
            self.t2 = ccsd.t2
        else:
            self.t2 = t2
        self.t2_aa  = self.t2 - self.t2.swapaxes(2,3)

        # Antisymmetrize the TEI
        tmp         = self.TEI.copy()
        self.VS     = tmp
        self.VS    -= tmp.swapaxes(2,3)
        self.V      = self.VS + self.TEI

        # Build persistent intermediates
        self.tau        = ccsd.build_tau()

        self.build_Hoo()
        self.build_Hvv()
        self.build_Hvo()

        # 0
        self.build_Hoooo()
        self.build_Hvvvv()
        self.build_Hovov()
        self.build_Hovvo()
        # -1
        self.build_Hovoo()
        self.build_Hvvvo()
        # +1 
        self.build_Hooov()
        self.build_Hvovv()

        if F==[]:
            Print(yellow+"\n..Hbar built in %.3f seconds\n" %(time.time()-time_init)+end)

    # 1-body Hbar
    def build_Hoo(self):
        o = self.o
        v = self.v

        self.Hoo  = self.F[o,o].copy()
        self.Hoo  = self.Hoo +  ndot('ie,ej->ij',self.t1,self.F[v,o])
        self.Hoo += ndot('me,jmie->ij',self.t1,self.V[o,o,o,v])
        self.Hoo += ndot('imef,jmef->ij',self.tau,self.V[o,o,v,v])
        return self.Hoo

    def build_Hvv(self):
        o = self.o
        v = self.v

        self.Hvv  = self.F[v,v].copy()
        self.Hvv  = self.Hvv - ndot('mb,am->ab',self.t1,self.F[v,o])
        self.Hvv += ndot('me,mbea->ab',self.t1,self.V[o,v,v,v])
        self.Hvv -= ndot('mnbe,mnae->ab',self.tau,self.V[o,o,v,v])
        return self.Hvv

    def build_Hvo(self):
        o = self.o
        v = self.v

        self.Hvo  = self.F[v,o].copy()
        self.Hvo  = self.Hvo + ndot('me,imae->ai',self.t1,self.V[o,o,v,v])
        return self.Hvo

    # 2-body hbar
    # 0 excitation rank
    def build_Hoooo(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hoooo  = self.TEI[o,o,o,o].copy()
        self.Hoooo  = self.Hoooo + ndot('je,klie->ijkl',self.t1,self.TEI[o,o,o,v],prefactor=2)
        self.Hoooo += ndot('ijef,klef->ijkl',self.tau,self.TEI[o,o,v,v])
        return self.Hoooo

    def build_Hvvvv(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hvvvv  = self.TEI[v,v,v,v].copy()
        self.Hvvvv  = self.Hvvvv - ndot('md,mcba->abcd',self.t1,self.TEI[o,v,v,v],prefactor=2)
        self.Hvvvv += ndot('mncd,mnab->abcd',self.tau,self.TEI[o,o,v,v])
        return self.Hvvvv

    def build_Hovov(self):
        o = self.o
        v = self.v

        self.Hovov  = self.TEI[o,v,o,v].copy()
        self.Hovov  = self.Hovov - ndot('mb,jmia->iajb',self.t1,self.TEI[o,o,o,v])
        self.Hovov += ndot('ie,jbea->iajb',self.t1,self.TEI[o,v,v,v]) 
        self.Hovov -= ndot('imeb,jmea->iajb',self.tau,self.TEI[o,o,v,v])
        return self.Hovov

    def build_Hovvo(self):  # alpha/beta/alpha/beta # exchange
        o = self.o
        v = self.v

        self.Hovvo  = self.TEI[o,v,v,o].copy()
        self.Hovvo  = self.Hovvo - ndot('ma,mjib->ibaj',self.t1,self.TEI[o,o,o,v])
        self.Hovvo += ndot('ie,jabe->ibaj',self.t1,self.TEI[o,v,v,v])
        self.Hovvo -= ndot('imea,mjeb->ibaj',self.tau,self.TEI[o,o,v,v])
        self.Hovvo += ndot('imae,mjeb->ibaj',self.t2,self.V[o,o,v,v])
        return self.Hovvo

    # -1 excitation rank
    def build_Hovoo(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hovoo  = self.TEI[o,v,o,o].copy()
        self.Hovoo  = self.Hovoo + ndot('ke,ijea->kaij',self.t1,self.TEI[o,o,v,v])
        return self.Hovoo

    def build_Hvvvo(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hvvvo  = self.TEI[v,v,v,o].copy()
        self.Hvvvo  = self.Hvvvo - ndot('mc,miab->abci',self.t1,self.TEI[o,o,v,v])
        return self.Hvvvo

    # +1 excitation rank
    def build_Hooov(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hooov  = self.TEI[o,o,o,v].copy()
        self.Hooov  = self.Hooov - ndot('ma,ijkm->ijka',self.t1,self.TEI[o,o,o,o])
        self.Hooov += ndot('je,ieka->ijka',self.t1,self.TEI[o,v,o,v])
        self.Hooov += ndot('ie,ejka->ijka',self.t1,self.TEI[v,o,o,v])

        self.Hooov += ndot('ijea,ek->ijka',self.t2,self.F[v,o])

        self.Hooov += ndot('ijef,kaef->ijka',self.tau,self.TEI[o,v,v,v])
        self.Hooov -= ndot('imea,mkje->ijka',self.tau,self.TEI[o,o,o,v])
        self.Hooov -= ndot('jmea,kmie->ijka',self.tau,self.TEI[o,o,o,v])
        self.Hooov += ndot('jmae,kmie->ijka',self.t2,self.V[o,o,o,v])

        tmp         = ndot('mf,kmef->ek',self.t1,self.V[o,o,v,v])
        self.Hooov += ndot('ijea,ek->ijka',self.t2,tmp)

        tmp         = ndot('ijef,kmef->ijkm',self.tau,self.TEI[o,o,v,v])
        self.Hooov -= ndot('ijkm,ma->ijka',tmp,self.t1)

        tmp         = ndot('ie,kmef->ifkm',self.t1,self.V[o,o,v,v])
        self.Hooov += ndot('jmaf,ifkm->ijka',self.t2,tmp)
        tmp         = ndot('ie,kmef->ifkm',self.t1,self.TEI[o,o,v,v])
        self.Hooov -= ndot('jmfa,ifkm->ijka',self.t2,tmp)
        tmp         = ndot('jf,kmef->ejkm',self.t1,self.TEI[o,o,v,v])
        self.Hooov -= ndot('imea,ejkm->ijka',self.t2,tmp)
        return self.Hooov

    def build_Hvovv(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hvovv  = self.TEI[v,o,v,v].copy()
        self.Hvovv  = self.Hvovv + ndot('ie,ceab->ciab',self.t1,self.TEI[v,v,v,v])
        self.Hvovv -= ndot('mb,ciam->ciab',self.t1,self.TEI[v,o,v,o])
        self.Hvovv -= ndot('ma,cimb->ciab',self.t1,self.TEI[v,o,o,v])

        self.Hvovv -= ndot('miab,cm->ciab',self.t2,self.F[v,o])

        self.Hvovv += ndot('mnab,nmic->ciab',self.tau,self.TEI[o,o,o,v])
        self.Hvovv -= ndot('miae,mbce->ciab',self.tau,self.TEI[o,v,v,v])
        self.Hvovv -= ndot('mibe,maec->ciab',self.tau,self.TEI[o,v,v,v])
        self.Hvovv += ndot('mieb,maec->ciab',self.t2,self.V[o,v,v,v])

        tmp         = ndot('ne,mnce->cm',self.t1,self.V[o,o,v,v])
        self.Hvovv -= ndot('miab,cm->ciab',self.t2,tmp)

        tmp         = ndot('mnab,mnce->ceab',self.tau,self.TEI[o,o,v,v])
        self.Hvovv += ndot('ie,ceab->ciab',self.t1,tmp)

        tmp         = ndot('ma,mnce->cean',self.t1,self.V[o,o,v,v])
        self.Hvovv -= ndot('inbe,cean->ciab',self.t2,tmp)
        tmp         = ndot('ma,mnce->cean',self.t1,self.TEI[o,o,v,v])
        self.Hvovv += ndot('ineb,cean->ciab',self.t2,tmp)
        tmp         = ndot('nb,mnce->cemb',self.t1,self.TEI[o,o,v,v])
        self.Hvovv += ndot('imea,cemb->ciab',self.t2,tmp)
        return self.Hvovv



class CCLambda(object):
    def __init__(self,ccsd,hbar):
        Print(yellow+"\nInitializing Lambda object..."+end)

        # Start timer
        time_init = time.time()

        # Read relevant data from ccsd class
        self.n_occ  = ccsd.n_occ
        self.n_virt = ccsd.n_virt
        self.o      = ccsd.o
        self.v      = ccsd.v

        self.TEI    = ccsd.TEI
        #self.F      = ccsd.F
        self.Dia    = ccsd.Dia.swapaxes(0,1)
        self.Dijab  = ccsd.Dijab.swapaxes(0,2).swapaxes(1,3)
        self.t1     = ccsd.t1
        self.t2     = ccsd.t2

        # Antisymmetrize the TEI
        tmp         = self.TEI.copy()
        self.VS     = tmp
        self.VS    -= tmp.swapaxes(2,3)
        self.V      = self.VS + self.TEI

        self.tau    = hbar.tau
        self.Hoo    = hbar.Hoo
        self.Hvv    = hbar.Hvv
        self.Hvo    = hbar.Hvo

        # 0
        self.Hoooo  = hbar.Hoooo
        self.Hvvvv  = hbar.Hvvvv
        self.Hovov  = hbar.Hovov
        self.Hovvo  = hbar.Hovvo
        # -1
        self.Hovoo  = hbar.Hovoo
        self.Hvvvo  = hbar.Hvvvo
        # +1 
        self.Hooov  = hbar.Hooov
        self.Hvovv  = hbar.Hvovv

        self.l1     = self.t1.swapaxes(0,1).copy()
        self.l1    *= 2.0
        tmp         = self.t2.copy()
        self.l2     = 2.0*(2.0*tmp - tmp.swapaxes(2,3))
        self.l2     = self.l2.swapaxes(0,2).swapaxes(1,3)

    def build_Goo(self):
        Goo = - ndot('mjef,efmi->ij',self.t2,self.l2)
        return Goo

    def build_Gvv(self):
        Gvv = ndot('mnea,ebmn->ab',self.t2,self.l2)
        return Gvv

    def update(self):
        o = self.o
        v = self.v

        Goo = self.build_Goo()
        Gvv = self.build_Gvv()

        l1_new  = 2*self.Hvo.copy()

        l1_new += ndot('ei,ae->ai',self.l1,self.Hvv)
        l1_new -= ndot('am,mi->ai',self.l1,self.Hoo)

        l1_new += ndot('em,maei->ai',self.l1,self.Hovvo,prefactor=2)
        l1_new -= ndot('em,maie->ai',self.l1,self.Hovov)

        l1_new += ndot('efim,amef->ai',self.l2,self.Hvovv)
        l1_new -= ndot('aemn,mnie->ai',self.l2,self.Hooov)

        l1_new += ndot('ef,eafi->ai',Gvv,self.Hvvvo,prefactor=2)
        l1_new -= ndot('ef,aefi->ai',Gvv,self.Hvvvo)
        l1_new += ndot('mn,mani->ai',Goo,self.Hovoo,prefactor=2)
        l1_new -= ndot('mn,main->ai',Goo,self.Hovoo)


        l2_new  = self.V[v,v,o,o].copy()

        l2_new  = l2_new + ndot('ai,bj->abij',self.l1,self.Hvo,prefactor=2)
        l2_new -= ndot('aj,bi->abij',self.l1,self.Hvo)

        l2_new -= ndot('bm,maji->abij',self.l1,self.Hovoo,prefactor=2)    # check sym
        l2_new += ndot('bm,maij->abij',self.l1,self.Hovoo)
        l2_new += ndot('ei,abej->abij',self.l1,self.Hvvvo,prefactor=2)
        l2_new -= ndot('ei,baej->abij',self.l1,self.Hvvvo)

        l2_new -= ndot('abim,mj->abij',self.l2,self.Hoo)
        l2_new += ndot('aeij,be->abij',self.l2,self.Hvv)

        l2_new += ndot('abmn,mnij->abij',self.l2,self.Hoooo,prefactor=0.5)
        l2_new += ndot('efij,abef->abij',self.l2,self.Hvvvv,prefactor=0.5)

        l2_new += ndot('ebmj,maei->abij',self.l2,self.Hovvo,prefactor=2)
        l2_new -= ndot('ebmj,maie->abij',self.l2,self.Hovov)
        l2_new -= ndot('ebmi,maej->abij',self.l2,self.Hovvo)
        l2_new -= ndot('ebim,maje->abij',self.l2,self.Hovov)

        l2_new -= ndot('ea,ijeb->abij',Gvv,self.V[o,o,v,v])
        l2_new += ndot('im,mjab->abij',Goo,self.V[o,o,v,v])

        self.l1 += l1_new*self.Dia

        tmp     = l2_new
        tmp     += l2_new.swapaxes(0,1).swapaxes(2,3)
        self.l2 += tmp*self.Dijab

        return

    # Split amplitude update into T1 and T2 functions (used for the time-propagation)
    def update_l1(self,hbar,t1,t2,l1,l2,F):
        o = self.o
        v = self.v

        self.F  = F.copy()
        self.t1 = t1.copy()
        self.t2 = t2.copy()
        self.l1 = l1.copy()
        self.l2 = l2.copy()

        Goo = self.build_Goo()
        Gvv = self.build_Gvv()

        l1_new  = 2*hbar.Hvo.copy()

        l1_new += ndot('ei,ae->ai',self.l1,hbar.Hvv)
        l1_new -= ndot('am,mi->ai',self.l1,hbar.Hoo)

        l1_new += ndot('em,maei->ai',self.l1,hbar.Hovvo,prefactor=2)
        l1_new -= ndot('em,maie->ai',self.l1,hbar.Hovov)

        l1_new += ndot('efim,amef->ai',self.l2,hbar.Hvovv)
        l1_new -= ndot('aemn,mnie->ai',self.l2,hbar.Hooov)

        l1_new += ndot('ef,eafi->ai',Gvv,hbar.Hvvvo,prefactor=2)
        l1_new -= ndot('ef,aefi->ai',Gvv,hbar.Hvvvo)
        l1_new += ndot('mn,mani->ai',Goo,hbar.Hovoo,prefactor=2)
        l1_new -= ndot('mn,main->ai',Goo,hbar.Hovoo)

        l1 = self.l1 + l1_new*self.Dia
        return l1

    def update_l2(self,hbar,t1,t2,l1,l2,F):
        o = self.o
        v = self.v

        self.F  = F.copy()
        self.t1 = t1.copy()
        self.t2 = t2.copy()
        self.l1 = l1.copy()
        self.l2 = l2.copy()

        Goo = self.build_Goo()
        Gvv = self.build_Gvv()

        l2_new  = self.V[v,v,o,o].copy()

        l2_new  = l2_new + ndot('ai,bj->abij',self.l1,hbar.Hvo,prefactor=2)
        l2_new -= ndot('aj,bi->abij',self.l1,hbar.Hvo)

        l2_new -= ndot('bm,maji->abij',self.l1,hbar.Hovoo,prefactor=2)    # check sym
        l2_new += ndot('bm,maij->abij',self.l1,hbar.Hovoo)
        l2_new += ndot('ei,abej->abij',self.l1,hbar.Hvvvo,prefactor=2)
        l2_new -= ndot('ei,baej->abij',self.l1,hbar.Hvvvo)

        l2_new -= ndot('abim,mj->abij',self.l2,hbar.Hoo)
        l2_new += ndot('aeij,be->abij',self.l2,hbar.Hvv)

        l2_new += ndot('abmn,mnij->abij',self.l2,hbar.Hoooo,prefactor=0.5)
        l2_new += ndot('efij,abef->abij',self.l2,hbar.Hvvvv,prefactor=0.5)

        l2_new += ndot('ebmj,maei->abij',self.l2,hbar.Hovvo,prefactor=2)
        l2_new -= ndot('ebmj,maie->abij',self.l2,hbar.Hovov)
        l2_new -= ndot('ebmi,maej->abij',self.l2,hbar.Hovvo)
        l2_new -= ndot('ebim,maje->abij',self.l2,hbar.Hovov)

        l2_new -= ndot('ea,ijeb->abij',Gvv,self.V[o,o,v,v])
        l2_new += ndot('im,mjab->abij',Goo,self.V[o,o,v,v])

        tmp     = l2_new
        tmp     += l2_new.swapaxes(0,1).swapaxes(2,3)
        l2 = self.l2 +  tmp*self.Dijab
        return l2

    def compute_pseudoenergy(self):
        o = self.o
        v = self.v
        
        e = ndot('abij,ijab->',self.l2,self.TEI[o,o,v,v],prefactor=0.5)
        return e

    def compute_lambda(self,maxiter=50,max_diis=8,start_diis=1):
        lambda_tstart = time.time()
        e_ccsd_p = self.compute_pseudoenergy()
        
        Print('\n\t  Summary of iterative solution of the ACC equations')
        Print('\t------------------------------------------------------')
        Print('\t\t\tPseudo\t\t      RMS')
        Print('\t Iteration\tEnergy\t\t     error')
        Print('\t------------------------------------------------------')
        Print('\t{:4d}{:26.15f}{:>22s}' .format(0,e_ccsd_p,'CCSD'))

        # Setup DIIS
        diis_object = helper_diis(self.l1,self.l2,max_diis)

        e_conv = psi4.core.get_option('CCLAMBDA','R_CONVERGENCE')
        # Iterate
        for iter in range(1,maxiter+1):
            l1_old = self.l1.copy()
            l2_old = self.l2.copy()
            e_old_p = e_ccsd_p

            self.update()
            e_ccsd_p = self.compute_pseudoenergy()
            rms = e_ccsd_p - e_old_p
            Print('\t{:4d}{:26.15f}{:15.5E}   DIIS={:d}' .format(iter,e_ccsd_p,rms,diis_object.diis_size))

            # Check convergence
            if abs(rms)<e_conv:
                Print('\t------------------------------------------------------')

                Print(yellow+"\n..The Lambda CCSD equations have converged in %.3f seconds" %(time.time()-lambda_tstart)+end)
                Print(blue+'The lambda pseudo-energy is'+end)
                Print(cyan+'\t%s \n' %e_ccsd_p+end)

                return

            # Add the new error vector
            diis_object.add_error_vector(self.l1,self.l2)
            if iter >= start_diis:
                self.l1,self.l2 = diis_object.extrapolate(self.l1,self.l2)

            # End Lambda CCSD iterations
    # End CCLambda class

class CCDensity(object):
    def __init__(self,ccsd,Lambda,memory=2):
        Print(yellow+"\nInitializing CCSD density object..."+end)

        # Start timer
        time_init = time.time()

        # Read relevant data
        self.n_occ  = ccsd.n_occ
        self.n_virt = ccsd.n_virt
        self.o      = ccsd.o
        self.v      = ccsd.v

        self.ints   = ccsd.mints.ao_dipole()

        self.P      = ccsd.P
        self.npC    = ccsd.npC

        self.t1     = ccsd.t1
        self.t2     = ccsd.t2

        self.l1     = Lambda.l1     # holds 2*lambda_a,i
        self.l2     = Lambda.l2     # holds 2*(2*lambda_ab,ij - lambda_ab,ji)

        D = self.compute_ccsd_density(self.t1,self.t2,self.l1,self.l2)
        Print(yellow+"\n..Density constructed in %.3f seconds\n" %(time.time()-time_init)+end)

        # Get nuclear dipole moments
        mol = psi4.core.get_active_molecule()
        dipoles_nuc = mol.nuclear_dipole()

        dipoles = self.compute_ccsd_dipole(D)
        Print(blue+'Dipole moment computed at the CCSD level'+end)
        Print(blue+'\tNuclear component (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles_nuc[0],'Y:',dipoles_nuc[1],'Z:',dipoles_nuc[2])+end)
        Print(blue+'\tElectronic component (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',float(dipoles[0]),'Y:',float(dipoles[1]),'Z:',float(dipoles[2]))+end)

        self.mu = dipoles

    def build_Doo(self,t1,t2,l1,l2):
        o = self.o
        v = self.v

        self.Doo  = ndot('jc,ci->ij',t1,l1,prefactor=-1)
        self.Doo -= ndot('kjcd,cdki->ij',t2,l2)
        return self.Doo

    def build_Dvv(self,t1,t2,l1,l2):
        o = self.o
        v = self.v

        self.Dvv  = ndot('ka,bk->ab',t1,l1)
        self.Dvv += ndot('klca,cbkl->ab',t2,l2)

        return self.Dvv

    def build_Dov(self,t1,t2,l1,l2):
        o = self.o
        v = self.v

        self.Dov  = 2*t1.copy()
        self.Dov += ndot('ikac,ck->ia',t2,l1,prefactor=2)
        self.Dov -= ndot('ikca,ck->ia',t2,l1)
        tmp = ndot('ka,ck->ca',t1,l1)
        self.Dov -= ndot('ic,ca->ia',t1,tmp)
        self.Dov -= contract('la,kicd,cdkl->ia',t1,t2,l2)
        self.Dov -= contract('id,klca,cdkl->ia',t1,t2,l2)
        return self.Dov

    def compute_ccsd_density(self,t1,t2,l1,l2):
        nmo     = self.n_occ + self.n_virt
        n_occ   = self.n_occ

        self.build_Doo(t1,t2,l1,l2)
        self.build_Dvv(t1,t2,l1,l2)
        self.build_Dov(t1,t2,l1,l2)

        D_corr_left     = np.vstack((self.Doo,l1))
        D_corr_right    = np.vstack((self.Dov,self.Dvv))
        D_corr          = np.hstack((D_corr_left,D_corr_right))     # CCSD correlation part of the density

        D = np.zeros((nmo,nmo))
        D[:n_occ,:n_occ] = 2*np.eye(n_occ)                          # HF MO density
        D = D + D_corr

        return D

    def compute_hf_dipole(self,P):
        # Compute HF electronic dipole moments
        dipoles_elec = []
        for n in range(3):
            mu  = ndot('uv,uv->',np.asarray(self.ints[n]),self.P)
            dipoles_elec.append(float(mu))

        return dipoles_elec

    def compute_ccsd_dipole(self,D):
        # Compute CCSD correlated dipole
        dipoles_elec = []
        for n in range(3):
            d = contract('ui,uv,vj->ij',self.npC,np.asarray(self.ints[n]),self.npC)
            mu = ndot('ij,ij->',d,D)
            dipoles_elec.append(mu)

        return dipoles_elec
