import time,sys
import numpy as np
from opt_einsum import contract
from helper_ndot import ndot
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

if sys.stdout.isatty():
    def Print(obj):  # Wrapper to ensure the Printout is only colorized in real terminal
        print(obj)
        return
else:
    def Print(obj):
        colorized = False
        for color in colors:
            if color in obj:
                colorized = True
        if colorized:
            string = obj[5:-4]
            print(string)
        else:
            print(obj)
        return

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
        tau  = self.t2.copy()
        tau += ndot('ia,jb->ijab',self.t1,self.t1)
        return tau      # Tau (alpha/beta)

    def build_tau_tilde(self):
        tau  = self.t2.copy()
        tau += ndot('ia,jb->ijab',self.t1,self.t1,prefactor=0.5)
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
        W += ndot('ijef,klef->ijkl',self.build_tau(),self.TEI[o,o,v,v])
        
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
            W += ndot('mjeb,kmce->cjkb',self.t2,self.VS[o,o,v,v],prefactor=0.5)
            W += ndot('je,kbce->cjkb',self.t1,self.TEI[o,v,v,v])
            W -= ndot('mb,mkjc->cjkb',self.t1,self.TEI[o,o,o,v])
            W -= ndot('jmeb,kmce->cjkb',tmp,self.TEI[o,o,v,v])
            return W
        elif string == 'aa':
            W  = self.VS[v,o,o,v].copy()
            W += ndot('mjeb,kmce->cjkb',self.t2,self.TEI[o,o,v,v],prefactor=0.5)
            W += ndot('je,kbce->cjkb',self.t1,self.VS[o,v,v,v])
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
        W += ndot('je,kaec->cjak',self.t1,self.TEI[o,v,v,v])
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

    def compute_ccsd_energy(self):
        o = self.o
        v = self.v
        e = ndot('ijab,abij->',self.build_tau(),self.V[v,v,o,o])
        self.e_ccsd = e
        self.E_ccsd = e + self.e_scf
        return e
    
    def compute_ccsd(self,maxiter=50,max_diis=8):
        ccsd_tstart = time.time()
        
        self.e_mp2 = self.compute_ccsd_energy()

        Print('\n\t  Summary of iterative solution of the CC equations')
        Print('\t------------------------------------------------------')
        Print('\t\t\tCorrelation\t      RMS')
        Print('\t Iteration\tEnergy\t\t     error')
        Print('\t------------------------------------------------------')
        Print('\t{:4d}{:26.15f}{:>25s}' .format(0,self.e_ccsd,'MBPT(2)'))

        # Setup DIIS
        diis_values_t1 = [self.t1.copy()]
        diis_values_t2 = [self.t2.copy()]
        diis_errors    = []
        diis_size      = 0

        e_conv = psi4.core.get_option('CCENERGY','E_CONVERGENCE')
        # Iterate
        for iter in range(1,maxiter+1):
            t1_old = self.t1.copy()
            t2_old = self.t2.copy()
            e_old = self.e_ccsd

            self.update()
            e_ccsd = self.compute_ccsd_energy()
            rms = e_ccsd - e_old
            Print('\t{:4d}{:26.15f}{:15.5E}   DIIS={:d}' .format(iter,e_ccsd,rms,diis_size))

            # Check convergence
            if abs(rms)<e_conv:
                Print('\t------------------------------------------------------')

                Print(yellow+"\n..The CCSD equations have converged in %.3f seconds" %(time.time()-ccsd_tstart)+end)
                Print(blue+'The ccsd correlation energy is'+end)
                Print(cyan+'\t%s \n' %e_ccsd+end)

                return

            if max_diis != 0:
                # Add DIIS vectors
                diis_values_t1.append(self.t1.copy())
                diis_values_t2.append(self.t2.copy())

                # Build new error vector
                error_t1 = (diis_values_t1[-1] - t1_old).ravel()
                error_t2 = (diis_values_t2[-1] - t2_old).ravel()
                diis_errors.append(np.concatenate((error_t1,error_t2)))

                if iter>=1:
                    if (len(diis_values_t1)>max_diis+1):
                        del diis_values_t1[0]
                        del diis_values_t2[0]
                        del diis_errors[0]

                    diis_size = len(diis_values_t1) - 1

                    # Build error matrix B
                    B = np.ones((diis_size+1,diis_size+1))*-1
                    B[-1,-1] = 0

                    for n1,e1 in enumerate(diis_errors):
                        B[n1,n1] = np.dot(e1,e1)
                        for n2,e2 in enumerate(diis_errors):
                            if n1 >= n2: continue
                            B[n1,n2] = np.dot(e1,e2)
                            B[n2,n1] = B[n1,n2]

                    B[:-1,:-1] /= np.abs(B[:-1,:-1]).max()

                    # Build residual vector
                    resid = np.zeros(diis_size+1)
                    resid[-1] = -1

                    # Solve pulay equations
                    ci = np.linalg.solve(B,resid)

                    # Calculate new amplitudes
                    self.t1[:] = 0
                    self.t2[:] = 0
                    for num in range(diis_size):
                        self.t1 += ci[num] * diis_values_t1[num+1]
                        self.t2 += ci[num] * diis_values_t2[num+1]

            self.t2_aa = self.t2 - self.t2.swapaxes(0,1)
            # End CCSD iterations
    # End CCEnergy class


class CCHbar(object):
    def __init__(self,ccsd,memory=2):  # ccsd input must be ccsd = CCEnergy(mol, memory=x)
        Print(yellow+"\nInitializing Hbar object..."+end)

        # Start timer
        time_init = time.time()

        # Read relevant data from ccsd class
        self.n_occ  = ccsd.n_occ
        self.n_virt = ccsd.n_virt
        self.o      = ccsd.o
        self.v      = ccsd.v

        self.TEI    = ccsd.TEI
        self.F      = ccsd.F
        self.t1     = ccsd.t1
        self.t2     = ccsd.t2
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

        Print(yellow+"\n..Hbar built in %.3f seconds\n" %(time.time()-time_init)+end)

    # 1-body Hbar
    def build_Hoo(self):
        o = self.o
        v = self.v

        self.Hoo  = self.F[o,o].copy()
        self.Hoo += ndot('ie,ej->ij',self.t1,self.F[v,o])
        self.Hoo += ndot('me,jmie->ij',self.t1,self.V[o,o,o,v])
        self.Hoo += ndot('imef,jmef->ij',self.tau,self.V[o,o,v,v])
        return self.Hoo

    def build_Hvv(self):
        o = self.o
        v = self.v

        self.Hvv  = self.F[v,v].copy()
        self.Hvv -= ndot('mb,am->ab',self.t1,self.F[v,o])
        self.Hvv += ndot('me,mbea->ab',self.t1,self.V[o,v,v,v])
        self.Hvv -= ndot('mnbe,mnae->ab',self.tau,self.V[o,o,v,v])
        return self.Hvv

    def build_Hvo(self):
        o = self.o
        v = self.v

        self.Hvo  = self.F[v,o].copy()
        self.Hvo += ndot('me,imae->ai',self.t1,self.V[o,o,v,v])
        return self.Hvo

    # 2-body hbar
    # 0 excitation rank
    def build_Hoooo(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hoooo  = self.TEI[o,o,o,o].copy()
        self.Hoooo += ndot('je,klie->ijkl',self.t1,self.TEI[o,o,o,v],prefactor=2)
        self.Hoooo += ndot('ijef,klef->ijkl',self.tau,self.TEI[o,o,v,v])
        return self.Hoooo

    def build_Hvvvv(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hvvvv  = self.TEI[v,v,v,v].copy()
        self.Hvvvv -= ndot('md,mcba->abcd',self.t1,self.TEI[o,v,v,v],prefactor=2)
        self.Hvvvv += ndot('mncd,mnab->abcd',self.tau,self.TEI[o,o,v,v])
        return self.Hvvvv

    def build_Hovov(self):
        o = self.o
        v = self.v

        self.Hovov  = self.TEI[o,v,o,v].copy()
        self.Hovov -= ndot('mb,jmia->iajb',self.t1,self.TEI[o,o,o,v])
        self.Hovov += ndot('ie,jbea->iajb',self.t1,self.TEI[o,v,v,v]) 
        self.Hovov -= ndot('imeb,jmea->iajb',self.tau,self.TEI[o,o,v,v])
        return self.Hovov

    def build_Hovvo(self):  # alpha/beta/alpha/beta # exchange
        o = self.o
        v = self.v

        self.Hovvo  = self.TEI[o,v,v,o].copy()
        self.Hovvo -= ndot('ma,mjib->ibaj',self.t1,self.TEI[o,o,o,v])
        self.Hovvo += ndot('ie,jabe->ibaj',self.t1,self.TEI[o,v,v,v])
        self.Hovvo -= ndot('imea,mjeb->ibaj',self.tau,self.TEI[o,o,v,v])
        self.Hovvo += ndot('imae,mjeb->ibaj',self.t2,self.V[o,o,v,v])
        return self.Hovvo

    # -1 excitation rank
    def build_Hovoo(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hovoo  = self.TEI[o,v,o,o].copy()
        self.Hovoo += ndot('ke,ijea->kaij',self.t1,self.TEI[o,o,v,v])
        return self.Hovoo

    def build_Hvvvo(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hvvvo  = self.TEI[v,v,v,o].copy()
        self.Hvvvo -= ndot('mc,miab->abci',self.t1,self.TEI[o,o,v,v])
        return self.Hvvvo

    # +1 excitation rank
    def build_Hooov(self):  # alpha/beta/alpha/beta
        o = self.o
        v = self.v

        self.Hooov  = self.TEI[o,o,o,v].copy()
        self.Hooov -= ndot('ma,ijkm->ijka',self.t1,self.TEI[o,o,o,o])
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
        self.Hvovv += ndot('ie,ceab->ciab',self.t1,self.TEI[v,v,v,v])
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

        self.l1     = self.t1.swapaxes(0,1)
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

        l2_new += ndot('ai,bj->abij',self.l1,self.Hvo,prefactor=2)
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

    def compute_pseudoenergy(self):
        o = self.o
        v = self.v
        
        e = ndot('abij,ijab->',self.l2,self.TEI[o,o,v,v],prefactor=0.5)
        return e

    def compute_lambda(self,maxiter=50,max_diis=8):
        lambda_tstart = time.time()
        e_ccsd_p = self.compute_pseudoenergy()
        
        Print('\n\t  Summary of iterative solution of the ACC equations')
        Print('\t------------------------------------------------------')
        Print('\t\t\tPseudo\t\t      RMS')
        Print('\t Iteration\tEnergy\t\t     error')
        Print('\t------------------------------------------------------')
        Print('\t{:4d}{:26.15f}{:>22s}' .format(0,e_ccsd_p,'CCSD'))

        # Setup DIIS
        diis_values_l1 = [self.l1.copy()]
        diis_values_l2 = [self.l2.copy()]
        diis_errors    = []
        diis_size      = 0

        e_conv = psi4.core.get_option('CCLAMBDA','R_CONVERGENCE')
        # Iterate
        diis_size = 0
        for iter in range(1,maxiter+1):
            l1_old = self.l1.copy()
            l2_old = self.l2.copy()
            e_old_p = e_ccsd_p

            self.update()
            e_ccsd_p = self.compute_pseudoenergy()
            rms = e_ccsd_p - e_old_p
            Print('\t{:4d}{:26.15f}{:15.5E}   DIIS={:d}' .format(iter,e_ccsd_p,rms,diis_size))

            # Check convergence
            if abs(rms)<e_conv:
                Print('\t------------------------------------------------------')

                Print(yellow+"\n..The Lambda CCSD equations have converged in %.3f seconds" %(time.time()-lambda_tstart)+end)
                Print(blue+'The lambda pseudo-energy is'+end)
                Print(cyan+'\t%s \n' %e_ccsd_p+end)

                return

            if max_diis != 0:
                # Add DIIS vectors
                diis_values_l1.append(self.l1.copy())
                diis_values_l2.append(self.l2.copy())

                # Build new error vector
                error_l1 = (diis_values_l1[-1] - l1_old).ravel()
                error_l2 = (diis_values_l2[-1] - l2_old).ravel()
                diis_errors.append(np.concatenate((error_l1,error_l2)))

                if iter>=1:
                    if (len(diis_values_l1)>max_diis+1):
                        del diis_values_l1[0]
                        del diis_values_l2[0]
                        del diis_errors[0]

                    diis_size = len(diis_values_l1) - 1

                    # Build error matrix B
                    B = np.ones((diis_size+1,diis_size+1))*-1
                    B[-1,-1] = 0

                    for n1,e1 in enumerate(diis_errors):
                        B[n1,n1] = np.dot(e1,e1)
                        for n2,e2 in enumerate(diis_errors):
                            if n1 >= n2: continue
                            B[n1,n2] = np.dot(e1,e2)
                            B[n2,n1] = B[n1,n2]

                    B[:-1,:-1] /= np.abs(B[:-1,:-1]).max()

                    # Build residual vector
                    resid = np.zeros(diis_size+1)
                    resid[-1] = -1

                    # Solve pulay equations
                    ci = np.linalg.solve(B,resid)

                    # Calculate new amplitudes
                    self.l1[:] = 0
                    self.l2[:] = 0
                    for num in range(diis_size):
                        self.l1 += ci[num] * diis_values_l1[num+1]
                        self.l2 += ci[num] * diis_values_l2[num+1]

            # End Lambda CCSD iterations
    # End CCLambda class
