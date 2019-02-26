# @BEGIN LICENSE
#
# RT-CC by Alexandre P. Bazante, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2018 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#
#
#   Real-time explicitly time dependent Coupled-Cluster Code
#       -- written by Alexandre P. Bazante, 2017
#

"""
This contains all the coupled-cluster machinery

It assumes RHF reference and C1 symmetry
"""

__authors__ = "Alexandre P. Bazante"
__credits__ = [
        "T.D. Crawford","Ashutosh Kumar","Alexandre P. Bazante"]

import time,sys
import contextlib
import numpy as np
import cmath
from opt_einsum import contract
from helper_Print import Print
from helper_diis import helper_diis
from helper_ndot import ndot
import psi4

@contextlib.contextmanager
def printoptions(*args, **kwargs):      # This helps printing nice arrays
    original = np.get_printoptions()
    np.set_printoptions(*args,**kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

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
        self.n_e     = int(sum(mol.Z(A) for A in range(N_atom))-mol.molecular_charge())
        self.ndocc   = int(self.n_e / 2) # can also be read as self.wfn.doccpi()[0] after an scf instance

        self.e_nuc = mol.nuclear_repulsion_energy()


        self.e_scf,self.wfn = psi4.energy('scf',return_wfn=True)    # This makes psi4 run the scf calculation
        Print(blue+'The SCF energy is'+end)
        Print(cyan+'\t%s\n'%self.e_scf+end)
        
        self.memory = memory
        self.nmo    = self.wfn.nmo()
        self.nso = self.nmo * 2
        self.n_occ = self.ndocc * 2
        self.n_virt = self.nso - self.n_occ
        # Make slices
        self.o = slice(self.n_occ)
        self.v = slice(self.n_occ,self.nso)

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

        ERI_size = (self.nmo**4) * 128e-9
        memory_footPrint = ERI_size * 5
        if memory_footPrint > self.memory:
            psi.clean()
            Print(red+"Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB."
                                                % (memory_footPrint, self.memory)+end)
            raise Exception

        self.F_ao   = self.F.copy()
        self.F      = contract('up,uv,vq->pq',self.npC,self.F,self.npC)
        # Tile for alpha/beta spin
        self.F      = np.repeat(self.F,2,axis=0)
        self.F      = np.repeat(self.F,2,axis=1)
        spin_ind    = np.arange(self.F.shape[0], dtype=np.int) % 2
        self.F      *= (spin_ind.reshape(-1, 1) == spin_ind)

        # Two Electron Integrals are stored as (left out,right out | left in,right in)
        self.TEI    = np.asarray(self.mints.mo_spin_eri(self.C, self.C))
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_size, self.nmo))

        # Build denominators
        eps         = np.diag(self.F)
        self.Dia    = 1/(eps[self.o].reshape(-1,1) - eps[self.v])
        self.Dijab  = 1/(eps[self.o].reshape(-1,1,1,1) + eps[self.o].reshape(-1,1,1) - eps[self.v].reshape(-1,1) - eps[self.v])

        # Build MBPT(2) initial guess (complex)
        Print(yellow+"\n..Building CCSD initial guess from MBPT(2) amplitudes...")

        self.t1 = np.zeros((self.n_occ,self.n_virt))                    + 1j*0.0    # t1 (ia)   <- 0
        self.t2 = self.TEI[self.o,self.o,self.v,self.v] * self.Dijab    + 1j*0.0    # t2 (iJaB) <- (ia|JB) * D(iJaB)

        Print(yellow+"\n..Initialized CCSD in %.3f seconds." %(time.time() - time_init)+end)

    def build_P(self):  # Build AO density
        o = slice(self.ndocc)
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

    # Compute the effective two-particle excitation operators tau and tau tilde
    # Tau is used in the T2 amplitude equations and in the 2 particle intermediates W
    # Tau tilde is used in the 1 particle intermediates F

    def build_tau(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        tau = t2.copy()
        tmp = ndot('ia,jb->ijab',t1,t1)
        tau += tmp - tmp.swapaxes(2,3)
        return tau      # Tau

    def build_tau_tilde(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        tau = t2.copy()
        tmp = ndot('ia,jb->ijab',t1,t1,prefactor=0.5)
        tau += tmp - tmp.swapaxes(2,3)
        return tau  # Tau tilde

    
    # Compute the effective Fock matrix
    def build_Foo(self, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v
        
        Foo  = F[o,o].copy() + 1j*0.0
        Foo += ndot('ie,me->mi',t1,F[o,v],prefactor=0.5)
        Foo += ndot('ne,mnie->mi',t1,self.TEI[o,o,o,v])
        Foo += ndot('inef,mnef->mi',self.build_tau_tilde(t1,t2),self.TEI[o,o,v,v],prefactor=0.5)
        return Foo      # Fmi

    def build_Fvv(self, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        Fvv  = F[v,v].copy() + 1j*0.0
        Fvv -= ndot('ma,me->ae',t1,F[o,v],prefactor=0.5)
        Fvv += ndot('mf,mafe->ae',t1,self.TEI[o,v,v,v])
        Fvv -= ndot('mnaf,mnef->ae',self.build_tau_tilde(t1,t2),self.TEI[o,o,v,v],prefactor=0.5)
        return Fvv      # Fae

    def build_Fov(self, F=None,t1=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1

        o = self.o
        v = self.v

        Fov  = F[o,v].copy() + 1j*0.0
        Fov += ndot('nf,mnef->me',t1,self.TEI[o,o,v,v])
        return Fov      # Fme


    # Compute the 2-body intermediates W
    def build_Woooo(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W = self.TEI[o,o,o,o].copy()   # V(ijkl)
        W = W + ndot('ijef,mnef->mnij',self.build_tau(t1,t2),self.TEI[o,o,v,v],prefactor=0.25)
        
        Pij = ndot('je,mnie->mnij',t1,self.TEI[o,o,o,v])
        W += Pij
        W -= Pij.swapaxes(2,3)
        return W           # Wmnij

    def build_Wvvvv(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W = self.TEI[v,v,v,v].copy()
        W = W + ndot('mnab,mnef->abef', self.build_tau(t1,t2), self.TEI[o,o,v,v], prefactor=0.25)

        Pab = ndot('mb,amef->abef', t1, self.TEI[v,o,v,v])
        W -= Pab
        W += Pab.swapaxes(0, 1)

        return W        # Wabef

    def build_Wovvo(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W = self.TEI[o,v,v,o].copy()
        W = W + ndot('jf,mbef->mbej', t1, self.TEI[o,v,v,v])
        W -= ndot('nb,mnej->mbej', t1, self.TEI[o,o,v,o])

        tmp  = (0.5 * t2)
        tmp += np.einsum('jf,nb->jnfb', t1, t1)
        W -= ndot('jnfb,mnef->mbej', tmp, self.TEI[o,o,v,v])

        return W        # Wmbej


    # Compute T1 and T2 residual
    def residual_t1(self, F,t1,t2):
        o = self.o
        v = self.v

        Foo = self.build_Foo(F,t1,t2)
        Fvv = self.build_Fvv(F,t1,t2)
        Fov = self.build_Fov(F,t1)

        #### Build RHS side of self.t1 equations
        rhs_T1 = F[o,v].copy() + 1j*0.0
        rhs_T1 += ndot('ie,ae->ia', t1,Fvv)
        rhs_T1 -= ndot('ma,mi->ia', t1,Foo)
        rhs_T1 += ndot('imae,me->ia', t2,Fov)
        rhs_T1 -= ndot('nf,naif->ia', t1, self.TEI[o,v,o,v])
        rhs_T1 -= ndot('imef,maef->ia', t2, self.TEI[o,v,v,v], prefactor=0.5)
        rhs_T1 -= ndot('mnae,nmei->ia', t2, self.TEI[o,o,v,o], prefactor=0.5)

        return rhs_T1

    def residual_t2(self, F,t1,t2):
        o = self.o
        v = self.v

        Foo = self.build_Foo(F,t1,t2)
        Fvv = self.build_Fvv(F,t1,t2)
        Fov = self.build_Fov(F,t1)

        tau     = self.build_tau(t1,t2)
        Woooo   = self.build_Woooo(t1,t2)
        Wvvvv   = self.build_Wvvvv(t1,t2)
        Wovvo   = self.build_Wovvo(t1,t2)

        ### Build RHS side of self.t2 equations
        rhs_T2 = self.TEI[o,o,v,v].copy() + 1j*0.0
        rhs_T2 += ndot('mnab,mnij->ijab', tau, Woooo, prefactor=0.5)
        rhs_T2 += ndot('ijef,abef->ijab', tau, Wvvvv, prefactor=0.5)

        tmp = ndot('ie,mbej->mbij', t1, self.TEI[o,v,v,o])
        tmp = ndot('ma,mbij->ijab', t1, tmp)
        Pijab = ndot('imae,mbej->ijab', t2, Wovvo) - tmp
        rhs_T2 += Pijab
        rhs_T2 -= Pijab.swapaxes(2, 3)
        rhs_T2 -= Pijab.swapaxes(0, 1)
        rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)

        tmp = Fvv - 0.5 * ndot('mb,me->be', t1, Fov)
        Pab = ndot('ijae,be->ijab', t2, tmp)
        rhs_T2 += Pab
        rhs_T2 -= Pab.swapaxes(2, 3)

        tmp = Foo + 0.5 * ndot('je,me->mj', t1, Fov)
        Pij = ndot('imab,mj->ijab', t2, tmp)
        rhs_T2 -= Pij
        rhs_T2 += Pij.swapaxes(0, 1)

        Pij = ndot('ie,abej->ijab', t1, self.TEI[v,v,v,o])
        rhs_T2 += Pij
        rhs_T2 -= Pij.swapaxes(0, 1)

        Pab = ndot('ma,mbij->ijab', t1, self.TEI[o,v,o,o])
        rhs_T2 -= Pab
        rhs_T2 += Pab.swapaxes(2, 3)

        return rhs_T2


    # Update amplitudes
    def update(self, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        Foo = self.build_Foo(F,t1,t2)
        Fvv = self.build_Fvv(F,t1,t2)
        Fov = self.build_Fov(F,t1)

        tau     = self.build_tau(t1,t2)
        Woooo   = self.build_Woooo(t1,t2)
        Wvvvv   = self.build_Wvvvv(t1,t2)
        Wovvo   = self.build_Wovvo(t1,t2)

        #### Build RHS side of t1 equations
        rhs_T1 = F[o,v].copy() + 1j*0.0
        rhs_T1 += ndot('ie,ae->ia', t1,Fvv)
        rhs_T1 -= ndot('ma,mi->ia', t1,Foo)
        rhs_T1 += ndot('imae,me->ia', t2,Fov)
        rhs_T1 -= ndot('nf,naif->ia', t1, self.TEI[o,v,o,v])
        rhs_T1 -= ndot('imef,maef->ia', t2, self.TEI[o,v,v,v], prefactor=0.5)
        rhs_T1 -= ndot('mnae,nmei->ia', t2, self.TEI[o,o,v,o], prefactor=0.5)

        ### Build RHS side of t2 equations
        rhs_T2 = self.TEI[o,o,v,v].copy() + 1j*0.0
        rhs_T2 += ndot('mnab,mnij->ijab', tau, Woooo, prefactor=0.5)
        rhs_T2 += ndot('ijef,abef->ijab', tau, Wvvvv, prefactor=0.5)

        tmp = ndot('ie,mbej->mbij', t1, self.TEI[o,v,v,o])
        tmp = ndot('ma,mbij->ijab', t1, tmp)
        Pijab = ndot('imae,mbej->ijab', t2, Wovvo) - tmp
        rhs_T2 += Pijab
        rhs_T2 -= Pijab.swapaxes(2, 3)
        rhs_T2 -= Pijab.swapaxes(0, 1)
        rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)

        tmp = Fvv - 0.5 * ndot('mb,me->be', t1, Fov)
        Pab = ndot('ijae,be->ijab', t2, tmp)
        rhs_T2 += Pab
        rhs_T2 -= Pab.swapaxes(2, 3)

        tmp = Foo + 0.5 * ndot('je,me->mj', t1, Fov)
        Pij = ndot('imab,mj->ijab', t2, tmp)
        rhs_T2 -= Pij
        rhs_T2 += Pij.swapaxes(0, 1)

        Pij = ndot('ie,abej->ijab', t1, self.TEI[v,v,v,o])
        rhs_T2 += Pij
        rhs_T2 -= Pij.swapaxes(0, 1)

        Pab = ndot('ma,mbij->ijab', t1, self.TEI[o,v,o,o])
        rhs_T2 -= Pab
        rhs_T2 += Pab.swapaxes(2, 3)

        ### Update T1 and T2 amplitudes
        self.t1 = t1 + rhs_T1*self.Dia
        self.t2 = t2 + rhs_T2*self.Dijab
        return

    def compute_corr_energy(self, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v
        e  = np.einsum('ia,ia->', F[o,v], t1)
        e += 0.25 * np.einsum('ijab,ijab->', self.TEI[o,o,v,v], t2)
        e += 0.5 * np.einsum('ijab,ia,jb->', self.TEI[o,o,v,v], t1, t1)
        self.e_ccsd = e.real
        self.E_ccsd = e.real + self.e_scf
        return e
    
    def compute_ccsd(self,maxiter=50,max_diis=8,start_diis=1):
        ccsd_tstart = time.time()
        
        self.e_mp2 = self.compute_corr_energy().real

        Print('\n\t  Summary of iterative solution of the CC equations')
        Print('\t------------------------------------------------------')
        Print('\t\t\tCorrelation\t      RMS')
        Print('\t Iteration\tEnergy\t\t     error')
        Print('\t------------------------------------------------------')
        Print('\t{:4d}{:26.15f}{:>25s}' .format(0,self.e_ccsd,'MBPT(2)'))

        e_conv = psi4.core.get_option('CCENERGY','E_CONVERGENCE')

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.t1,self.t2,max_diis)

        # Iterate
        for iter in range(1,maxiter+1):
            e_old = self.e_ccsd
            self.update()
            e_ccsd = self.compute_corr_energy().real
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
            diis_object.add_error_vector(self.t1,self.t2)

            if iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1,self.t2)

            # End CCSD iterations
    # End CCEnergy class



class CCLambda(object):
    def __init__(self,ccsd):
        Print(yellow+"\nInitializing Lambda object..."+end)

        # Start timer
        time_init = time.time()

        # Read relevant data from ccsd class
        self.n_occ  = ccsd.n_occ
        self.n_virt = ccsd.n_virt
        self.o      = ccsd.o
        self.v      = ccsd.v

        self.TEI    = ccsd.TEI
        self.Dia    = ccsd.Dia.swapaxes(0,1)
        self.Dijab  = ccsd.Dijab.swapaxes(0,2).swapaxes(1,3)
        self.t1     = ccsd.t1
        self.t2     = ccsd.t2
        self.F      = ccsd.F

        # Initialize l1 and l2 to the transpose of t1 and t2, respectively
        self.l1     = self.t1.swapaxes(0,1).copy()
        self.l2     = self.t2.swapaxes(0,2).swapaxes(1,3).copy()


        # Build intermediates independent of Lambda
        np.set_printoptions(precision=12, linewidth=200, suppress=True)
        self.Fov = ccsd.build_Fov()
        self.Foo = self.transform_Foo(ccsd)
        self.Fvv = self.transform_Fvv(ccsd)

        self.Woooo = self.transform_Woooo(ccsd)
        self.Wvvvv = self.transform_Wvvvv(ccsd)
        self.Wovvo = self.transform_Wovvo(ccsd)

        self.Wooov = self.build_Wooov(ccsd)
        self.Wvovv = self.build_Wvovv(ccsd)
        self.Wovoo = self.build_Wovoo(ccsd)
        self.Wvvvo = self.build_Wvvvo(ccsd)


    # Update the intermediates with Hbar elements
    def transform_Foo(self,ccsd, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        Foo  = ccsd.build_Foo(F,t1,t2)
        Foo += ndot('ie,me->mi',t1,self.Fov,prefactor=0.5)
        return Foo      # Fmi

    def transform_Fvv(self,ccsd, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        Fvv  = ccsd.build_Fvv(F,t1,t2)
        Fvv -= ndot('ma,me->ae',t1,self.Fov,prefactor=0.5)
        return Fvv      # Fae

    def transform_Woooo(self,ccsd, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)

        W  = ccsd.build_Woooo(t1,t2)
        W += ndot('ijef,mnef->mnij',tau,self.TEI[o,o,v,v],prefactor=0.25)
        return W           # Wmnij

    def transform_Wvvvv(self,ccsd, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)
        
        W  = ccsd.build_Wvvvv(t1,t2)
        W += ndot('mnab,mnef->abef',tau,self.TEI[o,o,v,v],prefactor=0.25)
        return W           # Wabef

    def transform_Wovvo(self,ccsd, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W  = ccsd.build_Wovvo(t1,t2)
        W -= ndot('jnfb,mnef->mbej',t2,self.TEI[o,o,v,v],prefactor=0.5)
        return W           # Wmbej


    # Build new intermediates
    def build_Wooov(self,ccsd, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W = self.TEI[o,o,o,v].copy()
        W = W + ndot('if,mnfe->mnie',t1,self.TEI[o,o,v,v])
        return W            # Wmnie

    def build_Wvovv(self,ccsd, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W = self.TEI[v,o,v,v].copy()
        W = W - ndot('na,nmef->amef',t1,self.TEI[o,o,v,v])
        return W            # Wamef

    def build_Wovoo(self,ccsd, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)

        W  = self.TEI[o,v,o,o].copy()
        W  = W - ndot('ijbe,me->mbij',t2,self.Fov)
        W -= ndot('nb,mnij->mbij',t1,self.Woooo)
        W += ndot('ijef,mbef->mbij',tau,self.TEI[o,v,v,v],prefactor=0.5)

        Pij = ndot('jnbe,mnie->mbij',t2,self.TEI[o,o,o,v])
        temp_mbej = self.TEI[o,v,v,o].copy()
        temp_mbej = temp_mbej - ndot('njbf,mnef->mbej',t2,self.TEI[o,o,v,v])
        Pij += ndot('ie,mbej->mbij',t1,temp_mbej)
        W += Pij
        W -= Pij.swapaxes(2,3)
        return W            # Wmbij

    def build_Wvvvo(self,ccsd, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)

        W  = self.TEI[v,v,v,o].copy()
        W  = W - ndot('miab,me->abei',t2,self.Fov)
        W += ndot('if,abef->abei',t1,self.Wvvvv)
        W += ndot('mnab,mnei->abei',tau,self.TEI[o,o,v,o],prefactor=0.5)

        Pab = ndot('miaf,mbef->abei',t2,self.TEI[o,v,v,v])
        temp_mbei = self.TEI[o,v,v,o].copy()
        temp_mbei = temp_mbei - ndot('nibf,mnef->mbei',t2,self.TEI[o,o,v,v])
        Pab += ndot('ma,mbei->abei',t1,temp_mbei)
        W -= Pab
        W += Pab.swapaxes(0,1)
        return W            # Wabei


    # Build 3-body intermediates
    def build_Goo(self, t2=None,l2=None):
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2

        Goo = ndot('mjef,efmi->ij',t2,l2,prefactor=0.5)
        return Goo          # Gmi

    def build_Gvv(self, t2=None,l2=None):
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2

        Gvv = - ndot('mnea,ebmn->ab',t2,l2,prefactor=0.5)
        return Gvv          # Gae



    # Compute L1 residual
    def residual_l1(self,ccsd, F,t1,t2,l1,l2):
        o = self.o
        v = self.v

        # compute intermediates independent of Lambda
        self.Fov = ccsd.build_Fov(F,t1)
        self.Foo = self.transform_Foo(ccsd, F,t1,t2)
        self.Fvv = self.transform_Fvv(ccsd, F,t1,t2)

        self.Woooo = self.transform_Woooo(ccsd, t1,t2)
        self.Wvvvv = self.transform_Wvvvv(ccsd, t1,t2)
        self.Wovvo = self.transform_Wovvo(ccsd, t1,t2)

        self.Wooov = self.build_Wooov(ccsd, t1,t2)
        self.Wvovv = self.build_Wvovv(ccsd, t1,t2)
        self.Wovoo = self.build_Wovoo(ccsd, t1,t2)
        self.Wvvvo = self.build_Wvvvo(ccsd, t1,t2)

        # compute lambda dependent intermediates
        Goo = self.build_Goo(t2,l2)
        Gvv = self.build_Gvv(t2,l2)

        #### Build RHS side of l1 equations
        rhs_L1  = self.Fov.swapaxes(0,1).copy()
        rhs_L1 += ndot('ei,ea->ai',l1,self.Fvv)
        rhs_L1 -= ndot('am,im->ai',l1,self.Foo)
        rhs_L1 += ndot('em,ieam->ai',l1,self.Wovvo)
        rhs_L1 += ndot('efim,efam->ai',l2,self.Wvvvo,prefactor=0.5)
        rhs_L1 -= ndot('aemn,iemn->ai',l2,self.Wovoo,prefactor=0.5)
        rhs_L1 -= ndot('ef,eifa->ai',Gvv,self.Wvovv)
        rhs_L1 -= ndot('mn,mina->ai',Goo,self.Wooov)
        
        return rhs_L1

    # Compute L2 residual
    def residual_l2(self,ccsd, F,t1,t2,l1,l2):
        o = self.o
        v = self.v

        # compute intermediates independent of Lambda
        self.Fov = ccsd.build_Fov(F,t1)
        self.Foo = self.transform_Foo(ccsd, F,t1,t2)
        self.Fvv = self.transform_Fvv(ccsd, F,t1,t2)

        self.Woooo = self.transform_Woooo(ccsd, t1,t2)
        self.Wvvvv = self.transform_Wvvvv(ccsd, t1,t2)
        self.Wovvo = self.transform_Wovvo(ccsd, t1,t2)

        self.Wooov = self.build_Wooov(ccsd, t1,t2)
        self.Wvovv = self.build_Wvovv(ccsd, t1,t2)
        self.Wovoo = self.build_Wovoo(ccsd, t1,t2)
        self.Wvvvo = self.build_Wvvvo(ccsd, t1,t2)

        # compute lambda dependent intermediates
        Goo = self.build_Goo(t2,l2)
        Gvv = self.build_Gvv(t2,l2)

        #### Build RHS side of l2 equations
        rhs_L2 = self.TEI[v,v,o,o].copy()

        Pab = ndot('aeij,eb->abij',l2,self.Fvv)
        rhs_L2  = rhs_L2 + Pab
        rhs_L2 -= Pab.swapaxes(0,1)

        Pij = ndot('abim,jm->abij',l2,self.Foo)
        rhs_L2 -= Pij
        rhs_L2 += Pij.swapaxes(2,3)

        rhs_L2 += ndot('abmn,ijmn->abij',l2,self.Woooo,prefactor=0.5)
        rhs_L2 += ndot('efij,efab->abij',l2,self.Wvvvv,prefactor=0.5)

        Pij = ndot('ei,ejab->abij',l1,self.Wvovv)
        rhs_L2 += Pij
        rhs_L2 -= Pij.swapaxes(2,3)

        Pab = ndot('am,ijmb->abij',l1,self.Wooov)
        rhs_L2 -= Pab
        rhs_L2 += Pab.swapaxes(0,1)

        Pijab = ndot('aeim,jebm->abij',l2,self.Wovvo)
        rhs_L2 += Pijab
        rhs_L2 -= Pijab.swapaxes(0,1)
        rhs_L2 -= Pijab.swapaxes(2,3)
        rhs_L2 += Pijab.swapaxes(0,1).swapaxes(2,3)

        Pijab = ndot('ai,jb->abij',l1,self.Fov)
        rhs_L2 += Pijab
        rhs_L2 -= Pijab.swapaxes(0,1)
        rhs_L2 -= Pijab.swapaxes(2,3)
        rhs_L2 += Pijab.swapaxes(0,1).swapaxes(2,3)

        Pab = ndot('be,ijae->abij',Gvv,self.TEI[o,o,v,v])
        rhs_L2 += Pab
        rhs_L2 -= Pab.swapaxes(0,1)

        Pij = ndot('mj,imab->abij',Goo,self.TEI[o,o,v,v])
        rhs_L2 -= Pij
        rhs_L2 += Pij.swapaxes(2,3)

        return rhs_L2


    # Update amplitudes
    def update(self, F=None,t1=None,t2=None,l1=None,l2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        o = self.o
        v = self.v

        Goo = self.build_Goo(t2,l2)
        Gvv = self.build_Gvv(t2,l2)

        #### Build RHS side of l1 equations
        rhs_L1  = self.Fov.swapaxes(0,1).copy()
        rhs_L1 += ndot('ei,ea->ai',l1,self.Fvv)
        rhs_L1 -= ndot('am,im->ai',l1,self.Foo)
        rhs_L1 += ndot('em,ieam->ai',l1,self.Wovvo)
        rhs_L1 += ndot('efim,efam->ai',l2,self.Wvvvo,prefactor=0.5)
        rhs_L1 -= ndot('aemn,iemn->ai',l2,self.Wovoo,prefactor=0.5)
        rhs_L1 -= ndot('ef,eifa->ai',Gvv,self.Wvovv)
        rhs_L1 -= ndot('mn,mina->ai',Goo,self.Wooov)

        #### Build RHS side of l2 equations
        rhs_L2 = self.TEI[v,v,o,o].copy()

        Pab = ndot('aeij,eb->abij',l2,self.Fvv)
        rhs_L2  = rhs_L2 + Pab
        rhs_L2 -= Pab.swapaxes(0,1)

        Pij = ndot('abim,jm->abij',l2,self.Foo)
        rhs_L2 -= Pij
        rhs_L2 += Pij.swapaxes(2,3)

        rhs_L2 += ndot('abmn,ijmn->abij',l2,self.Woooo,prefactor=0.5)
        rhs_L2 += ndot('efij,efab->abij',l2,self.Wvvvv,prefactor=0.5)

        Pij = ndot('ei,ejab->abij',l1,self.Wvovv)
        rhs_L2 += Pij
        rhs_L2 -= Pij.swapaxes(2,3)

        Pab = ndot('am,ijmb->abij',l1,self.Wooov)
        rhs_L2 -= Pab
        rhs_L2 += Pab.swapaxes(0,1)

        Pijab = ndot('aeim,jebm->abij',l2,self.Wovvo)
        rhs_L2 += Pijab
        rhs_L2 -= Pijab.swapaxes(0,1)
        rhs_L2 -= Pijab.swapaxes(2,3)
        rhs_L2 += Pijab.swapaxes(0,1).swapaxes(2,3)

        Pijab = ndot('ai,jb->abij',l1,self.Fov)
        rhs_L2 += Pijab
        rhs_L2 -= Pijab.swapaxes(0,1)
        rhs_L2 -= Pijab.swapaxes(2,3)
        rhs_L2 += Pijab.swapaxes(0,1).swapaxes(2,3)

        Pab = ndot('be,ijae->abij',Gvv,self.TEI[o,o,v,v])
        rhs_L2 += Pab
        rhs_L2 -= Pab.swapaxes(0,1)

        Pij = ndot('mj,imab->abij',Goo,self.TEI[o,o,v,v])
        rhs_L2 -= Pij
        rhs_L2 += Pij.swapaxes(2,3)


        ### Update T1 and T2 amplitudes
        self.l1 = l1 + rhs_L1*self.Dia
        self.l2 = l2 + rhs_L2*self.Dijab
        return


    def compute_pseudoenergy(self, F=None,l1=None,l2=None):
        if F  is None: F = self.F
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        o = self.o
        v = self.v
        
        e  = ndot('ai,ia->',l1,F[o,v])
        e += ndot('abij,ijab->',l2,self.TEI[o,o,v,v],prefactor=0.25)
        return e

    def compute_lambda(self,maxiter=50,max_diis=8,start_diis=1):
        lambda_tstart = time.time()
        e_ccsd_p = self.compute_pseudoenergy().real
        
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
            e_old_p = e_ccsd_p
            self.update()
            e_ccsd_p = self.compute_pseudoenergy().real
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


class CCProperties(object):
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

        self.l1     = Lambda.l1
        self.l2     = Lambda.l2

        D = self.compute_ccsd_density()
        Print(yellow+"\n..Density constructed in %.3f seconds\n" %(time.time()-time_init)+end)

        # Get nuclear dipole moments
        mol = psi4.core.get_active_molecule()
        dipoles_nuc = mol.nuclear_dipole()

        dipoles = self.compute_hf_dipole(self.P)
        Print(blue+'Dipole moment computed at the HF level'+end)
        Print(blue+'\tNuclear component (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles_nuc[0],'Y:',dipoles_nuc[1],'Z:',dipoles_nuc[2])+end)
        Print(blue+'\tElectronic component (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles[0],'Y:',dipoles[1],'Z:',dipoles[2])+end)
        dipoles = np.asarray([dipoles[i] + dipoles_nuc[i] for i in range(3)])
        Print(green+'\tTotal electric dipole (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles[0],'Y:',dipoles[1],'Z:',dipoles[2])+end)
        dipoles *= 1/0.393456
        Print(green+'\tTotal electric dipole (Debye)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}\n' .format('X:',dipoles[0],'Y:',dipoles[1],'Z:',dipoles[2])+end)

        dipoles = self.compute_ccsd_dipole()
        Print(blue+'Dipole moment computed at the CCSD level'+end)
        Print(blue+'\tNuclear component (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles_nuc[0],'Y:',dipoles_nuc[1],'Z:',dipoles_nuc[2])+end)
        Print(blue+'\tElectronic component (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles[0].real,'Y:',dipoles[1].real,'Z:',dipoles[2].real)+end)
        self.mu = dipoles.copy()
        dipoles = np.asarray([dipoles[i] + dipoles_nuc[i] for i in range(3)])
        Print(green+'\tTotal electric dipole (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles[0].real,'Y:',dipoles[1].real,'Z:',dipoles[2].real)+end)
        dipoles *= 1/0.393456
        Print(green+'\tTotal electric dipole (Debye)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}\n' .format('X:',dipoles[0].real,'Y:',dipoles[1].real,'Z:',dipoles[2].real)+end)


    def build_Doo(self, t1=None,t2=None,l1=None,l2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        self.Doo  = ndot('jc,ci->ij',t1,l1,prefactor=-1)
        self.Doo -= ndot('kjcd,cdki->ij',t2,l2,prefactor=0.5)
        return self.Doo

    def build_Dvv(self, t1=None,t2=None,l1=None,l2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        self.Dvv  = ndot('ka,bk->ab',t1,l1)
        self.Dvv += ndot('klca,cbkl->ab',t2,l2,prefactor=0.5)
        return self.Dvv

    def build_Dov(self, t1=None,t2=None,l1=None,l2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        self.Dov  = t1.copy()
        self.Dov += ndot('ikac,ck->ia',t2,l1)
        tmp = ndot('ka,ck->ca',t1,l1)
        self.Dov -= ndot('ic,ca->ia',t1,tmp)
        self.Dov -= 0.5*contract('la,kicd,cdkl->ia',t1,t2,l2)
        self.Dov -= 0.5*contract('id,klca,cdkl->ia',t1,t2,l2)
        return self.Dov

    def compute_ccsd_density(self, t1=None,t2=None,l1=None,l2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        nso     = self.n_occ + self.n_virt
        n_occ   = self.n_occ

        self.build_Doo(t1,t2,l1,l2)
        self.build_Dvv(t1,t2,l1,l2)
        self.build_Dov(t1,t2,l1,l2)

        D_corr_left     = np.vstack((self.Doo,l1))
        D_corr_right    = np.vstack((self.Dov,self.Dvv))
        D_corr          = np.hstack((D_corr_left,D_corr_right))     # CCSD correlation part of the density

        D = np.zeros((nso,nso))
        D[:n_occ,:n_occ] = np.eye(n_occ)                            # HF MO density
        D = D + D_corr

        # untile for alpha/beta spin
        Da = np.delete(D, list(range(1, D.shape[0], 2)), axis=1)
        Da = np.delete(Da, list(range(1, Da.shape[0], 2)), axis=0)
        Db = np.delete(D, list(range(0, D.shape[0], 2)), axis=1)
        Db = np.delete(Db, list(range(0, Db.shape[0], 2)), axis=0)

        return Da+Db

    def compute_hf_dipole(self,P):
        # Compute HF electronic dipole moments
        dipoles_elec = []
        for n in range(3):
            mu  = ndot('uv,vu->',np.asarray(self.ints[n]),self.P)
            dipoles_elec.append(mu)
        return dipoles_elec

    def compute_ccsd_dipole(self, t1=None,t2=None,l1=None,l2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        D = self.compute_ccsd_density(t1,t2,l1,l2)

        # Compute CCSD correlated dipole
        dipoles_elec = []
        for n in range(3):
            d = contract('ui,uv,vj->ij',self.npC,np.asarray(self.ints[n]),self.npC)
            mu = ndot('ij,ji->',d,D)
            dipoles_elec.append(mu)
        return dipoles_elec
