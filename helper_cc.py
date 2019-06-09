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
- CCSD Energy
- CCSD Lambdas
- CCSD Density and Dipole

Reference: 
1.  J. Gauss, J.F. Stanton, J. Chem. Phys. 103, 3561 (1995);
    https://doi.org/10.1063/1.470240

It assumes a closed-shell reference and C1 symmetry
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
from helper_ndot import ndot
from helper_diis import helper_diis
from helper_local import localize_occupied
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
        self.n_occ   = int(self.n_e / 2) # can also be read as self.wfn.doccpi()[0] after an scf instance

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

        # Overwrite H/F to match psi4numpy
        H = np.einsum('uj,vi,uv', self.npC, self.npC, self.H)
        self.F = H + 2.0 * np.einsum('pmqm->pq',self.TEI[:, self.o, :, self.o])
        self.F -= np.einsum('pmmq->pq',self.TEI[:, self.o, self.o, :])

        # Two Electron Integrals are stored as (left out,right out | left in,right in)
        Print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_size, self.nmo))

        # Build denominators
        eps         = np.diag(self.F)
        self.Dia    = 1/(eps[self.o].reshape(-1,1) - eps[self.v])
        self.Dijab  = 1/(eps[self.o].reshape(-1,1,1,1) + eps[self.o].reshape(-1,1,1) - eps[self.v].reshape(-1,1) - eps[self.v])

        # Build MBPT(2) initial guess (complex)
        Print(yellow+"\n..Building CCSD initial guess from MBPT(2) amplitudes...")

        self.t1 = np.zeros((self.n_occ,self.n_virt)) + 1j*0.0                   # t1 (ia)   <- 0
        self.t2 = self.TEI[self.o,self.o,self.v,self.v] * self.Dijab + 1j*0.0   # t2 (iJaB) <- (ia|JB) * D(iJaB)

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
##  The equations follow reference 1, but are spin integrated
##
##
##  i,j -> target occupied indeces
##  a,b -> target virtual indeces
##
##  m,n,o -> implicit (summed-over) occupied indeces
##  e,f,g -> implicit (summed-over) virtual indeces
##
##  F: effective 1-particle intermediate
##  W,Tau: effective 2 particle intermediates
##
##
##  Because equations are spin integrated, we only compute the necessary spin combinations of amplitudes:
##      t1  -> alpha block      -> t1 (ia)
##      t2  -> alpha/beta block -> t2 (iJaB)
##

    # Compute the effective two-particle excitation operators tau and tau tilde
    # Tau is used in the T2 amplitude equations and in the 2 particle intermediates W
    # Tau tilde is used in the 1 particle intermediates F

    def build_tau(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        tau = t2.copy()
        tau = tau + ndot('ia,jb->ijab',t1,t1)
        return tau      # Tau (alpha/beta)

    def build_tau_tilde(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        tau = t2.copy()
        tau = tau + ndot('ia,jb->ijab',t1,t1,prefactor=0.5)
        return tau  # Tau tilde (alpha/beta)


    # Compute the effective Fock matrix
    def build_Fmi(self, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v
        
        Fmi  = F[o,o].copy() + 1j*0.0
        Fmi += ndot('ie,me->mi',t1,F[o,v],prefactor=0.5)
        Fmi += ndot('ne,mnie->mi',t1,self.TEI[o,o,o,v],prefactor=2)
        Fmi += ndot('ne,nmie->mi',t1,self.TEI[o,o,o,v],prefactor=-1)
        Fmi += ndot('inef,mnef->mi',self.build_tau_tilde(t1,t2),self.TEI[o,o,v,v],prefactor=2)
        Fmi += ndot('inef,mnfe->mi',self.build_tau_tilde(t1,t2),self.TEI[o,o,v,v],prefactor=-1)
        return Fmi      # Fmi

    def build_Fae(self, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        Fae  = self.F[v,v].copy() + 1j*0.0
        Fae -= ndot('ma,me->ae',t1,F[o,v],prefactor=0.5)
        Fae += ndot('mf,amef->ae',t1,self.TEI[v,o,v,v],prefactor=2)
        Fae += ndot('mf,amfe->ae',t1,self.TEI[v,o,v,v],prefactor=-1)
        Fae -= ndot('mnaf,mnef->ae',self.build_tau_tilde(t1,t2),self.TEI[o,o,v,v],prefactor=2)
        Fae -= ndot('mnaf,mnfe->ae',self.build_tau_tilde(t1,t2),self.TEI[o,o,v,v],prefactor=-1)
        return Fae      # Fae

    def build_Fme(self, F=None,t1=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1

        o = self.o
        v = self.v

        Fme  = F[o,v].copy() + 1j*0.0
        Fme += ndot('nf,mnef->me',t1,self.TEI[o,o,v,v],prefactor=2)
        Fme += ndot('nf,mnfe->me',t1,self.TEI[o,o,v,v],prefactor=-1)
        return Fme      # Fme


    # Compute the 2-body intermediates W
    def build_Wmnij(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W  = self.TEI[o,o,o,o].copy()   # V(ijkl)
        W  = W + ndot('ijef,mnef->mnij',self.build_tau(t1,t2),self.TEI[o,o,v,v])
        W += ndot('je,mnie->mnij',t1,self.TEI[o,o,o,v])
        W += ndot('ie,mnej->mnij',t1,self.TEI[o,o,v,o])     ## test
        
        return W           # Wmnij

    def build_Z(self, t1=None,t2=None):      # Computes parts of Tau*Wabef to avoid computing Wabef
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        Z = ndot('ijef,mbef->mbij',self.build_tau(t1,t2),self.TEI[o,v,v,v])
        return Z

    def build_Wmbej(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W  = self.TEI[o,v,v,o].copy()       # V(mbej)
        W  = W + ndot('jf,mbef->mbej',t1,self.TEI[o,v,v,v])
        W -= ndot('nb,mnej->mbej',t1,self.TEI[o,o,v,o])

        tmp = 0.5 * t2 + ndot('jf,nb->jnfb',t1,t1)
        W -= ndot('jnfb,mnef->mbej',tmp,self.TEI[o,o,v,v])
        W += ndot('njfb,mnef->mbej',t2,self.TEI[o,o,v,v])
        W -= ndot('njfb,mnfe->mbej',t2,self.TEI[o,o,v,v],prefactor=0.5)
        return W        # Wmbej

    # This intermediate appears in the spin factorization of Wmbej terms.
    def build_Wmbje(self, t1=None,t2=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W  = -1.0 * self.TEI[o,v,o,v].copy()        # V(mbje)
        W  = W - ndot('jf,mbfe->mbje',t1,self.TEI[o,v,v,v])
        W += ndot('nb,mnje->mbje',t1,self.TEI[o,o,o,v])

        tmp = 0.5 * t2 + ndot('jf,nb->jnfb',t1,t1)
        W += ndot('jnfb,mnfe->mbje',tmp,self.TEI[o,o,v,v])
        return W        # Wmbje


    # Compute T1 and T2 residuals
    def residual_t1(self, F,t1,t2):
        o = self.o
        v = self.v

        Fmi = self.build_Fmi(F,t1,t2)
        Fae = self.build_Fae(F,t1,t2)
        Fme = self.build_Fme(F,t1)

        #### Build RHS side of t1 equations
        rhs_t1  = F[o,v].copy() + 1j*0.0
        rhs_t1 += ndot('ie,ae->ia',t1,Fae)
        rhs_t1 -= ndot('ma,mi->ia',t1,Fmi)
        rhs_t1 += ndot('imae,me->ia',t2,Fme,prefactor=2)
        rhs_t1 += ndot('imea,me->ia',t2,Fme,prefactor=-1)

        rhs_t1 += ndot('nf,nafi->ia',t1,self.TEI[o,v,v,o],prefactor=2)
        rhs_t1 += ndot('nf,naif->ia',t1,self.TEI[o,v,o,v],prefactor=-1)

        rhs_t1 += ndot('imef,amef->ia',t2,self.TEI[o,v,v,v],prefactor=2)
        rhs_t1 += ndot('imfe,amef->ia',t2,self.TEI[o,v,v,v],prefactor=-1)

        rhs_t1 -= ndot('mnae,mnie->ia',t2,self.TEI[o,o,o,v],prefactor=2)
        rhs_t1 -= ndot('mnae,nmie->ia',t2,self.TEI[o,o,o,v],prefactor=-1)
        return rhs_t1

    def residual_t2(self, F,t1,t2):
        o = self.o
        v = self.v

        Fmi = self.build_Fmi(F,t1,t2)
        Fae = self.build_Fae(F,t1,t2)
        Fme = self.build_Fme(F,t1)

        tau     = self.build_tau(t1,t2)
        Wmnij   = self.build_Wmnij(t1,t2)
        Z       = self.build_Z(t1,t2)
        Wmbej   = self.build_Wmbej(t1,t2)
        Wmbje   = self.build_Wmbje(t1,t2)

        #### Build RHS side of t2 equations
        rhs_t2  = self.TEI[o,o,v,v].copy() + 1j*0.0

        rhs_t2 += ndot('mnab,mnij->ijab',tau,Wmnij)
        rhs_t2 += ndot('ijef,abef->ijab',tau,self.TEI[v,v,v,v])

        Pijab = -1.0 * ndot('ma,mbij->ijab',t1,Z)

        tmp  = Fae - ndot('mb,me->be',t1,Fme,prefactor=0.5)
        Pijab  += ndot('ijae,be->ijab',t2,tmp)

        tmp  = Fmi + ndot('je,me->mj',t1,Fme,prefactor=0.5)
        Pijab  -= ndot('imab,mj->ijab',t2,tmp)

        Pijab += ndot('imae,mbej->ijab',t2,Wmbej,prefactor=2)
        Pijab -= ndot('imea,mbej->ijab',t2,Wmbej)
        Pijab += ndot('imae,mbje->ijab',t2,Wmbje)
        Pijab += ndot('mjae,mbie->ijab',t2,Wmbje)
    
        tmp = ndot('ie,ma->imea',t1,t1)
        Pijab -= ndot('imea,mbej->ijab',tmp,self.TEI[o,v,v,o])
        Pijab -= ndot('imeb,maje->ijab',tmp,self.TEI[o,v,o,v])

        Pijab += ndot('ie,abej->ijab',t1,self.TEI[v,v,v,o])
        Pijab -= ndot('ma,mbij->ijab',t1,self.TEI[o,v,o,o])

        rhs_t2 += Pijab
        rhs_t2 += Pijab.swapaxes(0,1).swapaxes(2,3)
        return rhs_t2

    # Update amplitudes
    def update(self, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        Fmi = self.build_Fmi(F,t1,t2)
        Fae = self.build_Fae(F,t1,t2)
        Fme = self.build_Fme(F,t1)

        #### Build RHS side of t1 equations
        rhs_t1  = F[o,v].copy() + 1j*0.0
        rhs_t1 += ndot('ie,ae->ia',t1,Fae)
        rhs_t1 -= ndot('ma,mi->ia',t1,Fmi)
        rhs_t1 += ndot('imae,me->ia',t2,Fme,prefactor=2)
        rhs_t1 += ndot('imea,me->ia',t2,Fme,prefactor=-1)

        rhs_t1 += ndot('nf,nafi->ia',t1,self.TEI[o,v,v,o],prefactor=2)
        rhs_t1 += ndot('nf,naif->ia',t1,self.TEI[o,v,o,v],prefactor=-1)

        rhs_t1 += ndot('imef,amef->ia',t2,self.TEI[v,o,v,v],prefactor=2)
        rhs_t1 += ndot('imfe,amef->ia',t2,self.TEI[v,o,v,v],prefactor=-1)

        rhs_t1 -= ndot('mnae,mnie->ia',t2,self.TEI[o,o,o,v],prefactor=2)
        rhs_t1 -= ndot('mnae,nmie->ia',t2,self.TEI[o,o,o,v],prefactor=-1)

        tau     = self.build_tau(t1,t2)
        Wmnij   = self.build_Wmnij(t1,t2)
        Z       = self.build_Z(t1,t2)
        Wmbej   = self.build_Wmbej(t1,t2)
        Wmbje   = self.build_Wmbje(t1,t2)

        #### Build RHS side of t2 equations
        rhs_t2  = self.TEI[o,o,v,v].copy() + 1j*0.0

        rhs_t2 += ndot('mnab,mnij->ijab',tau,Wmnij)
        rhs_t2 += ndot('ijef,abef->ijab',tau,self.TEI[v,v,v,v])

        Pijab = -1.0 * ndot('ma,mbij->ijab',t1,Z)

        tmp  = Fae - ndot('mb,me->be',t1,Fme,prefactor=0.5)
        Pijab  += ndot('ijae,be->ijab',t2,tmp)

        tmp  = Fmi + ndot('je,me->mj',t1,Fme,prefactor=0.5)
        Pijab  -= ndot('imab,mj->ijab',t2,tmp)

        Pijab += ndot('imae,mbej->ijab',t2,Wmbej,prefactor=2)
        Pijab -= ndot('imea,mbej->ijab',t2,Wmbej)
        Pijab += ndot('imae,mbje->ijab',t2,Wmbje)
        Pijab += ndot('mjae,mbie->ijab',t2,Wmbje)
    
        tmp = ndot('ie,ma->imea',t1,t1)
        Pijab -= ndot('imea,mbej->ijab',tmp,self.TEI[o,v,v,o])
        Pijab -= ndot('imeb,maje->ijab',tmp,self.TEI[o,v,o,v])

        Pijab += ndot('ie,abej->ijab',t1,self.TEI[v,v,v,o])
        Pijab -= ndot('ma,mbij->ijab',t1,self.TEI[o,v,o,o])

        rhs_t2 += Pijab
        rhs_t2 += Pijab.swapaxes(0,1).swapaxes(2,3)

        self.t1 += rhs_t1*self.Dia
        self.t2 += rhs_t2*self.Dijab
        return


    def compute_corr_energy(self, F=None,t1=None,t2=None):
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        e  = ndot('ia,ia->',F[o,v],t1)
        e += ndot('ijab,abij->',self.build_tau(t1,t2),self.TEI[v,v,o,o],prefactor=2)
        e += ndot('ijab,abji->',self.build_tau(t1,t2),self.TEI[v,v,o,o],prefactor=-1)
        self.e_ccsd = e.real
        self.E_ccsd = e.real + self.e_scf
        return e
    
    def print_amplitudes(self):
        t1 = self.t1.real.copy()
        # unpack tensor, remove zeros, sort and select top 10
        t = np.ravel(t1)
        t = sorted(t[np.nonzero(t)],key=abs,reverse=True)
        if len(t) > 10:
            t = t[:10]
        # Disentangle degenerate amplitudes
        indeces = []
        amplitudes = []
        for amplitude in t:
            index = np.argwhere(t1==amplitude)
            if index.shape[0]==1:
                indeces.append(index[0])
                amplitudes.append(amplitude)
            else:
                n = index.shape[0]
                for i in range(n):
                    if not next((True for elem in indeces if elem.size == index[i].size and np.allclose(elem, index[i])), False):
                        indeces.append(index[i])
                        amplitudes.append(amplitude)
        Print(green+'Largest t(I,A) amplitudes'+end)
        for i in range(len(amplitudes)):
            index = indeces[i]
            amplitude = amplitudes[i]
            Print(cyan+'\t{:2d}{:2d}{:>24.10f}'.format(index[0]+1,index[1]+1,amplitude)+end)

        t2 = self.t2.real.copy()
        # unpack tensor, remove zeros, sort and select top 10
        t = np.ravel(t2)
        t = sorted(t[np.nonzero(t)],key=abs,reverse=True)
        if len(t) > 10:
            t = t[:10]
        # Disentangle degenerate amplitudes
        indeces = []
        amplitudes = []
        for amplitude in t:
            index = np.argwhere(t2==amplitude)
            if index.shape[0]==1:
                indeces.append(index[0])
                amplitudes.append(amplitude)
            else:
                n = index.shape[0]
                for i in range(n):
                    if not next((True for elem in indeces if elem.size == index[i].size and np.allclose(elem, index[i])), False):
                        indeces.append(index[i])
                        amplitudes.append(amplitude)
        Print(green+'Largest t(I,j,A,b) amplitudes'+end)
        for i in range(len(amplitudes)):
            index = indeces[i]
            amplitude = amplitudes[i]
            Print(cyan+'\t{:2d}{:2d}{:2d}{:2d}{:>20.10f}'.format(index[0]+1,index[1]+1,index[2]+1,index[3]+1,amplitude)+end)

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
        diis_object = helper_diis(self.t1, self.t2, max_diis)

        # Iterate
        for iter in range(1,maxiter+1):
            t1_old = self.t1.copy()
            t2_old = self.t2.copy()
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

                self.print_amplitudes()
                self.t1 = np.around(self.t1,decimals=10)
                self.t2 = np.around(self.t2,decimals=10)

                return

            #  Add the new error vector
            diis_object.add_error_vector(self.t1, self.t2)

            if iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1, self.t2)

            # End CCSD iterations
    # End CCEnergy class


class CCLambda(object):
    def __init__(self,ccsd):
        Print(yellow+"\nInitializing Lambda object..."+end)

##------------------------------------------------------
##  CCSD Lambda equations
##------------------------------------------------------
##
##  The equations follow reference 1, but are spin integrated using the unitary group approach.
##
##  i,j -> target occupied indeces
##  a,b -> target virtual indeces
##
##  m,n,o -> implicit (summed-over) occupied indeces
##  e,f,g -> implicit (summed-over) virtual indeces
##
##  G: effective 1-particle intermediate
##  W,Tau: effective 2 particle intermediates
##
##
##  Because equations are spin integrated, we only compute the necessary spin combinations of amplitudes:
##      l1  -> alpha block      -> l1 (ai)
##      l2  -> alpha/beta block -> l2 (aBiJ)
##

        # Start timer
        time_init = time.time()

        # Read relevant data from ccsd class
        self.n_occ  = ccsd.n_occ
        self.n_virt = ccsd.n_virt
        self.o      = ccsd.o
        self.v      = ccsd.v

        self.TEI    = ccsd.TEI
        #self.Dia    = ccsd.Dia.swapaxes(0,1)
        #self.Dijab  = ccsd.Dijab.swapaxes(0,2).swapaxes(1,3)
        self.Dia    = ccsd.Dia
        self.Dijab  = ccsd.Dijab
        self.t1     = ccsd.t1
        self.t2     = ccsd.t2
        self.F      = ccsd.F

        # Initialize l1 and l2 to the transpose of t1 and t2 respectively
        #self.l1     = self.t1.swapaxes(0,1).copy()
        #self.l2     = self.t2.swapaxes(0,2).swapaxes(1,3).copy()
        self.l1     = 2*self.t1.copy()
        self.l2     = 4*self.t2.copy()
        self.l2    -= 2*self.t2.swapaxes(2,3)

        # Build intermediates independent of Lambda
        #   the following Hbar elements are similar to CCSD intermediates; in spin orpbitals, they are easily obtained from the CCSD intermediates directly
        #   When spin integrated, it's easier to recompute them from scratch.
        self.Fme = ccsd.build_Fme()
        self.Fmi = self.build_Fmi(ccsd)
        self.Fae = self.build_Fae(ccsd)

        self.Wmnij = ccsd.build_Wmnij()
        self.Wabef = self.build_Wabef(ccsd)
        self.Wmbej = self.build_Wmbej(ccsd)
        self.Wmbje = self.build_Wmbje(ccsd)

        #   the following Hbar elements have to be computed from scratch as they don't correspond to transformed CCSD intermediates.
        self.Wmnie = self.build_Wmnie(ccsd)
        self.Wamef = self.build_Wamef(ccsd)
        self.Wmbij = self.build_Wmbij(ccsd)
        self.Wabei = self.build_Wabei(ccsd)

        #np.set_printoptions(precision=12,linewidth=200,suppress=True)


    # Build the Hbar intermediates
    def build_Fmi(self,ccsd, F=None,t1=None,t2=None):       #   < m | Hbar | i >
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v
        
        Fmi  = F[o,o].copy() + 1j*0.0
        Fmi += ndot('ie,me->mi',t1,F[o,v])
        Fmi += ndot('ne,mnie->mi',t1,self.TEI[o,o,o,v],prefactor=2)
        Fmi += ndot('ne,nmie->mi',t1,self.TEI[o,o,o,v],prefactor=-1)
        Fmi += ndot('inef,mnef->mi',ccsd.build_tau(t1,t2),self.TEI[o,o,v,v],prefactor=2)
        Fmi += ndot('inef,mnfe->mi',ccsd.build_tau(t1,t2),self.TEI[o,o,v,v],prefactor=-1)
        return Fmi      # Fmi

    def build_Fae(self,ccsd, F=None,t1=None,t2=None):       #   < a | Hbar | e >
        if F  is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        Fae  = F[v,v].copy() + 1j*0.0
        Fae -= ndot('ma,me->ae',t1,F[o,v])
        Fae += ndot('mf,amef->ae',t1,self.TEI[v,o,v,v],prefactor=2)
        Fae += ndot('mf,amfe->ae',t1,self.TEI[v,o,v,v],prefactor=-1)
        Fae -= ndot('mnfa,mnfe->ae',ccsd.build_tau(t1,t2),self.TEI[o,o,v,v],prefactor=2)
        Fae -= ndot('mnfa,mnef->ae',ccsd.build_tau(t1,t2),self.TEI[o,o,v,v],prefactor=-1)
        return Fae      # Fae

    def build_Wabef(self,ccsd, t1=None,t2=None):            #   < ab | Hbar | ef >
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)

        W = self.TEI[v,v,v,v].copy() + 1j*0.0
        W += ndot('mnab,mnef->abef',tau,self.TEI[o,o,v,v])
        W -= ndot('mb,amef->abef',t1,self.TEI[v,o,v,v])
        W -= ndot('ma,bmfe->abef',t1,self.TEI[v,o,v,v])
        return W        # Wabef

    def build_Wmbej(self,ccsd, t1=None,t2=None):            #   < mb | Hbar | ej >
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)

        W  = self.TEI[o,v,v,o].copy() + 1j*0.0
        W += ndot('jnbf,nmfe->mbej',t2,self.TEI[o,o,v,v],prefactor=2)
        W += ndot('jnbf,nmef->mbej',t2,self.TEI[o,o,v,v],prefactor=-1)
        W -= ndot('jnfb,nmfe->mbej',tau,self.TEI[o,o,v,v])
        W += ndot('jf,mbef->mbej',t1,self.TEI[o,v,v,v])
        W -= ndot('nb,mnej->mbej',t1,self.TEI[o,o,v,o])
        return W        # Wmbej

    def build_Wmbje(self,ccsd, t1=None,t2=None):            #   < mb | Hbar | je >
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)

        W  = self.TEI[o,v,o,v].copy() + 1j*0.0
        W -= ndot('jnfb,nmef->mbje',tau,self.TEI[o,o,v,v])
        W += ndot('jf,bmef->mbje',t1,self.TEI[v,o,v,v])
        W -= ndot('nb,mnje->mbje',t1,self.TEI[o,o,o,v])
        return W        # Wmbje


    def build_Wmnie(self,ccsd, t1=None,t2=None):            #   < mn | Hbar | ie >
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W  = self.TEI[o,o,o,v].copy() + 1j*0.0
        W += ndot('if,mnfe->mnie',t1,self.TEI[o,o,v,v])
        return W        # Wmnie

    def build_Wamef(self,ccsd, t1=None,t2=None):            #   < am | Hbar | ef >
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        W  = self.TEI[v,o,v,v].copy() + 1j*0.0
        W -= ndot('na,nmef->amef',t1,self.TEI[o,o,v,v])
        return W        # Wamef

    def build_Wmbij(self,ccsd, F=None,t1=None,t2=None):     #   < mb | Hbar | ij >
        if F is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)

        W  = self.TEI[o,v,o,o].copy() + 1j*0.0
        W += ndot('mnfe,if->iemn',t2,self.Fme)
        W -= ndot('oe,iomn->iemn',t1,self.Wmnij)
        W += ndot('mnfg,iefg->iemn',tau,self.TEI[o,v,v,v])

        W += ndot('noef,iomf->iemn',t2,self.TEI[o,o,o,v],prefactor=2)
        W += ndot('noef,oimf->iemn',t2,self.TEI[o,o,o,v],prefactor=-1)
        W -= ndot('nofe,iomf->iemn',t2,self.TEI[o,o,o,v])
        W -= ndot('mofe,iofn->iemn',t2,self.TEI[o,o,v,o])

        W += ndot('mf,iefn->iemn',t1,self.TEI[o,v,v,o])
        W += ndot('nf,iemf->iemn',t1,self.TEI[o,v,o,v])

        tmp = ndot('njbf,mnef->mbej',t2,self.TEI[o,o,v,v])
        W -= ndot('mf,iefn->iemn',t1,tmp)
        tmp = ndot('njbf,mnfe->mbje',t2,self.TEI[o,o,v,v])
        W -= ndot('nf,iemf->iemn',t1,tmp)
        tmp  = ndot('njfb,mnef->mbej',t2,self.TEI[o,o,v,v],prefactor=2)
        tmp += ndot('njfb,mnfe->mbej',t2,self.TEI[o,o,v,v],prefactor=-1)
        W += ndot('mf,iefn->iemn',t1,tmp)
        return W        # Wmbij

    def build_Wabei(self,ccsd, F=None,t1=None,t2=None):     #   < ab | Hbar | ei >
        if F is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2

        o = self.o
        v = self.v

        tau = ccsd.build_tau(t1,t2)

        W  = self.TEI[v,v,v,o].copy() + 1j*0.0
        W -= ndot('mnfe,na->efam',t2,self.Fme)
        W += ndot('mg,efag->efam',t1,self.Wabef)
        W += ndot('noef,amno->efam',tau,self.TEI[v,o,o,o])

        W += ndot('mnfg,enag->efam',t2,self.TEI[v,o,v,v],prefactor=2)
        W += ndot('mnfg,enga->efam',t2,self.TEI[v,o,v,v],prefactor=-1)
        W -= ndot('mngf,enag->efam',t2,self.TEI[v,o,v,v])
        W -= ndot('mnge,nfag->efam',t2,self.TEI[o,v,v,v])

        W -= ndot('ne,nfam->efam',t1,self.TEI[o,v,v,o])
        W -= ndot('nf,enam->efam',t1,self.TEI[v,o,v,o])

        tmp = ndot('njbf,mnef->mbej',t2,self.TEI[o,o,v,v])
        W += ndot('ne,nfam->efam',t1,tmp)
        tmp = ndot('njbf,nmef->bmej',t2,self.TEI[o,o,v,v])
        W += ndot('nf,enam->efam',t1,tmp)
        tmp  = ndot('njfb,mnef->mbej',t2,self.TEI[o,o,v,v],prefactor=2)
        tmp += ndot('njfb,mnfe->mbej',t2,self.TEI[o,o,v,v],prefactor=-1)
        W -= ndot('ne,nfam->efam',t1,tmp)
        return W        # Wabei


    # Build Lambda intermediates
    def build_Gmi(self, t2=None,l2=None):
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2

        Gmi = ndot('mnef,inef->mi',t2,l2)
        return Gmi

    def build_Gae(self, t2=None,l2=None):
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2

        Gae = - ndot('mnef,mnaf->ae',t2,l2)
        return Gae


    # Compute L1 residual
    def residual_l1(self,ccsd, F,t1,t2,l1,l2):
        o = self.o
        v = self.v

        rhs_l1  = self.Fov.swapaxes(0,1).copy()

        return rhs_l1

    def update(self, F=None,t1=None,t2=None,l1=None,l2=None):
        if F is None: F = self.F
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        o = self.o
        v = self.v

        Gmi = self.build_Gmi(t2,l2)
        Gae = self.build_Gae(t2,l2)
        
        #### Build RHS side of l1 equations
        rhs_l1  = 2*self.Fme.copy()

        rhs_l1 -= ndot('ma,im->ia',l1,self.Fmi)
        rhs_l1 += ndot('ie,ea->ia',l1,self.Fae)

        rhs_l1 += ndot('me,ieam->ia',l1,self.Wmbej,prefactor=2)
        rhs_l1 += ndot('me,iema->ia',l1,self.Wmbje,prefactor=-1)

        rhs_l1 -= ndot('mnae,iemn->ia',l2,self.Wmbij)
        rhs_l1 += ndot('imef,efam->ia',l2,self.Wabei)

        rhs_l1 -= ndot('mn,mina->ia',Gmi,self.Wmnie,prefactor=2)
        rhs_l1 -= ndot('mn,imna->ia',Gmi,self.Wmnie,prefactor=-1)
        rhs_l1 -= ndot('ef,eifa->ia',Gae,self.Wamef,prefactor=2)
        rhs_l1 -= ndot('ef,eiaf->ia',Gae,self.Wamef,prefactor=-1)


        #### Build RHS side of l2 equations
        rhs_l2  = 2*self.TEI[o,o,v,v] - self.TEI[o,o,v,v].swapaxes(2,3) + 1j*0.0

        rhs_l2 += ndot('ia,jb->ijab',l1,self.Fme,prefactor=2)
        rhs_l2 += ndot('ja,ib->ijab',l1,self.Fme,prefactor=-1)

        rhs_l2 -= ndot('mjab,im->ijab',l2,self.Fmi)
        rhs_l2 += ndot('ijeb,ea->ijab',l2,self.Fae)

        rhs_l2 += ndot('mnab,ijmn->ijab',l2,self.Wmnij,prefactor=0.5)
        rhs_l2 += ndot('ijef,efab->ijab',l2,self.Wabef,prefactor=0.5)

        rhs_l2 += ndot('ie,ejab->ijab',l1,self.Wamef,prefactor=2)
        rhs_l2 += ndot('ie,ejba->ijab',l1,self.Wamef,prefactor=-1)

        rhs_l2 -= ndot('mb,jima->ijab',l1,self.Wmnie,prefactor=2)
        rhs_l2 -= ndot('mb,ijma->ijab',l1,self.Wmnie,prefactor=-1)

        rhs_l2 += ndot('mjeb,ieam->ijab',l2,self.Wmbej,prefactor=2)
        rhs_l2 += ndot('mjeb,iema->ijab',l2,self.Wmbje,prefactor=-1)
        rhs_l2 -= ndot('mieb,jeam->ijab',l2,self.Wmbej)
        rhs_l2 -= ndot('mibe,jema->ijab',l2,self.Wmbje)

        rhs_l2 -= ndot('mi,mjab->ijab',Gmi,self.TEI[o,o,v,v],prefactor=2)
        rhs_l2 -= ndot('mi,mjba->ijab',Gmi,self.TEI[o,o,v,v],prefactor=-1)
        rhs_l2 += ndot('ae,ijeb->ijab',Gae,self.TEI[o,o,v,v],prefactor=2)
        rhs_l2 += ndot('ae,ijbe->ijab',Gae,self.TEI[o,o,v,v],prefactor=-1)

        self.l1 += rhs_l1*self.Dia

        tmp  = rhs_l2
        tmp += rhs_l2.swapaxes(0,1).swapaxes(2,3)
        self.l2 += tmp*self.Dijab

        return

    def compute_pseudoenergy(self):
        o = self.o
        v = self.v
        
        e = ndot('ijab,ijab->',self.l2,self.TEI[o,o,v,v],prefactor=0.5).real
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
        np.set_printoptions(precision=12, linewidth=200, suppress=True)

        # Read relevant data
        self.n_occ  = ccsd.n_occ
        self.n_virt = ccsd.n_virt
        self.o      = ccsd.o
        self.v      = ccsd.v

        self.ints   = ccsd.mints.ao_dipole()

        self.P      = ccsd.P
        self.npC    = ccsd.npC

        self.t1     = ccsd.t1.real
        self.t2     = ccsd.t2.real

        self.l1     = Lambda.l1.real     # holds 2*lambda_a,i
        self.l2     = Lambda.l2.real     # holds 2*(2*lambda_ab,ij - lambda_ab,ji)

        D = self.compute_ccsd_density(self.t1,self.t2,self.l1,self.l2)
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

        dipoles = self.compute_ccsd_dipole(D)
        Print(blue+'Dipole moment computed at the CCSD level'+end)
        Print(blue+'\tNuclear component (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles_nuc[0],'Y:',dipoles_nuc[1],'Z:',dipoles_nuc[2])+end)
        Print(blue+'\tElectronic component (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles[0],'Y:',dipoles[1],'Z:',dipoles[2])+end)
        self.mu = dipoles.copy()
        dipoles = np.asarray([dipoles[i] + dipoles_nuc[i] for i in range(3)])
        Print(green+'\tTotal electric dipole (a.u.)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}' .format('X:',dipoles[0],'Y:',dipoles[1],'Z:',dipoles[2])+end)
        dipoles *= 1/0.393456
        Print(green+'\tTotal electric dipole (Debye)'+end)
        Print(cyan+'\t{:>6s}{:10.5f}{:>6s}{:10.5f}{:>6s}{:10.5f}\n' .format('X:',dipoles[0],'Y:',dipoles[1],'Z:',dipoles[2])+end)


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
            mu  = ndot('uv,vu->',np.asarray(self.ints[n]),self.P)
            dipoles_elec.append(mu)
        return dipoles_elec

    def compute_ccsd_dipole(self,D):
        # Compute CCSD correlated dipole
        dipoles_elec = []
        for n in range(3):
            d = contract('ui,uv,vj->ij',self.npC,np.asarray(self.ints[n]),self.npC)
            mu = ndot('ij,ji->',d,D)
            dipoles_elec.append(mu)
        return dipoles_elec
