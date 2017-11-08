#
# @BEGIN LICENSE
#
# rtcc by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2017 The Psi4 Developers.
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

import psi4
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util
from copy import deepcopy
import numpy as np
from opt_einsum import contract

def run_rtcc(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    rtcc can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('rtcc')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    psi4.core.set_local_option('MYPLUGIN', 'PRINT', 1)
    psi4.core.set_local_option('TRANSQT2','WFN','CCSD')
    psi4.core.set_local_option('CCTRANSORT','WFN','CCSD')
    psi4.core.set_local_option('CCENERGY','WFN','CCSD')
    psi4.core.set_local_option('CCSORT','WFN','CCSD')

    # Compute a SCF reference, a wavefunction is returned which holds the molecule used, orbitals
    # Fock matrices, and more
    print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Ensure IWL files have been written when not using DF/CD
    proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    if psi4.core.get_global_option('RUN_CCTRANSORT'):
        psi4.core.cctransort(ref_wfn)
    else:
        psi4.core.transqt2(ref_wfn)
        psi4.core.ccsort()

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
#    rtcc_wfn = psi4.core.ccenergy(ref_wfn)

#    return rtcc_wfn
    pass


# Integration with driver routines
psi4.driver.procedures['energy']['rtcc'] = run_rtcc


def ccdiags(calc):
    # Written by Alexandre Bazante, 2017
    # This is NOT an efficient code! Each diagram is implemented individually for debug purposes
    # This assumes RHF reference (for now)
    bold        = '\033[1m'
    underline   = '\033[4m'
    blue        = '\033[94m'
    green       = '\033[92m'
    cyan        = '\033[96m'
    end         = '\033[0m'

    print_data  = False
    # Set up the logic to compute subsets of T1 and T2 equations and intermediates
    if calc=='ccsd':
        ccsd    = True
        ccd     = True
        lccsd   = True
        lccd    = True
    elif calc=='ccd':
        ccsd    = False
        ccd     = True
        lccsd   = False
        lccd    = True
    elif calc=='lccsd':
        ccsd    = False
        ccd     = False
        lccsd   = True
        lccd    = True
    elif calc=='lccd':
        ccsd    = False
        ccd     = False
        lccsd   = False
        lccd    = True
    else:
        ccsd    = True
        ccd     = True
        lccsd   = True
        lccd    = True


    print(blue+'\nCCSD Program'+end)
    print(blue+'-- Written by Alexandre P. Bazante, 2017\n'+end)

    # Read molecule data
    mol     = psi4.core.get_active_molecule()
    N_atom  = mol.natom()
    N_e     = int(sum(mol.Z(A) for A in range(N_atom))-mol.molecular_charge())
    n_occ   = int(N_e / 2)

    #Read basis set
    #Much of the basis construction is being moved to python side, for now it's convenient to read it after psi4 runs an scf
    e_scf,wfn   = psi4.energy('scf', return_wfn = True) # This makes psi4 run the scf calculation
    print(blue+'The SCF energy is'+end)
    print(cyan+'\t%s\n'%e_scf+end)

    basis   = wfn.basisset()
    n_ao    = psi4.core.BasisSet.nao(basis)

    n_virt  = n_ao - n_occ

    # Read SCF data
    mints   = psi4.core.MintsHelper(wfn.basisset())
    TEI_ao  = np.asarray(mints.ao_eri())
    S       = np.asarray(mints.ao_overlap())
    V       = np.asarray(mints.ao_potential())
    T       = np.asarray(mints.ao_kinetic())
    H       = V + T

    if(mol.schoenflies_symbol()=='c1'):
        C   = wfn.Ca()
    else:
        C   = wfn.Ca_subset('AO','ALL')
    TEI     = np.asarray(mints.mo_eri(C,C,C,C))
    TEI     = TEI.swapaxes(1,2) # change indexing i,a,j,b -> i,j,a,b
    C       = np.asarray(C)

    def form_ao_density(C,N,n): # I'm sure this can be done using contract, will think about later
        P = np.zeros((N,N))
        for u in range(N):
            for v in range(N):
                for i in range(n):
                    P[u,v] = P[u,v] + 2*C[u,i]*C[v,i]
        return(P)

    def form_Fock_matrix(H,V,P):
        F   = H
        F   += contract('ls,uvls->uv',P,V)
        F   -= 0.5*contract('ls,ulvs->uv',P,V)
        return(F)

    P       = form_ao_density(C,n_ao,n_occ)
    F       = form_Fock_matrix(H,TEI_ao,P)

    #Update Fock matrix to MO basis
    F = contract('vi,uv,uj->ij',C,F,C)
    e,evcs = np.linalg.eigh(F)

    ## MAIN PROGRAM ##

    o = slice(n_occ)
    v = slice(n_occ,n_ao)
    Dia     = 1/(e[o].reshape(-1,1)-e[v])
    Dijab   = 1/(e[o].reshape(-1,1,1,1)+e[o].reshape(-1,1,1)-e[v].reshape(-1,1)-e[v])

    #T1 GUESS
    t1old = np.zeros((n_occ,n_virt))
    #T2 GUESS
    t2old_ab = TEI[o,o,v,v].copy()
    t2old_ab *= 0.5
    t2old_ab = t2old_ab + t2old_ab.swapaxes(0,1).swapaxes(2,3)
    #DENOMINATOR
    t2old_ab = contract('ijab,ijab->ijab',t2old_ab,Dijab)
    #ENERGY
    tmp = 2*TEI[o,o,v,v] - TEI.swapaxes(2,3)[o,o,v,v]
    e_corr = contract('ijab,ijab->',t2old_ab,tmp)

    print(blue+'The MBPT(2) correlation energy is'+end)
    print(cyan+'\t%s \n' %e_corr+end)

    print('\t Summary of iterative solution of the CC equations')
    print('\t------------------------------------------------------')
    print('\t\t\t Correlation')
    print('\t  Iteration \t Energy')
    print('\t------------------------------------------------------')
    print('\t  %s \t\t %s' %(0,e_corr))

    e_conv = 1e-14
    maxiter = 100
    for iter in range(0,maxiter):
        e_old = e_corr
        t2old_aa = t2old_ab - t2old_ab.swapaxes(2,3)

        t1new = np.zeros((n_occ,n_virt))
        #LCCSD
        if lccsd:
            VS = TEI[v,v,v,o] - TEI.swapaxes(2,3)[v,v,v,o]
            t1new += 0.5*contract('ikdc,dcak->ia',t2old_aa,VS)+contract('ikdc,dcak->ia',t2old_ab,TEI[v,v,v,o])
            VS = TEI[o,v,o,o] - TEI.swapaxes(2,3)[o,v,o,o]
            t1new -= 0.5*contract('lkac,iclk->ia',t2old_aa,VS)+contract('lkac,iclk->ia',t2old_ab,TEI[o,v,o,o])
            V = 2*TEI[v,o,o,v]-TEI.swapaxes(2,3)[v,o,o,v]
            t1new += contract('kc,cika->ia',t1old,V)
        #CCSD
        if ccsd:
            V = 2*TEI[v,v,o,v] - TEI.swapaxes(2,3)[v,v,o,v]
            t1new += contract('kc,id,cdka->ia',t1old,t1old,V)
            V = 2*TEI[v,o,o,o] - TEI.swapaxes(2,3)[v,o,o,o]
            t1new -= contract('kc,la,cikl->ia',t1old,t1old,V)

            V = 2*TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
            t1new -= contract('kc,id,la,cdkl->ia',t1old,t1old,t1old,V)

            T = t2old_aa + t2old_ab
            t1new += contract('kc,ilad,cdkl->ia',t1old,T,V)
            VS = TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
            tmp = contract('kicd,cdkl->il',t2old_ab,TEI[v,v,o,o])+0.5*contract('kicd,cdkl->il',t2old_aa,VS)
            t1new -= contract('il,la->ia',tmp,t1old)
            tmp = contract('klad,cdkl->ca',t2old_ab,TEI[v,v,o,o])+0.5*contract('klad,cdkl->ca',t2old_aa,VS)
            t1new -= contract('ca,ic->ia',tmp,t1old)

        #LCCD
        t2new  = TEI[o,o,v,v].copy()
        t2new *= 0.5
        t2new -= contract('ikcb,cjak->ijab',t2old_ab,TEI[v,o,v,o])
        VS = TEI[v,o,o,v]-TEI.swapaxes(2,3)[v,o,o,v]
        t2new += contract('ikac,cjkb->ijab',t2old_aa,TEI[v,o,o,v]) + contract('ikac,cjkb->ijab',t2old_ab,VS)
        t2new = t2new + t2new.swapaxes(0,1).swapaxes(2,3)
        t2new += contract('ijcd,cdab->ijab',t2old_ab,TEI[v,v,v,v])
        t2new += contract('klab,ijkl->ijab',t2old_ab,TEI[o,o,o,o])
        #LCCSD
        if lccsd:
            tmp = contract('ic,cjab->ijab',t1old,TEI[v,o,v,v])
            tmp -= contract('ka,ijkb->ijab',t1old,TEI[o,o,o,v])
            t2new += tmp + tmp.swapaxes(0,1).swapaxes(2,3)
        #CCD
        if ccd:
            t2new += contract('ijcd,klab,cdkl->ijab',t2old_ab,t2old_ab,TEI[v,v,o,o])
            VS = TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
            tmp = - contract('ilab,jkdc,cdkl->ijab',t2old_ab,t2old_ab,TEI[v,v,o,o])
            tmp -= 0.5*contract('ilab,jkdc,cdkl->ijab',t2old_ab,t2old_aa,VS)
            tmp -= contract('ijad,lkbc,cdkl->ijab',t2old_ab,t2old_ab,TEI[v,v,o,o])
            tmp -= 0.5*contract('ijad,lkbc,cdkl->ijab',t2old_ab,t2old_aa,VS)
            tmp += 0.5*contract('ikac,ljdb,cdkl->ijab',t2old_aa,t2old_ab,VS)
            tmp += 0.5*contract('ikac,ljdb,cdkl->ijab',t2old_aa,t2old_aa,TEI[v,v,o,o])
            tmp += 0.5*contract('ikac,ljdb,cdkl->ijab',t2old_ab,t2old_ab,TEI[v,v,o,o])
            tmp += 0.5*contract('ikac,ljdb,cdkl->ijab',t2old_ab,t2old_aa,VS)
            tmp += 0.5*contract('kjac,ildb,dckl->ijab',t2old_ab,t2old_ab,TEI[v,v,o,o])
            t2new += tmp + tmp.swapaxes(0,1).swapaxes(2,3)
        #CCSD
        if ccsd:
            tmp = 0.5*contract('ic,jd,cdab->ijab',t1old,t1old,TEI[v,v,v,v])
            tmp += 0.5* contract('ka,lb,ijkl->ijab',t1old,t1old,TEI[o,o,o,o])
            tmp -= contract('ic,kb,cjak->ijab',t1old,t1old,TEI[v,o,v,o])
            tmp -= contract('ic,ka,cjkb->ijab',t1old,t1old,TEI[v,o,o,v])

            tmp -= contract('ic,ka,jd,cdkb->ijab',t1old,t1old,t1old,TEI[v,v,o,v])
            tmp += contract('ic,ka,lb,cjkl->ijab',t1old,t1old,t1old,TEI[v,o,o,o])

            tmp += contract('ic,klab,cjkl->ijab',t1old,t2old_ab,TEI[v,o,o,o])
            tmp -= contract('ka,ijcd,cdkb->ijab',t1old,t2old_ab,TEI[v,v,o,v])
            tmp -= contract('ic,kjad,cdkb->ijab',t1old,t2old_ab,TEI[v,v,o,v])
            tmp += contract('ka,ilcb,cjkl->ijab',t1old,t2old_ab,TEI[v,o,o,o])
            V = 2*TEI[v,v,o,v] - TEI.swapaxes(2,3)[v,v,o,v]
            tmp += contract('kc,ijdb,cdka->ijab',t1old,t2old_ab,V)
            V = 2*TEI[v,o,o,o] - TEI.swapaxes(2,3)[v,o,o,o]
            tmp -= contract('kc,ljab,cikl->ijab',t1old,t2old_ab,V)
            VS = TEI[v,v,v,o] - TEI.swapaxes(2,3)[v,v,v,o]
            tmp += contract('ic,kjdb,cdak->ijab',t1old,t2old_ab,VS) + contract('ic,kjdb,cdak->ijab',t1old,t2old_aa,TEI[v,v,v,o])
            VS = TEI[o,v,o,o] - TEI.swapaxes(2,3)[o,v,o,o]
            tmp -= contract('ka,ljcb,ickl->ijab',t1old,t2old_ab,VS) + contract('ka,ljcb,ickl->ijab',t1old,t2old_aa,TEI[o,v,o,o])

            tmp += 0.5*contract('ka,ijcd,lb,cdkl->ijab',t1old,t2old_ab,t1old,TEI[v,v,o,o])
            tmp += 0.5*contract('ic,klab,jd,cdkl->ijab',t1old,t2old_ab,t1old,TEI[v,v,o,o])
            tmp += contract('ic,kjad,lb,cdkl->ijab',t1old,t2old_ab,t1old,TEI[v,v,o,o])

            tmp += 0.5*contract('ic,ka,jd,lb,cdkl->ijab',t1old,t1old,t1old,t1old,TEI[v,v,o,o])

            VS = TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
            V = 2*TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
            tmp1 = contract('ljdb,cdkl->cjkb',t2old_ab,VS) + contract('ljdb,cdkl->cjkb',t2old_aa,TEI[v,v,o,o])
            tmp -= contract('ic,ka,cjkb->ijab',t1old,t1old,tmp1)
            tmp -= contract('kc,id,ljab,cdkl->ijab',t1old,t1old,t2old_ab,V)
            tmp -= contract('kc,la,ijdb,cdkl->ijab',t1old,t1old,t2old_ab,V)

            t2new += tmp + tmp.swapaxes(0,1).swapaxes(2,3)
        
        #DENOMINATOR
        t1new = contract('ia,ia->ia',t1new,Dia)
        t2new = contract('ijab,ijab->ijab',t2new,Dijab)

        #ENERGY
        tmp1    = 2*TEI[o,o,v,v] - TEI.swapaxes(2,3)[o,o,v,v]
        tau     = t2new + contract('ia,jb->ijab',t1new,t1new)
        if ccsd:
            e_corr   = contract('ijab,ijab->',tau,tmp1)
        else:
            e_corr   = contract('ijab,ijab->',t2new,tmp1)

        #UPDATE
        t1old = t1new.copy()
        t2old_ab = t2new.copy()

        print('\t  %s \t\t %s' %(iter+1,e_corr))
        if (abs(e_corr-e_old) < e_conv):
            break

    print('\t------------------------------------------------------')
    print('The CC equations have converged')
    print(blue+'The %s correlation energy is' %calc+end)
    print(cyan+'\t%s \n' %e_corr+end)
    pass
