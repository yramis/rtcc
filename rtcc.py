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


def rtccsd(calc):
    # Written by Alexandre Bazante, 2017
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

    print(blue+'\nTime Dependent CCSD Program'+end)
    print(blue+'-- Written by Alexandre P. Bazante, 2017\n'+end)

    # Read molecule data
    mol     = psi4.core.get_active_molecule()
    N_atom  = mol.natom()
    N_e     = int(sum(mol.Z(A) for A in range(N_atom))-mol.molecular_charge())
    n_occ   = int(N_e / 2)

    if(print_data):
        print(underline+'Reading Molecular Data'+end)
        print('\t# of atoms')
        print('\t%s \n' %N_atom)

        print('\t# of electrons')
        print('\t%s \n' %N_e)

        print('\t# of occupied orbitals')
        print('\t%s \n' %n_occ)

    #Read basis set
    #Much of the basis construction is being moved to python side, for now it's convenient to read it after psi4 runs an scf
    e_scf,wfn   = psi4.energy('scf', return_wfn = True) # This makes psi4 run the scf calculation
    print(blue+'The SCF energy is'+end)
    print(cyan+'\t%s\n'%e_scf+end)

    basis   = wfn.basisset()
    n_ao    = psi4.core.BasisSet.nao(basis)

    n_virt  = n_ao - n_occ

    if(print_data):
        print(underline+'Reading Basis Set Data'+end)
        print('\t# of AOs')
        print('\t%s \n' %n_ao)

        print('\t# of virtual orbitals')
        print('\t%s \n' %n_virt)

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

        # Build Fock matrix
        # F(mu,nu) = H(mu,nu) + ∑ P(lambda,sigma) {(mu nu|sigma lambda) - 0.5*(mu lambda| sigma nu)} , (sigma,lamda = 1 .. #AOs)
        # P(mu,nu) = ∑ C(mu,i)*C(nu,i) , (i = 1 .. #Occupied MOs)  ## Don't forget factor of 2 for RHF

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

    if(print_data):
        print(bold+'AO overlap matrix'+end)
        print(green+'%s \n' %S+end)

        print(bold+'AO -> MO Coefficients'+end)
        print(green+'%s \n' %C+end)

        print(bold+'Fock matrix'+end)
        print(green+'%s \n' %F+end)

        print(bold+'SCF energy'+end)
        print(green+'%s \n' %e_scf+end)

        print(bold+'Density matrix'+end)
        print(green+'%s \n' %P+end)

        print(bold+'TEI'+end)
        print(green+'%s \n' %TEI+end)

    #Alternartively, the fock and density matrices can be read from the wfn object
    #F      = np.asarray(wfn.Fa())
    #P      = 2*np.asarray(wfn.Da())
    #If symmetry is on, the above commands will return symmetry adapted blocks!
    # wfn.Fa().to_array[0] -> relevant block for irrep 1

    #Update Fock matrix to MO basis
    F = contract('vi,uv,uj->ij',C,F,C)
    e,evcs = np.linalg.eigh(F)

    # CCSD Procedures
    # All equations are spin integrated
    def t_denominator(t1new,t2new): # Computes the denominator for the T array and updates T
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        Dia     = 1/(e[o].reshape(-1,1)-e[v])
        Dijab   = 1/(e[o].reshape(-1,1,1,1)+e[o].reshape(-1,1,1)-e[v].reshape(-1,1)-e[v])
        t1      = contract('ia,ia->ia',t1new,Dia)
        t2_ab   = contract('ijab,ijab->ijab',t2new,Dijab)
        return(t1,t2_ab)

    def guess_rhf():    # Creates the MBPT(2) guess for T2
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        t1      = np.zeros((n_occ,n_virt))
        t2_ab   = TEI[o,o,v,v] + TEI.swapaxes(0,1).swapaxes(2,3)[o,o,v,v] # This isn't necessary, but helps enforce clean permutation symmetry
        t2_ab  *= 0.5
        t1,t2_ab= t_denominator(t1,t2_ab)
        return(t1,t2_ab)

    def energy_rhf():   # Computes the CCSD energy
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        tau     = t2_ab + contract('ia,jb->ijab',t1,t1)
        tmp1    = 2*TEI[o,o,v,v] - TEI.swapaxes(2,3)[o,o,v,v]
        if ccsd:
            E   = contract('ijab,ijab->',tau,tmp1)
        else:
            E   = contract('ijab,ijab->',t2_ab,tmp1)
        return(E)

    def antisymmetrize_T2():
        t2_aa = t2_ab - t2_ab.swapaxes(0,1)
        return(t2_aa)

    def tau():  # Computes the effective two-particle excitation operators tau and tau tilde
                # Tau is used in the T2 amplitude equations and in the 2 particle intermediates W
                # Tau tilde is used in the 1 particle intermediates F
        Tau_ab  = t2_ab.copy()
        TauP_ab = t2_ab.copy()
        TauP_aa = t2_aa.copy()
        if ccsd:
            tmp1    = contract('ia,jb->ijab',t1,t1)
            Tau_ab += tmp1
            tmp2        = contract('ib,ja->ijab',t1,t1)
            TauP_ab    += 0.5*tmp1
            TauP_aa    += 0.5*(tmp1-tmp2)
        return(Tau_ab, TauP_ab, TauP_aa)

    def init_T():
        o       = n_occ
        v       = n_virt
        t1new  = np.zeros((o,v))
        t2new  = np.zeros((o,o,v,v))
        return(t1new,t2new)

    def form_Fac():  # Computes the Fae intermediate
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        Fac     = np.zeros((n_virt,n_virt))
        
        if ccsd:
            tmp1    = 2*TEI[v,v,v,o] - TEI.swapaxes(2,3)[v,v,v,o]
            Fac    += contract('kd,cdak->ac',t1,tmp1)

        Fac    -= contract('klad,cdkl->ac',TauP_ab,TEI[v,v,o,o])

        tmp2    = TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
        Fac    -= 0.5*contract('klad,cdkl->ac',TauP_aa,tmp2)

        return(Fac)

    def form_Fki():   # Computes the Fmi intermediate
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        Fki     = np.zeros((n_occ,n_occ))

        if ccsd:
            tmp1    = 2*TEI[o,v,o,o] - TEI.swapaxes(2,3)[o,v,o,o]
            Fki    += contract('lc,ickl->ki',t1,tmp1)

        Fki    += contract('ilcd,cdkl->ki',TauP_ab,TEI[v,v,o,o])

        tmp2    = TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
        Fki    += 0.5*contract('ilcd,cdkl->ki',TauP_aa,tmp2)

        return(Fki)

    def form_Fkc():   # Computes the Fme intermediate
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        Fkc     = np.zeros((n_occ,n_virt))

        tmp1    = 2*TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
        Fkc    += contract('ld,cdkl->kc',t1,tmp1)

        return(Fkc)

    def form_Wijkl():   # Computes the Wmnij intermediate
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        Wijkl   = TEI[o,o,o,o].copy()

        if ccd:
            Wijkl  += 0.5*contract('ijcd,cdkl->ijkl',Tau_ab,TEI[v,v,o,o])
        if ccsd:
            tmp1    = contract('jd,idkl->ijkl',t1,TEI[o,v,o,o])
            Wijkl  += tmp1 + tmp1.swapaxes(0,1).swapaxes(2,3)
        
        return(Wijkl)

    def form_Wcdab():   # Computes the Wabef intermediate
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        Wcdab   = TEI[v,v,v,v].copy()

        if ccd:
            Wcdab  += 0.5*contract('klab,cdkl->cdab',Tau_ab,TEI[v,v,o,o])
        if ccsd:
            tmp1    = contract('lb,cdal->cdab',t1,TEI[v,v,v,o])
            Wcdab  -= tmp1 + tmp1.swapaxes(0,1).swapaxes(2,3)

        return(Wcdab)

    def form_Wcjkb():   # Computes the Wmbej intermediates
        o           = slice(n_occ)
        v           = slice(n_occ,n_ao)
        Wcjak       = np.zeros((n_virt,n_occ,n_virt,n_occ))
        Wcjkb_ab    = np.zeros((n_virt,n_occ,n_occ,n_virt))
        Wcjkb_aa    = Wcjkb_ab.copy()

        Wcjkb_ab   += TEI[v,o,o,v]
        Wcjkb_aa   += TEI[v,o,o,v] - TEI.swapaxes(2,3)[v,o,o,v]
        if ccd:
            tmp1        = TEI[v,v,o,o] - TEI.swapaxes(2,3)[v,v,o,o]
            Wcjkb_ab   += 0.5*contract('ljdb,cdkl->cjkb',t2_ab,tmp1)
            Wcjkb_aa   += 0.5*contract('ljdb,cdkl->cjkb',t2_ab,TEI[v,v,o,o])
        if ccd and not ccsd:
            Wcjkb_ab   += 0.5*contract('ljdb,cdkl->cjkb',t2_aa,TEI[v,v,o,o])
            Wcjkb_aa   += 0.5*contract('ljdb,cdkl->cjkb',t2_aa,tmp1)
        if ccsd:
            Wcjkb_ab   += contract('jd,cdkb->cjkb',t1,TEI[v,v,o,v])
            Wcjkb_ab   -= contract('lb,cjkl->cjkb',t1,TEI[v,o,o,o])
            tmp2        = TEI[v,v,o,v] - TEI.swapaxes(2,3)[v,v,o,v]
            Wcjkb_aa   += contract('jd,cdkb->cjkb',t1,tmp2)
            tmp3        = TEI[v,o,o,o] - TEI.swapaxes(2,3)[v,o,o,o]
            Wcjkb_aa   -= contract('lb,cjkl->cjkb',t1,tmp3)

            tmp4        = 0.5*t2_aa + contract('jd,lb->jldb',t1,t1)
            Wcjkb_ab   -= contract('jldb,cdkl->cjkb',tmp4,TEI[v,v,o,o])
            Wcjkb_aa   -= contract('jldb,cdkl->cjkb',tmp4,tmp1)

        Wcjak      += TEI[v,o,v,o]
        if ccd and not ccsd:
            Wcjak  -= 0.5*contract('ljad,cdlk->cjak',t2_ab,TEI[v,v,o,o])
        if ccsd:
            Wcjak  += contract('jd,cdak->cjak',t1,TEI[v,v,v,o])
            Wcjak  -= contract('la,cjlk->cjak',t1,TEI[v,o,o,o])
            tmp5    = 0.5*t2_ab + contract('la,jd->ljad',t1,t1)
            Wcjak  -= contract('ljad,cdlk->cjak',tmp5,TEI[v,v,o,o])

        return(Wcjkb_ab, Wcjkb_aa, Wcjak)

    def T1_new():   # Computes all contributions to the singles amplitudes
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        t1new   = np.zeros((n_occ,n_virt))

        if ccsd:
            t1new  += contract('ic,ac->ia',t1,Fac)
            t1new  -= contract('ka,ki->ia',t1,Fki)
            tmp1    = t2_ab + t2_aa
            t1new  += contract('ikac,kc->ia',tmp1,Fkc)

        tmp2    = 2*TEI[o,v,v,o] - TEI.swapaxes(2,3)[o,v,v,o]
        t1new  += contract('ld,idal->ia',t1,tmp2)

        tmp3    = TEI[v,v,v,o] - TEI.swapaxes(2,3)[v,v,v,o]
        t1new  += 0.5*contract('ikcd,cdak->ia',t2_aa,tmp3)
        t1new  += contract('ikcd,cdak->ia',t2_ab,TEI[v,v,v,o])

        tmp4    = TEI[o,v,o,o] - TEI.swapaxes(2,3)[o,v,o,o]
        t1new  -= 0.5*contract('klac,ickl->ia',t2_aa,tmp4)
        t1new  -= contract('klac,ickl->ia',t2_ab,TEI[o,v,o,o])

        return(t1new)

    def T2_new():   # Computes all contributions to the doubles amplitudes
        o       = slice(n_occ)
        v       = slice(n_occ,n_ao)
        t2new   = np.zeros((n_occ,n_occ,n_virt,n_virt))

        tmp     = 0.5*TEI[o,o,v,v]

        if lccd:
            t2new  += contract('ijkl,klab->ijab',Wijkl,Tau_ab)
            t2new  += contract('cdab,ijcd->ijab',Wcdab,Tau_ab)

            tmp    += contract('ikac,cjkb->ijab',t2_ab,Wcjkb_aa)
            tmp    += contract('ikac,cjkb->ijab',t2_aa,Wcjkb_ab)
            tmp    -= contract('ikcb,cjak->ijab',t2_ab,Wcjak)

        if lccsd:
            tmp    += contract('ic,cjab->ijab',t1,TEI[v,o,v,v])
            tmp    -= contract('ka,ijkb->ijab',t1,TEI[o,o,o,v])

        if ccd and not ccsd:
            tmp    += contract('ijac,bc->ijab',t2_ab,Fac)
            tmp    -= contract('ikab,kj->ijab',t2_ab,Fki)

        if ccsd:
            tmp1    = Fac-0.5*contract('kb,kc->bc',t1,Fkc)
            tmp    += contract('ijac,bc->ijab',t2_ab,tmp1)

            tmp2    = Fki+0.5*contract('jc,kc->kj',t1,Fkc)
            tmp    -= contract('ikab,kj->ijab',t2_ab,tmp2)

        t2new  += tmp + tmp.swapaxes(0,1).swapaxes(2,3)

        if ccsd:
            tmp3    = contract('ic,kb,cjak->ijab',t1,t1,TEI[v,o,v,o])+contract('jc,ka,cibk->ijab',t1,t1,TEI[v,o,v,o])
            tmp4    = contract('ic,ka,cjkb->ijab',t1,t1,TEI[v,o,o,v])+contract('jc,kb,cika->ijab',t1,t1,TEI[v,o,o,v])
            t2new  -= tmp3 + tmp4

        return(t2new)

    def update_t():
        tmp         = t2new + t2new.swapaxes(0,1).swapaxes(2,3)
        tmp        *= 0.5
        t1,t2_ab    = t_denominator(t1new,tmp)

        return(t1,t2_ab)

    ## MAIN PROGRAM ##

    t1,t2_ab = guess_rhf()
    e_corr  = energy_rhf()

    print(blue+'The MBPT(2) correlation energy is'+end)
    print(cyan+'\t%s \n' %e_corr+end)

    print('\t Summary of iterative solution of the CC equations')
    print('\t------------------------------------------------------')
    print('\t\t\t Correlation')
    print('\t  Iteration \t Energy')
    print('\t------------------------------------------------------')
    print('\t  %s \t\t %s' %(0,e_corr))

    maxiter = 100
    e_conv  = 1e-14
    for iter in range(0,maxiter):
        e_old   = e_corr

        t2_aa                       = antisymmetrize_T2()
        Tau_ab, TauP_ab, TauP_aa    = tau()
        t1new,t2new                 = init_T()

        if ccd:
            Fac     = form_Fac()
            Fki     = form_Fki()
        if ccsd:
            Fkc     = form_Fkc()

        if lccsd:
            t1new   = T1_new()

        Wijkl   = form_Wijkl()
        Wcdab   = form_Wcdab()
        Wcjkb_ab,Wcjkb_aa,Wcjak   = form_Wcjkb()

        t2new   = T2_new()

        t1,t2_ab    = update_t()

        e_corr      = energy_rhf()

        print('\t  %s \t\t %s' %(iter+1,e_corr))

        if (abs(e_corr-e_old) < e_conv):
            break

    print('\t------------------------------------------------------')
    print('The CC equations have converged')
    print(blue+'The %s correlation energy is' %calc+end)
    print(cyan+'\t%s \n' %e_corr+end)
    pass
