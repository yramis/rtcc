import psi4
import numpy as np
from helper_ndot import ndot
from opt_einsum import contract

def localize_occupied(self):    # Pipek mezey scheme
    C_occ   = self.wfn.Ca_subset('AO','OCC')
    basis   = self.wfn.basisset()
    scheme  = psi4.core.Localizer.build("PIPEK_MEZEY",basis,C_occ)
    psi4.core.Localizer.localize(scheme)
    if scheme.converged:
        Local_C = np.asarray(scheme.L)
        U       = np.asarray(scheme.U)
    # end PM localization
    o = self.o
    self.npC        = np.asarray(self.C)
    self.npC[:,o]   = Local_C
    self.C          = psi4.core.Matrix.from_array(self.npC)
    return

def localize_virtual(self):     # PAO scheme
    o = self.o
    v = self.v

    # Compute the projectors
    D   = ndot('ui,vi->uv',self.npC[:,o],self.npC[:,o])     # 1/2 SCF MO density matrix
    P_o = ndot('uv,vl->ul',D,self.S_ao)                     # Projector: AO -> Occupied MO
    P_v = np.eye(self.nmo) - P_o                            # Projector: AO -> Virtual  MO

    # Compute the PAO overlap (metric)
    S_pao = contract('uP,uv,vQ->PQ',P_v,self.S_ao,P_v)

    # Count the number of zero eigenvalues
    eps,evcs = np.linalg.eigh(S_PAO)
    count = 0
    for i in range(self.nmo):
        if (eps[i] < 1e-6):
            count += 1

    # Compute transformation: redundant PAO -> non-redundant PAO
    X = np.zeros((self.nmo,self.nmo-count))
    I = 0
    for i in range(self.nmo):
        if (eps[i] > 1e-6):
            for j in range(self.nmo):
                X[j,I] = evcs[j,i]/np.sqrt(eps[i])          # Transformation: PAO -> nr-PAO
            I += 1

    Rt = ndot('uA,AB->uB',P_v,X)                            # Transformation: AO -> nr-PAO
    
    C_pao       = np.zeros((self.nmo,self.nmo))
    C_pao[:,o]  = self.npC[:,o]
    C_pao[:,v]  = Rt                                        # Transformation: AO -> LMO/nr-PAO

    U_pao   = ndot('pu,uA->pA',np.linalg.inv(self.npC),Rt)  # Transformation: MO -> nr-PAO
    U_pao_v = U_pao[v,:]                                    # Transformation: virtual MO -> nr-PAO

    return
