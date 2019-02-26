# Run this with:
#   python input.py
# Make sure helper files are in the same directory
# Everything should work as intended if downloaded from github.com/yramis/rtcc
#   If not, feel free to write me a strongly worded email for not keepting this repo up to date.

import psi4
from helper_tdcc import rtcc
from opt_einsum import contract
import numpy as np

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'sto-3g'})
psi4.set_options({'reference':'rhf'})

rtcc()

#e,wfn = psi4.energy('ccsd',return_wfn=True)
#e,wfn = psi4.properties('ccsd','dipole',return_wfn=True)
#C = np.asarray(wfn.Ca())
#P = 2*np.asarray(wfn.Da())
#D = contract('iu,uv,jv->ij',np.linalg.inv(C),P,np.linalg.inv(C))
#np.set_printoptions(precision=5,suppress=True)
#print('psi4 D')
#print(D)
#print('trace')
#print(np.trace(D))
# Water SCF/cc-pVDZ Single Point Energy
