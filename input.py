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
Li
H 1 1
symmetry c1
""")

psi4.set_options({'basis': 'sto-3g'})
psi4.set_options({'reference':'rhf'})

rtcc()

psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-14})
psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-14})
psi4.set_module_options('CCENERGY', {'E_CONVERGENCE':1e-16})
psi4.set_module_options('CCLAMBDA', {'R_CONVERGENCE':1e-16})

e_ccsd = psi4.properties('ccsd','dipole')
