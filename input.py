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
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
""")

psi4.set_options({'basis': 'sto-3g'})
psi4.set_options({'reference':'rhf'})

rtcc()
