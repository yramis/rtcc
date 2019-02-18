# Run this with:
#   python input.py
# Make sure helper files are in the same directory
# Everything should work as intended if downloaded from github.com/yramis/rtcc
#   If not, feel free to write me a strongly worded email for not keepting this repo up to date.

import psi4
from helper_tdcc import rtcc

mol = psi4.geometry("""
He
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz'})
psi4.set_options({'reference':'rhf'})

rtcc()
