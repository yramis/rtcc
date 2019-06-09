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
This is the main rtcc driver, which calls the individual steps and handles the various options
The heavy work is done in:
    helper_ndot     -> contraction routine
    helper_diis     -> DIIS procedure
    helper_cc       -> CCSD equations, HBAR, Lambda equations, 1-PDM, dipole moments
    helper_local    -> orbital localizations such as pipek mezey, PAO
    helper_prop     -> time propagation of the CC equations

This assumes RHF reference and C1 symmetry
"""

__authors__ = "Alexandre P. Bazante"
__credits__ = [
        "T.D. Crawford","Ashutosh Kumar","Alexandre P. Bazante"]

import sys
import psi4
import numpy as np
from helper_Print import Print
from helper_cc import CCEnergy
from helper_cc import CCLambda
from helper_cc import CCProperties
from helper_prop import RK4
import contextlib
import time
from opt_einsum import contract

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

print_data  = True
localize    = False
use_hbar    = True      # flag to be used later for some testing

Print(blue+'\nTime Dependent CCSD Program'+end)
Print(blue+'-- Written by Alexandre P. Bazante, 2017\n'+end)

psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)
numpy_memory = 2

class rtcc(object):
    def __init__(self,memory=2):

        psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
        psi4.set_module_options('SCF', {'E_CONVERGENCE':1e-14})
        psi4.set_module_options('SCF', {'D_CONVERGENCE':1e-14})

        psi4.set_module_options('CCENERGY', {'E_CONVERGENCE':1e-14})
        psi4.set_module_options('CCLAMBDA', {'R_CONVERGENCE':1e-14})

        mol = psi4.core.get_active_molecule()
        ccsd = CCEnergy(mol, memory=2)
        ccsd.compute_ccsd()

        Lambda = CCLambda(ccsd)
        Lambda.compute_lambda()
        density = 0

        prop = CCProperties(ccsd,Lambda)

        options = {
            'timestep'          : 0.1,
            'number of steps'   : 10000,
            'timelength'        : np.inf,
            'field amplitude'   : 0.0002,
            'field frequency'   : 0.5}
        #td = RK4(ccsd,Lambda,prop,options)
