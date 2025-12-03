#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 2025

@author: Eric W. Fischer

Input mean-field linearised CRP-CCSD (w/ and w/o energy correction) 
and CRP-CCD 
"""

import numpy as np
from pyscf import gto
from crpcc.lin_crp_ccsd import mf_lin_crp_ccsd, lambda_lin_crp_ccsd
from crpcc.iterative_crp_ccsd import crp_ccsd

mol = gto.M(
    atom = '''
Li        0.00   0.00   0.00
H         0.00   0.00   1.64
           ''',
    basis = 'def2svpd',
    symmetry = False,
    charge = 0,
    verbose=4
)


coup = 0.03
polarization = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
max_crpccsd_iteration = 10

my_mflin_crp_ccsd = mf_lin_crp_ccsd(mol, polarization[2], coup)
my_lambdalin_crp_ccsd = lambda_lin_crp_ccsd(mol, polarization[2], coup)
my_crp_ccsd = crp_ccsd(mol, polarization[2], coup, max_crpccsd_iteration)

etot_mflin_crpccsd, ecorr_mflin_crpccsd = my_mflin_crp_ccsd[0], my_mflin_crp_ccsd[1]
etot_lambdalin_crpccsd, ecorr_lambdalin_crpccsd = my_lambdalin_crp_ccsd[0], my_lambdalin_crp_ccsd[1]
etot_crpccsd, ecorr_crpccsd = my_crp_ccsd[0][-1], my_crp_ccsd[1][-1]
iterations = len(my_crp_ccsd[0])


print("etot_mflin_crpccsd, ecorr_mflin_crpccsd: \n", \
       etot_mflin_crpccsd, ecorr_mflin_crpccsd)
        
print("etot_lambdalin_crpccsd, ecorr_lambdalin_crpccsd: \n", \
       etot_lambdalin_crpccsd, ecorr_lambdalin_crpccsd)
       
print("etot_crpccsd, ecorr_crpccsd, iterations: \n", \
       etot_crpccsd, ecorr_crpccsd, iterations)