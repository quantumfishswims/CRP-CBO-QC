#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 2025

@author: Eric W. Fischer

Runs Cavity Born-Oppenheimer Coupled Cluster theory
with singles and doubles in cavity reaction potential (CRP) 
formulation for correlated dipole fluctuation corrections in
ab initio vibro-polaritonic chemistry.

CRP-formulation minimizes electronic energy in cavity subspace.

Contains approximate linearized CRP-CCSD methods as well as
an iterative (self-consistent) CRP-CCSD implementation.

Literature:
Fischer, J. Chem. Phys. 161, 164112 (2024), doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
from pyscf import gto, scf, cc
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
    verbose = 4
)


# Light-matter coupling strength in sqrt(Eh)/e Bohr
coup = 0.03
# Normalized Polarization vectors
polarization = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
# Maximum number of CRP-CCSD iterations
max_crpccsd_iteration = 10

# Standard CCSD calculation for reference
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)
e_ccsd = mycc.kernel()[0]

# CRP-CCSD calculations: mean-field linearised, lambda linearised, iterative
my_mflin_crp_ccsd = mf_lin_crp_ccsd(mol, polarization[2], coup)
my_lambdalin_crp_ccsd = lambda_lin_crp_ccsd(mol, polarization[2], coup)
my_crp_ccsd = crp_ccsd(mol, polarization[2], coup, max_crpccsd_iteration)

# Correlated dipole fluctuation corrections on different levels of CRP-CCSD theory (vanishes for coup = 0.0)
delta_mflin_crpccsd = my_mflin_crp_ccsd[0] - e_ccsd
delta_lambdalin_crpccsd = my_lambdalin_crp_ccsd[0] - e_ccsd
delta_crpccsd = my_crp_ccsd[0][-1] - e_ccsd
iterations = len(my_crp_ccsd[0])


print("delta_mflin_crpccsd: \n", delta_mflin_crpccsd)
        
print("delta_lambdalin_crpccsd: \n", delta_lambdalin_crpccsd)
       
print("delta_crpccsd, iterations: \n", delta_crpccsd, iterations)