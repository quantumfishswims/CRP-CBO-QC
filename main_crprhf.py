#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:04:31 2024

@author: Eric W. Fischer

Runs Cavity Born-Oppenheimer Hartree-Fock in 
cavity reaction potential (CRP) formulation for 
mean-field dipole fluctuation corrections in
ab initio vibro-polaritonic chemistry.

CRP-formulation minimizes electronic energy in cavity subspace.

Literature:
Fischer, J. Chem. Phys. 161, 164112 (2024), doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
from pyscf import gto, scf
from crprhf.crp_rhf import crprhf

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
# Mean-field dipole fluctuation correction (vanishes for coup = 0.0)
delta_crprhf = []

# Standard RHF calculation for reference
rhf_mol = scf.RHF(mol).run()
e_rhf_mol = rhf_mol.e_tot

# Cavity Born-Oppenheimer RHF calculations for different polarization directions
for i in range(len(polarization)):
    my_crprhf = crprhf(mol, polarization[i], coup)
    e_crprhf = my_crprhf[0]
    delta_crprhf.append(np.abs(e_crprhf-e_rhf_mol))

print("e_rhf_mol: \n", e_rhf_mol)
print("delta_crprhf: \n", delta_crprhf)

