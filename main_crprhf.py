#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:04:31 2024

@author: eric

Cavity Born-Oppenheimer Hartree-Fock
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
    verbose=4
)


coup = 0.03
polarization = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
delta_crprhf = []

rhf_mol = scf.RHF(mol).run()
e_rhf_mol = rhf_mol.e_tot

for i in range(len(polarization)):
    my_crprhf = crprhf(mol, polarization[i], coup)
    e_crprhf = my_crprhf[0]
    delta_crprhf.append(np.abs(e_crprhf-e_rhf_mol))

print("e_rhf_mol: \n", e_rhf_mol)
print("delta_crprhf: \n", delta_crprhf)

