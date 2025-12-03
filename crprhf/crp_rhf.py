#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:39:06 2024

@author: Eric W. Fischer

Implementation of cavity reaction potential (CRP) formulation of the 
cavity Born-Oppenheimer (CBO) restricted Hartree-Fock (RHF) method (CBO-RHF)

CRP-formulation minimizes electronic energy in cavity subspace
"""

import numpy as np
from pyscf import scf

def get_crp_hcore(mol, polarization, coupling):
    """
    Compute the CRP-corrected Hartree-Fock core Hamiltonian.

    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : array_like, shape (3,)
        Cavity polarization vector
    coupling : float
       Light–matter coupling strength (scalar) used to scale the DSE terms.

    Returns
    -------
    crp_hcore : numpy.ndarray
        The CRP-corrected core Hamiltonian matrix (shape: (nao, nao)).
    """

    int1e_rr_array = mol.intor_symmetric('int1e_rr').reshape(3,3,mol.nao, mol.nao)
    rr = np.einsum('i,ijkl,j->kl', polarization, int1e_rr_array, polarization)

    # canonical core Hamiltonian   
    hmol  = mol.intor_symmetric('int1e_kin') 
    hmol += mol.intor_symmetric('int1e_nuc') 
       
    # 1e DSE correction
    hdse = 0.5*coupling*coupling*rr
    
    crp_hcore = hmol + hdse

    return crp_hcore



def get_crp_veff(mol, dm, polarization, coupling):
    """
    Compute the CRP effective two-index potential (veff)
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    dm : ndarray, shape (n_ao, n_ao)
        One-particle density matrix in AO basis.
    polarization : array_like, shape (3,)
        Cavity polarization vector
    coupling : float
       Light–matter coupling strength (scalar) used to scale the DSE terms.

    Returns
    -------
    crp_veff : ndarray, shape (n_ao, n_ao)
        The CRP-effective potential matrix (veff) in the AO basis:
    """

    int1e_r_array = -mol.intor('int1e_r')
    de = np.einsum('i,ijk->jk', polarization, int1e_r_array)
        
    dde = np.outer(de,de).reshape((len(de), len(de), len(de), len(de)), order='C')
    eri = mol.intor('int2e') 
    
    # canonical Coulom integral
    J    = np.einsum('pqrs,qp->rs', eri, dm)
    # canonical exchange integral
    K    = np.einsum('pqrs,qr->ps', eri, dm)
    # DSE exchange contribution
    Kdse = np.einsum('pqrs,qr->ps', dde, dm)

    crp_veff =  J - 0.5*K - 0.5*coupling**2*Kdse
    
    return crp_veff

    
def crprhf(mol, polarization, coupling, bare_mo=False, df=False, newton=False):
    """
    Cavity Born-Oppenheimer Hartree-Fock in Cavity Reaction Potential formulation

    Parameters
        ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : array_like, shape (3,)
        Cavity polarization vector
    coupling : float
       Light–matter coupling strength (scalar) used to scale the DSE terms.
    bare_mo : bool, optional
        If True, run one-shot CRP-RHF calculation on top of RHF MOs.
    df : bool, optional
        If True, use density-fitting approximation for electron repulsion integrals.
    newton : bool, optional 
        If True, use Newton solver for RHF convergence.
    Returns
    -------
    crp_rhf_energy : float
        CBO-RHF energy in CRP formulation
    
    crp_mf : pyscf.scf.hf.RHF
        CBO-RHF mean-field object

    """     

    if newton == True:
        crp_mf = scf.RHF(mol).newton()
    else:
        crp_mf = scf.RHF(mol)
    crp_mf.init_guess = 'minao'
    crp_mf.kernel()
    dm1     = crp_mf.make_rdm1()

    if bare_mo == False and df == False:
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm1, polarization, coupling)
        if newton == True:
            crp_mf.newton()
        crp_mf.kernel(dm0=dm1)

    if bare_mo == False and df == True:
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm1, polarization, coupling)
        if newton == True:
            crp_mf.newton().density_fit()
        crp_mf.kernel(dm0=dm1)

    if bare_mo == True:
        # There is no additional SCF run need here to keep RHF MOs
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm1, polarization, coupling)

    #Energy
    dm = crp_mf.make_rdm1()
    h1e  = crp_mf.get_hcore(mol, polarization, coupling)
    vhf  = crp_mf.get_veff(crp_mf.mol, dm, polarization, coupling)

    e1 = np.einsum('ij,ji->', h1e, dm, optimize=True).real
    e2 = np.einsum('ij,ji->', vhf, dm, optimize=True).real * .5
    nuc = crp_mf.energy_nuc()

    crp_rhf_energy = e1 + e2 + nuc

    return crp_rhf_energy, crp_mf

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
