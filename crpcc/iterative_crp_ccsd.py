#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed March 19 2025

@author: Eric W. Fischer

Implementation of self-consistent cavity reaction potential 
coupled cluster (CRP-CC) approach based on the PySCF package for CCSD.

CRP-formulation minimizes electronic energy in cavity subspace. 

Literature: 
Fischer, J. Chem. Phys. 161, 164112 (2024), doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
from pyscf import ao2mo
import sys
from crprhf.crp_rhf import crprhf, get_crp_hcore, get_crp_veff


def eri_ao2mo(mol,orb):
    """
    AO to MO transformation of 2-electron integrals
    
    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    orb : np.array
        molecular orbital coefficents
    
    Returns
    -------
    eri1_fold_mo : np.array
        2-electron integrals in MO basis
    """

    eri = mol.intor('int2e', aosym=1)
    eri1 = np.einsum('pi,pqrs->iqrs', orb, eri, optimize=True)
    eri1 = np.einsum('qj,iqrs->ijrs', orb, eri1, optimize=True)
    eri1 = np.einsum('ijrs,rk->ijks', eri1, orb, optimize=True)
    eri1_fold_mo = np.einsum('ijks,sl->ijkl', eri1, orb, optimize=True)
    return eri1_fold_mo


def de_ao2mo(mol, orb, polarization):
    """ 
    AO to MO transformation of polarization-projected dipole integrals

        Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    orb : np.array
        molecular orbital coefficents
    polarization : np.array
        single polarization vector

    Returns
    -------
    de1_fold_mo : np.array
         polarization-projected dipole integrals in MO basis
    """
        
    int1e_r_array = -mol.intor('int1e_r')
    de = np.einsum('i,ijk->jk', polarization, int1e_r_array, optimize=True)
        
    de1 = np.einsum('pi,pq->iq', orb, de, optimize=True)
    de1_fold_mo = np.einsum('iq,qj->ij', de1, orb, optimize=True)
    
    return de1_fold_mo


def cbo_eri(mol, polarization, coupling):
    """
    DSE-augmented 2-electron integrals

    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : np.array
        single polarization vector
    coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.

    Returns
    -------
    cbo_eri : np.array
       DSE-augmented 2-electron integrals
    """ 
        
    int1e_r_array = -mol.intor('int1e_r')
    de = np.einsum('i,ijk->jk', polarization, int1e_r_array)
    dde = np.outer(de,de).reshape((len(de), len(de), len(de), len(de)), order='C')

    cbo_eri  = mol.intor('int2e')  
    cbo_eri += coupling**2*dde
    
    return cbo_eri


def gamma1_intermediates(t1, t2, l1, l2):
    """
    CCSD response 1-RDM

    Parameters
    ----------
    t1 : np.array[nocc, nvir]
         Singles amplitudes
    t2 : np.array[nocc, nocc, nvir, nvir]
        Doubles amplitudes
    l1 : np.array[nocc, nvir]
        Singles multipliers
    l2 : np.array[nocc, nocc, nvir, nvir]
        Doubles multipliers

    Returns
    -------
    doo : np.array[nocc, nocc]
        Gamma1 occ/occ block
    dvv : np.array[nvir, nvir]
        Gamma1 vir/vir block
    dov : np.array[nocc, nvir]
        Gamma1 occ/vir block - DeXcite
    dvo : np.array[nvir, nocc]
        Gamma1 vir/occ block - Xcite
    tdvo : np.array[nvir, nocc]
        Multiplier gamma1 vir/occ block (no t^a_i)

    """
    nocc, nvir = t1.shape
    
    # --- occ/occ ---
    doo  = -np.einsum('jc,ic->ij', l1, t1, optimize=True)
    doo -= 0.5*np.einsum('jkcd,ikcd->ij', l2, t2, optimize=True)
    
    # --- virt/virt ---
    dvv  = np.einsum('kb,ka->ab',l1, t1, optimize=True)
    dvv += 0.5*np.einsum('klbc,klac->ab', l2, t2, optimize=True)
    
    # --- occ/virt ---
    dov = l1
    
    # --- virt/occ ---
    dvo  = t1.T # np.array[nvir, nocc]
    
    # l1-terms
    dvo += np.einsum('jb,jiba->ai', l1, t2, optimize=True)
    dvo -= np.einsum('jb,ib,ja->ai', l1, t1, t1, optimize=True)
    
    # l2-terms w/ intermediates
    dvo_x1 = np.einsum('kjcb,kicb->ji', l2, t2, optimize=True)
    dvo_x2 = np.einsum('kjcb,kjca->ba', l2, t2, optimize=True)
    
    dvo -= 0.5*np.einsum('ji,ja->ai', dvo_x1, t1, optimize=True)
    dvo += 0.5*np.einsum('ba,ib->ai', dvo_x2, t1, optimize=True)
    
    # --- lambda-only virt/occ ---
    tdvo = dvo - t1.T
    
    return doo, dvv, dov, dvo, tdvo


def ccsd_dipole(de_mo, t1, t2, l1, l2):
    """
    Compute CCSD electronic dipole expectation values .

    Parameters
    ----------
    de_mo : numpy.ndarray
        One-electron dipole integral matrix in MO basis. Shape (n_orb, n_orb), n_orb = nocc + nvir.
    t1 : numpy.ndarray
        CCSD single excitation amplitudes. Shape (nocc, nvir).
    t2 : numpy.ndarray
        CCSD double excitation amplitudes. Typical shape (nocc, nocc, nvir, nvir).
    l1 : numpy.ndarray
        Lambda (de-excitation) single amplitudes. Shape (nocc, nvir).
    l2 : numpy.ndarray
        Lambda (de-excitation) double amplitudes. Typical shape (nocc, nocc, nvir, nvir).
   
   Returns
    -------
    tuple(float, float)
        (response_dipole, lambda_dipole)
        - response_dipole : float
            The CC electronic dipole expectation value
        - lambda_dipole : float
            The multiplier-dependent component of the CC electronic dipole expectation value.
    """
    

    nocc, nvir  = t1.shape

    # --- response CC electronic dipole expectation value ---
    response_dipole  = np.einsum('ij,ij->', de_mo[:nocc,:nocc], gamma1_intermediates(t1, t2, l1, l2)[0], optimize=True)
    response_dipole += np.einsum('ab,ab->', de_mo[nocc:,nocc:], gamma1_intermediates(t1, t2, l1, l2)[1], optimize=True)
    response_dipole += np.einsum('ia,ia->', de_mo[:nocc,nocc:], gamma1_intermediates(t1, t2, l1, l2)[2], optimize=True)
    response_dipole += np.einsum('ai,ai->', de_mo[nocc:,:nocc], gamma1_intermediates(t1, t2, l1, l2)[3], optimize=True)

    # --- multiplier-dependent component of response CC electronic dipole expectation value ---
    lambda_dipole  = np.einsum('ij,ij->', de_mo[:nocc,:nocc], gamma1_intermediates(t1, t2, l1, l2)[0], optimize=True)
    lambda_dipole += np.einsum('ab,ab->', de_mo[nocc:,nocc:], gamma1_intermediates(t1, t2, l1, l2)[1], optimize=True)
    lambda_dipole += np.einsum('ia,ia->', de_mo[:nocc,nocc:], gamma1_intermediates(t1, t2, l1, l2)[2], optimize=True)
    lambda_dipole += np.einsum('ai,ai->', de_mo[nocc:,:nocc], gamma1_intermediates(t1, t2, l1, l2)[4], optimize=True)


    return response_dipole, lambda_dipole


def get_crp_hcore_lambda(mol, polarization, coupling, response_dipole):
    """
    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : np.array
        single polarization vector
    coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.
    response_dipole : float
        CC response dipole/ changes each iteration
    Returns
    -------
    hcore_iter : TYPE
        DESCRIPTION.

    """ 

    int1e_rr_array = mol.intor_symmetric('int1e_rr').reshape(3,3,mol.nao, mol.nao)
    rr = np.einsum('i,ijkl,j->kl', polarization, int1e_rr_array, polarization)
    
    int1e_r_array = -mol.intor('int1e_r')
    de = np.einsum('i,ijk->jk', polarization, int1e_r_array)
    
    # canonical hcore
    hmol  = mol.intor_symmetric('int1e_kin') 
    hmol += mol.intor_symmetric('int1e_nuc') 
    
    # DSE quadrupole term
    hdse = 0.5*coupling**2*rr
    
    # lambda-dependent CC dipole term
    hlambda = -coupling**2*response_dipole*de
    
    hcore_iter = hmol + hdse + hlambda

    return hcore_iter


# --- Iterative CRP-CCSD approach --------

def crp_ccsd_iter(crp_mf, dm, mol, polarization, coupling, response_dipole, iteration):
    """
    Run one CRP-CCSD iteration for an effective CRP Hamiltonian.

    Parameters
    ----------
    crp_mf :
            CRP mean-field object
    dm : array_like
            Electronic density matrix used to build the effective mean-field potential.
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : np.array
        single polarization vector
    coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.
    response_dipole :
            CCSD response dipole expectation value
    iteration : int
            Iteration index. When iteration == 0, the standard CRP hcore is used;
            for iteration > 0 a lambda-dependent hcore (get_crp_hcore_lambda) is used.
    Returns
    -------
    tuple
            (ecorr_iter_crp_ccsd, response_dipole_iter)
            - ecorr_iter_crp_ccsd (float): Total CRP-CCSD correlation energy for this
                iteration.
            - response_dipole_iter (array_like or scalar): Dipole expectation value
                returned by the CCSD response quantity computed by ccsd_dipole(...)[0].
    """


    # --- Generate effective CBO electronic Hamiltonian: get_crp_hcore, get_crp_veff, cbo_eri
    if iteration == 0:
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        crp_mf.kernel()
        crp_rhf_orb  = crp_mf.mo_coeff

    else:
        crp_mf.get_hcore = lambda *args: get_crp_hcore_lambda(mol, polarization, coupling, response_dipole)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        crp_mf.kernel()
        crp_rhf_orb  = crp_mf.mo_coeff

    mol.incore_anyway = True
    
    
    # --- CCSD run with effective lambda-crp hamiltonian
    my_iter_crp_ccsd = crp_mf.CCSD()
    my_iter_crp_ccsd.kernel()

    t1, t2 = my_iter_crp_ccsd.t1, my_iter_crp_ccsd.t2
    l1, l2 = my_iter_crp_ccsd.solve_lambda()

    # ---  Calculate CRP-CCSD correlation energy ---
    nocc, nvir  = t1.shape
    eri_mo = eri_ao2mo(mol, crp_rhf_orb)
    de_mo = de_ao2mo(mol, crp_rhf_orb, polarization)

    eri_ovov = eri_mo[:nocc,nocc:,:nocc,nocc:]
    de_ov = de_mo[:nocc, nocc:]
    
    # --- Canonical CCSD energy
    ecorr_coul  = 2*np.einsum('ia,jb,iajb', t1, t1, eri_ovov, optimize=True)
    ecorr_coul += 2*np.einsum('ijab,iajb', t2, eri_ovov, optimize=True)
    ecorr_coul -= np.einsum('ia,jb,ibja', t1, t1, eri_ovov, optimize=True)
    ecorr_coul -= np.einsum('ijab, ibja', t2, eri_ovov, optimize=True)
    
    # --- CRP-DSE CCSD energy
    ecorr_dse = 2*coupling**2*np.einsum('ijab, ia, jb', t2, de_ov, de_ov, optimize=True)
    ecorr_dse -= coupling**2*np.einsum('ia,jb,ib, ja', t1, t1, de_ov, de_ov, optimize=True)
    ecorr_dse -= coupling**2*np.einsum('ijab, ib, ja', t2, de_ov, de_ov, optimize=True)
    
    # ---  
    lambda_dipole = ccsd_dipole(de_mo, t1, t2, l1, l2)[1]
    ecorr_lambda = 0.5*coupling**2*lambda_dipole**2

    ecorr_iter_crp_ccsd = ecorr_coul + ecorr_dse + ecorr_lambda

    # --- CRP-CCSD dipole expectation value ---
    
    response_dipole_iter = ccsd_dipole(de_mo, t1, t2, l1, l2)[0]
    
    return ecorr_iter_crp_ccsd, response_dipole_iter


def crp_ccsd(mol, polarization, coupling, iteration):
    """
    Perform self-consistent CRP-CCSD calculation.

    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : np.array
        single polarization vector
    coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.
    iteration : int
        Number of CRP-CCSD iterations to perform. If 0, a single-shot calculation is performed.

    Returns
    -------
    etot_crpccsd : ndarray
        Total CRP-CCSD energies for each iteration.
    ecorr_crpccsd : ndarray
        CRP-CCSD correlation energies for each iteration.
    response_dipole_iter : array_like
        Final response dipole vector from the last iteration.
    """
    
    
    if abs(1-np.dot(polarization, polarization)) > 1e-15:
        print('Error: Cavity polarization vector is not normalized!')
        sys.exit()
    
    # --- CRP-RHF run ---
    crp_rhf = crprhf(mol, polarization, coupling)
    e_crp_rhf = crp_rhf[0]
    crp_mf = crp_rhf[1]
    dm     = crp_mf.make_rdm1() 

    etot_crpccsd  = []
    ecorr_crpccsd = []
    
    # --- single-shot Lambda0-linearized CRP-CCSD ---
    if iteration == 0:
        my_crp_ccsd_iter      = crp_ccsd_iter(crp_mf, dm, mol, polarization, coupling, 0, 0)
        ecorr_crpcc_iter      = my_crp_ccsd_iter[0]
        response_dipole_iter  = my_crp_ccsd_iter[1]
        
        etot_crpccsd = e_crp_rhf + ecorr_crpcc_iter
        ecorr_crpccsd = ecorr_crpcc_iter
        
    
    # --- iterative CRP-CCSD --- 
    else:
        for i in range(iteration):
            print('CRP-CCSD iteration nr.'+str(i)+'/'+str(iteration))
            if i == 0:
                my_crp_ccsd_iter      = crp_ccsd_iter(crp_mf, dm, mol, polarization, coupling, 0.0, i)
                ecorr_crpcc_iter      = my_crp_ccsd_iter[0]
                response_dipole_iter  = my_crp_ccsd_iter[1]
            
            else:
                my_crp_ccsd_iter      = crp_ccsd_iter(crp_mf, dm, mol, polarization, coupling, response_dipole_iter, i)
                ecorr_crpcc_iter      = my_crp_ccsd_iter[0]
                response_dipole_iter  = my_crp_ccsd_iter[1]            

            etot_crpcc_iter = e_crp_rhf + ecorr_crpcc_iter
        
            ecorr_crpccsd.append(ecorr_crpcc_iter)
            etot_crpccsd.append(etot_crpcc_iter)
        
            if i > iteration:
                print('Maximum number of iterations '+str(iteration)+' reached w/o convergence')
                break
        
            if i > 0 and abs(ecorr_crpcc_iter-ecorr_crpccsd[i-1]) < 5e-8:
                print('Iterative CRP-CCSD converged for iteration='+str(i)+'/'+str(iteration))
                break
        
    return np.asarray(etot_crpccsd), np.asarray(ecorr_crpccsd), response_dipole_iter



















