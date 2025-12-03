#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 06 2025

@author: Eric W. Fischer

Implementation of linearized cavity reaction potential 
coupled cluster (CRP-CC) approach based on the PySCF package for CCSD 

CRP-formulation minimizes electronic energy in cavity subspace. 
Mean-field linearized CRP-CC approximates stationary cavity coordinate on 
mean-field (CRP-RHF) level of theory.

Literature: 
Fischer, J. Chem. Phys. 161, 164112 (2024), doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
import scipy as sp
import sys
from pyscf import cc, df, lib, ao2mo
from crprhf.crp_rhf import crprhf, get_crp_hcore, get_crp_veff

# --- AO to MO integral transformations ---
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
    de = np.einsum('i,ijk->jk', polarization, int1e_r_array)
        
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



# --- Mean-Field-linearised CRP-CCSD approach ---

def mf_lin_crp_ccsd(mol, polarization, coupling, l0corr=False, bare_mo=False, df=False):
    """
    Compute mean-field linearised CRP-CCSD energy.
    Treats energy optimization in cavity coordinate space on mean-field level
    l0-correction adds correlation correction to energy (no changes in amplitude eqs)

    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : array-like, shape (3,)
        Cavity polarization vector. This vector is expected to be normalized;
        the function will exit (sys.exit) if its norm differs from unity by
        more than 1e-15.
    coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.
    l0corr : bool, optional (default False)
        When True, adjust the DSE contribution according to the
        "l0-corrected" prescription implemented in the function. 
    bare_mo : bool, optional (default False)
        If True, compute and use "bare" molecular orbitals returned by
        crprhf(..., bare_mo=True) instead of performing a full CRP-RHF
        orbital optimization prior to CCSD.
    df : bool, optional (default False)
        If True and bare_mo is False, run the CRP-RHF mean-field with
        density-fitting (via density_fit().kernel()) rather than the
        standard kernel.
    Returns
    -------
    mflin_crp_ccsd_e_tot : float
        Total energy: sum of the CRP-RHF reference energy and the computed
        linearised CRP-CCSD correlation energy.
    mflin_crp_ccsd_ecorr : float
        Explicit linearised CRP-CCSD correlation energy. 
    my_mflin_crp_ccsd : pyscf.cc.ccsd.CCSD-like
        The CCSD solver object after convergence. It contains amplitude
        arrays (t1, t2) and other intermediate data. 
    """
    

    if abs(1-np.dot(polarization, polarization)) > 1e-15:
        print('Error: Cavity polarization vector is not normalized!')
        sys.exit()
        
    if bare_mo == False and df == False: 
        # --- CRP-RHF run ---
        crp_rhf = crprhf(mol, polarization, coupling)
        e_crp_rhf = crp_rhf[0]
        crp_mf = crp_rhf[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        # --- Define get_crp_hcore, get_crp_veff, cbo_eri
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        crp_mf.kernel()
        crp_rhf_orb  = crp_mf.mo_coeff 
        
    elif bare_mo == False and df == True:
        # --- CRP-RHF run ---
        crp_rhf = crprhf(mol, polarization, coupling, df=True)
        e_crp_rhf = crp_rhf[0]
        crp_mf = crp_rhf[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        # --- Define get_crp_hcore, get_crp_veff, cbo_eri
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        crp_mf.density_fit().kernel()
        crp_rhf_orb  = crp_mf.mo_coeff 
    
    else:
        crp_rhf_bmo = crprhf(mol, polarization, coupling, bare_mo=True)
        e_crp_rhf = crp_rhf_bmo[0]
        crp_mf = crp_rhf_bmo[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        
    mol.incore_anyway = True
    
    # --- Define mean-field linearised amplitude eqs
    my_mflin_crp_ccsd = cc.CCSD(crp_mf)       
    my_mflin_crp_ccsd.kernel()
    
    # --- explicit mean-field linearized CRP-CCSD correlation energy ---
       
    t1, t2 = my_mflin_crp_ccsd.t1, my_mflin_crp_ccsd.t2
    nocc, nvir  = t1.shape
    
    eri_mo = eri_ao2mo(mol, crp_rhf_orb)
    de_mo = de_ao2mo(mol, crp_rhf_orb, polarization)
    
    eri_ovov = eri_mo[:nocc,nocc:,:nocc,nocc:]
    de_ov = de_mo[:nocc,nocc:]
    
    ecorr_coul  = 2*np.einsum('ia,jb,iajb', t1, t1, eri_ovov)
    ecorr_coul += 2*np.einsum('ijab,iajb', t2, eri_ovov)
    ecorr_coul -= np.einsum('ia,jb,ibja', t1, t1, eri_ovov)
    ecorr_coul -= np.einsum('ijab, ibja', t2, eri_ovov)
       
    ecorr_dse  = 2*coupling**2*np.einsum('ia,jb,ia,jb', t1, t1, de_ov, de_ov)
    ecorr_dse += 2*coupling**2*np.einsum('ijab, ia,jb', t2, de_ov, de_ov)
    ecorr_dse -= coupling**2*np.einsum('ia,jb,ib,ja', t1, t1, de_ov, de_ov)
    ecorr_dse -= coupling**2*np.einsum('ijab, ib,ja', t2, de_ov, de_ov)
    
    if l0corr == True:
        ecorr_dse = 2*coupling**2*np.einsum('ijab, ia, jb', t2, de_ov, de_ov)
        ecorr_dse -= coupling**2*np.einsum('ia,jb,ib, ja', t1, t1, de_ov, de_ov)
        ecorr_dse -= coupling**2*np.einsum('ijab, ib, ja', t2, de_ov, de_ov)
    
    
    mflin_crp_ccsd_ecorr = ecorr_coul + ecorr_dse 
    mflin_crp_ccsd_e_tot  = e_crp_rhf + mflin_crp_ccsd_ecorr
        
    return mflin_crp_ccsd_e_tot, mflin_crp_ccsd_ecorr, my_mflin_crp_ccsd


# --- Lambda-linearised CRP-CCSD approach ---
 
def lambda_lin_crp_ccsd(mol, polarization, coupling, bare_mo=False, df=False):
    """
    Compute lambda-linearised CRP-CCSD energy.
    Treats energy optimization in cavity coordinate space with correlation
    corrections, specifically, the l0-correction and modified amplitude eqs.

    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : array-like, shape (3,)
        Cavity polarization vector. This vector is expected to be normalized;
        the function will exit (sys.exit) if its norm differs from unity by
        more than 1e-15.
    coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.
    bare_mo : bool, optional (default False)
        If True, compute and use "bare" molecular orbitals returned by
        crprhf(..., bare_mo=True) instead of performing a full CRP-RHF
        orbital optimization prior to CCSD.
    df : bool, optional (default False)
        If True and bare_mo is False, run the CRP-RHF mean-field with
        density-fitting (via density_fit().kernel()) rather than the
        standard kernel. 
    Returns
    -------
    lambda_lin_crp_ccsd_e_tot : float
        Total energy: sum of the CRP-RHF reference energy and the computed
        linearised CRP-CCSD correlation energy.
    lambda_lin_crp_ccsd_ecorr : float
        Explicit linearised CRP-CCSD correlation energy. 
    my_lambda_lin_crp_ccsd : pyscf.cc.ccsd.CCSD-like
        The CCSD solver object after convergence. It contains amplitude
        arrays (t1, t2) and other intermediate data. 
    """
    
    if abs(1-np.dot(polarization, polarization)) > 1e-15:
        print('Error: Cavity polarization vector is not normalized!')
        sys.exit()
          
    if bare_mo == False and df == False: 
        # --- CRP-RHF run ---
        crp_rhf = crprhf(mol, polarization, coupling)
        e_crp_rhf = crp_rhf[0]
        crp_mf = crp_rhf[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        # --- Define get_crp_hcore, get_crp_veff, cbo_eri
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        crp_mf.kernel()
        crp_rhf_orb  = crp_mf.mo_coeff 
        
    elif bare_mo == False and df == True:
        # --- DF-CRP-RHF run ---
        crp_rhf = crprhf(mol, polarization, coupling, df=True)
        e_crp_rhf = crp_rhf[0]
        crp_mf = crp_rhf[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        # --- Define get_crp_hcore, get_crp_veff, cbo_eri
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        crp_mf.density_fit().kernel()
        crp_rhf_orb  = crp_mf.mo_coeff 
    
    else:
        crp_rhf_bmo = crprhf(mol, polarization, coupling, bare_mo=True)
        e_crp_rhf = crp_rhf_bmo[0]
        crp_mf = crp_rhf_bmo[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        
    mol.incore_anyway = True
    
    
    # ---  lambda-linearized CRP-CC amplitude equations ---
    my_lambda_lin_crp_ccsd = cc.CCSD(crp_mf)
    old_update_amps = my_lambda_lin_crp_ccsd.update_amps
    
    def update_amps_lambda_ccsd(t1, t2, eris):
        """
        Lambda linearisation correction of CRP-CCSD amplitude eqs 

        """
        t1, t2 = old_update_amps(t1, t2, eris)
        
        nocc, nvir  = t1.shape

        de_mo = de_ao2mo(mol, crp_rhf_orb, polarization)
        de_oo = de_mo[:nocc,:nocc]
        de_ov = de_mo[:nocc,nocc:]
        de_vv = de_mo[nocc:,nocc:]
        
        de_1 = 2*np.einsum('ia,ia', t1, de_ov)
        
        # --- Lambda-Lin T1-equations ---
        t1_new = t1
        
        t1_la = de_ov
        t1_la += 2*np.einsum('ikac,kc->ia', t2, de_ov)
        t1_la += np.einsum('ic,ac->ia', t1, de_vv)
        t1_la -= np.einsum('ka,ki->ia', t1, de_oo)
        t1_la -= np.einsum('ic,ka,kc->ia', t1, t1, de_ov)
        
        t1_new -= coupling**2*de_1*t1_la      # Lambda-linear T1-correct
        
        # --- Lambda-Lin T2-equations ---
        t2_new = t2
        
        t2_la  = np.einsum('ijac,bc->ijab', t2, de_vv)
        t2_la -= np.einsum('ijbc,ac->ijba', t2, de_vv)
        t2_la -= np.einsum('ikab,kj->ijab', t2, de_oo)
        t2_la += np.einsum('jkab,ki->jiab', t2, de_oo)
        t2_la -= np.einsum('ic,kjab,kc->ijab', t1, t2, de_ov)
        t2_la += np.einsum('jc,kiab,kc->jiab', t1, t2, de_ov)
        t2_la -= np.einsum('ka,ijcb,kc->ijab', t1, t2, de_ov)
        t2_la += np.einsum('kb,ijca,kc->ijba', t1, t2, de_ov)
        
        t2_new -= coupling**2*de_1*t2_la     # Lambda-linear T2-correct
        
        return t1_new, t2_new 
    
    my_lambda_lin_crp_ccsd.update_amps = update_amps_lambda_ccsd
    my_lambda_lin_crp_ccsd.kernel()
    
    # --- explicit lambda-linearized CRP-CCSD correlation energy ---
    
    t1, t2 = my_lambda_lin_crp_ccsd.t1, my_lambda_lin_crp_ccsd.t2  
    nocc, nvir  = t1.shape
    
    eri_mo = eri_ao2mo(mol, crp_rhf_orb)
    de_mo = de_ao2mo(mol, crp_rhf_orb, polarization)
    
    eri_ovov = eri_mo[:nocc,nocc:,:nocc,nocc:]
    de_ov = de_mo[:nocc,nocc:]
    
    ecorr_coul  = 2*np.einsum('ia,jb,iajb', t1, t1, eri_ovov)
    ecorr_coul += 2*np.einsum('ijab,iajb', t2, eri_ovov)
    ecorr_coul -= np.einsum('ia,jb,ibja', t1, t1, eri_ovov)
    ecorr_coul -= np.einsum('ijab, ibja', t2, eri_ovov)
          
    ecorr_dse = 2*coupling**2*np.einsum('ijab, ia, jb', t2, de_ov, de_ov)
    ecorr_dse -= coupling**2*np.einsum('ia,jb,ib, ja', t1, t1, de_ov, de_ov)
    ecorr_dse -= coupling**2*np.einsum('ijab, ib, ja', t2, de_ov, de_ov)
    
    lambda_lin_crp_ccsd_ecorr = ecorr_coul + ecorr_dse
    lambda_lin_crp_ccsd_e_tot = e_crp_rhf + lambda_lin_crp_ccsd_ecorr
        
    return lambda_lin_crp_ccsd_e_tot, lambda_lin_crp_ccsd_ecorr, my_lambda_lin_crp_ccsd
    

# --- LambdaT-linearised CRP-CCSD approach ---

def lambdaT_lin_crp_ccsd(mol, polarization, coupling, bare_mo=False, df=False):
    """
    Compute lambdaT-linearised CRP-CCSD energy.
    Treats energy optimization in cavity coordinate space with correlation
    corrections, specifically, the l0-correction and additional corrections
    in modified amplitude eqs. of lambda-linerised approach

    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : array-like, shape (3,)
        Cavity polarization vector. This vector is expected to be normalized;
        the function will exit (sys.exit) if its norm differs from unity by
        more than 1e-15.
    coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.
    bare_mo : bool, optional (default False)
        If True, compute and use "bare" molecular orbitals returned by
        crprhf(..., bare_mo=True) instead of performing a full CRP-RHF
        orbital optimization prior to CCSD.
    df : bool, optional (default False)
        If True and bare_mo is False, run the CRP-RHF mean-field with
        density-fitting (via density_fit().kernel()) rather than the
        standard kernel. 
    Returns
    -------
    lambda_lin_crp_ccsd_e_tot : float
        Total energy: sum of the CRP-RHF reference energy and the computed
        linearised CRP-CCSD correlation energy.
    lambda_lin_crp_ccsd_ecorr : float
        Explicit linearised CRP-CCSD correlation energy. 
    my_lambda_lin_crp_ccsd : pyscf.cc.ccsd.CCSD-like
        The CCSD solver object after convergence. It contains amplitude
        arrays (t1, t2) and other intermediate data. 
    """
    
    if abs(1-np.dot(polarization, polarization)) > 1e-15:
        print('Error: Cavity polarization vector is not normalized!')
        sys.exit()
        
    if bare_mo == False and df == False: 
        # --- CRP-RHF run ---
        crp_rhf = crprhf(mol, polarization, coupling)
        e_crp_rhf = crp_rhf[0]
        crp_mf = crp_rhf[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        # --- Define get_crp_hcore, get_crp_veff, cbo_eri
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        crp_mf.kernel()
        crp_rhf_orb  = crp_mf.mo_coeff 
        
    elif bare_mo == False and df == True:
        # --- DF-CRP-RHF run ---
        crp_rhf = crprhf(mol, polarization, coupling, df=True)
        e_crp_rhf = crp_rhf[0]
        crp_mf = crp_rhf[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        # --- Define get_crp_hcore, get_crp_veff, cbo_eri
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        crp_mf.density_fit().kernel()
        crp_rhf_orb  = crp_mf.mo_coeff 
    
    else:
        crp_rhf_bmo = crprhf(mol, polarization, coupling, bare_mo=True)
        e_crp_rhf = crp_rhf_bmo[0]
        crp_mf = crp_rhf_bmo[1]
        crp_rhf_orb  = crp_mf.mo_coeff 
        
        dm     = crp_mf.make_rdm1()  
        crp_mf.get_hcore = lambda *args: get_crp_hcore(mol, polarization, coupling)
        crp_mf.get_veff  = lambda *args: get_crp_veff(mol, dm, polarization, coupling)
        crp_mf._eri = ao2mo.restore(1, cbo_eri(mol, polarization, coupling), mol.nao)
        
    mol.incore_anyway = True
    
    # ---  lambdaT-linearized CRP-CC amplitude equations ---
    my_lambdaT_lin_crp_ccsd = cc.CCSD(crp_mf)
    old_update_amps = my_lambdaT_lin_crp_ccsd.update_amps 
    
    def update_amps_lambdaT_ccsd(t1, t2, eris):
        """
        Lambda-T linearisation correction of CRP-CCSD amplitude eqs 

        """
        
        t1, t2 = old_update_amps(t1, t2, eris)
        
        nocc, nvir  = t1.shape

        de_mo = de_ao2mo(mol, crp_rhf_orb, polarization)
        de_oo = de_mo[:nocc,:nocc]
        de_ov = de_mo[:nocc,nocc:]
        de_vv = de_mo[nocc:,nocc:]
        
        de_1 = 2*np.einsum('ia,ia', t1, de_ov)
        
        # --- lambda-T dipole-contractions ---
              
        # singles dipole contractions
        t1_con  = 2.0*np.einsum('kc,kc', t1, de_ov)
        t1_con += 4.0*np.einsum('kc,klcd,ld', t1, t2, de_ov)
        t1_con += 2.0*np.einsum('kc,kd,cd', t1, t1, de_vv)
        t1_con -= 2.0*np.einsum('kc,lc,lk', t1, t1, de_oo)
        t1_con -= 2.0*np.einsum('kc,kd,lc,ld', t1, t1, t1, de_ov)
        
        # doubles dipole contractions
        t2_con_x1 = np.einsum('klce,de->klcd', t2, de_vv)
        t2_con = np.einsum('klcd, klcd', t2, t2_con_x1)
        t2_con_x2 = np.einsum('klde,ce->kldc', t2, de_vv)
        t2_con -= np.einsum('klcd, kldc', t2, t2_con_x2)
        t2_con_x3 = np.einsum('kncd,nl->klcd', t2, de_oo)
        t2_con -= np.einsum('klcd, klcd', t2, t2_con_x3)
        t2_con_x4 = np.einsum('lncd,nk->lkcd', t2, de_oo)
        t2_con += np.einsum('klcd, lkcd', t2, t2_con_x4)
        t2_con_x5 = np.einsum('ke,nlcd,ne->klcd', t1, t2, de_ov)
        t2_con -= np.einsum('klcd, klcd', t2, t2_con_x5)
        t2_con_x6 = np.einsum('le,nkcd,ne->lkcd', t1, t2, de_ov)
        t2_con += np.einsum('klcd, lkcd', t2, t2_con_x6)
        t2_con_x7 = np.einsum('nc,kled,ne->klcd', t1, t2, de_ov)
        t2_con -= np.einsum('klcd, klcd', t2, t2_con_x7)
        t2_con_x8 = np.einsum('nd,klec,ne->kldc', t1, t2, de_ov)
        t2_con += np.einsum('klcd, klcd', t2, t2_con_x8)
        
        # --- Lambda-T-Lin T1-equations ---
        t1_new = t1
        
        t1_la = de_ov
        t1_la += 2*np.einsum('ikac,kc->ia', t2, de_ov)
        t1_la += np.einsum('ic,ac->ia', t1, de_vv)
        t1_la -= np.einsum('ka,ki->ia', t1, de_oo)
        t1_la -= np.einsum('ic,ka,kc->ia', t1, t1, de_ov)
        
        t1_new -= coupling**2*de_1*t1_la        # Lambda-linear T1-correct
        t1_new -= 0.5*coupling**2*t1_la*t1_con  # Lambda-T-Linear T1-correct t1-con
        t1_new -= 0.5*coupling**2*t1_la*t2_con  # Lambda-T-Linear T1-correct t2-con
        
        # --- Lambda-T-Lin T2-equations ---
        t2_new = t2
        
        t2_la  = np.einsum('ijac,bc->ijab', t2, de_vv)
        t2_la -= np.einsum('ijbc,ac->ijba', t2, de_vv)
        t2_la -= np.einsum('ikab,kj->ijab', t2, de_oo)
        t2_la += np.einsum('jkab,ki->jiab', t2, de_oo)
        t2_la -= np.einsum('ic,kjab,kc->ijab', t1, t2, de_ov)
        t2_la += np.einsum('jc,kiab,kc->jiab', t1, t2, de_ov)
        t2_la -= np.einsum('ka,ijcb,kc->ijab', t1, t2, de_ov)
        t2_la += np.einsum('kb,ijca,kc->ijba', t1, t2, de_ov)
        
        t2_new -= coupling**2*de_1*t2_la        # Lambda-linear T2-correct
        t2_new -= 0.5*coupling**2*t2_la*t1_con  # Lambda-T-Linear T2-correct t1-con
        t2_new -= 0.5*coupling**2*t2_la*t2_con  # Lambda-T-Linear T2-correct t2-con

        
        return t1_new, t2_new
    
    my_lambdaT_lin_crp_ccsd.update_amps = update_amps_lambdaT_ccsd
    my_lambdaT_lin_crp_ccsd.kernel()
    
    # --- explicit mean-field linearized CRP-CCSD correlation energy ---
    
    t1, t2 = my_lambdaT_lin_crp_ccsd.t1, my_lambdaT_lin_crp_ccsd.t2  
    nocc, nvir  = t1.shape
    
    eri_mo = eri_ao2mo(mol, crp_rhf_orb)
    de_mo = de_ao2mo(mol, crp_rhf_orb, polarization)
    
    eri_ovov = eri_mo[:nocc,nocc:,:nocc,nocc:]
    de_ov = de_mo[:nocc,nocc:]
    
    ecorr_coul  = 2*np.einsum('ia,jb,iajb', t1, t1, eri_ovov)
    ecorr_coul += 2*np.einsum('ijab,iajb', t2, eri_ovov)
    ecorr_coul -= np.einsum('ia,jb,ibja', t1, t1, eri_ovov)
    ecorr_coul -= np.einsum('ijab, ibja', t2, eri_ovov)
          
    ecorr_dse = 2*coupling**2*np.einsum('ijab, ia, jb', t2, de_ov, de_ov)
    ecorr_dse -= coupling**2*np.einsum('ia,jb,ib, ja', t1, t1, de_ov, de_ov)
    ecorr_dse -= coupling**2*np.einsum('ijab, ib, ja', t2, de_ov, de_ov)
    
    lambdaT_lin_crp_ccsd_ecorr = ecorr_coul + ecorr_dse
    lambdaT_lin_crp_ccsd_e_tot = e_crp_rhf + lambdaT_lin_crp_ccsd_ecorr
    

    return lambdaT_lin_crp_ccsd_e_tot, lambdaT_lin_crp_ccsd_ecorr, my_lambdaT_lin_crp_ccsd
















