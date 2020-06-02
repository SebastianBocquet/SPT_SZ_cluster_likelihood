from __future__ import division
import numpy as np
import scipy.integrate

DIST_H = 2997.92458
RHOCRIT = 2.77528233987e11

def Ez(z, cosmology):
    """Return the dimensionless Hubble parameter."""
    return (cosmology['Omega_m']*(1+z)**3 + cosmology['Omega_l']*(1+z)**(3*(1+cosmology['w0']+cosmology['wa']))*np.exp(-3*cosmology['wa']*z/(1+z)))**.5

def Omega_m_z(z, cosmology):
    """Return Omega_m(z)."""
    return cosmology['Omega_m'] * (1+z)**3 / Ez(z, cosmology)**2

def dA(z, cosmology):
    """Return angular diameter distance in Mpc/h."""
    integrand = lambda z_int: 1/Ez(z_int, cosmology)
    return scipy.integrate.quad(integrand, 0., z)[0] *DIST_H/(1+z)

def dA_two_z(z1, z2, cosmology):
    """Return angular diameter distance between two redshifts (z1<z2) in Mpc/h."""
    integrand = lambda z_int: 1/Ez(z_int, cosmology)
    return scipy.integrate.quad(integrand, z1, z2)[0] * DIST_H/(1+z2)

def deltaV(z_arr, cosmology):
    """Return solid angle volume as a function of redshift [(Mpc/h)^3]."""
    dA_arr = [dA(z, cosmology) for z in z_arr]
    return DIST_H*((1+z_arr)*dA_arr)**2 / Ez(z_arr, cosmology)
