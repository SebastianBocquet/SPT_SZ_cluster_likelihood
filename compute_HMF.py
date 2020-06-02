from __future__ import division
import numpy as np
import imp
from scipy.interpolate import interp1d
import scipy.integrate
from cosmosis.datablock import option_section
import cosmo

class HMFCalculator:
    def __init__(self, options):
        """Initialize Tinker interpolation functions and critical overdensity,
        and store in self."""
        self.Deltacrit = options.get_double(option_section, 'Deltacrit', default=500.)
        # Initialize Tinker interpolation (A, a, b, c)
        x = np.log((200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.))
        y = (1.858659e-01, 1.995973e-01, 2.115659e-01, 2.184113e-01, 2.480968e-01, 2.546053e-01, 2.600000e-01, 2.600000e-01, 2.600000e-01)
        self.interp_A = interp1d(x, y, kind='cubic')
        y = (1.466904e+00, 1.521782e+00, 1.559186e+00, 1.614585e+00, 1.869936e+00, 2.128056e+00, 2.301275e+00, 2.529241e+00, 2.661983e+00)
        self.interp_a = interp1d(x, y, kind='cubic')
        y = (2.571104e+00, 2.254217e+00, 2.048674e+00, 1.869559e+00, 1.588649e+00, 1.507134e+00, 1.464374e+00, 1.436827e+00, 1.405210e+00)
        self.interp_b = interp1d(x, y, kind='cubic')
        y = (1.193958e+00, 1.270316e+00, 1.335191e+00, 1.446266e+00, 1.581345e+00, 1.795050e+00, 1.965613e+00, 2.237466e+00, 2.439729e+00)
        self.interp_c = interp1d(x, y, kind='cubic')


    def compute_HMF(self, block):
        """Compute Tinker HMF, apply redshift volume, and add to block."""
        ##### Setup
        # Only need cosmo for E(z)-type stuff
        cosmology = {
            'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_nu': block.get_double('cosmological_parameters', 'Omega_nu'),
            'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa')}
        rho_m = (cosmology['Omega_m'] - cosmology['Omega_nu']) * cosmo.RHOCRIT
        # Data arrays
        M_arr = np.logspace(13, 16, 301)
        z_arr = block.get_double_array_1d('matter_power_lin', 'z')
        k_arr = block.get_double_array_1d('matter_power_lin', 'k_h')
        Pk = block.get_double_array_nd('matter_power_lin', 'p_k')
        # Mean overdensity at each redshift
        Deltamean = self.Deltacrit / cosmo.Omega_m_z(z_arr, cosmology)

        ##### Compute sigma(M)
        # Radius [M_arr]
        R = (3 * M_arr / (4 * np.pi * rho_m))**(1/3)
        # [M_arr, k_arr]
        kR = k_arr[None,:] * R[:,None]
        # Window functions [M_arr, k_arr]
        window = 3 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
        dwindow = 3/kR**4 * (3*kR*np.cos(kR) + ((kR**2 - 3)*np.sin(kR)))
        # Integrands [z_arr, M_arr, k_arr]
        integrand_sigma2 = Pk[:,None,:] * window[None,:,:]**2 * k_arr[None,None,:]**3
        integrand_dsigma2dM = Pk[:,None,:] * window[None,:,:] * dwindow[None,:,:] * k_arr[None,None,:]**4
        # Sigma^2 and dsigma^2/dM [z_arr, M_arr]
        sigma2 = .5/np.pi**2 * np.trapz(integrand_sigma2, np.log(k_arr), axis=-1)
        dsigma2dM = np.pi**-2 * R[None,:]/M_arr[None,:]/3 * np.trapz(integrand_dsigma2dM, np.log(k_arr), axis=-1)

        ##### Compute Tinker HMF (unit volume)
        A, a, b, c = np.array([self.Tinker_params(z_arr[i], Deltamean[i]) for i in range(len(z_arr))]).T
        # HMF [z_arr, M_arr]
        fsigma = A[:,None] * ((sigma2**.5/b[:,None])**-a[:,None] + 1) * np.exp(-c[:,None]/sigma2)
        dNdlnM_noVol = - fsigma * rho_m * dsigma2dM/2/sigma2

        ##### Apply redshift volume
        deltaV = cosmo.deltaV(z_arr[1:], cosmology)
        deltaV = np.insert(deltaV, 0, deltaV[1])
        dNdlnM = dNdlnM_noVol * deltaV[:,None]

        ##### Store HMF
        block.put_double_array_1d('HMF', 'M_arr', M_arr)
        block.put_double_array_1d('HMF', 'z_arr', z_arr)
        block.put_double_array_nd('HMF', 'dNdlnM_unitVol', dNdlnM_noVol)
        block.put_double_array_nd('HMF', 'dNdlnM', dNdlnM)


    def Tinker_params(self, z, Deltamean):
        """For given redshift and mean overdensity, return list of four Tinker
        parameters. If outside defined overdensity, return last valid number."""
        # Parameters at z=0
        if Deltamean<200:
            A, a, b, c = [.186, 1.47, 2.57, 1.19]
        elif Deltamean>3200:
            A, a, b, c = [.26, 2.66, 1.41, 2.44]
        else:
            lnDeltamean = np.log(Deltamean)
            A, a, b, c = [self.interp_A(lnDeltamean), self.interp_a(lnDeltamean), self.interp_b(lnDeltamean), self.interp_c(lnDeltamean)]
        # Redshift evolution
        logalpha = -(.75/np.log10(Deltamean/75.))**1.2
        alpha = 10**logalpha
        A*= (1+z)**-.14
        a*= (1+z)**-.06
        b*= (1+z)**-alpha
        return A, a, b, c
