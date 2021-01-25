from __future__ import division
import numpy as np
import os
import imp
from multiprocessing import Pool
import scipy.ndimage
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from astropy.table import Table

from cosmosis.datablock import option_section
import cosmo

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return NumberCount.lnlike_field(*arg)

################################################################################
class NumberCount:
    def __init__(self, options):
        ##### Global variables
        self.NPROC = options.get_int(option_section, 'NPROC')
        self.SZmPivot = options.get_double(option_section, 'SZmPivot')
        self.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
        self.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
        ##### SPT survey
        SPTdatafile = options.get_string(option_section, 'SPTdatafile')
        SPTdata = imp.load_source('SPTdata', SPTdatafile)
        SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
        assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
        self.catalog = Table.read(SPTcatalogfile)
        self.SPTfieldNames = SPTdata.SPTfieldNames
        self.SPTfieldCorrection = SPTdata.SPTfieldCorrection
        self.SPTfieldSize = SPTdata.SPTfieldSize
        self.SPTnFalse_alpha = SPTdata.SPTnFalse_alpha
        self.SPTnFalse_beta = SPTdata.SPTnFalse_beta
        ##### Various observable arrays
        # Lin spaced for convo with unit scatter (+3 sigma margin)
        Nxi = int((self.surveyCutSZ[1]+3 - 2.7)/.1 + 1)
        self.xi_bins = np.linspace(2.7, self.surveyCutSZ[1]+3, Nxi)
        self.dxi = self.xi_bins[1] - self.xi_bins[0]
        # ln(zeta(xi_bins))
        self.ln_zeta_xi_arr = np.log(self.xi2zeta(self.xi_bins))
        # dlnzeta/dxi (xi_bins)
        self.dlnzeta_dxi_arr = self.dlnzeta_dxi(self.xi_bins)
        # Arrays over which we'll integrate (survey cuts applied)
        Nxi = int(np.log10(self.surveyCutSZ[1]/self.surveyCutSZ[0])/.005 + 1)
        self.xi_arr = np.logspace(np.log10(self.surveyCutSZ[0]), np.log10(self.surveyCutSZ[1]), Nxi)
        dz = .01
        Nz = int((self.surveyCutRedshift[1]-self.surveyCutRedshift[0])/dz + 1)
        self.z_arr = np.linspace(self.surveyCutRedshift[0], self.surveyCutRedshift[1], Nz)



    ##########
    def lnlike(self, block):
        """Return ln-likelihood for SPT cluster abundance."""
        # Only need cosmo for E(z)-type stuff
        self.cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa')}
        # SZ scaling relation parameters
        self.Asz = block.get_double('mor_parameters', 'Asz')
        self.Bsz = block.get_double('mor_parameters', 'Bsz')
        self.Csz = block.get_double('mor_parameters', 'Csz')
        self.Dsz = block.get_double('mor_parameters', 'Dsz')
        # Halo mass function
        self.HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
            'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
            'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
        self.HMF['len_z'] = len(self.HMF['z_arr'])

        ##### Convert HMF to dN/dln(zeta) = dN/dlog10(M) * dlog10(M)/dln(zeta)
        dlnM_dlnzeta = 1/self.Bsz
        dN_dlnzeta_noScatter = self.HMF['dNdlnM'] * dlnM_dlnzeta
        # Concolve with intrinsic scatter
        dlnzeta = self.Bsz*np.log(self.HMF['M_arr'][1]/self.HMF['M_arr'][0])
        Nbin = self.Dsz / dlnzeta
        self.dN_dlnzeta_unitSolidAng = scipy.ndimage.gaussian_filter1d(dN_dlnzeta_noScatter, Nbin, axis=1, mode='constant')

        ##### Evaluate (log)-likelihood for each SPT field (optional multiprocessing)
        num_fields = len(self.SPTfieldNames)
        if self.NPROC==0:
            field_results = [self.lnlike_field(fieldidx) for fieldidx in range(num_fields)]
        else:
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*num_fields, range(num_fields))
            field_results = pool.map(unwrap_self_f, argin)
            pool.close()
        field_results = np.array(field_results)
        lnlike = np.sum(field_results[:,0])
        Ntotal = np.sum(field_results[:,1])

        return lnlike


    ##########
    def lnlike_field(self, fieldidx):
        """Returns (ln-likelihood, Ntotal) for a given SPT field (index)."""
        # dN/dln(zeta)
        dN_dlnzeta = self.dN_dlnzeta_unitSolidAng * self.SPTfieldSize[fieldidx] * (np.pi/180)**2
        if np.any(dN_dlnzeta==0):
            dN_dlnzeta[np.where(dN_dlnzeta==0)] = np.nextafter(0, 1)

        # zeta[z,M]
        zeta_m = self.mass2zeta(self.HMF['M_arr'], self.HMF['z_arr'])

        # Apply field scaling factor
        zeta_m*= self.SPTfieldCorrection[fieldidx]

        # dN/dxi = dN/dlnzeta dlnzeta/dxi (unconvolved)
        # Unfortunately, the zeta_m table is not regular
        # and repeated spline interp is way too slow (1.6sec per field)
        # So we do linear interpolation (in ln(M), and for ln(dN/dlnzeta))
        dN_dxi = self.dlnzeta_dxi_arr\
            * np.exp(np.array([np.interp(self.ln_zeta_xi_arr, np.log(zeta_m[i]), np.log(dN_dlnzeta[i]))
            for i in range(self.HMF['len_z'])]))

        # Convolve with unit scatter (measurement uncertainty)
        dN_dxi = scipy.ndimage.gaussian_filter1d(dN_dxi, 1/self.dxi, axis=1, mode='constant')
        if np.any(dN_dxi==0):
            dN_dxi[np.where(dN_dxi==0)] = np.nextafter(0, 1)

        # Set up interpolation for cluster list below
        lndNdxi = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.xi_bins), np.log(dN_dxi[1:,:]))

        # Ntotal (trapz except that we sum in log-space)
        integrand = np.exp(.5*(lndNdxi(np.log(self.z_arr), np.log(self.xi_arr[1:])) + lndNdxi(np.log(self.z_arr), np.log(self.xi_arr[:-1]))))\
             * (self.xi_arr[1:]-self.xi_arr[:-1])
        dNdz = np.sum(integrand, axis=1)
        Ntotal = np.trapz(dNdz, self.z_arr)

        # Likelihood contribution from Ntotal
        lnlike_this_field = -Ntotal

        ##### confirmed clusters
        thisfield_conf = np.where((self.catalog['field']==self.SPTfieldNames[fieldidx])
            & (self.catalog['xi']>=self.surveyCutSZ[0]) & (self.catalog['xi']<=self.surveyCutSZ[1])
            & (self.catalog['redshift']>=self.surveyCutRedshift[0]) & (self.catalog['redshift']<=self.surveyCutRedshift[1]))[0]
        for i in thisfield_conf:
            # spec-z: Evaluate dN/dxi/dz at exact location
            if self.catalog['redshift_err'][i]==0.:
                this_lnlike = lndNdxi(np.log(self.catalog['redshift'][i]), np.log(self.catalog['xi'][i]))[0,0]
                lnlike_this_field+= this_lnlike
            # photo-z: \int dz dN/dxi/dz, choose limits to encompass +/- 4 sigma of photo-z error
            elif self.catalog['redshift_err'][i]>0.:
                zlo = max((self.surveyCutRedshift[0], self.catalog['redshift'][i]-4*self.catalog['redshift_err'][i]))
                zhi = min((self.surveyCutRedshift[1], self.catalog['redshift'][i]+4*self.catalog['redshift_err'][i]))
                zarr = np.linspace(zlo, zhi, 15)
                integrand = np.exp(lndNdxi(np.log(zarr), np.log(self.catalog['xi'][i])))[:,0] * norm.pdf(zarr, self.catalog['redshift'][i], self.catalog['redshift_err'][i])
                this_lnlike = np.log(np.trapz(integrand, zarr))
                lnlike_this_field+= this_lnlike

        ##### unconfirmed candidates
        thisfield_unconf = np.where((self.catalog['field']==self.SPTfieldNames[fieldidx])
            & (self.catalog['xi']>=self.surveyCutSZ[0]) & (self.catalog['xi']<=self.surveyCutSZ[1])
            & (self.catalog['redshift']==0.) & (self.catalog['redshift_lim']<=self.surveyCutRedshift[1]))[0]
        for i in thisfield_unconf:
            # If it's a false detection, it's drawn from dN_false/dxi
            dNdxifalse = self.SPTnFalse_beta[fieldidx] * self.SPTfieldSize[fieldidx]/2500 * self.SPTnFalse_alpha[fieldidx]\
                * np.exp(-self.SPTnFalse_beta[fieldidx]*(self.catalog['xi'][i]-5.))
            # If it's a true, unconfirmed cluster, it's drawn from \int_redshift_lim^inf dz dN/dxi/dz
            zarr = np.linspace(self.catalog['redshift_lim'][i], self.HMF['z_arr'][-1], 25)
            dNdxitrue = np.trapz(np.exp(lndNdxi(np.log(zarr), np.log(self.catalog['xi'][i])))[:,0], zarr)
            # Either way, it's drawn from one of these
            this_lnlike = np.log(dNdxifalse + dNdxitrue)
            lnlike_this_field+= this_lnlike

        return lnlike_this_field, Ntotal


    ########## Utility functions

    def dlnzeta_dxi(self, xi):
        return xi/(xi**2 - 3)

    def xi2zeta(self, xi):
        return (xi**2 - 3)**.5

    def mass2zeta(self, mass, z):
        # [redshift][mass]
        massterm = (mass/self.SZmPivot)**self.Bsz
        zterm = (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.Csz
        return self.Asz * massterm[None,:] * zterm[:,None]
