from __future__ import division
import numpy as np
from numpy.lib import scimath as sm
from scipy.stats import norm
import pickle
import imp
import os

from cosmosis.datablock import option_section
import cosmo

########################################
##### This class reads and stores shear data and calculates P(shear|P(M))
class SPTlensing:

    def __init__(self, options, catalog):
        # WL simulation calibration data
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration
        # Lensing data
        self.HSTfile = options.get_string(option_section, 'HSTfile')
        self.MegacamDir = options.get_string(option_section, 'MegacamDir')
        # I don't know how to pass a None
        if self.HSTfile=='None': self.HSTfile = None
        if self.MegacamDir=='None': self.MegacamDir = None
        self.readdata(catalog)


    ########################################
    # Get P(Mwl) from dP/dMwl and shear data
    def like(self, data, dataindex, mArr, cosmology, MCrel, lnM500_to_lnM200):
        """Return likelihood of shear profile for a given cluster (index) given
        an array of cluster masses."""
        self.name = data['SPT_ID'][dataindex]
        self.zcluster = data['redshift'][dataindex]
        self.WLdata = data['WLdata'][dataindex]

        ##### Precalculate M and r independent stuff, everything in h units
        self.rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.zcluster, cosmology)**2 # [h^2 Msun/Mpc^3]
        Dl = cosmo.dA(self.zcluster, cosmology)
        self.get_beta(cosmology)

        ##### M200 and scale radius, wrt critical density, everything in h units
        M200c = np.exp(lnM500_to_lnM200(self.zcluster, np.log(mArr)))[0]
        r200c = (3.*M200c/4./np.pi/200./self.rho_c_z)**(1./3.)
        c200c = MCrel.calC200(M200c, self.zcluster)
        self.delta_c = 200./3. * c200c**3. / (np.log(1.+c200c) - c200c/(1.+c200c))
        self.rs = r200c/c200c

        ##### dimensionless radial distance [Radius][Mass]
        self.x_2d = self.WLdata['r_deg'][:,None] * Dl * np.pi/180. / self.rs[None,:]


        #################### Megacam: no magnitude bin stuff
        if self.WLdata['datatype']!='HST':
            # Sigma_crit, with c^2/4piG [h Msun/Mpc^2]
            Sigma_c = 1.6624541593797974e+18/Dl/self.beta_avg

            # gamma_t [Radius][Mass]
            gamma_2d = self.get_Delta_Sigma() / Sigma_c

            # kappa [Radius][Mass]
            kappa_2d = self.get_Sigma() / Sigma_c

            # Reduced shear g_t [Radius][Mass]
            g_2d = gamma_2d/(1-kappa_2d) * (1 + kappa_2d*(self.beta2_avg/self.beta_avg**2-1))

            # Keep all radial bins (make cut in data)
            rInclude = range(len(self.WLdata['r_deg']))


        #################### HST data
        ##### HST: beta(r) because of magnitude bins
        else:
            # Sigma_crit, with c^2/4piG [h Msun/Mpc^2] [Radius]
            rangeR = range(len(self.WLdata['r_deg']))
            betaR = np.array([self.beta_avg[self.WLdata['magbinids'][i]] for i in rangeR])
            beta2R = np.array([self.beta2_avg[self.WLdata['magbinids'][i]] for i in rangeR])
            Sigma_c = 1.6624541593797974e+18/Dl/betaR

            # gamma_t [Radius][Mass]
            gamma_2d = self.get_Delta_Sigma() / Sigma_c[:,None]

            # kappa [Radius][Mass]
            kappa_2d = self.get_Sigma() / Sigma_c[:,None]

            # [Radius][Mass]
            mu0_2d = 1./((1.-kappa_2d)**2 - gamma_2d**2)
            kappaFake = (mu0_2d-1)/2.

            # Magnification correction [Radius][Mass]
            mykappa = kappaFake * 0.3/betaR[:,None]

            magcorr = [np.interp(mykappa[i], self.WLdata['magcorr'][self.WLdata['magbinids'][i]][0], self.WLdata['magcorr'][self.WLdata['magbinids'][i]][1]) for i in rangeR]

            # Beta correction [Radius][Mass]
            betaratio = beta2R/betaR**2
            betaCorr = (1 + kappa_2d*(betaratio[:,None]-1))

            # Reduced shear g_t [Radius][Mass]
            g_2d = np.array(magcorr) * gamma_2d/(1-kappa_2d) * betaCorr

            # Only consider 500<r/kpc/1500 in reference cosmology
            cosmoRef = {'Omega_m':.3, 'Omega_l':.7, 'h':.7, 'w0':-1., 'wa':0}
            DlRef = cosmo.dA(self.zcluster, cosmoRef)
            rPhysRef = self.WLdata['r_deg'] * DlRef * np.pi/180. /cosmoRef['h']
            rInclude = np.where((rPhysRef>.5)&(rPhysRef<1.5))[0]


        #################### Back to common code

        ##### Compare with data [Radius][Mass]
        # Likelihood grid [Radius][Mass]
        likelihood = norm.pdf(g_2d[rInclude,:], self.WLdata['shear'][rInclude,None], self.WLdata['shearerr'][rInclude,None])

        # Return array of P(data|MassArray)
        # Note that this is not normalized wrt the mArr for a good reason:
        # In general, the mArr will not cover the full pOfMass range, and it varies as a function of SZ parameters.
        # However, pOfMass is a product of normalized distributions, and so its normalization is constant
        # throughout parameter space.
        pOfMass = np.prod(likelihood, axis=0)

        return pOfMass


    ########################################
    # dA [Mpc/h]
    def get_dAs(self, cosmology):
        """Precompute angular diameter distances for an array of redshifts."""
        zs = np.logspace(-1,np.log10(5),100)
        dA = np.array([cosmo.dA(z, cosmology) for z in zs])
        self.dAs = {'lnz':np.log(zs), 'lndA':np.log(dA)}


    ########################################
    def get_beta(self, cosmology):
        """Compute <beta> and <beta^2> from distribution of redshift galaxies."""
        ##### Only consider redshift bins behind the cluster
        betaArr = np.zeros(len(self.WLdata['redshifts']))
        bgIdx = np.where(self.WLdata['redshifts']>self.zcluster)[0]

        ##### Calculate beta(z_source)
        betaArr[bgIdx] = np.array([cosmo.dA_two_z(self.zcluster, z, cosmology) for z in self.WLdata['redshifts'][bgIdx]])
        betaArr[bgIdx]/= np.exp(np.interp(np.log(self.WLdata['redshifts'][bgIdx]), self.dAs['lnz'], self.dAs['lndA']))

        ##### Weight beta(z) with N(z) distribution to get <beta> and <beta^2>
        if self.WLdata['datatype']!='HST':
            self.beta_avg = np.sum(self.WLdata['Nz']*betaArr)/self.WLdata['Ntot']
            self.beta2_avg = np.sum(self.WLdata['Nz']*betaArr**2)/self.WLdata['Ntot']
        else:
            self.beta_avg, self.beta2_avg = {}, {}
            for i in self.WLdata['pzs'].keys():
                self.beta_avg[i] = np.sum(self.WLdata['pzs'][i]*betaArr)/self.WLdata['Ntot'][i]
                self.beta2_avg[i] = np.sum(self.WLdata['pzs'][i]*betaArr**2)/self.WLdata['Ntot'][i]


    ########################################
    ##### Compute the inverse sec of the complex number z.
    # by Joerg Dietrich
    def arcsec(self, z):
        val1 = 1j / z
        val2 = sm.sqrt(1 - 1./z**2)
        val = 1j * np.log(val2 + val1)
        return .5 * np.pi + val


    ########################################
    ##### Delta Sigma[Radius][Mass]
    # by Joerg Dietrich
    def get_Delta_Sigma(self):
        fac = 2 * self.rs * self.rho_c_z * self.delta_c
        val1 = 1. / (1 - self.x_2d**2)
        num = ((3 * self.x_2d**2) - 2) * self.arcsec(self.x_2d)
        div = self.x_2d**2 * (sm.sqrt(self.x_2d**2 - 1))**3
        val2 = (num / div).real
        val3 = 2 * np.log(self.x_2d / 2) / self.x_2d**2
        return fac * (val1+val2+val3)


    ########################################
    ##### Sigma_NFW[Radius][Mass]
    # by Joerg Dietrich
    def get_Sigma(self):
        val1 = 1. / (self.x_2d**2 - 1)
        val2 = (self.arcsec(self.x_2d) / (sm.sqrt(self.x_2d**2 - 1))**3).real
        return 2 * self.rs * self.rho_c_z * self.delta_c * (val1-val2)




    ########################################
    def readdata(self, catalog):
        """Read and load weak-lensing data into `WLdata` field in `catalog` if
        the corresponding path-variables lead to valid files on disk. Otherwise,
        no data is read, so you better be careful."""
        # "Allocate" empty data field
        catalog['WLdata'] = [None for i in range(len(catalog['SPT_ID']))]

        ##### Check for HST data
        if self.HSTfile is not None:
            assert os.path.isfile(self.HSTfile), "HST shear data %s not found"%self.HSTfile
            # Load weak lensing data
            # HSTdataclass = HSTCluster(self.HSTfile)
            HSTdata = pickle.load(open(self.HSTfile, 'rb'))
            for i,name in enumerate(catalog['SPT_ID']):
                if name in HSTdata.keys():
                    # Total number of background galaxies
                    Ntot, pzs = {}, {}
                    for j in HSTdata[name]['pzs'].keys():
                        pzs[j] = np.sum(HSTdata[name]['pzs'][j], axis=0)
                        Ntot[j] = np.sum(HSTdata[name]['pzs'][j])
                    catalog['WLdata'][i] = {'datatype':'HST', 'center':HSTdata[name]['center'],
                        'r_deg':HSTdata[name]['r_deg'], 'shear':HSTdata[name]['shear'], 'shearerr':HSTdata[name]['shearerr'],
                        'magbinids':HSTdata[name]['magbinids'], 'redshifts':HSTdata[name]['redshifts'], 'pzs':pzs, 'magcorr':HSTdata[name]['magnificationcorr'], 'Ntot':Ntot,
                        'massModelErr': (self.WLcalib['HSTsim'][name][1]**2 + self.WLcalib['HSTmcErr']**2 + self.WLcalib['HSTcenterErr']**2)**.5,
                        'zDistShearErr': (self.WLcalib['HSTzDistErr']**2 + self.WLcalib['HSTshearErr']**2)**.5}

        ##### Megacam data
        if self.MegacamDir is not None:
            assert os.path.isdir(self.MegacamDir), "Megacam data %s not found"%self.MegacamDir
            for i,name in enumerate(catalog['SPT_ID']):
                prefix = self.MegacamDir+'/'+name+'/'+name
                if os.path.isfile(prefix+'_shear.txt'):
                    shear = np.loadtxt(prefix+'_shear.txt', unpack=True)
                    Nz = np.loadtxt(prefix+'_Nz.txt', unpack=True)
                    catalog['WLdata'][i] = {'datatype':'Megacam', 'r_deg':shear[0], 'shear':shear[1], 'shearerr':shear[2],
                        'redshifts':Nz[0], 'Nz':Nz[1], 'Ntot':np.sum(Nz[1]),
                        'massModelErr': (self.WLcalib['MegacamSim'][1]**2 + self.WLcalib['MegacamMcErr']**2 + self.WLcalib['MegacamCenterErr']**2)**.5,
                        'zDistShearErr': (self.WLcalib['MegacamzDistErr']**2 + self.WLcalib['MegacamShearErr']**2)**.5}


    #######################################
    def set_scaling(self, scaling):
        """Set total (or effective) bias and scatter for Megacam using the
        simulation calibration numbers and the nuissance parameters."""
        # Megacam
        if self.MegacamDir is not None:
            massModelErr = (self.WLcalib['MegacamSim'][1]**2 + self.WLcalib['MegacamMcErr']**2 + self.WLcalib['MegacamCenterErr']**2)**.5
            zDistShearErr = (self.WLcalib['MegacamzDistErr']**2 + self.WLcalib['MegacamShearErr']**2 + self.WLcalib['MegacamContamCorr']**2)**.5
            # bias = bSim + bMassModel + (bN(z)+bShearCal)
            scaling['bWL_Megacam'] = self.WLcalib['MegacamSim'][0] + scaling['WLbias']*massModelErr + scaling['MegacamBias']*zDistShearErr
            # lognormal scatter
            scaling['DWL_Megacam'] = self.WLcalib['MegacamSim'][2]+scaling['WLscatter']*self.WLcalib['MegacamSim'][3]
