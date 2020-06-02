from __future__ import division
import numpy as np
import os
import imp
from multiprocessing import Pool
import scipy.ndimage
import scipy.special as ss
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
from scipy import signal
from scipy.stats import norm
from scipy.stats import multivariate_normal
from astropy.table import Table

from cosmosis.datablock import option_section
import cosmo, Mconversion_concentration, lensing, observablecovmat

cosmologyRef = {'Omega_m':.272, 'Omega_l':.728, 'h':.702, 'w0':-1, 'wa':0}
getpull = False

# Because multiprocessing within classes doesn't really work...
def unwrap_self_f(arg):
    return MassCalibration.clusterlike(*arg)

################################################################################
class MassCalibration:

    def __init__(self, options):
        ##### Config parameters
        self.todo = {
            'WL': options.get_bool(option_section, 'doWL'),
            'Yx': options.get_bool(option_section, 'doYx'),
            'Mgas': options.get_bool(option_section, 'doMgas'),
            }
        self.SZmPivot = options.get_double(option_section, 'SZmPivot')
        self.XraymPivot = options.get_double(option_section, 'XraymPivot')
        self.mcType = options.get_string(option_section, 'mcType')
        self.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
        self.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
        self.NPROC = options.get_int(option_section, 'NPROC')
        ##### SPT survey
        # Data
        SPTdatafile = options.get_string(option_section, 'SPTdatafile')
        SPTdata = imp.load_source('SPTdata', SPTdatafile)
        SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
        assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
        self.catalog = Table.read(SPTcatalogfile)
        # Survey specs
        self.SPTfieldNames = SPTdata.SPTfieldNames
        self.SPTfieldCorrection = SPTdata.SPTfieldCorrection
        self.SPTdoubleCount = SPTdata.SPTdoubleCount
        ##### WL simulation calibration
        WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
        WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)
        self.WLcalib = WLsimcalib.WLcalibration

        # Weak lensing
        if self.todo['WL']:
            self.WL = lensing.SPTlensing(options, self.catalog)


    ############################################################################
    def lnlike(self, block):
        """Returns ln-likelihood for mass calibration of the whole cluster sample."""
        ##### Extract from datablock
        self.cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
            'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
            'Omega_b': block.get_double('cosmological_parameters', 'Omega_b'),
            'h': block.get_double('cosmological_parameters', 'hubble')/100,
            'ns': block.get_double('cosmological_parameters', 'n_s'),
            'w0': block.get_double('cosmological_parameters', 'w'),
            'wa': block.get_double('cosmological_parameters', 'wa'),
            'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
        self.scaling = {
            # SZ
            'Asz': block.get_double('mor_parameters', 'Asz'),
            'Bsz': block.get_double('mor_parameters', 'Bsz'),
            'Csz': block.get_double('mor_parameters', 'Csz'),
            'Dsz': block.get_double('mor_parameters', 'Dsz'),
            # X-ray
            'Ax': block.get_double('mor_parameters', 'Ax'),
            'Bx': block.get_double('mor_parameters', 'Bx'),
            'Cx': block.get_double('mor_parameters', 'Cx'),
            'Dx': block.get_double('mor_parameters', 'Dx'),
            'dlnMg_dlnr': block.get_double('mor_parameters', 'dlnMg_dlnr'),
            # WL
            'WLbias': block.get_double('mor_parameters', 'WLbias'),
            'WLscatter': block.get_double('mor_parameters', 'WLscatter'),
            'HSTbias': block.get_double('mor_parameters', 'HSTbias'),
            'HSTscatterLSS': block.get_double('mor_parameters', 'HSTscatterLSS'),
            'MegacamBias': block.get_double('mor_parameters', 'MegacamBias'),
            'MegacamScatterLSS': block.get_double('mor_parameters', 'MegacamScatterLSS'),
            # Correlation coefficients
            'rhoSZWL': block.get_double('mor_parameters', 'rhoSZWL'),
            'rhoWLX': block.get_double('mor_parameters', 'rhoWLX'),
            'rhoSZX': block.get_double('mor_parameters', 'rhoSZX'),
            }
        # Halo mass function
        self.HMF = {'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
            'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
            'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
        self.HMF['len_z'] = len(self.HMF['z_arr'])

        ##### Setup stuff for WL
        if self.todo['WL']:
            # Set bias and scatter for Megacam from sim calibration
            # and nuissance parameters
            self.WL.set_scaling(self.scaling)
            # Precompute array of angular diameter distances
            self.WL.get_dAs(self.cosmology)

        ##### Populate and check observable covariance matrices
        self.covmat = {'invertible': True}
        if not observablecovmat.set_covmats(self.todo, self.scaling, self.covmat):
            self.covmat['invertible'] = False
            return -np.inf

        ##### Set up interpolation for HMF
        HMF_in = self.HMF['dNdlnM'][1:,:]
        if np.any(HMF_in==0):
            HMF_in[np.where(HMF_in==0)] = np.nextafter(0, 1)
        self.HMF_interp = RectBivariateSpline(np.log(self.HMF['z_arr'][1:]), np.log(self.HMF['M_arr']), np.log(HMF_in))

        ##### Initialize mass-concentration relation class (for WL)
        if self.todo['WL']:
            self.MCrel = Mconversion_concentration.ConcentrationConversion(self.mcType, self.cosmology)

        ##### Compute interpolation table for M500-M200
        if self.todo['WL']:
            z_arr = np.linspace(.1, 2, 20)
            M500 = np.logspace(np.log10(self.HMF['M_arr'][0]), np.log10(self.HMF['M_arr'][-1]), 20)
            M200 = np.array([np.array([self.MCrel.MDelta_to_M200(m, 500., z) for m in M500]) for z in z_arr])
            self.lnM500_to_lnM200 = RectBivariateSpline(z_arr, np.log(M500), np.log(M200))

        ##### Evaluate the individual likelihoods
        len_data = len(self.catalog['SPT_ID'])

        if self.NPROC==0:
            # Iterate through cluster list
            likelihoods = np.array([self.clusterlike(i) for i in range(len_data)])
        else:
            # Launch a multiprocessing pool and get the likelihoods
            pool = Pool(processes=self.NPROC)
            argin = zip([self]*len_data, range(len_data))
            likelihoods = pool.map(unwrap_self_f, argin)
            pool.close()

        # If likelihood computation failed it returned 0
        if np.count_nonzero(likelihoods)<len_data:
            return -np.inf

        lnlike = np.sum(np.log(likelihoods))

        return lnlike



    ############################################################################
    def clusterlike(self, i):
        """Return multi-wavelength mass-calibration likelihood (no log!) for a
        given cluster (index) by calling get_P_1obs_xi or get_P_2obs_xi or
        returning 1 if no follow-up data is available."""
        name = self.catalog['SPT_ID'][i]

        ##### Do we actually want this guy? (some clusters in SPT-SZ are at field boundaries)
        if (name,self.catalog['field'][i]) in self.SPTdoubleCount: return 1.
        if not self.surveyCutSZ[0]<self.catalog['xi'][i]<self.surveyCutSZ[1] or not self.surveyCutRedshift[0]<self.catalog['redshift'][i]<self.surveyCutRedshift[1]: return 1

        ##### Check if follow-up is available
        nobs = 0
        obsnames = []
        if self.todo['WL'] and self.catalog['WLdata'][i] is not None:
            nobs+= 1
            if self.catalog['WLdata'][i]['datatype']=='Megacam':
                obsnames.append('WLMegacam')
            elif self.catalog['WLdata'][i]['datatype']=='HST':
                obsnames.append('WLHST')
                # bias = bSim + bMassModel + (bN(z)+bShearCal)
                self.scaling['bWL_HST'] = self.WLcalib['HSTsim'][name][0] + self.scaling['WLbias']*self.catalog['WLdata'][i]['massModelErr'] + self.scaling['HSTbias']*self.catalog['WLdata'][i]['zDistShearErr']
                # lognormal scatter
                self.scaling['DWL_HST'] = self.WLcalib['HSTsim'][name][2]+self.scaling['WLscatter']*self.WLcalib['HSTsim'][name][3]
                cov = [[self.scaling['DWL_HST']**2, self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST']],
                    [self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST'], self.scaling['Dsz']**2]]
                if np.linalg.det(cov)<observablecovmat.THRESHOLD:
                    return 0.
                self.covmat['WLHST'] = cov

        if self.todo['Yx'] and self.catalog['Mg_fid'][i]!=0:
            nobs+= 1
            obsnames.append('Yx')
        if self.todo['Mgas'] and self.catalog['Mg_fid'][i]!=0:
            nobs+= 1
            obsnames.append('Mgas')
        if nobs==0:
            return 1.

        ##### Set SPT field scaling factor
        self.thisSPTfieldCorrection = self.SPTfieldCorrection[self.SPTfieldNames.index(self.catalog['field'][i])]

        #####
        if nobs==1:
            probability = self.get_P_1obs_xi(obsnames[0], i)

        elif nobs==2:
            if 'WLMegacam' in obsnames: covname = 'XrayMegacam'
            elif 'WLHST' in obsnames:
                covname = 'XrayHST'
                cov = [[self.scaling['DWL_HST']**2, self.scaling['rhoWLX']*self.scaling['DWL_HST']*self.scaling['Dx'], self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST']],
                [self.scaling['rhoWLX']*self.scaling['DWL_HST']*self.scaling['Dx'], self.scaling['Dx']**2, self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx']],
                [self.scaling['rhoSZWL']*self.scaling['Dsz']*self.scaling['DWL_HST'], self.scaling['rhoSZX']*self.scaling['Dsz']*self.scaling['Dx'], self.scaling['Dsz']**2]]
                if np.linalg.det(cov)<observablecovmat.THRESHOLD:
                    return 0.
                self.covmat[covname] = cov
            if self.scaling['rhoWLX']==0:
                probability = self.get_P_1obs_xi(obsnames[0], i) * self.get_P_1obs_xi(obsnames[1], i)
            else:
                probability = self.get_P_2obs_xi(obsnames[:2], i, covname)

        else:
            raise ValueError(name,"has",nobs,"follow-up observables. I don't know what to do!")

        if (probability<0) | (np.isnan(probability)):
            return 0
            # raise ValueError("P(obs|xi) =", probability, name)

        return probability




    ############################################################################
    def get_P_1obs_xi(self, obsname, dataID):
        """Returns P(obs|xi,z,p) for a single type of follow-up data."""
        covmat = self.covmat[obsname]

        ##### Get the follow-up observable, obsintr is used for setting up mass range
        if obsname=='Yx':
            obsmeas, obsintr, obserr = self.catalog['Yx_fid'][dataID], self.scaling['Dx'], self.catalog['Yx_err'][dataID]
        elif obsname=='Mgas':
            obsmeas, obsintr, obserr = self.catalog['Mg_fid'][dataID], self.scaling['Dx'], self.catalog['Mg_err'][dataID]
        elif obsname=='WLMegacam':
            LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
            obsmeas, obserr, obsintr = .8*self.scaling['bWL_Megacam']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_Megacam']
        elif obsname=='WLHST':
            LSSnoise = self.WLcalib['HST_LSS'][0] + self.scaling['HSTscatterLSS'] * self.WLcalib['HST_LSS'][1]
            obsmeas, obserr, obsintr = .8*self.scaling['bWL_HST']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_HST']

        ##### Define reasonable mass range
        # xi -> M(xi)
        xi_minmax = np.array([max(2.6,self.catalog['xi'][dataID]-5), self.catalog['xi'][dataID]+3])
        M_xi_minmax = self.obs2mass('zeta', self.xi2zeta(xi_minmax), self.catalog['redshift'][dataID])
        if M_xi_minmax[0]>self.HMF['M_arr'][-1]:
            print "cluster mass exceeds HMF mass range", self.catalog['SPT_ID'][dataID],\
                M_xi_minmax[0], self.HMF['M_arr'][-1]
            return 0

        # obs: prediction
        lnobs0 = np.log(self.mass2obs(obsname, self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), self.catalog['redshift'][dataID]))
        SZscatterobs = self.dlnM_dlnobs('zeta') / self.dlnM_dlnobs(obsname, self.SZmPivot, self.catalog['redshift'][dataID]) * self.scaling['Dsz']
        intrscatter = (SZscatterobs**2 + obsintr**2)**.5
        obsthminmax = np.exp(np.array([lnobs0-5.*intrscatter, lnobs0+3.5*intrscatter]))
        M_obsth_minmax = self.obs2mass(obsname, obsthminmax, self.catalog['redshift'][dataID])
        # obs: measurement
        if obsname in ('Mgas', 'Yx'):
            obsmeasminmax = np.amax((.1, obsmeas-3*obserr)), obsmeas+3*obserr
        else:
            obsmeasminmax = np.exp(np.log(obsmeas)-4*obserr), np.exp(np.log(obsmeas)+3*obserr)
        M_obsmeas_minmax = self.obs2mass(obsname, np.array(obsmeasminmax), self.catalog['redshift'][dataID])

        ##### Define grid in mass
        Mmin, Mmax = min(M_xi_minmax[0], M_obsth_minmax[0], M_obsmeas_minmax[0]), max(M_xi_minmax[1], M_obsth_minmax[1], M_obsmeas_minmax[1])
        Mmin, Mmax = max(.5*Mmin, self.HMF['M_arr'][0]), min(Mmax, self.HMF['M_arr'][-1])
        lenObs = 54
        M_obsArr = np.logspace(np.log10(Mmin), np.log10(Mmax), lenObs)

        ##### Observable arrays
        lnzeta_arr = np.log(self.mass2obs('zeta', M_obsArr, self.catalog['redshift'][dataID]))
        xi_arr = self.zeta2xi(np.exp(lnzeta_arr))
        obsArr = self.mass2obs(obsname, M_obsArr, self.catalog['redshift'][dataID])

        ##### Add radial dependence for X-ray observables
        if obsname in ('Mgas','Yx'):
            # Angular diameter distances in current and reference cosmology [Mpc]
            dA = cosmo.dA(self.catalog['redshift'][dataID], self.cosmology)/self.cosmology['h']
            dAref = cosmo.dA(self.catalog['redshift'][dataID], cosmologyRef)/cosmologyRef['h']
            # R500 [kpc]
            rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['redshift'][dataID], self.cosmology)**2
            r500 = 1000 * (3*M_obsArr/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
            # r500 in reference cosmology [kpc]
            r500ref = r500 * dAref/dA
            # Xray observable at fiducial r500...
            obsArr*= (self.catalog['r500'][dataID]/r500ref)**self.scaling['dlnMg_dlnr']
            # ... corrected to reference cosmology
            obsArr*= (dAref/dA)**2.5

        lnobsArr = np.log(obsArr)

        ##### HMF array for convolution
        M_HMF_arr = M_obsArr

        ##### Convert self.HMF to dN/(dlnzeta dlnobs) = dN/dlnM * dlnM/dlnzeta * dlnM/dlnobs
        # This only matter if dlnM/dlnobs is mass-dependent, as for dispersions
        dN_dlnzeta_dlnobs = np.exp(self.HMF_interp(np.log(self.catalog['redshift'][dataID]), np.log(M_HMF_arr)))[0]

        ##### HMF on 2D observable grid
        HMF_2d_in = np.zeros((lenObs, lenObs))
        np.fill_diagonal(HMF_2d_in, dN_dlnzeta_dlnobs)

        ##### 2D convolution with correlated scatter [lnobs,lnzeta]
        pos = np.empty((lenObs,lenObs,2))
        pos[:,:,0], pos[:,:,1] = np.meshgrid(lnobsArr, lnzeta_arr, indexing='ij')
        kernel = multivariate_normal.pdf(pos, mean=(lnobsArr[27], lnzeta_arr[27]), cov=covmat)
        HMF_2d = signal.fftconvolve(HMF_2d_in, kernel, mode='same')

        # set to 0 if zeta<2
        HMF_2d[:,np.where(lnzeta_arr<np.log(2.))] = 0.

        # Set small negative values to zero (FFT noise)
        if np.any(HMF_2d<-1e-7):
            if np.abs(np.amin(HMF_2d))/np.amax(HMF_2d)>1e-6:
                print "HMF_2d has negative entries:",np.amin(HMF_2d), np.amax(HMF_2d)
        HMF_2d[np.where(HMF_2d<0)] = 0.

        # Safety check
        if np.all(HMF_2d==0.):
            print self.catalog['SPT_ID'][dataID],'HMF_2d is zero, det',np.linalg.det(covmat),self.scaling['Dsz'],obsintr,self.scaling['rhoSZX']
            return 0.

        ##### dN/(dxi dlnobs) = dN/(dlnzeta dlnobs) * dlnzeta/dxi [lnobs,xi]
        HMF_2d*= self.dlnzeta_dxi(xi_arr)[None,:]

        #### Convolve with xi measurement error [lnobs]
        dP_dlnobs = np.trapz(HMF_2d * norm.pdf(self.catalog['xi'][dataID], xi_arr[None,:], 1.), xi_arr, axis=1)


        ##### Evaluate likelihood
        #dP/dobs = dP/dlnobs * dlnobs/dobs = dP/dlnobs /obs
        dP_dobs = dP_dlnobs/obsArr
        # normalize
        dP_dobs/= np.trapz(dP_dobs, obsArr)

        ##### WL
        if obsname in ('WLHST', 'WLMegacam'):
            # Concolve with Gaussian LSS scatter
            if LSSnoise>0.:
                integrand = dP_dobs[None,:] * norm.pdf(obsArr[:,None], obsArr[None,:], LSSnoise)
                dP_dobs = np.trapz(integrand, obsArr, axis=1)
                dP_dobs/= np.trapz(dP_dobs, obsArr)
            # P(Mwl) from data
            Pwl = self.WL.like(self.catalog, dataID, obsArr, self.cosmology, self.MCrel, self.lnM500_to_lnM200)
            # Get likelihood
            likeli = np.trapz(Pwl*dP_dobs, obsArr)


        ##### X-ray
        else:
            # Get likelihood
            likeli = np.trapz(dP_dobs*norm.pdf(obsmeas, obsArr, obserr), obsArr)

            if getpull:
                integrand = dP_dobs[None,:] * norm.pdf(obsArr[:,None], obsArr[None,:], obserr)
                dP_dobs_obs = np.trapz(integrand, obsArr, axis=1)
                dP_dobs_obs/= np.trapz(dP_dobs_obs,obsArr)
                cumtrapz = integrate.cumtrapz(dP_dobs_obs,obsArr)
                perc = np.interp(obsmeas, obsArr[1:], cumtrapz)
                print self.catalog['SPT_ID'][dataID], '%.4f %.4f %.4f %.4e'%(self.catalog['xi'][dataID], self.catalog['redshift'][dataID], obsmeas, 2**.5 * ss.erfinv(2*perc-1))

        if ((likeli<0)|(np.isnan(likeli))):
            print self.catalog['SPT_ID'][dataID], obsname, likeli
            #np.savetxt(self.catalog['SPT_ID'][dataID],np.transpose((obsArr, dP_dobs)))
            return 0.


        return likeli



    ############################################################################
    def get_P_2obs_xi(self, obsnames, dataID, covname):
        """Returns P(obs1, obs2|xi,z,p) for two types of follow-up data (e.g.,
        WL and X-ray)."""
        ##### Get observables, obsintr is used for setting up mass range
        obsmeas, obserr, obsintr = np.empty(2), np.empty(2), np.empty(2)
        for i in range(2):
            if obsnames[i]=='Yx':
                obsmeas[i], obsintr[i], obserr[i] = self.catalog['Yx_fid'][dataID], self.scaling['Dx'], self.catalog['Yx_err'][dataID]
            elif obsnames[i]=='Mgas':
                obsmeas[i], obsintr[i], obserr[i] = self.catalog['Mg_fid'][dataID], self.scaling['Dx'], self.catalog['Mg_err'][dataID]
            elif obsnames[i]=='WLMegacam':
                LSSnoise = self.WLcalib['Megacam_LSS'][0] + self.scaling['MegacamScatterLSS'] * self.WLcalib['Megacam_LSS'][1]
                obsmeas[i], obserr[i], obsintr[i] = .8*self.scaling['bWL_Megacam']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_Megacam']
            elif obsnames[i]=='WLHST':
                LSSnoise = self.WLcalib['HST_LSS'][0] + self.scaling['HSTscatterLSS'] * self.WLcalib['HST_LSS'][1]
                obsmeas[i], obserr[i], obsintr[i] = .8*self.scaling['bWL_HST']*self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), .3, self.scaling['DWL_HST']

        covmat = self.covmat[covname]


        ##### Define reasonable mass range
        # xi -> M(xi)
        xi_minmax = np.array((np.amax((2.6,self.catalog['xi'][dataID]-5)), self.catalog['xi'][dataID]+3))
        M_xi_minmax = self.obs2mass('zeta', self.xi2zeta(xi_minmax), self.catalog['redshift'][dataID])
        if M_xi_minmax[0]>self.HMF['M_arr'][-1]:
            print "cluster mass exceeds HMF mass range", self.catalog['SPT_ID'][dataID],\
                M_xi_minmax[0], self.HMF['M_arr'][-1]
            return 0

        M_obsminmax = []
        for i in range(2):
            # obs: prediction
            lnobs0 = np.log(self.mass2obs(obsnames[i], self.obs2mass('zeta', self.xi2zeta(self.catalog['xi'][dataID]), self.catalog['redshift'][dataID]), self.catalog['redshift'][dataID]))
            SZscatterobs = self.dlnM_dlnobs('zeta')/self.dlnM_dlnobs(obsnames[i])*self.scaling['Dsz']
            intrscatter = (SZscatterobs**2 + obsintr[i]**2)**.5
            obsthminmax = np.exp(np.array((lnobs0-5*intrscatter, lnobs0+3.5*intrscatter)))
            # obs: measurement
            if obsnames[i] in ('Mgas', 'Yx'):
                obsmeasminmax = np.amax((.1, obsmeas[i]-3*obserr[i])), obsmeas[i]+3*obserr[i]
            else:
                obsmeasminmax = np.exp(np.log(obsmeas[i])-4*obserr[i]), np.exp(np.log(obsmeas[i])+3*obserr[i])
            # put together
            obsminmax = np.array((min(obsthminmax[0],obsmeasminmax[0]), max(obsthminmax[1],obsmeasminmax[1])))
            M_obsminmax.append(self.obs2mass(obsnames[i], obsminmax, self.catalog['redshift'][dataID]))

        ##### Define grid in mass
        Mmin, Mmax = min(M_xi_minmax[0],M_obsminmax[0][0],M_obsminmax[1][0]), max(M_xi_minmax[1],M_obsminmax[0][1],M_obsminmax[1][1])
        Mmin, Mmax = max(.5*Mmin, self.HMF['M_arr'][0]), min(Mmax, self.HMF['M_arr'][-1])
        lenObs = 54
        M_obsArr = np.logspace(np.log10(Mmin), np.log10(Mmax), lenObs)
        M_HMF_arr = M_obsArr


        ##### Observable arrays
        lnzeta_arr = np.log(self.mass2obs('zeta', M_obsArr, self.catalog['redshift'][dataID]))
        xi_arr = self.zeta2xi(np.exp(lnzeta_arr))
        obsArr, lnobsArr = [], []
        for i in range(2):
            obsArrTemp = self.mass2obs(obsnames[i], M_obsArr, self.catalog['redshift'][dataID])
            ##### Add radial dependence for X-ray observables
            if obsnames[i] in ('Mgas','Yx'):
                # Angular diameter distances in current and reference cosmology [Mpc]
                dA = cosmo.dA(self.catalog['redshift'][dataID], self.cosmology)/self.cosmology['h']
                dAref = cosmo.dA(self.catalog['redshift'][dataID], cosmologyRef)/cosmologyRef['h']
                # R500 [kpc]
                rho_c_z = cosmo.RHOCRIT * cosmo.Ez(self.catalog['redshift'][dataID], self.cosmology)**2
                r500 = 1000 * (3*M_obsArr/(4*np.pi*500*rho_c_z))**(1/3) / self.cosmology['h']
                # r500 in reference cosmology [kpc]
                r500ref = r500 * dAref/dA
                # Xray observable at rFid...
                obsArrTemp*= (self.catalog['r500'][dataID]/r500ref)**self.scaling['dlnMg_dlnr']
                # ... corrected to reference cosmology
                obsArrTemp*= (dAref/dA)**2.5
            obsArr.append( obsArrTemp )
            lnobsArr.append( np.log(obsArrTemp) )


        ##### HMF to dN/(dlnzeta dlnobs0 dlnobs1) = dN/dlnM * dlnM/dlnzeta * dlnM/dlnobs0 * dlnM/dlnobs1
        # This only matter if dlnM/dlnobs is mass-dependent, as for dispersions
        dN_dlnzeta_dlnobs = np.exp(self.HMF_interp(np.log(self.catalog['redshift'][dataID]), np.log(M_HMF_arr)))[0]

        ##### HMF on 3D observable grid [lnobs0,lnobs1,lnzeta]
        HMF_3d_in = np.zeros((lenObs, lenObs, lenObs))
        np.fill_diagonal(HMF_3d_in, dN_dlnzeta_dlnobs)

        ##### 3D convolution with correlated scatter
        # kernel is min(lenObs, max(20bins, +/-5sigma))
        Nbin_obs0 = int(np.amin((lenObs, 10 * np.amax((2, covmat[0][0]**.5/(lnobsArr[0][1] - lnobsArr[0][0]))))))
        Nbin_obs1 = int(np.amin((lenObs, 10 * np.amax((2, covmat[1][1]**.5/(lnobsArr[1][1] - lnobsArr[1][0]))))))
        Nbin_zeta = int(np.amin((lenObs, 10 * np.amax((2, covmat[-1][-1]**.5/(lnzeta_arr[1] - lnzeta_arr[0]))))))
        pos = np.empty((Nbin_obs0, Nbin_obs1, Nbin_zeta, 3))
        pos[:,:,:,0], pos[:,:,:,1], pos[:,:,:,2] = np.meshgrid(lnobsArr[0][:Nbin_obs0], lnobsArr[1][:Nbin_obs1], lnzeta_arr[:Nbin_zeta], indexing='ij')
        kernel = multivariate_normal.pdf(pos, mean=(lnobsArr[0][int(Nbin_obs0/2)], lnobsArr[1][int(Nbin_obs1/2)], lnzeta_arr[int(Nbin_zeta/2)]), cov=covmat)
        HMF_3d = signal.fftconvolve(HMF_3d_in, kernel, mode='same')

        # set to 0 if zeta<2
        HMF_3d[:,:,np.where(lnzeta_arr<np.log(2.))] = 0.
        # Safety check
        if np.any(HMF_3d<-1e-6):
            print np.amin(HMF_3d), np.amax(HMF_3d)
            print self.catalog['SPT_ID'][dataID],'HMF_3d<0, det',np.linalg.det(covmat),self.scaling['Dsz'],obsintr
        HMF_3d[np.where(HMF_3d<0)] = 0
        if np.all(HMF_3d==0.):
            print self.catalog['SPT_ID'][dataID],'HMF_3d<=0, det',np.linalg.det(covmat),self.scaling['Dsz'],obsintr
            return 0

        ##### dN/(dxi dlnobs) = dN/(dlnzeta dlnobs) * dlnzeta/dxi [lnobs0][lnobs1][xi]
        HMF_3d*= self.dlnzeta_dxi(xi_arr)[None,None,:]

        #### Convolve with xi measurement error [lnobs0][lnobs1]
        dP_dlnobs = np.trapz(HMF_3d * norm.pdf(self.catalog['xi'][dataID], xi_arr[None,None,:], 1.), xi_arr, axis=2)

        ##### Go to linear space [obs0][obs1]
        dP_dobs01 = dP_dlnobs/obsArr[0][:,None]/obsArr[1][None,:]

        ##### P0
        dP_dobs0 = np.trapz(dP_dobs01, obsArr[1], axis=1)
        dP_dobs0/= np.trapz(dP_dobs0, obsArr[0])

        if obsnames[0] in ('WLHST', 'WLMegacam'):
            # Concolve with Gaussian LSS scatter
            if LSSnoise>0.:
                integrand = dP_dobs0[None,:] * norm.pdf(obsArr[0][:,None], obsArr[0][None,:], LSSnoise)
                dP_dobs0 = np.trapz(integrand, obsArr[0], axis=1)
                dP_dobs0/= np.trapz(dP_dobs0, obsArr[0])
            # P(Mwl) from data
            Pobs = self.WL.like(self.catalog, dataID, obsArr[0], self.cosmology, self.MCrel, self.lnM500_to_lnM200)
        else: print "not ready!"

        likeli0 = np.trapz(dP_dobs0*Pobs, obsArr[0])

        ##### P1 (Yx)
        dP_dobs1 = np.trapz(dP_dobs01, obsArr[0], axis=0)

        # Normalize (in principe, multiply with dlnX/dlnXfid, but this is mass-independent)
        dP_dobs1/= np.trapz(dP_dobs1, obsArr[1])
        likeli1 = np.trapz(dP_dobs1*norm.pdf(obsmeas[1], obsArr[1], obserr[1]), obsArr[1])


        ##### Probability
        likeli = likeli0*likeli1

        return likeli




    ############################################################################
    ##### Utility functions
    def xi2zeta(self, xi): return (xi**2 - 3)**.5
    def zeta2xi(self, zeta): return (zeta**2 + 3)**.5
    def dlnzeta_dxi(self, xi): return xi / (xi**2 - 3)
    def dxi_dzeta(self, zeta): return zeta / (zeta**2 + 3)


    ####################
    def obs2mass(self, name, obs, z):
        """Returns mass given (observable, z) using scaling relation."""
        if name=='zeta':
            Asz = self.thisSPTfieldCorrection * self.scaling['Asz']
            return self.SZmPivot * (obs / Asz \
                / (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Csz'])**(1/self.scaling['Bsz'])
        elif name=='Yx':
            return 1e14 * .7**(3/2) * self.scaling['Ax'] \
                * (obs * (self.cosmology['h']/.7)**2.5 / 3)**self.scaling['Bx'] \
                * cosmo.Ez(z, self.cosmology)**self.scaling['Cx']
        elif name=='Mgas':
            return self.XraymPivot/.7  * (obs / self.XraymPivot / self.scaling['Ax'] \
                * (self.cosmology['h']/.7)**2.5 \
                /(cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx'])**(1/self.scaling['Bx'])
        elif name=='WLMegacam':
            return obs/self.scaling['bWL_Megacam']
        elif name=='WLHST':
            return obs/self.scaling['bWL_HST']
        else:
            raise ValueError("Observable not known:",name)


    ####################
    def mass2obs(self, name, mass, z):
        """Returns observable given (mass, z) using scaling relation."""
        if name=='zeta':
            return self.scaling['Asz']*self.thisSPTfieldCorrection \
                * (mass/self.SZmPivot)**self.scaling['Bsz'] \
                * (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Csz']
        elif name=='Yx':
            return 3 * (self.cosmology['h']/.7)**-2.5 * (mass/1e14 /.7**(3/2) / self.scaling['Ax'] \
                / cosmo.Ez(z, self.cosmology)**self.scaling['Cx'])**(1/self.scaling['Bx'])
        elif name=='Mgas':
            return self.scaling['Ax'] * self.XraymPivot * (self.cosmology['h']/.7)**-2.5 \
                * (mass/self.XraymPivot/.7)**self.scaling['Bx'] \
                * (cosmo.Ez(z, self.cosmology)/cosmo.Ez(.6, self.cosmology))**self.scaling['Cx']
        elif name=='WLMegacam':
            return self.scaling['bWL_Megacam'] * mass
        elif name=='WLHST':
            return self.scaling['bWL_HST'] * mass
        else:
            raise ValueError("Observable not known:",name)


    ####################
    def dlnM_dlnobs(self, name, M0_arr=None, z=None):
        """Returns dlnM/dln(obs) for a given observable."""
        if name=='zeta':
            return 1/self.scaling['Bsz']
        elif name=='Yx':
            return 1/(1/self.scaling['Bx'] - self.scaling['dlnMg_dlnr']/3)
        elif name=='Mgas':
            return 1/(self.scaling['Bx'] - self.scaling['dlnMg_dlnr']/3)
        elif (name=='WLMegacam')|(name=='WLHST'):
            return 1.
