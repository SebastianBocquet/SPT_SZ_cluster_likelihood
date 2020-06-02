import numpy as np

THRESHOLD = 1e-8

def set_covmats(todo, scaling, covmat):
    """Populate `covmat` dict with all possible covariance matrices between all
    observables we're currently analyzing. The scatter in velocity dispersions
    depends on cluster properties and therefore cannot be pre-computed.
    Return: (bool) whether or not all covariance matrices can be inverted (by
    checking whether all determinants are >= THRESHOLD)
    """

    ##### one follow-up observable
    if todo['Yx'] or todo['Mgas']:
        cov = [[scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
        [scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
        if np.linalg.det(cov) < THRESHOLD:
            return False
        covmat['Yx'] = cov
        covmat['Mgas'] = cov

    if todo['WL']:
        if 'DWL_Megacam' in scaling.keys():
            cov = [[scaling['DWL_Megacam']**2, scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
                [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['Dsz']**2]]
            if np.linalg.det(cov) < THRESHOLD:
                return False
            covmat['WLMegacam'] = cov

    ##### two follow-up observables

    if (todo['Yx'] or todo['Mgas']) and todo['WL']:
        # Megacam
        if 'DWL_Megacam' in scaling.keys():
            cov = [[scaling['DWL_Megacam']**2, scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam']],
                [scaling['rhoWLX']*scaling['DWL_Megacam']*scaling['Dx'], scaling['Dx']**2, scaling['rhoSZX']*scaling['Dsz']*scaling['Dx']],
                [scaling['rhoSZWL']*scaling['Dsz']*scaling['DWL_Megacam'], scaling['rhoSZX']*scaling['Dsz']*scaling['Dx'], scaling['Dsz']**2]]
            if np.linalg.det(cov) < THRESHOLD:
                return False
            covmat['XrayMegacam'] = cov

    return True
