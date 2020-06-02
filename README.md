# SPT-SZ Cluster Likelihood

This is the likelihood code used to produce the results obtained from the SPT-SZ
cluster sample with weak-lensing follow-up data from Magellan/Megacam and HST
and X-ray follow-up data from Chandra, as presented in Bocquet et al. (2019).

## Quickstart

The code is meant to be used within the [CosmoSIS framework]
(https://bitbucket.org/joezuntz/cosmosis)

To reproduce the baseline SPTcl results, just run CosmoSIS using the setup file
`setup_nuLCDM_SPTcl.ini` that is provided with this likelihood code.

## (A Little) More Details

This package contains three modules (in the CosmoSIS sense):
* `compute_HMF` computes the halo mass function on a grid of redshifts and masses
* `abundance` computes the likelihood of the cluster abundance (SPT-SZ SNR and redshifts)
* `mass_calibration` computes the likelihood using the clusters' follow-up mass calibration data
Take a look at the three `.yaml` files and reach out to me if you have further
questions.

## References

* Cluster cosmology constraints and analysis pipeline: Bocquet et al. 2019, DOI 10.3847/1538-4357/ab1f10
* SPT-SZ cluster catalog: Bleem et al. 2015, DOI 10.1088/0067-0049/216/2/27
* Weak-lensing dataset from the Hubble Space Telescope: Schrabback et al. 2018, DOI 10.1093/mnras/stx2666
* Weak-lensing dataset from Magellan/Megacam: Dietrich et al. 2019, DOI 10.1093/mnras/sty3088
* X-ray Chandra dataset: McDonald et al. 2013, DOI 10.1088/0004-637X/774/1/23
* X-ray Chandra z>1.2 dataset: McDonald et al. 2017, DOI 10.3847/1538-4357/aa7740
