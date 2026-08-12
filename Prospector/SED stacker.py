import argparse
from prospect.io import write_results
from prospect.fitting import fit_model
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from multiprocessing import Pool
import time

# -----------------------
# adjust_continuity_agebins
# -----------------------
def adjust_continuity_agebins(parset, nbins=8):
    '''
    Define the agebins. The first bin goes from 0-5 Myr, the next from
    5-10 Myr, the next from 10-30 Myr, and the rest are evenly spaced in
    logarithmic time
    '''
    from prospect.models import priors

    if nbins < 4:
        raise ValueError('Must have nbins >= 4, returning')

    tuniv = cosmo.age(parset['zred']['init']).value * 1e9
    # Limits of 5 Myr and 10 Myr
    lim1, lim2 = 6.6990, 7.0
    agelims = ([0, lim1] + np.linspace(lim2, np.log10(tuniv), nbins-1).tolist())
    agebins = np.array([agelims[:-1], agelims[1:]])

    ncomp = nbins
    mean = np.zeros(ncomp-1)
    scale = np.ones_like(mean)*0.5
    df = np.ones_like(mean)*2
    rprior = priors.StudentT(mean=mean, scale=scale, df=df)

    parset['mass']['N'] = ncomp
    parset['agebins']['N'] = ncomp
    parset['agebins']['init'] = agebins.T
    parset['logsfr_ratios']['N'] = ncomp-1
    parset['logsfr_ratios']['init'] = mean
    parset['logsfr_ratios']['prior'] = rprior

    return parset

# -----------------------
# zred_to_agebins
# -----------------------
def zred_to_agebins(zred=None, nbins_sfh=8, **extras):
    '''
    Construct `nbins_sfh` bins in lookback time from 0 to the age of the
    Universe at the redshift under consideration. The first bin goes from
    0-10 Myr, the next from 10-30 Myr, and the rest are evenly spaced in
    logarithmic time
    '''

    tuniv = cosmo.age(zred[0]).value * 1e9
    lim1, lim2 = 6.6990, 7.0
    agelims = ([0, lim1] +
        np.linspace(lim2, np.log10(tuniv), nbins_sfh[0]-1).tolist())
    agebins = np.array([agelims[:-1], agelims[1:]])

    return agebins.T

# -----------------------
# zlogsfr_ratios_to_masses
# -----------------------
def zlogsfr_ratios_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
    '''
    This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
    log10(Sum(M_i) over i) to values of M_i.  j=0 is the most recent bin in
    lookback time; it incorporates changes in the agebins due to changing
    redshift.
    '''
    agebins = zred_to_agebins(zred, **extras)
    nbins = agebins.shape[0]
    sratios = 10**np.clip(logsfr_ratios, -100, 100)  # numerical issues...
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    coeffs = np.array([
        (1. / np.prod(sratios[:i])) * (np.prod(dt[1: i+1]) / np.prod(dt[: i]))
        for i in range(nbins)
    ])
    m1 = (10**logmass) / coeffs.sum()

    return m1 * coeffs

# -----------------------
# to_dust1
# -----------------------
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction * dust2

# -----------------------
# build_model
# -----------------------
# Set fixable parameters as params
def build_model(nbins_sfh=8, **kwargs):
    from prospect.models.sedmodel import SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    model_params = TemplateLibrary["continuity_sfh"]

    # redshift:
    model_params['zred']['isfree'] = True
    model_params['zred']['init'] = 0.5*(zlo+zup)
    model_params['zred']['prior'] = priors.TopHat(mini=zlo, maxi=zup)
    print(f'Redshift range: z={zlo}-{zup}\n\n')

    # Deal with the age bins - let them vary based on the redshift being
    # considered
    model_params['nbins_sfh'] = dict(N=1, isfree=False, init=nbins_sfh)
    model_params = adjust_continuity_agebins(model_params, nbins=nbins_sfh)
    model_params['agebins']['N'] = nbins_sfh
    model_params['agebins']['depends_on'] = zred_to_agebins


    # Let log(mass) vary (not mass, Prospector calculates that)
    model_params['logmass']['isfree'] = True
    model_params['logmass']['init'] = 9
    model_params['logmass']['prior'] = priors.Uniform(mini=6, maxi=12)
    # Set up the mass parameter that Prospector calculates
    model_params['mass']['isfree'] = False
    model_params['mass']['init'] = np.array([nbins_sfh *[10**model_params['logmass']['init']/nbins_sfh]])
    model_params['mass']['depends_on'] = zlogsfr_ratios_to_masses

    # Choose the IMF and the upper and lower limits
    model_params['imf_type']['init'] = 1  # Chabrier
    model_params['imf_upper_limit'] = {'name': 'imf_upper_limit',
                                       'N': 1,
                                       'isfree': False,
                                       'init': 300.0,
                                       'units': 'type'}
    model_params['imf_lower_limit'] = {'name': 'imf_lower_limit',
                                       'N': 1,
                                       'isfree': False,
                                       'init': 0.1,
                                       'units': 'type'}

    # Choose the dust model and let diffuse dust optical depth and power law
    # modifier to Calzetti+2000 attenuateion curve vary
    model_params['dust_type']['init'] = 4  # Kriek & Conroy (2013)
    model_params['dust2']['isfree'] = True
    model_params['dust2']['prior'] = priors.ClippedNormal(
        mean=0.3, sigma=1, mini=0, maxi=4)
    model_params['dust_index'] = dict(N=1, isfree=True, init=0,
        prior=priors.TopHat(mini=-1, maxi=0.4))
    # Also allow differential attenuation of young (<10 Myr old) stars by their
    # birth cloud to vary
    
    model_params['dust1'] = {
        "N": 1, "isfree": False, "init": 0.0,
        "units": "optical depth towards young stars",
        "prior": None, 'depends_on': to_dust1
    }

    model_params['dust1_index'] = {
        "N": 1, "isfree": False, "init": -1.0,
        "units": "power-law index of the birth-cloud attenuation"
    }

    model_params['dust1_fraction'] = dict(N=1, isfree=True, init=1.0,
        prior=priors.ClippedNormal(mean=1, sigma=0.3, mini=0, maxi=2))

    # Give stellar metallicity a log-uniform prior between 0.01-1 Zsun
    model_params['logzsol']['isfree'] = True
    model_params['logzsol']['init'] = -1
    model_params['logzsol']['prior'] = priors.TopHat(mini=-2, maxi=0.19)

    # Add in nebular emission and give log(U) a log-uniform prior between -4 and -1
    model_params.update(TemplateLibrary['nebular'])
    model_params['add_neb_emission']['init'] = True
    model_params['nebemlineinspec']['init'] = False
    model_params['gas_logu']['isfree'] = True
    model_params['gas_logu']['init'] = -2.5
    model_params['gas_logu']['prior'] = priors.TopHat(mini=-4, maxi=-1)

    # Ignore Lyman alpha
    lines_to_ignore = ['Ly-alpha 1215']
    model_params['elines_to_ignore'] = dict(init=lines_to_ignore, isfree=False)

    # Add IGM attenuation and let the scaling (relative to the Madau+1995
    # model) range a bit
    model_params.update(TemplateLibrary['igm'])
    model_params['igm_factor']['isfree'] = True
    model_params['igm_factor']['init'] = 1
    model_params['igm_factor']['prior'] = priors.ClippedNormal(mean=1, sigma=0.3, mini=0, maxi=2)

    # Instantiate the model using this dictionary of parameter specifications
    model = SpecModel(model_params)
    # for i in model.config_dict:
    #     print(i)
    # print('\n\n\n')
    return model

# -----------------------
# build_sps
# -----------------------
def build_sps(**kwargs):
    from prospect.sources import FastStepBasis
    sps = FastStepBasis()
    return sps


zlo = 9
zup = 10

start_time = time.perf_counter()

model = build_model()
print(f'model built after {time.perf_counter() - start_time:.2f} seconds \n\n')

sps = build_sps()
print(f'sps built after {time.perf_counter() - start_time:.2f} seconds \n\n')

import prospect.io.read_results as reader
# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files `dangerous=True`

withMIRIfile = f'Prospector/z={zlo}-{zup} with MIRI.h5'
noMIRIfile = f'Prospector/z={zlo}-{zup} no MIRI.h5'

withMIRIresult, obs, _ = reader.results_from(withMIRIfile, dangerous=False)
noMIRIresult, _, _ = reader.results_from(noMIRIfile, dangerous=False)

withMIRI_imax = np.argmax(withMIRIresult['lnprobability'])
withMIRI_theta_max = withMIRIresult["chain"][withMIRI_imax, :]
noMIRI_imax = np.argmax(noMIRIresult['lnprobability'])
noMIRI_theta_max = noMIRIresult["chain"][noMIRI_imax, :]
thin = 1

start_time = time.perf_counter()

### PLOT SEDs AND RESIDUALS

# generate models
withMIRI_mspec_map, withMIRI_mphot_map, _ = model.mean_model(withMIRI_theta_max, obs, sps=sps)
noMIRI_mspec_map, noMIRI_mphot_map, _ = model.mean_model(noMIRI_theta_max, obs, sps=sps)

# Make plot of data and model
plt.figure(figsize=(16,8))

a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
# photometric effective wavelengths
wphot = obs["phot_wave"]
# spectroscopic wavelengths
if obs["wavelength"] is None:
    # *restframe* spectral wavelengths, since obs["wavelength"] is None
    wspec = sps.wavelengths
    wspec *= a #redshift them
else:
    wspec = obs["wavelength"]

# establish bounds
# Can change this for visuals!
ARTIFICIAL_Y_LIM = 1e-17

xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
temp = np.interp(np.linspace(xmin,xmax,10000), wspec, withMIRI_mspec_map)
print(f'temp.min={temp.min()}, temp.max={temp.max()}')
ymin, ymax = max(temp.min()*(0.1),ARTIFICIAL_Y_LIM), temp.max()/0.1

mask = withMIRI_mphot_map > 0

plt.loglog(wspec, withMIRI_mspec_map, label='Model spectrum with MIRI',
    lw=0.7, color='blue', alpha=0.7)
plt.loglog(wspec, noMIRI_mspec_map, label='Model spectrum no MIRI',
    lw=0.7, color='hotpink', alpha=0.7)
plt.errorbar(wphot, withMIRI_mphot_map, label='Model photometry with MIRI',
        marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
        markerfacecolor='none', markeredgecolor='blue', 
        markeredgewidth=3)
plt.errorbar(wphot, noMIRI_mphot_map, label='Model photometry no MIRI',
        marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
        markerfacecolor='none', markeredgecolor='hotpink', 
        markeredgewidth=3)
plt.errorbar(wphot[mask], obs['maggies'][mask], yerr=obs['maggies_unc'][mask], 
        label='Observed photometry', ecolor='gray', 
        marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
        markerfacecolor='none', markeredgecolor='gray', 
        markeredgewidth=3)

# plot transmission curves
for f in obs['filters']:
    w, t = f.wavelength.copy(), f.transmission.copy()
    t = t / t.max()
    t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
    plt.loglog(w, t, lw=3, color='gray', alpha=0.7)

plt.xlabel('Wavelength [A]')
plt.ylabel('Flux Density [maggies]')
# plt.xlim([xmin, xmax])
plt.xlim([xmin, xmax*5])
plt.ylim([ymin, ymax])
### add vertical stripes on plot to show min and max wavelengths of Al V, Pf-5, Ca IV, Pf-delta
# cws = [29053, 30392, 32070, 32970][2:]
# labels = ['Ca IV', 'Pf-delta']
# colors = ['green', 'red']
# # for i in range(2):
#     plt.axvspan(cws[i]*9, cws[i]*10, color=colors[i], alpha=0.3)
#     plt.axvline(cws[i]*9.32, color=colors[i], alpha=0.7, ls='--', lw=2, label=labels[i])
plt.legend(loc='best', fontsize=20)
fig = plt.gcf()
fig.suptitle(f'SED: z={zlo}-{zup}', y=0.94)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'Prospector/Plots/z={zlo}-{zup} joint SED.png', dpi=300)
#plt.show()
plt.close()