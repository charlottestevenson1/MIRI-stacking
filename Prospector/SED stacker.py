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

zlo = 11
zup = 12

start_time = time.perf_counter()

model = build_model()
print(f'model built after {time.perf_counter() - start_time:.2f} seconds \n\n')

from prospect.models.templates import TemplateLibrary

sps = build_sps()
print(f'sps built after {time.perf_counter() - start_time:.2f} seconds \n\n')

import prospect.io.read_results as reader
# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files `dangerous=True`

withMIRIfile = f'Prospector/z={zlo}-{zup} with MIRI.h5'
noMIRIfile = f'Prospector/z={zlo}-{zup} no MIRI.h5'

# ============================================================
# LOAD RESULTS
# ============================================================

withMIRIresult, obs, _ = reader.results_from(
    withMIRIfile,
    dangerous=False
)

noMIRIresult, _, _ = reader.results_from(
    noMIRIfile,
    dangerous=False
)


# ============================================================
# HELPER: WEIGHTED MEDIAN
# ============================================================

def weighted_median(values, weights):

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]

    return np.interp(
        0.5,
        cumulative,
        values
    )


# ============================================================
# HELPER: GET POSTERIOR SED + UNCERTAINTY
# ============================================================

def posterior_sed(
    result,
    obs,
    model,
    sps,
    nsamples=500,
    n_wavelengths=5000
):

    chain = np.asarray(result["chain"])

    weights = np.asarray(
        result["weights"],
        dtype=float
    )

    weights /= weights.sum()

    # --------------------------------------------------------
    # Draw posterior samples according to Dynesty weights
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(chain),
        size=min(nsamples, len(chain)),
        replace=True,
        p=weights
    )

    theta_samples = chain[indices]

    # --------------------------------------------------------
    # Common observed-frame wavelength grid
    # --------------------------------------------------------

    z_index = list(
        result["theta_labels"]
    ).index("zred")

    z_samples = theta_samples[:, z_index]

    z_min = np.percentile(z_samples, 1)
    z_max = np.percentile(z_samples, 99)

    wave_min = sps.wavelengths.min() * (1 + z_min)
    wave_max = sps.wavelengths.max() * (1 + z_max)

    common_wave = np.logspace(
        np.log10(wave_min),
        np.log10(wave_max),
        n_wavelengths
    )

    # --------------------------------------------------------
    # Evaluate every posterior model
    # --------------------------------------------------------

    spectra = []
    photometry = []

    for theta in theta_samples:

        mspec, mphot, _ = model.mean_model(
            theta,
            obs,
            sps=sps
        )

        zred = theta[z_index]

        wave = sps.wavelengths * (1 + zred)

        # Interpolate to common wavelength grid
        spectrum_interp = np.interp(
            common_wave,
            wave,
            mspec,
            left=np.nan,
            right=np.nan
        )

        spectra.append(spectrum_interp)
        photometry.append(mphot)

    spectra = np.asarray(spectra)
    photometry = np.asarray(photometry)

    # --------------------------------------------------------
    # Percentiles of posterior SED
    # --------------------------------------------------------

    sed_p16 = np.nanpercentile(
        spectra,
        16,
        axis=0
    )

    sed_p50 = np.nanpercentile(
        spectra,
        50,
        axis=0
    )

    sed_p84 = np.nanpercentile(
        spectra,
        84,
        axis=0
    )

    # --------------------------------------------------------
    # Percentiles of posterior photometry
    # --------------------------------------------------------

    phot_p16 = np.nanpercentile(
        photometry,
        16,
        axis=0
    )

    phot_p50 = np.nanpercentile(
        photometry,
        50,
        axis=0
    )

    phot_p84 = np.nanpercentile(
        photometry,
        84,
        axis=0
    )

    return (
        common_wave,
        sed_p16,
        sed_p50,
        sed_p84,
        phot_p16,
        phot_p50,
        phot_p84
    )


# ============================================================
# COMPUTE POSTERIOR SEDS
# ============================================================

(
    withMIRI_wspec,
    withMIRI_sed_p16,
    withMIRI_sed_p50,
    withMIRI_sed_p84,
    withMIRI_phot_p16,
    withMIRI_phot_p50,
    withMIRI_phot_p84
) = posterior_sed(
    withMIRIresult,
    obs,
    model,
    sps,
    nsamples=500
)

(
    noMIRI_wspec,
    noMIRI_sed_p16,
    noMIRI_sed_p50,
    noMIRI_sed_p84,
    noMIRI_phot_p16,
    noMIRI_phot_p50,
    noMIRI_phot_p84
) = posterior_sed(
    noMIRIresult,
    obs,
    model,
    sps,
    nsamples=500
)


# Photometric wavelengths
wphot = obs["phot_wave"]

def maggies_to_ab(flux):
    flux = np.asarray(flux, dtype=float)
    mag = np.full_like(flux, np.nan)
    valid = np.isfinite(flux) & (flux > 0)
    mag[valid] = -2.5 * np.log10(flux[valid])
    return mag

# Convert SED percentiles to AB magnitudes.
# Flux percentile ordering reverses in magnitude space.
withMIRI_sed_mag_p16 = maggies_to_ab(withMIRI_sed_p84)
withMIRI_sed_mag_p50 = maggies_to_ab(withMIRI_sed_p50)
withMIRI_sed_mag_p84 = maggies_to_ab(withMIRI_sed_p16)

noMIRI_sed_mag_p16 = maggies_to_ab(noMIRI_sed_p84)
noMIRI_sed_mag_p50 = maggies_to_ab(noMIRI_sed_p50)
noMIRI_sed_mag_p84 = maggies_to_ab(noMIRI_sed_p16)

# Convert model photometry
withMIRI_phot_mag_p16 = maggies_to_ab(withMIRI_phot_p84)
withMIRI_phot_mag_p50 = maggies_to_ab(withMIRI_phot_p50)
withMIRI_phot_mag_p84 = maggies_to_ab(withMIRI_phot_p16)

noMIRI_phot_mag_p16 = maggies_to_ab(noMIRI_phot_p84)
noMIRI_phot_mag_p50 = maggies_to_ab(noMIRI_phot_p50)
noMIRI_phot_mag_p84 = maggies_to_ab(noMIRI_phot_p16)

plt.figure(figsize=(16, 8))

# Posterior SEDs
plt.fill_between(
    withMIRI_wspec, withMIRI_sed_mag_p16, withMIRI_sed_mag_p84,
    color="lightblue", alpha=0.35, label="With MIRI 16–84%"
)
plt.plot(
    withMIRI_wspec, withMIRI_sed_mag_p50,
    color="blue", lw=1.5, label="With MIRI posterior median"
)

plt.fill_between(
    noMIRI_wspec, noMIRI_sed_mag_p16, noMIRI_sed_mag_p84,
    color="lightpink", alpha=0.35, label="No MIRI 16–84%"
)
plt.plot(
    noMIRI_wspec, noMIRI_sed_mag_p50,
    color="hotpink", lw=1.5, label="No MIRI posterior median"
)

# Posterior model photometry
plt.errorbar(
    wphot, withMIRI_phot_mag_p50,
    yerr=[
        withMIRI_phot_mag_p50 - withMIRI_phot_mag_p16,
        withMIRI_phot_mag_p84 - withMIRI_phot_mag_p50
    ],
    label="With MIRI median photometry",
    marker="s", markersize=10, ls="",
    markerfacecolor="none", markeredgecolor="blue", markeredgewidth=3,
    ecolor="blue", alpha=0.6, capsize=3
)

plt.errorbar(
    wphot, noMIRI_phot_mag_p50,
    yerr=[
        noMIRI_phot_mag_p50 - noMIRI_phot_mag_p16,
        noMIRI_phot_mag_p84 - noMIRI_phot_mag_p50
    ],
    label="No MIRI median photometry",
    marker="s", markersize=10, ls="",
    markerfacecolor="none", markeredgecolor="hotpink", markeredgewidth=3,
    ecolor="hotpink", alpha=0.6, capsize=3
)

# Observed photometry
obs_flux = np.asarray(obs["maggies"], dtype=float)
obs_unc = np.asarray(obs["maggies_unc"], dtype=float)
snr = obs_flux / obs_unc

detected = (
    (snr >= 1.5) & np.isfinite(obs_flux) & np.isfinite(obs_unc)
    & (obs_unc > 0) & (obs_flux > obs_unc)
)
upper_limits = (
    (snr < 1.5) & np.isfinite(obs_unc) & (obs_unc > 0)
)

obs_mag = np.full_like(obs_flux, np.nan)
obs_mag_err_lower = np.full_like(obs_flux, np.nan)
obs_mag_err_upper = np.full_like(obs_flux, np.nan)

obs_mag[detected] = maggies_to_ab(obs_flux[detected])
mag_bright = maggies_to_ab(obs_flux[detected] + obs_unc[detected])
mag_faint = maggies_to_ab(obs_flux[detected] - obs_unc[detected])

obs_mag_err_lower[detected] = obs_mag[detected] - mag_bright
obs_mag_err_upper[detected] = mag_faint - obs_mag[detected]

plt.errorbar(
    wphot[detected], obs_mag[detected],
    yerr=[obs_mag_err_lower[detected], obs_mag_err_upper[detected]],
    label="Observed photometry",
    ecolor="gray", marker="o", markersize=10, ls="",
    markerfacecolor="none", markeredgecolor="gray", markeredgewidth=3,
    capsize=3
)

# 1.5-sigma limits
upper_limit_mag = maggies_to_ab(1.5 * obs_unc[upper_limits])

plt.scatter(
    wphot[upper_limits], upper_limit_mag,
    marker="v", s=100, color="gray",
    label="Observed < 1.5σ", zorder=5
)

# Axis limits
xmin, xmax = np.min(wphot) * 0.8, np.max(wphot) / 0.8

all_mags = np.concatenate([
    withMIRI_sed_mag_p16[np.isfinite(withMIRI_sed_mag_p16)],
    withMIRI_sed_mag_p84[np.isfinite(withMIRI_sed_mag_p84)],
    noMIRI_sed_mag_p16[np.isfinite(noMIRI_sed_mag_p16)],
    noMIRI_sed_mag_p84[np.isfinite(noMIRI_sed_mag_p84)],
    obs_mag[np.isfinite(obs_mag)],
    upper_limit_mag[np.isfinite(upper_limit_mag)]
])

brightest = np.nanmin(all_mags)
faintest = np.nanmax(all_mags)

plt.xlabel("Wavelength [Å]")
plt.ylabel("AB magnitude")
plt.xscale("log")
plt.yscale("linear")
plt.xlim(xmin, xmax * 5)
plt.ylim(20, 40)
plt.gca().invert_yaxis()

plt.legend(loc="best", fontsize=14)

fig = plt.gcf()
fig.suptitle(f"Posterior median SED: z={zlo}-{zup}", y=0.94)
fig.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig(
    f"Prospector/Plots/z={zlo}-{zup} posterior median SED.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()