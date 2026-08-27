import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import prospect.io.read_results as reader
import time
from astropy.cosmology import WMAP9 as cosmo
import os

# ============================================================
# MODEL SETTINGS
# ============================================================

zlo = 8
zup = 9

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


# ============================================================
# SETTINGS
# ============================================================

readfile = "Prospector/z=8-9 with MIRI.h5"

cache_file = (
    "Prospector/Animations/"
    "z8-9_nested_seds.npz"
)

output_file = (
    "Prospector/Animations/"
    "z8-9_nested_sed_convergence.gif"
)

stride = 500
nsamples_per_frame = 1
fps = 2

# ============================================================
# LOAD RESULTS
# ============================================================

start_time = time.perf_counter()

model = build_model()
print(f'model built after {time.perf_counter() - start_time:.2f} seconds \n\n')

from prospect.models.templates import TemplateLibrary

sps = build_sps()
print(f'sps built after {time.perf_counter() - start_time:.2f} seconds \n\n')


readfile = "Prospector/Fits/z=8-9 with MIRI.h5"

result, obs, _ = reader.results_from(
    readfile,
    dangerous=False
)

chain = np.asarray(result["chain"])
labels = list(result["theta_labels"])

zidx = labels.index("zred")

# Thin the chain
chain = chain[::stride]

# ============================================================
# LOAD CACHE IF IT EXISTS
# ============================================================

if os.path.exists(cache_file):

    print(f"Loading cached SEDs from:\n{cache_file}")

    data = np.load(
        cache_file,
        allow_pickle=False
    )

    waves = data["waves"]
    spectra = data["spectra"]

    print(
        f"Loaded {len(spectra)} cached SEDs."
    )

else:

    print("No cached SEDs found.")
    print("Evaluating Prospector models...")

    spectra = []
    waves = []

    for i, theta in enumerate(chain):

        mspec, mphot, _ = model.mean_model(
            theta,
            obs,
            sps=sps
        )

        zred = theta[zidx]

        wave = (
            sps.wavelengths
            * (1 + zred)
        )

        spectra.append(mspec)
        waves.append(wave)

        if (i + 1) % 100 == 0:
            print(
                f"Calculated {i + 1}/{len(chain)} SEDs"
            )

    spectra = np.asarray(spectra)
    waves = np.asarray(waves)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(cache_file),
        exist_ok=True
    )

    np.savez_compressed(
        cache_file,
        waves=waves,
        spectra=spectra
    )

    print(
        f"Saved SEDs to:\n{cache_file}"
    )


# ============================================================
# COMMON WAVELENGTH GRID
# ============================================================

wave_min = np.nanmin(waves)
wave_max = np.nanmax(waves)

common_wave = np.logspace(
    np.log10(wave_min),
    np.log10(wave_max),
    3000
)

# Interpolate every SED onto the common grid
interp_spectra = np.full(
    (
        len(spectra),
        len(common_wave)
    ),
    np.nan
)

for i in range(len(spectra)):

    valid = (
        np.isfinite(waves[i])
        & np.isfinite(spectra[i])
        & (spectra[i] > 0)
    )

    if np.sum(valid) < 2:
        continue

    interp_spectra[i] = np.interp(
        common_wave,
        waves[i][valid],
        spectra[i][valid],
        left=np.nan,
        right=np.nan
    )


# ============================================================
# FRAME INDICES
# ============================================================

frame_indices = np.arange(
    nsamples_per_frame,
    len(interp_spectra) + 1,
    nsamples_per_frame
)

if frame_indices[-1] != len(interp_spectra):
    frame_indices = np.append(
        frame_indices,
        len(interp_spectra)
    )

# ============================================================
# GLOBAL AXIS LIMITS
# ============================================================

valid_flux = interp_spectra[
    np.isfinite(interp_spectra)
    & (interp_spectra > 0)
]

ymin = np.nanpercentile(
    valid_flux,
    1
)

ymax = np.nanpercentile(
    valid_flux,
    99.5
)

# ============================================================
# OBSERVED PHOTOMETRY
# ============================================================

wphot = obs["phot_wave"]

obs_flux = np.asarray(
    obs["maggies"],
    dtype=float
)

obs_unc = np.asarray(
    obs["maggies_unc"],
    dtype=float
)

detected = (
    np.isfinite(obs_flux)
    & np.isfinite(obs_unc)
    & (obs_unc > 0)
    & (obs_flux > 0)
)

# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 7),
    facecolor="black"
)

ax.set_facecolor("black")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(
    common_wave.min(),
    common_wave.max()
)

ax.set_ylim(
    ymin,
    ymax
)

ax.set_xlabel(
    "Observed wavelength [Å]",
    color="white",
    fontsize=16
)

ax.set_ylabel(
    "Flux density [maggies]",
    color="white",
    fontsize=16
)

ax.tick_params(
    colors="white",
    labelsize=13
)

for spine in ax.spines.values():
    spine.set_color("white")

ax.grid(
    color="white",
    alpha=0.12
)

# ============================================================
# OBSERVED DATA
# ============================================================

ax.errorbar(
    wphot[detected],
    obs_flux[detected],
    yerr=obs_unc[detected],
    fmt="o",
    color="white",
    ecolor="white",
    markersize=7,
    alpha=0.8,
    label="Observed"
)

# ============================================================
# INITIAL SED / BAND
# ============================================================

band = ax.fill_between(
    common_wave,
    np.ones_like(common_wave) * np.nan,
    np.ones_like(common_wave) * np.nan,
    color="hotpink",
    alpha=0.2
)

median_line, = ax.plot(
    common_wave,
    np.ones_like(common_wave) * np.nan,
    color="hotpink",
    lw=2.5,
    label="Posterior median"
)

title = ax.text(
    0.5,
    0.95,
    "",
    transform=ax.transAxes,
    ha="center",
    va="top",
    color="white",
    fontsize=18,
    fontweight="bold"
)

# ============================================================
# UPDATE
# ============================================================

def update(frame):

    n = frame_indices[frame]

    current = interp_spectra[:n]

    # --------------------------------------------------------
    # Posterior percentiles
    # --------------------------------------------------------

    p16 = np.nanpercentile(
        current,
        16,
        axis=0
    )

    p50 = np.nanpercentile(
        current,
        50,
        axis=0
    )

    p84 = np.nanpercentile(
        current,
        84,
        axis=0
    )

    # --------------------------------------------------------
    # Update shaded region
    # --------------------------------------------------------

    global band

    band.remove()

    band = ax.fill_between(
        common_wave,
        p16,
        p84,
        color="hotpink",
        alpha=0.25
    )

    # --------------------------------------------------------
    # Update median
    # --------------------------------------------------------

    median_line.set_data(
        common_wave,
        p50
    )

    # --------------------------------------------------------
    # Update title
    # --------------------------------------------------------

    title.set_text(
        f"Nested sampling convergence\n"
        f"{n * stride:,} samples"
    )

    return (
        median_line,
        band,
        title
    )


# ============================================================
# ANIMATION
# ============================================================

anim = FuncAnimation(
    fig,
    update,
    frames=len(frame_indices),
    interval=1000 / fps,
    blit=False
)

# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)

anim.save(
    output_file,
    writer=PillowWriter(
        fps=fps
    ),
    dpi=200
)

plt.show()