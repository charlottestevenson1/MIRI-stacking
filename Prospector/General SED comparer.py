import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis

from astropy.cosmology import WMAP9 as cosmo


# ============================================================
# SETTINGS
# ============================================================

# ------------------------------------------------------------
# FILE 1
# ------------------------------------------------------------

file1 = "Prospector/Fits/z=8-9 with MIRI.h5"
label1 = "z = 8-9 with MIRI, no logzsol cap"

zlo1 = 8
zup1 = 9


# ------------------------------------------------------------
# FILE 2
# ------------------------------------------------------------

file2 = "Prospector/Fits/z=8-9 with MIRI logzsol<-1.h5"
label2 = "z = 8-9 with MIRI, logzsol capped at -1"

zlo2 = 8
zup2 = 9


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

nsamples = 500
n_wavelengths = 5000

colour1 = "steelblue"
colour2 = "red"

output_file = (
    "Prospector/Plots/"
    "SED_capped_zsol_comparison.png"
)


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def adjust_continuity_agebins(parset, nbins=8):

    from prospect.models import priors

    if nbins < 4:
        raise ValueError(
            "Must have nbins >= 4"
        )

    tuniv = (
        cosmo.age(
            parset["zred"]["init"]
        ).value
        * 1e9
    )

    lim1 = 6.6990
    lim2 = 7.0

    agelims = (
        [0, lim1]
        +
        np.linspace(
            lim2,
            np.log10(tuniv),
            nbins - 1
        ).tolist()
    )

    agebins = np.array([
        agelims[:-1],
        agelims[1:]
    ])

    ncomp = nbins

    mean = np.zeros(ncomp - 1)
    scale = np.ones_like(mean) * 0.5
    df = np.ones_like(mean) * 2

    rprior = priors.StudentT(
        mean=mean,
        scale=scale,
        df=df
    )

    parset["mass"]["N"] = ncomp
    parset["agebins"]["N"] = ncomp
    parset["agebins"]["init"] = agebins.T

    parset["logsfr_ratios"]["N"] = ncomp - 1
    parset["logsfr_ratios"]["init"] = mean
    parset["logsfr_ratios"]["prior"] = rprior

    return parset


def zred_to_agebins(
    zred=None,
    nbins_sfh=8,
    **extras
):

    tuniv = (
        cosmo.age(
            zred[0]
        ).value
        * 1e9
    )

    lim1 = 6.6990
    lim2 = 7.0

    if np.isscalar(nbins_sfh):
        nbins = int(nbins_sfh)
    else:
        nbins = int(nbins_sfh[0])

    agelims = (
        [0, lim1]
        +
        np.linspace(
            lim2,
            np.log10(tuniv),
            nbins - 1
        ).tolist()
    )

    agebins = np.array([
        agelims[:-1],
        agelims[1:]
    ])

    return agebins.T


def zlogsfr_ratios_to_masses(
    logmass=None,
    logsfr_ratios=None,
    zred=None,
    **extras
):

    agebins = zred_to_agebins(
        zred,
        **extras
    )

    nbins = agebins.shape[0]

    sratios = 10 ** np.clip(
        logsfr_ratios,
        -100,
        100
    )

    dt = (
        10 ** agebins[:, 1]
        -
        10 ** agebins[:, 0]
    )

    coeffs = np.array([
        (
            (1.0 / np.prod(sratios[:i]))
            *
            (
                np.prod(dt[1:i + 1])
                /
                np.prod(dt[:i])
            )
        )
        for i in range(nbins)
    ])

    m1 = (
        10 ** logmass
    ) / coeffs.sum()

    return m1 * coeffs


def to_dust1(
    dust1_fraction=None,
    dust1=None,
    dust2=None,
    **extras
):

    return dust1_fraction * dust2


def build_model(
    zlo,
    zup,
    nbins_sfh=8
):

    from prospect.models.sedmodel import SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    model_params = TemplateLibrary[
        "continuity_sfh"
    ]

    # --------------------------------------------------------
    # Redshift
    # --------------------------------------------------------

    model_params["zred"]["isfree"] = True

    model_params["zred"]["init"] = (
        0.5 * (zlo + zup)
    )

    model_params["zred"]["prior"] = (
        priors.TopHat(
            mini=zlo,
            maxi=zup
        )
    )

    # --------------------------------------------------------
    # SFH
    # --------------------------------------------------------

    model_params["nbins_sfh"] = dict(
        N=1,
        isfree=False,
        init=nbins_sfh
    )

    model_params = adjust_continuity_agebins(
        model_params,
        nbins=nbins_sfh
    )

    model_params["agebins"]["N"] = nbins_sfh

    model_params["agebins"]["depends_on"] = (
        zred_to_agebins
    )

    # --------------------------------------------------------
    # Mass
    # --------------------------------------------------------

    model_params["logmass"]["isfree"] = True

    model_params["logmass"]["init"] = 9

    model_params["logmass"]["prior"] = (
        priors.Uniform(
            mini=6,
            maxi=12
        )
    )

    model_params["mass"]["isfree"] = False

    model_params["mass"]["init"] = np.array([
        [
            10 ** model_params["logmass"]["init"]
            / nbins_sfh
        ] * nbins_sfh
    ])

    model_params["mass"]["depends_on"] = (
        zlogsfr_ratios_to_masses
    )

    # --------------------------------------------------------
    # IMF
    # --------------------------------------------------------

    model_params["imf_type"]["init"] = 1

    model_params["imf_upper_limit"] = {
        "name": "imf_upper_limit",
        "N": 1,
        "isfree": False,
        "init": 300.0,
        "units": "type"
    }

    model_params["imf_lower_limit"] = {
        "name": "imf_lower_limit",
        "N": 1,
        "isfree": False,
        "init": 0.1,
        "units": "type"
    }

    # --------------------------------------------------------
    # Dust
    # --------------------------------------------------------

    model_params["dust_type"]["init"] = 4

    model_params["dust2"]["isfree"] = True

    model_params["dust2"]["prior"] = (
        priors.ClippedNormal(
            mean=0.3,
            sigma=1,
            mini=0,
            maxi=4
        )
    )

    model_params["dust_index"] = dict(
        N=1,
        isfree=True,
        init=0,
        prior=priors.TopHat(
            mini=-1,
            maxi=0.4
        )
    )

    model_params["dust1"] = {
        "N": 1,
        "isfree": False,
        "init": 0.0,
        "units": (
            "optical depth towards young stars"
        ),
        "prior": None,
        "depends_on": to_dust1
    }

    model_params["dust1_index"] = {
        "N": 1,
        "isfree": False,
        "init": -1.0,
        "units": (
            "power-law index of the "
            "birth-cloud attenuation"
        )
    }

    model_params["dust1_fraction"] = dict(
        N=1,
        isfree=True,
        init=1.0,
        prior=priors.ClippedNormal(
            mean=1,
            sigma=0.3,
            mini=0,
            maxi=2
        )
    )

    # --------------------------------------------------------
    # Metallicity
    # --------------------------------------------------------

    model_params["logzsol"]["isfree"] = True
    model_params["logzsol"]["init"] = -1

    model_params["logzsol"]["prior"] = (
        priors.TopHat(
            mini=-2,
            maxi=0.19
        )
    )

    # --------------------------------------------------------
    # Nebular emission
    # --------------------------------------------------------

    model_params.update(
        TemplateLibrary["nebular"]
    )

    model_params["add_neb_emission"]["init"] = True

    model_params["nebemlineinspec"]["init"] = False

    model_params["gas_logu"]["isfree"] = True
    model_params["gas_logu"]["init"] = -2.5

    model_params["gas_logu"]["prior"] = (
        priors.TopHat(
            mini=-4,
            maxi=-1
        )
    )

    model_params["elines_to_ignore"] = {
        "init": ["Ly-alpha 1215"],
        "isfree": False
    }

    # --------------------------------------------------------
    # IGM
    # --------------------------------------------------------

    model_params.update(
        TemplateLibrary["igm"]
    )

    model_params["igm_factor"]["isfree"] = True
    model_params["igm_factor"]["init"] = 1

    model_params["igm_factor"]["prior"] = (
        priors.ClippedNormal(
            mean=1,
            sigma=0.3,
            mini=0,
            maxi=2
        )
    )

    return SpecModel(model_params)


def build_sps():

    return FastStepBasis()


# ============================================================
# POSTERIOR SED FUNCTION
# ============================================================

def posterior_sed(
    result,
    obs,
    model,
    sps,
    nsamples=500,
    n_wavelengths=5000
):

    chain = np.asarray(
        result["chain"]
    )

    weights = np.asarray(
        result["weights"],
        dtype=float
    )

    weights /= weights.sum()

    # --------------------------------------------------------
    # Draw posterior samples
    # --------------------------------------------------------

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(chain),
        size=min(
            nsamples,
            len(chain)
        ),
        replace=True,
        p=weights
    )

    theta_samples = chain[indices]

    # --------------------------------------------------------
    # Redshift range represented by the samples
    # --------------------------------------------------------

    theta_labels = list(
        result["theta_labels"]
    )

    z_index = theta_labels.index(
        "zred"
    )

    z_samples = theta_samples[:, z_index]

    z_min = np.percentile(
        z_samples,
        1
    )

    z_max = np.percentile(
        z_samples,
        99
    )

    # --------------------------------------------------------
    # Common wavelength grid
    # --------------------------------------------------------

    wave_min = (
        sps.wavelengths.min()
        * (1 + z_min)
    )

    wave_max = (
        sps.wavelengths.max()
        * (1 + z_max)
    )

    common_wave = np.logspace(
        np.log10(wave_min),
        np.log10(wave_max),
        n_wavelengths
    )

    # --------------------------------------------------------
    # Generate spectra
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

        wave = (
            sps.wavelengths
            * (1 + zred)
        )

        spectrum_interp = np.interp(
            common_wave,
            wave,
            mspec,
            left=np.nan,
            right=np.nan
        )

        spectra.append(
            spectrum_interp
        )

        photometry.append(
            mphot
        )

    spectra = np.asarray(
        spectra
    )

    photometry = np.asarray(
        photometry
    )

    # --------------------------------------------------------
    # Percentiles
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

    phot_p16 = np.percentile(
        photometry,
        16,
        axis=0
    )

    phot_p50 = np.percentile(
        photometry,
        50,
        axis=0
    )

    phot_p84 = np.percentile(
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
# LOAD FIRST FILE
# ============================================================

print(f"Loading: {file1}")

result1, obs1, _ = reader.results_from(
    file1,
    dangerous=False
)

model1 = build_model(
    zlo1,
    zup1
)

sps1 = build_sps()


# ============================================================
# LOAD SECOND FILE
# ============================================================

print(f"Loading: {file2}")

result2, obs2, _ = reader.results_from(
    file2,
    dangerous=False
)

model2 = build_model(
    zlo2,
    zup2
)

sps2 = build_sps()


# ============================================================
# GET POSTERIOR SEDS
# ============================================================

print(
    f"Generating posterior SED: {label1}"
)

(
    wave1,
    sed1_p16,
    sed1_p50,
    sed1_p84,
    phot1_p16,
    phot1_p50,
    phot1_p84
) = posterior_sed(
    result1,
    obs1,
    model1,
    sps1,
    nsamples=nsamples,
    n_wavelengths=n_wavelengths
)


print(
    f"Generating posterior SED: {label2}"
)

(
    wave2,
    sed2_p16,
    sed2_p50,
    sed2_p84,
    phot2_p16,
    phot2_p50,
    phot2_p84
) = posterior_sed(
    result2,
    obs2,
    model2,
    sps2,
    nsamples=nsamples,
    n_wavelengths=n_wavelengths
)


# ============================================================
# CONVERT TO AB MAGNITUDES
# ============================================================

def maggies_to_ab(flux):

    flux = np.asarray(
        flux,
        dtype=float
    )

    result = np.full_like(
        flux,
        np.nan
    )

    valid = (
        np.isfinite(flux)
        &
        (flux > 0)
    )

    result[valid] = (
        -2.5
        * np.log10(
            flux[valid]
        )
    )

    return result


# ------------------------------------------------------------
# Continuous SEDs
# ------------------------------------------------------------

sed1_mag_p16 = maggies_to_ab(
    sed1_p84
)

sed1_mag_p50 = maggies_to_ab(
    sed1_p50
)

sed1_mag_p84 = maggies_to_ab(
    sed1_p16
)


sed2_mag_p16 = maggies_to_ab(
    sed2_p84
)

sed2_mag_p50 = maggies_to_ab(
    sed2_p50
)

sed2_mag_p84 = maggies_to_ab(
    sed2_p16
)


# ------------------------------------------------------------
# Model photometry
# ------------------------------------------------------------

phot1_mag_p16 = maggies_to_ab(
    phot1_p84
)

phot1_mag_p50 = maggies_to_ab(
    phot1_p50
)

phot1_mag_p84 = maggies_to_ab(
    phot1_p16
)


phot2_mag_p16 = maggies_to_ab(
    phot2_p84
)

phot2_mag_p50 = maggies_to_ab(
    phot2_p50
)

phot2_mag_p84 = maggies_to_ab(
    phot2_p16
)


# ============================================================
# OBSERVED PHOTOMETRY
# ============================================================

# The observations should be the same physical data if the
# two files are fits to the same object.

wphot = obs1["phot_wave"]

obs_flux = np.asarray(
    obs1["maggies"],
    dtype=float
)

obs_unc = np.asarray(
    obs1["maggies_unc"],
    dtype=float
)

snr = (
    obs_flux / obs_unc
)

detected = (
    (snr >= 1.5)
    &
    np.isfinite(obs_flux)
    &
    np.isfinite(obs_unc)
    &
    (obs_unc > 0)
    &
    (obs_flux > obs_unc)
)

upper_limits = (
    (snr < 1.5)
    &
    np.isfinite(obs_unc)
    &
    (obs_unc > 0)
)


obs_mag = np.full_like(
    obs_flux,
    np.nan
)

obs_mag[detected] = (
    maggies_to_ab(
        obs_flux[detected]
    )
)


mag_bright = maggies_to_ab(
    obs_flux[detected]
    + obs_unc[detected]
)

mag_faint = maggies_to_ab(
    obs_flux[detected]
    - obs_unc[detected]
)

obs_mag_err_lower = (
    obs_mag[detected]
    - mag_bright
)

obs_mag_err_upper = (
    mag_faint
    - obs_mag[detected]
)


upper_limit_mag = maggies_to_ab(
    1.5 * obs_unc[upper_limits]
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 8)
)


# ------------------------------------------------------------
# SED 1
# ------------------------------------------------------------

ax.fill_between(
    wave1,
    sed1_mag_p16,
    sed1_mag_p84,
    color=colour1,
    alpha=0.25,
    label=f"{label1} 16–84%"
)

ax.plot(
    wave1,
    sed1_mag_p50,
    color=colour1,
    linewidth=2,
    label=f"{label1} median"
)


# ------------------------------------------------------------
# SED 2
# ------------------------------------------------------------

ax.fill_between(
    wave2,
    sed2_mag_p16,
    sed2_mag_p84,
    color=colour2,
    alpha=0.25,
    label=f"{label2} 16–84%"
)

ax.plot(
    wave2,
    sed2_mag_p50,
    color=colour2,
    linewidth=2,
    label=f"{label2} median"
)


# ------------------------------------------------------------
# Observed photometry
# ------------------------------------------------------------

ax.errorbar(
    wphot[detected],
    obs_mag[detected],
    yerr=[
        obs_mag_err_lower,
        obs_mag_err_upper
    ],
    fmt="o",
    color="black",
    markersize=7,
    capsize=3,
    label="Observed photometry",
    zorder=10
)


# ------------------------------------------------------------
# Upper limits
# ------------------------------------------------------------

ax.scatter(
    wphot[upper_limits],
    upper_limit_mag,
    marker="v",
    s=80,
    color="black",
    label="Observed < 1.5σ",
    zorder=10
)


# ============================================================
# FORMATTING
# ============================================================

ax.set_xlim(4000,300000)

ax.set_xscale("log")

ax.set_xlabel(
    r"Wavelength [$\AA$]"
)

ax.set_ylim(23, 35)

ax.set_ylabel(
    r"$m_{\mathrm{AB}}$"
)

ax.set_title(
    "SED comparison"
)

ax.invert_yaxis()

ax.legend(
    fontsize=11
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()