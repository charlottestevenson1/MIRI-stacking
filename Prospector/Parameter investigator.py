import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as reader
from prospect.sources import FastStepBasis
import copy

from astropy.cosmology import WMAP9 as cosmo


# ============================================================
# MODEL SETTINGS
# ============================================================

# -----------------------
# adjust_continuity_agebins
# -----------------------

def adjust_continuity_agebins(parset, nbins=8):
    from prospect.models import priors

    if nbins < 4:
        raise ValueError(
            "Must have nbins >= 4, returning"
        )

    tuniv = (
        cosmo.age(
            parset["zred"]["init"]
        ).value * 1e9
    )

    lim1, lim2 = 6.6990, 7.0

    agelims = (
        [0, lim1]
        + np.linspace(
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

    mean = np.zeros(
        ncomp - 1
    )

    scale = (
        np.ones_like(mean)
        * 0.5
    )

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


# -----------------------
# zred_to_agebins
# -----------------------

def zred_to_agebins(
    zred=None,
    nbins_sfh=8,
    **extras
):

    tuniv = (
        cosmo.age(
            zred[0]
        ).value * 1e9
    )

    lim1, lim2 = 6.6990, 7.0

    agelims = (
        [0, lim1]
        + np.linspace(
            lim2,
            np.log10(tuniv),
            nbins_sfh[0] - 1
        ).tolist()
    )

    agebins = np.array([
        agelims[:-1],
        agelims[1:]
    ])

    return agebins.T


# -----------------------
# zlogsfr_ratios_to_masses
# -----------------------

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
        - 10 ** agebins[:, 0]
    )

    coeffs = np.array([
        (
            (1.0 / np.prod(sratios[:i]))
            * (
                np.prod(dt[1:i + 1])
                / np.prod(dt[:i])
            )
        )
        for i in range(nbins)
    ])

    m1 = (
        10 ** logmass
    ) / coeffs.sum()

    return m1 * coeffs


# -----------------------
# to_dust1
# -----------------------

def to_dust1(
    dust1_fraction=None,
    dust1=None,
    dust2=None,
    **extras
):
    return dust1_fraction * dust2


# -----------------------
# build_model
# -----------------------

def build_model(
    nbins_sfh=8,
    **kwargs
):

    from prospect.models.sedmodel import SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    model_params = TemplateLibrary[
        "continuity_sfh"
    ]

    # Redshift
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

    # Age bins
    model_params["nbins_sfh"] = dict(
        N=1,
        isfree=False,
        init=nbins_sfh
    )

    model_params = adjust_continuity_agebins(
        model_params,
        nbins=nbins_sfh
    )

    model_params["agebins"]["N"] = (
        nbins_sfh
    )

    model_params["agebins"]["depends_on"] = (
        zred_to_agebins
    )

    # Mass
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

    # IMF
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

    # Dust
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

    # Metallicity
    model_params["logzsol"]["isfree"] = True
    model_params["logzsol"]["init"] = -1

    model_params["logzsol"]["prior"] = (
        priors.TopHat(
            mini=-2,
            maxi=0.19
        )
    )

    # Nebular emission
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

    # Ignore Lyman alpha
    model_params["elines_to_ignore"] = {
        "init": ["Ly-alpha 1215"],
        "isfree": False
    }

    # IGM
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


# -----------------------
# build_sps
# -----------------------

def build_sps(**kwargs):
    return FastStepBasis()


# ============================================================
# SETTINGS
# ============================================================

zlo = 8
zup = 9

filename = (
    f"Prospector/Fits/"
    f"z={zlo}-{zup} with MIRI.h5"
)

logzsol_values = [
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0
]

nsamples = 1

colours = [
    "#2166AC",
    "#67A9CF",
    "#F7F7F7",
    "#EF8A62",
    "#B2182B"
]

output_dir = (
    "Prospector/Plots/"
    "Metallicity SEDs"
)


# ============================================================
# LOAD RESULTS
# ============================================================

result, obs, _ = reader.results_from(
    filename,
    dangerous=False
)

theta_labels = list(
    result["theta_labels"]
)

chain = np.asarray(
    result["chain"]
)

weights = np.asarray(
    result["weights"],
    dtype=float
)

weights /= weights.sum()


# ============================================================
# BUILD MODEL
# ============================================================

model = build_model()
sps = build_sps()


# ============================================================
# SELECT REPRESENTATIVE POSTERIOR SAMPLE
# ============================================================

imax = np.argmax(
    result["lnprobability"]
)

theta_base = chain[
    imax
].copy()

print(
    "Using MAP posterior sample as baseline."
)


# ============================================================
# IDENTIFY LOGZSOL
# ============================================================

logzsol_index = theta_labels.index(
    "logzsol"
)


# ============================================================
# COMMON WAVELENGTH GRID
# ============================================================

common_wave = np.logspace(
    np.log10(
        sps.wavelengths.min()
        * (1 + zlo)
    ),
    np.log10(
        sps.wavelengths.max()
        * (1 + zup)
    ),
    5000
)


# ============================================================
# OBS DICTIONARY FOR CONTINUOUS SPECTRUM
# ============================================================

obs_sed = copy.deepcopy(
    obs
)

obs_sed["wavelength"] = (
    common_wave
)

obs_sed["spectrum"] = (
    np.zeros_like(common_wave)
)

obs_sed["unc"] = (
    np.ones_like(common_wave)
)

obs_sed["mask"] = np.ones(
    len(common_wave),
    dtype=bool
)


# ============================================================
# GENERATE SEDS
# ============================================================

spectra = []

for logzsol, colour in zip(
    logzsol_values,
    colours
):

    theta = theta_base.copy()

    # Change only metallicity
    theta[logzsol_index] = logzsol

    print(
        f"Generating SED for "
        f"logzsol = {logzsol}"
    )

    spec, phot, _ = model.predict(
        theta,
        obs=obs_sed,
        sps=sps
    )

    spectra.append(
        spec
    )


# ============================================================
# CONVERT TO AB MAGNITUDES
# ============================================================

spectra = np.asarray(
    spectra
)

spectra = np.where(
    spectra > 0,
    spectra,
    np.nan
)

spectra_ab = (
    -2.5
    * np.log10(spectra)
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

valid = (
    np.isfinite(obs_flux)
    & np.isfinite(obs_unc)
    & (obs_flux > 0)
    & (obs_unc > 0)
)

obs_mag = np.full_like(
    obs_flux,
    np.nan
)

obs_mag[valid] = (
    -2.5
    * np.log10(
        obs_flux[valid]
    )
)

obs_mag_err = (
    2.5
    / np.log(10)
    * obs_unc[valid]
    / obs_flux[valid]
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 8),
    facecolor="none"
)

ax.set_facecolor("none")

for logzsol, sed, colour in zip(
    logzsol_values,
    spectra_ab,
    colours
):

    ax.plot(
        common_wave,
        sed,
        color=colour,
        lw=2.5,
        label=fr"$\log Z/Z_\odot = {logzsol:.1f}$"
    )


# ------------------------------------------------------------
# Observed photometry
# ------------------------------------------------------------

ax.errorbar(
    wphot[valid],
    obs_mag[valid],
    yerr=obs_mag_err,
    fmt="o",
    color="white",
    ecolor="white",
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=1.5,
    capsize=4,
    elinewidth=1.5,
    label="Observed photometry"
)


# ============================================================
# FORMATTING
# ============================================================

ax.set_xscale("log")

ax.set_xlim(
    4000,
    300000
)

ax.set_ylim(
    23,
    35
)

ax.invert_yaxis()

ax.set_xlabel(
    r"Observed-frame wavelength [$\AA$]",
    color="white",
    fontsize=20,
    labelpad=12
)

ax.set_ylabel(
    r"$m_{\mathrm{AB}}$",
    color="white",
    fontsize=20,
    labelpad=12
)

ax.tick_params(
    axis="both",
    colors="white",
    labelsize=15,
    width=1.5,
    length=6
)

for spine in ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1.5)

ax.grid(
    color="white",
    alpha=0.12
)

ax.text(
    0.02,
    0.95,
    f"$z = {zlo}-{zup}$",
    transform=ax.transAxes,
    color="white",
    fontsize=18,
    fontweight="bold",
    ha="left",
    va="top"
)

legend = ax.legend(
    loc="best",
    fontsize=14,
    frameon=True,
    facecolor="black",
    edgecolor="white"
)

for text in legend.get_texts():
    text.set_color("white")

legend.get_frame().set_alpha(0.75)

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_file = (
    f"{output_dir}/"
    f"logzsol_SEDs_z={zlo}-{zup}.png"
)

plt.savefig(
    output_file,
    dpi=400,
    bbox_inches="tight",
    transparent=True
)

plt.show()