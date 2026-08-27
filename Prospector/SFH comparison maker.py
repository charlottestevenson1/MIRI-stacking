import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as reader
import time
import os
from astropy.cosmology import WMAP9 as cosmo

# ============================================================
# SETTINGS
# ============================================================

zlo = 12
zup = 15

with_miri_file = (
    f"Prospector/Fits/z={zlo}-{zup} with MIRI.h5"
)

no_miri_file = (
    f"Prospector/Fits/z={zlo}-{zup} no MIRI.h5"
)

n_sfh_samples = 1000

colour_miri = "steelblue"
colour_no_miri = "hotpink"

output_dir = "Prospector/Plots/SFHs"
os.makedirs(output_dir, exist_ok=True)

title = f"SFH comparison: z = {zlo}-{zup}"


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def adjust_continuity_agebins(parset, nbins=8):
    from prospect.models import priors

    if nbins < 4:
        raise ValueError("Must have nbins >= 4")

    tuniv = (
        cosmo.age(
            parset["zred"]["init"]
        ).value * 1e9
    )

    lim1 = 6.6990
    lim2 = 7.0

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

    tuniv = cosmo.age(
        zred[0]
    ).value * 1e9

    lim1, lim2 = 6.6990, 7.0

    if np.isscalar(nbins_sfh):
        nbins = int(nbins_sfh)
    else:
        nbins = int(nbins_sfh[0])

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
        - 10 ** agebins[:, 0]
    )

    coeffs = np.array([
        (
            (1.0 / np.prod(sratios[:i]))
            *
            (
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


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(nbins_sfh=8):

    from prospect.models.sedmodel import SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    model_params = TemplateLibrary[
        "continuity_sfh"
    ]

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
        "units": "optical depth towards young stars",
        "prior": None,
        "depends_on": lambda dust1_fraction=None,
        dust1=None,
        dust2=None,
        **extras: dust1_fraction * dust2
    }

    model_params["dust1_index"] = {
        "N": 1,
        "isfree": False,
        "init": -1.0,
        "units": "power-law index of the birth-cloud attenuation"
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

    model_params["logzsol"]["isfree"] = True
    model_params["logzsol"]["init"] = -1
    model_params["logzsol"]["prior"] = (
        priors.TopHat(
            mini=-2,
            maxi=0.19
        )
    )

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


# ============================================================
# BUILD MODEL
# ============================================================

print(
    f"Building model for z={zlo}-{zup}..."
)

start_time = time.perf_counter()

model = build_model()

print(
    f"Model built after "
    f"{time.perf_counter() - start_time:.2f} s"
)


# ============================================================
# LOAD BOTH RESULTS
# ============================================================

print(
    f"Loading With MIRI:\n{with_miri_file}"
)

with_miri_result, _, _ = reader.results_from(
    with_miri_file,
    dangerous=False
)

print(
    f"Loading No MIRI:\n{no_miri_file}"
)

no_miri_result, _, _ = reader.results_from(
    no_miri_file,
    dangerous=False
)


# ============================================================
# GET CHAINS AND WEIGHTS
# ============================================================

with_miri_chain = np.asarray(
    with_miri_result["chain"]
)

with_miri_weights = np.asarray(
    with_miri_result["weights"],
    dtype=float
)

no_miri_chain = np.asarray(
    no_miri_result["chain"]
)

no_miri_weights = np.asarray(
    no_miri_result["weights"],
    dtype=float
)


# ============================================================
# DRAW POSTERIOR SAMPLES
# ============================================================

def draw_thetas(
    chain,
    weights,
    nsamples=1000,
    seed=42
):

    weights = weights / weights.sum()

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        len(chain),
        size=min(
            nsamples,
            len(chain)
        ),
        replace=True,
        p=weights
    )

    return chain[indices]


theta_with_miri = draw_thetas(
    with_miri_chain,
    with_miri_weights,
    nsamples=n_sfh_samples,
    seed=42
)

theta_no_miri = draw_thetas(
    no_miri_chain,
    no_miri_weights,
    nsamples=n_sfh_samples,
    seed=43
)


# ============================================================
# CONSTRUCT ONE SFH
# ============================================================

def get_sfh(
    theta,
    theta_labels
):

    z_index = theta_labels.index(
        "zred"
    )

    logmass_index = theta_labels.index(
        "logmass"
    )

    sfr_indices = [
        theta_labels.index(
            f"logsfr_ratios_{i}"
        )
        for i in range(1, 8)
    ]

    zred = theta[z_index]
    logmass = theta[logmass_index]

    logsfr_ratios = theta[
        sfr_indices
    ]

    agebins = zred_to_agebins(
        zred=np.array([zred]),
        nbins_sfh=8
    )

    age_low = (
        10 ** agebins[:, 0]
    )

    age_high = (
        10 ** agebins[:, 1]
    )

    dt = age_high - age_low

    masses = zlogsfr_ratios_to_masses(
        logmass=logmass,
        logsfr_ratios=logsfr_ratios,
        zred=np.array([zred])
    )

    sfr = masses / dt

    t0 = age_low / 1e6
    t1 = age_high / 1e6

    return t0, t1, sfr


# ============================================================
# CALCULATE POSTERIOR SFH
# ============================================================

def posterior_sfh(
    theta_samples,
    theta_labels
):

    all_t0 = []
    all_t1 = []
    all_sfr = []

    for theta in theta_samples:

        t0, t1, sfr = get_sfh(
            theta,
            theta_labels
        )

        all_t0.append(t0)
        all_t1.append(t1)
        all_sfr.append(sfr)

    all_t0 = np.asarray(
        all_t0
    )

    all_t1 = np.asarray(
        all_t1
    )

    all_sfr = np.asarray(
        all_sfr
    )

    t0_median = np.median(
        all_t0,
        axis=0
    )

    t1_median = np.median(
        all_t1,
        axis=0
    )

    sfr_p16 = np.percentile(
        all_sfr,
        16,
        axis=0
    )

    sfr_p50 = np.percentile(
        all_sfr,
        50,
        axis=0
    )

    sfr_p84 = np.percentile(
        all_sfr,
        84,
        axis=0
    )

    return (
        t0_median,
        t1_median,
        sfr_p16,
        sfr_p50,
        sfr_p84
    )


# ============================================================
# CALCULATE BOTH
# ============================================================

with_miri_labels = list(
    with_miri_result["theta_labels"]
)

no_miri_labels = list(
    no_miri_result["theta_labels"]
)

(
    t0_miri,
    t1_miri,
    sfr_miri_p16,
    sfr_miri_p50,
    sfr_miri_p84
) = posterior_sfh(
    theta_with_miri,
    with_miri_labels
)

(
    t0_no_miri,
    t1_no_miri,
    sfr_no_miri_p16,
    sfr_no_miri_p50,
    sfr_no_miri_p84
) = posterior_sfh(
    theta_no_miri,
    no_miri_labels
)


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_sfh(ax, t0, t1, p16, p50, p84, colour, label, linestyle="-"):
    # Step boundaries
    x = np.concatenate([t0, [t1[-1]]])

    # Median step function
    y = np.concatenate([p50, [p50[-1]]])

    # Uncertainty step functions
    y16 = np.concatenate([p16, [p16[-1]]])
    y84 = np.concatenate([p84, [p84[-1]]])

    ax.step(
        x,
        y,
        where="post",
        color=colour,
        lw=3,
        linestyle=linestyle,
        label=label
    )

    ax.fill_between(
        x,
        y16,
        y84,
        step="post",
        color=colour,
        alpha=0.18
    )


fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 10),
    sharex=True,
    sharey=True,
    facecolor="none"
)

for ax in axes:
    ax.set_facecolor("none")

# With MIRI
plot_sfh(
    axes[0],
    t0_miri,
    t1_miri,
    sfr_miri_p16,
    sfr_miri_p50,
    sfr_miri_p84,
    "deepskyblue",
    "With MIRI"
)

# No MIRI
plot_sfh(
    axes[1],
    t0_no_miri,
    t1_no_miri,
    sfr_no_miri_p16,
    sfr_no_miri_p50,
    sfr_no_miri_p84,
    "hotpink",
    "No MIRI"
)

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

for ax in axes:

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()

    ax.tick_params(
        axis="both",
        colors="white",
        labelsize=14
    )

    for spine in ax.spines.values():
        spine.set_color("white")

    ax.grid(
        color="white",
        alpha=0.12
    )

    # legend = ax.legend(
    #     loc="upper right",
    #     fontsize=15,
    #     frameon=True,
    #     facecolor="black",
    #     edgecolor="white"
    # )

    # for text in legend.get_texts():
    #     text.set_color("white")

    # legend.get_frame().set_alpha(0.7)

axes[0].set_title(
    "With MIRI",
    color="deepskyblue",
    fontsize=20,
    fontweight="bold"
)

axes[1].set_title(
    "No MIRI",
    color="hotpink",
    fontsize=20,
    fontweight="bold"
)

axes[0].set_ylabel(
    r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]",
    color="white",
    fontsize=18
)

axes[1].set_ylabel(
    r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]",
    color="white",
    fontsize=18
)

axes[1].set_xlabel(
    "Lookback time [Myr]",
    color="white",
    fontsize=18
)

max_time = cosmo.age(zlo).value * 1e3

axes[1].set_xlim(
    max_time,
    1
)

fig.suptitle(
    f"Star-formation histories: z = {zlo}-{zup}",
    color="white",
    fontsize=22,
    fontweight="bold"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.savefig(
    f"{output_dir}/SFH_MIRI_comparison_z={zlo}-{zup}.png",
    dpi=400,
    bbox_inches="tight",
    transparent=True
)

plt.show()