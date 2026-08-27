import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as reader
import time
import os
from astropy.cosmology import WMAP9 as cosmo


# ============================================================
# SETTINGS
# ============================================================

zlo = 8
zup = 9

selected_zbin = f"z = {zlo}-{zup}"

MIRI_desc = 'no MIRI'

title = f"SFH comparison ({MIRI_desc}): z = {zlo}-{zup}"

# Parameter used for the selection
cut_param = "logmass"

# Split:
# sample 1: cut_param <= cut_value
# sample 2: cut_param > cut_value
cut_value = 8.15

gt_or_lt = ">"

sample1_name = fr"{cut_param} $\leq$ {cut_value}" # less than CV
sample2_name = fr"{cut_param} > {cut_value}" # greater than CV

# Number of posterior samples used
n_sfh_samples = 1000

# Colours
# steelblue: majority
# red: minority
colour1 = "steelblue" if gt_or_lt==">" else "red"
colour2 = "red" if gt_or_lt==">" else "steelblue"

# Output
main_dir = "Prospector/Plots/Bimodality Checks"

# CHANGE?
sub_dir = f"z={zlo}-{zup}, {cut_param} {gt_or_lt} {cut_value}"

output_dir = f"{main_dir}/{sub_dir}"

os.makedirs(output_dir, exist_ok=True)


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


def zred_to_agebins(zred=None, nbins_sfh=8, **extras):

    tuniv = cosmo.age(zred[0]).value * 1e9

    lim1, lim2 = 6.6990, 7.0

    # Allow either nbins_sfh=8 or nbins_sfh=np.array([8])
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


def build_model(nbins_sfh=8):

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
    # Age bins
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
# LOAD RESULTS
# ============================================================

# CHANGE?

readfile = (
    f"Prospector/Fits/"
    f"z={zlo}-{zup} {MIRI_desc}.h5"
)

print(f"loading file: {readfile}")

result, obs, _ = reader.results_from(
    readfile,
    dangerous=False
)


# ============================================================
# CHAIN AND WEIGHTS
# ============================================================

chain = np.asarray(
    result["chain"]
)

weights = np.asarray(
    result["weights"],
    dtype=float
)

theta_labels = list(
    result["theta_labels"]
)


# ============================================================
# MAKE MASKS
# ============================================================

cut_index = theta_labels.index(
    cut_param
)

cut_values = chain[:, cut_index]

mask1 = cut_values <= cut_value
mask2 = cut_values > cut_value


# ============================================================
# PRINT SELECTION INFORMATION
# ============================================================

weights_norm = (
    weights / weights.sum()
)

posterior_weight1 = (
    weights_norm[mask1].sum()
)

posterior_weight2 = (
    weights_norm[mask2].sum()
)

print()
print("=" * 60)
print("SELECTION")
print("=" * 60)

print(
    f"Parameter:              {cut_param}"
)

print(
    f"Cut:                    {cut_value}"
)

print(
    f"Sample 1 entries:       {mask1.sum()}"
)

print(
    f"Sample 2 entries:       {mask2.sum()}"
)

print(
    f"Sample 1 fraction:      {mask1.mean():.3f}"
)

print(
    f"Sample 2 fraction:      {mask2.mean():.3f}"
)

print(
    f"Sample 1 posterior:     {posterior_weight1:.3f}"
)

print(
    f"Sample 2 posterior:     {posterior_weight2:.3f}"
)


# ============================================================
# FUNCTION: SAMPLE POSTERIOR
# ============================================================

def draw_thetas(
    chain,
    weights,
    mask,
    nsamples=1000,
    seed=42
):

    chain = chain[mask]
    weights = weights[mask]

    if len(chain) == 0:
        raise ValueError(
            "No samples remain after masking."
        )

    weights = (
        weights / weights.sum()
    )

    rng = np.random.default_rng(seed)

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


# ============================================================
# DRAW POSTERIOR SAMPLES
# ============================================================

theta_sample1 = draw_thetas(
    chain,
    weights,
    mask1,
    nsamples=n_sfh_samples,
    seed=42
)

theta_sample2 = draw_thetas(
    chain,
    weights,
    mask2,
    nsamples=n_sfh_samples,
    seed=43
)


# ============================================================
# FUNCTION: CONSTRUCT ONE SFH
# ============================================================

def get_sfh(theta):

    # --------------------------------------------------------
    # Get redshift
    # --------------------------------------------------------

    z_index = theta_labels.index(
        "zred"
    )

    zred = theta[z_index]

    # --------------------------------------------------------
    # Get logmass
    # --------------------------------------------------------

    logmass_index = theta_labels.index(
        "logmass"
    )

    logmass = theta[logmass_index]

    # --------------------------------------------------------
    # Get logsfr ratios
    # --------------------------------------------------------

    sfr_indices = [
        theta_labels.index(
            f"logsfr_ratios_{i}"
        )
        for i in range(1, 8)
    ]

    logsfr_ratios = theta[
        sfr_indices
    ]

    # --------------------------------------------------------
    # Age bins
    # --------------------------------------------------------

    agebins = zred_to_agebins(
        zred=np.array([zred]),
        nbins_sfh=8
    )

    # log10(years) -> years
    age_low = 10 ** agebins[:, 0]
    age_high = 10 ** agebins[:, 1]

    # Duration of each bin
    dt = age_high - age_low

    # --------------------------------------------------------
    # Stellar mass formed in each bin
    # --------------------------------------------------------

    masses = zlogsfr_ratios_to_masses(
        logmass=logmass,
        logsfr_ratios=logsfr_ratios,
        zred=np.array([zred])
    )

    # --------------------------------------------------------
    # SFR
    # --------------------------------------------------------

    sfr = masses / dt

    # Convert to Myr for plotting
    t0 = age_low / 1e6
    t1 = age_high / 1e6

    return t0, t1, sfr


# ============================================================
# FUNCTION: CALCULATE POSTERIOR SFH
# ============================================================

def posterior_sfh(theta_samples):

    all_t0 = []
    all_t1 = []
    all_sfr = []

    for theta in theta_samples:

        t0, t1, sfr = get_sfh(
            theta
        )

        all_t0.append(t0)
        all_t1.append(t1)
        all_sfr.append(sfr)

    all_t0 = np.asarray(all_t0)
    all_t1 = np.asarray(all_t1)
    all_sfr = np.asarray(all_sfr)

    # --------------------------------------------------------
    # Use the median age-bin boundaries
    #
    # Since zred varies slightly between posterior samples,
    # the age-bin boundaries also vary. For plotting, use
    # their medians.
    # --------------------------------------------------------

    t0_median = np.median(
        all_t0,
        axis=0
    )

    t1_median = np.median(
        all_t1,
        axis=0
    )

    # --------------------------------------------------------
    # SFR percentiles
    # --------------------------------------------------------

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
# CALCULATE BOTH SFHs
# ============================================================

(
    t0_1,
    t1_1,
    sfr1_p16,
    sfr1_p50,
    sfr1_p84
) = posterior_sfh(
    theta_sample1
)

(
    t0_2,
    t1_2,
    sfr2_p16,
    sfr2_p50,
    sfr2_p84
) = posterior_sfh(
    theta_sample2
)


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_sfh(
    ax,
    t0,
    t1,
    p16,
    p50,
    p84,
    colour,
    label
):

    # --------------------------------------------------------
    # Posterior uncertainty
    # --------------------------------------------------------

    for lo, hi, y16, y84 in zip(
        t0,
        t1,
        p16,
        p84
    ):

        ax.fill_between(
            [lo, hi],
            [y16, y16],
            [y84, y84],
            color=colour,
            alpha=0.25
        )

    # --------------------------------------------------------
    # Posterior median
    # --------------------------------------------------------

    for lo, hi, y in zip(
        t0,
        t1,
        p50
    ):

        ax.plot(
            [lo, hi],
            [y, y],
            color=colour,
            lw=2
        )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    sharex=True,
    sharey=True
)


# ============================================================
# SAMPLE 1
# ============================================================

plot_sfh(
    axes[0],
    t0_1,
    t1_1,
    sfr1_p16,
    sfr1_p50,
    sfr1_p84,
    colour1,
    sample1_name
)

axes[0].set_title(
    sample1_name
)

axes[0].set_xlabel(
    "Lookback time [Myr]"
)

axes[0].set_ylabel(
    r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]"
)

axes[0].set_xscale(
    "log"
)

axes[0].set_yscale(
    "log"
)

axes[0].invert_xaxis()


# ============================================================
# SAMPLE 2
# ============================================================

plot_sfh(
    axes[1],
    t0_2,
    t1_2,
    sfr2_p16,
    sfr2_p50,
    sfr2_p84,
    colour2,
    sample2_name
)

axes[1].set_title(
    sample2_name
)

axes[1].set_xlabel(
    "Lookback time [Myr]"
)

axes[1].set_xscale(
    "log"
)

axes[1].set_yscale(
    "log"
)

axes[1].invert_xaxis()


# ============================================================
# AXIS LIMITS
# ============================================================

max_time = cosmo.age(
    zlo
).value * 1e3

axes[0].set_xlim(
    max_time,
    1
)


# Find sensible common SFR range
all_sfr = np.concatenate([
    sfr1_p16,
    sfr1_p50,
    sfr1_p84,
    sfr2_p16,
    sfr2_p50,
    sfr2_p84
])

valid_sfr = (
    np.isfinite(all_sfr)
    &
    (all_sfr > 0)
)

if valid_sfr.any():

    ymin = (
        np.nanmin(
            all_sfr[valid_sfr]
        )
        / 2
    )

    ymax = (
        np.nanmax(
            all_sfr[valid_sfr]
        )
        * 2
    )

    axes[0].set_ylim(
        ymin,
        ymax
    )


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    title,
    fontsize=16
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    f"{output_dir}/"
    f"SFH_{selected_zbin.replace(' ', '_')}_"
    f"{cut_param}_comparison.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SFH COMPARISON COMPLETE")
print("=" * 60)

print(
    f"Sample 1: {sample1_name}"
)

print(
    f"Sample 2: {sample2_name}"
)

print(
    f"Posterior weight, sample 1: "
    f"{posterior_weight1:.3f}"
)

print(
    f"Posterior weight, sample 2: "
    f"{posterior_weight2:.3f}"
)

print(
    f"Saved to:\n{output_file}"
)