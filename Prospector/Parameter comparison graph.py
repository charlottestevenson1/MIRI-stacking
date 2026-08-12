import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as reader


# ==========================================
# PARAMETERS
# ==========================================

params = [
    "zred",
    "logzsol",
    "dust2",
    "logmass",
    "logsfr_ratios_1",
    "logsfr_ratios_2",
    "logsfr_ratios_3",
    "logsfr_ratios_4",
    "logsfr_ratios_5",
    "logsfr_ratios_6",
    "logsfr_ratios_7",
    "dust_index",
    "dust1_fraction",
    "gas_logu",
    "igm_factor"
]


redshift_bins = [
    {
        "label": "z = 8–9",
        "with_miri": "Prospector/z=8-9 with MIRI.h5",
        "without_miri": "Prospector/z=8-9 no MIRI.h5",
    },
    {
        "label": "z = 9–10",
        "with_miri": "Prospector/z=9-10 with MIRI.h5",
        "without_miri": "Prospector/z=9-10 no MIRI.h5",
    },
    {
        "label": "z = 10–11",
        "with_miri": "Prospector/z=10-11 with MIRI.h5",
        "without_miri": "Prospector/z=10-11 no MIRI.h5",
    },
    {
        "label": "z = 11–12",
        "with_miri": "Prospector/z=11-12 with MIRI.h5",
        "without_miri": "Prospector/z=11-12 no MIRI.h5",
    },
    {
        "label": "z = 12–15",
        "with_miri": "Prospector/z=12-15 with MIRI.h5",
        "without_miri": "Prospector/z=12-15 no MIRI.h5",
    },
]


# ==========================================
# FUNCTIONS
# ==========================================

def weighted_percentile(values, weights, percentile):

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]

    return np.interp(
        percentile / 100,
        cumulative,
        values
    )


def load_parameter(filename, parameter):

    result, _, _ = reader.results_from(
        filename,
        dangerous=False
    )

    index = list(result["theta_labels"]).index(parameter)

    samples = np.asarray(
        result["chain"][:, index]
    )

    weights = np.asarray(
        result["weights"],
        dtype=float
    )

    weights /= weights.sum()

    return samples, weights


def get_summary(samples, weights):

    p16 = weighted_percentile(
        samples, weights, 16
    )

    p50 = weighted_percentile(
        samples, weights, 50
    )

    p84 = weighted_percentile(
        samples, weights, 84
    )

    width = p84 - p16

    sigma = width / 2

    return {
        "p16": p16,
        "p50": p50,
        "p84": p84,
        "width": width,
        "sigma": sigma
    }


# ==========================================
# LOAD RESULTS
# ==========================================

all_results = {}

for zbin in redshift_bins:

    label = zbin["label"]

    # With MIRI
    miri_result, _, _ = reader.results_from(
        zbin["with_miri"],
        dangerous=False
    )

    # No MIRI
    try:

        nomiri_result, _, _ = reader.results_from(
            zbin["without_miri"],
            dangerous=False
        )

    except (FileNotFoundError, OSError):

        print(
            f"No MIRI file not found for {label}"
        )

        nomiri_result = None

    all_results[label] = {
        "miri": miri_result,
        "nomiri": nomiri_result
    }


# ==========================================
# CALCULATE COMPARISON METRICS
# ==========================================

comparison = {}

for zbin in redshift_bins:

    label = zbin["label"]

    miri_result = all_results[label]["miri"]
    nomiri_result = all_results[label]["nomiri"]

    # Skip bins without a no-MIRI result
    if nomiri_result is None:
        continue

    comparison[label] = {}

    miri_labels = list(
        miri_result["theta_labels"]
    )

    nomiri_labels = list(
        nomiri_result["theta_labels"]
    )

    for parameter in params:

        # Make sure parameter exists in both models
        if (
            parameter not in miri_labels
            or parameter not in nomiri_labels
        ):
            continue

        # ------------------------------
        # MIRI posterior
        # ------------------------------

        i_miri = miri_labels.index(parameter)

        samples_miri = np.asarray(
            miri_result["chain"][:, i_miri]
        )

        weights_miri = np.asarray(
            miri_result["weights"],
            dtype=float
        )

        weights_miri /= weights_miri.sum()

        summary_miri = get_summary(
            samples_miri,
            weights_miri
        )

        # ------------------------------
        # No MIRI posterior
        # ------------------------------

        i_nomiri = nomiri_labels.index(parameter)

        samples_nomiri = np.asarray(
            nomiri_result["chain"][:, i_nomiri]
        )

        weights_nomiri = np.asarray(
            nomiri_result["weights"],
            dtype=float
        )

        weights_nomiri /= weights_nomiri.sum()

        summary_nomiri = get_summary(
            samples_nomiri,
            weights_nomiri
        )

        # ------------------------------
        # Comparison
        # ------------------------------

        median_shift = (
            summary_miri["p50"]
            - summary_nomiri["p50"]
        )

        combined_sigma = np.sqrt(
            summary_miri["sigma"]**2
            + summary_nomiri["sigma"]**2
        )

        # Normalised shift
        significance = (
            median_shift / combined_sigma
            if combined_sigma > 0
            else np.nan
        )

        # > 1 means MIRI gives a narrower posterior
        width_ratio = (
            summary_nomiri["width"]
            / summary_miri["width"]
            if summary_miri["width"] > 0
            else np.nan
        )

        comparison[label][parameter] = {
            "median_shift": median_shift,
            "significance": significance,
            "width_ratio": width_ratio,
            "miri_summary": summary_miri,
            "nomiri_summary": summary_nomiri
        }


# ==========================================
# PRINT RESULTS
# ==========================================

for label, results in comparison.items():

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    for parameter, values in results.items():

        print(
            f"{parameter:20s} "
            f"shift = {values['median_shift']:+.3f}   "
            f"shift/sigma = {values['significance']:+.2f}   "
            f"width ratio = {values['width_ratio']:.2f}"
        )


# ==========================================
# PLOT
# ==========================================

available_bins = list(comparison.keys())

fig, axes = plt.subplots(
    1,
    len(available_bins),
    figsize=(7 * len(available_bins), 8),
    sharey=True
)

if len(available_bins) == 1:
    axes = [axes]


for ax, label in zip(axes, available_bins):

    results = comparison[label]

    x = []
    y = []
    names = []

    for parameter, values in results.items():

        if not np.isfinite(values["width_ratio"]):
            continue

        if not np.isfinite(values["significance"]):
            continue

        x.append(values["width_ratio"])
        y.append(values["significance"])
        names.append(parameter)

    x = np.asarray(x)
    y = np.asarray(y)


    # --------------------------------------
    # Points
    # --------------------------------------

    ax.scatter(
        x,
        y,
        s=60,
        color="pink",
        zorder=3
    )


    # --------------------------------------
    # Parameter labels
    # --------------------------------------

    for xi, yi, name in zip(
        x, y, names
    ):

        ax.annotate(
            name,
            (xi, yi),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )


    # --------------------------------------
    # Reference lines
    # --------------------------------------

    # No change in posterior width
    ax.axvline(
        1,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )

    # No change in median
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )


    # --------------------------------------
    # Axes
    # --------------------------------------

    ax.set_xscale("log")

    ax.set_xlabel(
        "Width ratio\n"
        r"$W_{\rm no\,MIRI}/W_{\rm MIRI}$"
    )

    ax.set_title(label)

    ax.grid(
        alpha=0.2
    )


axes[0].set_ylabel(
    r"Median shift / combined $1\sigma$"
)


fig.suptitle(
    "Effect of MIRI on posterior constraints",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "Prospector/Plots/MIRI_parameter_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()