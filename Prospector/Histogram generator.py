import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as reader


# ==========================================
# SETTINGS
# ==========================================

parameter = "logmass"   # <-- change this

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
    # Add more bins here:
    
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

nbins = 50

label1 = "With MIRI"
label2 = "Without MIRI"


# ==========================================
# FUNCTIONS
# ==========================================

def weighted_median(values, weights):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)

    return values[
        np.searchsorted(
            cumulative,
            0.5 * cumulative[-1]
        )
    ]


def load_parameter(filename, parameter):

    result, _, _ = reader.results_from(
        filename,
        dangerous=False
    )

    index = list(result["theta_labels"]).index(parameter)

    samples = np.asarray(result["chain"])[:, index]
    weights = np.asarray(result["weights"], dtype=float)

    weights /= weights.sum()

    return samples, weights


# ==========================================
# LOAD ALL BINS
# ==========================================

data = []

for zbin in redshift_bins:

    # With MIRI should always exist
    samples_miri, weights_miri = load_parameter(
        zbin["with_miri"],
        parameter
    )

    # Without MIRI is optional
    try:
        samples_nomiri, weights_nomiri = load_parameter(
            zbin["without_miri"],
            parameter
        )
    except (FileNotFoundError, OSError):
        print(f"No MIRI file not found for {zbin['label']}")
        samples_nomiri = None
        weights_nomiri = None

    data.append({
        "label": zbin["label"],
        "miri_samples": samples_miri,
        "miri_weights": weights_miri,
        "nomiri_samples": samples_nomiri,
        "nomiri_weights": weights_nomiri,
    })


# ==========================================
# CREATE FIGURE
# ==========================================

nplots = len(data)

fig, axes = plt.subplots(
    nplots,
    1,
    figsize=(8, 4 * nplots),
    sharex=True
)

fig.suptitle(f'{parameter} fits: with/without MIRI data', fontsize=16)

if nplots == 1:
    axes = [axes]

# Give the panels a little more breathing room
fig.subplots_adjust(
    hspace=0.28,
    top=0.93,
    bottom=0.05,
    left=0.10,
    right=0.98
)


# ==========================================
# FIND COMMON X RANGE
# ==========================================

all_samples = []

for d in data:
    all_samples.extend(d["miri_samples"])

    if d["nomiri_samples"] is not None:
        all_samples.extend(d["nomiri_samples"])

xmin = min(all_samples)
xmax = max(all_samples)

bins = np.linspace(xmin, xmax, nbins)


# ==========================================
# PLOT EACH BIN
# ==========================================

for ax, d in zip(axes, data):

    samples_miri = d["miri_samples"]
    weights_miri = d["miri_weights"]

    samples_nomiri = d["nomiri_samples"]
    weights_nomiri = d["nomiri_weights"]

    # Weighted medians
    median_miri = weighted_median(
        samples_miri,
        weights_miri
    )

    # --------------------------------------
    # With MIRI
    # --------------------------------------

    ax.hist(
        samples_miri,
        bins=bins,
        weights=weights_miri,
        density=True,
        histtype="stepfilled",
        color="lightblue",
        alpha=0.5,
        edgecolor="steelblue",
        linewidth=1.5,
        label=label1
    )

    ax.axvline(
        median_miri,
        color="steelblue",
        linestyle="--",
        linewidth=2,
        label=f"{label1} median = {median_miri:.2f}"
    )

    # --------------------------------------
    # Without MIRI (ONLY if available)
    # --------------------------------------

    if samples_nomiri is not None:

        median_nomiri = weighted_median(
            samples_nomiri,
            weights_nomiri
        )

        ax.hist(
            samples_nomiri,
            bins=bins,
            weights=weights_nomiri,
            density=True,
            histtype="stepfilled",
            color="lightpink",
            alpha=0.5,
            edgecolor="hotpink",
            linewidth=1.5,
            label=label2
        )

        ax.axvline(
            median_nomiri,
            color="hotpink",
            linestyle="--",
            linewidth=2,
            label=f"{label2} median = {median_nomiri:.2f}"
        )

        print(f"\n{d['label']}")
        print(f"{label1} median:    {median_miri:.3f}")
        print(f"{label2} median: {median_nomiri:.3f}")
        print(f"Difference:       {median_miri - median_nomiri:.3f}")

    else:

        print(f"\n{d['label']}")
        print(f"{label1} median:    {median_miri:.3f}")
        print(f"{label2}: not available")

    ax.set_ylabel("Posterior density")
    ax.set_title(d["label"])
    ax.legend()

    # Print values
    print(f"\n{d['label']}")
    print(f"{label1} median:    {median_miri:.3f}")
    print(f"{label2} median: {median_nomiri:.3f}")
    print(f"Difference:       {median_miri - median_nomiri:.3f}")


# ==========================================
# FINAL FORMATTING
# ==========================================

axes[-1].set_xlabel(parameter)

plt.savefig(
    f"Prospector/Plots/{parameter}_histograms.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()