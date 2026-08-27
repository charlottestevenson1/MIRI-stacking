import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as reader
import os


# ============================================================
# PARAMETERS TO PLOT
# ============================================================

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


# ============================================================
# SETTINGS
# ============================================================

zlo = 9
zup = 10

selected_zbin = f"z = {zlo}-{zup}"

# Parameter used to select chain entries
cut_param = "logmass"

# Keep entries satisfying:
# cut_param > cut_value
cut_value = 8.3

gt_or_lt = '<'

MIRI_desc = 'with MIRI'

nbins = 50

original_colour = "white"
selected_colour = "steelblue"

main_dir = "Prospector/Plots/Bimodality Checks"

# CHANGE THIS
sub_dir = f"z={zlo}-{zup}, {cut_param} {gt_or_lt} {cut_value}"

output_dir = f"{main_dir}/{sub_dir}"

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# REDSHIFT BINS
# ============================================================

redshift_bins = {
    f"z = 8-9": f"Prospector/Fits/z=8-9 {MIRI_desc}.h5",
    f"z = 9-10": f"Prospector/Fits/z=9-10 {MIRI_desc}.h5",
    f"z = 10-11": f"Prospector/Fits/z=10-11 {MIRI_desc}.h5",
    f"z = 11-12": f"Prospector/Fits/z=11-12 {MIRI_desc}.h5",
    f"z = 12-15": f"Prospector/Fits/z=12-15 {MIRI_desc}.h5",
}


# ============================================================
# FUNCTIONS
# ============================================================

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


def get_percentiles(samples, weights):

    return (
        weighted_percentile(samples, weights, 16),
        weighted_percentile(samples, weights, 50),
        weighted_percentile(samples, weights, 84)
    )


# ============================================================
# LOAD CHAIN
# ============================================================

filename = redshift_bins[selected_zbin]

print(f"loading file: {filename}")

result, _, _ = reader.results_from(
    filename,
    dangerous=False
)

theta_labels = list(result["theta_labels"])
chain = np.asarray(result["chain"])

weights = np.asarray(
    result["weights"],
    dtype=float
)

# Normalise the full posterior weights
weights /= weights.sum()


# ============================================================
# CREATE SELECTION
# ============================================================

if cut_param not in theta_labels:
    raise ValueError(
        f"'{cut_param}' is not in theta_labels."
        f"\nAvailable parameters are:\n{theta_labels}"
    )

cut_index = theta_labels.index(cut_param)
cut_values = chain[:, cut_index]

# CHANGE?
mass_mask = (cut_values > cut_value) if gt_or_lt==">" else (cut_values < cut_value)

samples_selected_count = mass_mask.sum()

if samples_selected_count == 0:
    raise ValueError(
        f"No chain entries satisfy "
        f"{cut_param} {gt_or_lt} {cut_value}"
    )

# IMPORTANT:
# Keep the original weights.
# Do NOT renormalise the selected weights.
weights_selected = weights[mass_mask]


# ============================================================
# SELECTION INFORMATION
# ============================================================

print(f"\nRedshift bin: {selected_zbin}")
print(f"Selection: {cut_param} {gt_or_lt} {cut_value}")
print(f"Total chain entries: {len(chain)}")
print(f"Selected chain entries: {samples_selected_count}")
print(
    f"Fraction of entries selected: "
    f"{mass_mask.mean():.3f}"
)
print(
    f"Posterior weight selected: "
    f"{weights_selected.sum():.3f}"
)


# ============================================================
# LOOP THROUGH PARAMETERS
# ============================================================

for parameter in params:

    print(f"\n{'=' * 50}")
    print(parameter)
    print("=" * 50)


    # --------------------------------------------------------
    # Extract parameter
    # --------------------------------------------------------

    if parameter not in theta_labels:
        print(
            f"Skipping {parameter}: "
            f"not found in theta_labels"
        )
        continue

    parameter_index = theta_labels.index(parameter)

    samples = chain[:, parameter_index]

    # Apply same selection mask
    samples_selected = samples[mass_mask]


    # --------------------------------------------------------
    # Statistics: original
    # --------------------------------------------------------

    p16, p50, p84 = get_percentiles(
        samples,
        weights
    )


    # --------------------------------------------------------
    # Statistics: selected
    # --------------------------------------------------------

    p16_selected, p50_selected, p84_selected = (
        get_percentiles(
            samples_selected,
            weights_selected
        )
    )


    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print(
        f"Original: "
        f"{p50:.3f} "
        f"+{p84-p50:.3f} "
        f"-{p50-p16:.3f}"
    )

    print(
        f"Selected: "
        f"{p50_selected:.3f} "
        f"+{p84_selected-p50_selected:.3f} "
        f"-{p50_selected-p16_selected:.3f}"
    )

    print(
        f"Median shift: "
        f"{p50_selected-p50:+.3f}"
    )


    # --------------------------------------------------------
    # Histogram range
    # --------------------------------------------------------

    xmin = weighted_percentile(
        samples,
        weights,
        5
    )

    xmax = weighted_percentile(
        samples,
        weights,
        95
    )

    padding = 0.05 * (xmax - xmin)

    xmin -= padding
    xmax += padding

    bins = np.linspace(
        xmin,
        xmax,
        nbins + 1
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6),
        facecolor='black'
    )

    ax.set_facecolor("black")


    # Original posterior
    ax.hist(
        samples,
        bins=bins,
        weights=weights,
        histtype="step",
        color=original_colour,
        linewidth=2,
        label="Original"
    )


    # Selected posterior
    ax.hist(
        samples_selected,
        bins=bins,
        weights=weights_selected,
        histtype="stepfilled",
        color=selected_colour,
        alpha=0.5,
        edgecolor=selected_colour,
        linewidth=1.5,
        label=fr"${cut_param} {gt_or_lt} {cut_value}$"
    )


    # --------------------------------------------------------
    # Medians
    # --------------------------------------------------------

    ax.axvline(
        p50,
        color=original_colour,
        linestyle="--",
        linewidth=2
    )

    ax.axvline(
        p50_selected,
        color=selected_colour,
        linestyle="--",
        linewidth=2
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlabel(parameter, color='white')
    ax.set_ylabel("Posterior weight", color='white')

    # ax.set_title(
    #     f"{parameter} — {selected_zbin} ({MIRI_desc})\n"
    #     f"Selection: {cut_param} {gt_or_lt} {cut_value}",
    #     color='white'
    # )

    ax.tick_params(
        axis="both",
        colors="white",
        labelsize=14,
        width=1.5,
        length=6
    )

    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.5)

    ax.grid(
        axis="y",
        color="white",
        alpha=0.12
    )

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.legend()

    plt.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        f"{output_dir}/"
        f"{parameter}_"
        f"{cut_param}_lt_{cut_value}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    #plt.show()
    plt.close()