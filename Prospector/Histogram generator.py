import numpy as np
import matplotlib.pyplot as plt
import prospect.io.read_results as reader
import pickle

from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    TextArea,
    VPacker,
    HPacker
)


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


# ==========================================
# SETTINGS
# ==========================================

for parameter in params:

    redshift_bins = [
        {
            "label": "z = 8–9",
            "with_miri": "Prospector/Fits/z=8-9 with MIRI.h5",
            "without_miri": "Prospector/Fits/z=8-9 no MIRI.h5",
        },
        {
            "label": "z = 9–10",
            "with_miri": "Prospector/Fits/z=9-10 with MIRI.h5",
            "without_miri": "Prospector/Fits/z=9-10 no MIRI.h5",
        },
        {
            "label": "z = 10–11",
            "with_miri": "Prospector/Fits/z=10-11 with MIRI.h5",
            "without_miri": "Prospector/Fits/z=10-11 no MIRI.h5",
        },
        {
            "label": "z = 11–12",
            "with_miri": "Prospector/Fits/z=11-12 with MIRI.h5",
            "without_miri": "Prospector/Fits/z=11-12 no MIRI.h5",
        },
        {
            "label": "z = 12–15",
            "with_miri": "Prospector/Fits/z=12-15 with MIRI.h5",
            "without_miri": "Prospector/Fits/z=12-15 no MIRI.h5",
        },
    ]

    nbins = 50

    # Plot colours
    miri_fill = "lightblue"
    miri_edge = "steelblue"

    nomiri_fill = "lightpink"
    nomiri_edge = "hotpink"


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

        samples = np.asarray(result["chain"])[:, index]

        weights = np.asarray(
            result["weights"],
            dtype=float
        )

        weights /= weights.sum()

        return samples, weights


    # ==========================================
    # LOAD ALL BINS
    # ==========================================

    data = []
    priors = []

    for zbin in redshift_bins:

        # ------------------------------
        # With MIRI
        # ------------------------------

        samples_miri, weights_miri = load_parameter(
            zbin["with_miri"],
            parameter
        )

        # ------------------------------
        # Without MIRI (optional)
        # ------------------------------

        try:

            samples_nomiri, weights_nomiri = load_parameter(
                zbin["without_miri"],
                parameter
            )

        except (FileNotFoundError, OSError):

            print(
                f"No MIRI file not found for {zbin['label']}"
            )

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
        # PRIOR
        # ==========================================

        readfile = zbin['with_miri']
        result, _, _ = reader.results_from(readfile, dangerous=False)
        model_params = result['model_params']

        entry = next(
        (p for p in result["model_params"] if p["name"] == parameter), 
        model_params[9]
        )

        try:
            prior = pickle.loads(entry['prior'])
            priors.append(prior)

        except:
            print(entry['prior'])


    # ==========================================
    # CREATE FIGURE
    # ==========================================

    nplots = len(data)

    fig, axes = plt.subplots(
        nplots,
        1,
        figsize=(10, 4 * nplots),
        sharex=True
    )

    fig.suptitle(
        f"{parameter} fits: with/without MIRI data",
        fontsize=16
    )

    if nplots == 1:
        axes = [axes]


    # Leave room on the right for statistics boxes

    fig.subplots_adjust(
        hspace=0.28,
        top=0.93,
        bottom=0.05,
        left=0.10,
        right=0.78
    )


    # ==========================================
    # FIND COMMON X RANGE
    # ==========================================

    # First determine the common 5th–95th percentile
    # plotting range.

    lower_limits = []
    upper_limits = []

    for d, prior in zip(data, priors):

        # With MIRI

        lower_limits.append(
            weighted_percentile(
                d["miri_samples"],
                d["miri_weights"],
                5
            )
        )

        upper_limits.append(
            weighted_percentile(
                d["miri_samples"],
                d["miri_weights"],
                95
            )
        )

        # Without MIRI, if available

        if d["nomiri_samples"] is not None:

            lower_limits.append(
                weighted_percentile(
                    d["nomiri_samples"],
                    d["nomiri_weights"],
                    5
                )
            )

            upper_limits.append(
                weighted_percentile(
                    d["nomiri_samples"],
                    d["nomiri_weights"],
                    95
                )
            )


    xmin = min(lower_limits)
    xmax = max(upper_limits)

    bins = np.linspace(
        xmin,
        xmax,
        nbins + 1
    )

    # Small padding

    padding = 0.05 * (xmax - xmin)

    xmin -= padding
    xmax += padding

    # ==========================================
    # HISTOGRAM BINS
    # ==========================================

    # Use the full sample ranges to construct the
    # histogram, while only displaying the common
    # 5–95% region.

    all_samples = []

    for d in data:

        all_samples.extend(
            d["miri_samples"]
        )

        if d["nomiri_samples"] is not None:

            all_samples.extend(
                d["nomiri_samples"]
            )

    full_min = min(all_samples)
    full_max = max(all_samples)


    # ==========================================
    # PLOT EACH BIN
    # ==========================================

    for ax, d, prior in zip(axes, data, priors):

        samples_miri = d["miri_samples"]
        weights_miri = d["miri_weights"]

        samples_nomiri = d["nomiri_samples"]
        weights_nomiri = d["nomiri_weights"]

        # --------------------------------------
        # PRIOR VALUES
        # --------------------------------------

        x = np.linspace(xmin, xmax, 50)

        if not isinstance(prior(0), np.ndarray):
            prior_pdf = np.exp(prior(x))
        else:
            values = [prior(x)[i][0] for i in range(len(x))]
            prior_pdf = np.exp(values)

        # --------------------------------------
        # WITH MIRI POSTERIOR
        # --------------------------------------

        p16_miri = weighted_percentile(
            samples_miri,
            weights_miri,
            16
        )

        p50_miri = weighted_percentile(
            samples_miri,
            weights_miri,
            50
        )

        p84_miri = weighted_percentile(
            samples_miri,
            weights_miri,
            84
        )

        width_miri = p84_miri - p16_miri


        ax.hist(
            samples_miri,
            bins=bins,
            weights=weights_miri,
            density=True,
            histtype="stepfilled",
            color=miri_fill,
            alpha=0.5,
            edgecolor=miri_edge,
            linewidth=1.5
        )


        ax.axvline(
            p50_miri,
            color=miri_edge,
            linestyle="--",
            linewidth=2
        )


        # --------------------------------------
        # WITHOUT MIRI POSTERIOR
        # --------------------------------------

        if samples_nomiri is not None:

            p16_nomiri = weighted_percentile(
                samples_nomiri,
                weights_nomiri,
                16
            )

            p50_nomiri = weighted_percentile(
                samples_nomiri,
                weights_nomiri,
                50
            )

            p84_nomiri = weighted_percentile(
                samples_nomiri,
                weights_nomiri,
                84
            )

            width_nomiri = p84_nomiri - p16_nomiri

            median_shift = p50_miri - p50_nomiri

            width_ratio = (
                width_nomiri / width_miri
            )


            ax.hist(
                samples_nomiri,
                bins=bins,
                weights=weights_nomiri,
                density=True,
                histtype="stepfilled",
                color=nomiri_fill,
                alpha=0.5,
                edgecolor=nomiri_edge,
                linewidth=1.5
            )


            ax.axvline(
                p50_nomiri,
                color=nomiri_edge,
                linestyle="--",
                linewidth=2
            )
        
        # --------------------------------------
        # PRIOR PLOTTING
        # --------------------------------------
        ax.step(
            x,
            prior_pdf,
            where = 'mid',
            color = 'gray',
            label = 'Prior',
            linestyle = '--',
            linewidth = 1.5,
            alpha = 1,
            zorder = 1
        )

        # --------------------------------------
        # AXIS LIMITS
        # --------------------------------------

        ax.set_xlim(
            xmin,
            xmax
        )


        # --------------------------------------
        # STATISTICS BOX
        # --------------------------------------

        if samples_nomiri is not None:

            # With MIRI line
            miri_label = TextArea(
                "With MIRI: ",
                textprops={
                    "color": miri_edge,
                    "fontsize": 10,
                    "fontweight": "bold"
                }
            )

            miri_value = TextArea(
                f"{p50_miri:.2f}"
                f"$^{{+{p84_miri-p50_miri:.2f}}}"
                f"_{{-{p50_miri-p16_miri:.2f}}}$",
                textprops={
                    "color": miri_edge,
                    "fontsize": 10
                }
            )

            miri_line = HPacker(
                children=[miri_label, miri_value],
                align="center",
                pad=0,
                sep=2
            )


            # No MIRI line
            nomiri_label = TextArea(
                "No MIRI: ",
                textprops={
                    "color": nomiri_edge,
                    "fontsize": 10,
                    "fontweight": "bold"
                }
            )

            nomiri_value = TextArea(
                f"{p50_nomiri:.2f}"
                f"$^{{+{p84_nomiri-p50_nomiri:.2f}}}"
                f"_{{-{p50_nomiri-p16_nomiri:.2f}}}$",
                textprops={
                    "color": nomiri_edge,
                    "fontsize": 10
                }
            )

            nomiri_line = HPacker(
                children=[nomiri_label, nomiri_value],
                align="center",
                pad=0,
                sep=2
            )


            # Median shift
            shift_label = TextArea(
                "Median shift: ",
                textprops={
                    "color": "black",
                    "fontsize": 10,
                    "fontweight": "bold"
                }
            )

            shift_value = TextArea(
                f"{median_shift:+.2f}",
                textprops={
                    "color": "black",
                    "fontsize": 10
                }
            )

            shift_line = HPacker(
                children=[shift_label, shift_value],
                align="center",
                pad=0,
                sep=2
            )


            # Width ratio
            width_label = TextArea(
                "Width ratio: ",
                textprops={
                    "color": "black",
                    "fontsize": 10,
                    "fontweight": "bold"
                }
            )

            width_value = TextArea(
                f"{width_ratio:.2f}$\\times$",
                textprops={
                    "color": "black",
                    "fontsize": 10
                }
            )

            width_line = HPacker(
                children=[width_label, width_value],
                align="center",
                pad=0,
                sep=2
            )


            # Stack all four lines vertically
            packed_text = VPacker(
                children=[
                    miri_line,
                    nomiri_line,
                    shift_line,
                    width_line
                ],
                align="left",
                pad=0,
                sep=7
            )


            # Single rounded box
            stats_box = AnchoredOffsetbox(
                loc="center left",
                child=packed_text,
                pad=0.6,
                frameon=True,
                bbox_to_anchor=(1.01, 0.5),
                bbox_transform=ax.transAxes,
                borderpad=0.5
            )

            stats_box.patch.set_boxstyle("round,pad=0.5")
            stats_box.patch.set_facecolor("whitesmoke")
            stats_box.patch.set_edgecolor("0.7")
            stats_box.patch.set_alpha(0.8)

            ax.add_artist(stats_box)


        else:

            # Only With MIRI available
            miri_label = TextArea(
                "With MIRI: ",
                textprops={
                    "color": miri_edge,
                    "fontsize": 10,
                    "fontweight": "bold"
                }
            )

            miri_value = TextArea(
                f"{p50_miri:.2f}"
                f"$^{{+{p84_miri-p50_miri:.2f}}}"
                f"_{{-{p50_miri-p16_miri:.2f}}}$",
                textprops={
                    "color": miri_edge,
                    "fontsize": 10
                }
            )

            miri_line = HPacker(
                children=[miri_label, miri_value],
                align="center",
                pad=0,
                sep=2
            )

            stats_box = AnchoredOffsetbox(
                loc="center left",
                child=miri_line,
                pad=0.6,
                frameon=True,
                bbox_to_anchor=(1.01, 0.5),
                bbox_transform=ax.transAxes,
                borderpad=0.5
            )

            stats_box.patch.set_boxstyle("round,pad=0.5")
            stats_box.patch.set_facecolor("whitesmoke")
            stats_box.patch.set_edgecolor("0.7")
            stats_box.patch.set_alpha(0.8)

            ax.add_artist(stats_box)


            print(f"\n{d['label']}")

            print(
                f"With MIRI: "
                f"{p50_miri:.3f} "
                f"+{p84_miri-p50_miri:.3f} "
                f"-{p50_miri-p16_miri:.3f}"
            )

            print(
                "Without MIRI: not available"
            )


        # --------------------------------------
        # Labels
        # --------------------------------------

        ax.set_ylabel(
            "Posterior density"
        )

        ax.set_title(
            d["label"]
        )


    # ==========================================
    # FINAL FORMATTING
    # ==========================================

    axes[-1].set_xlabel(
        parameter
    )


    plt.savefig(
        f"Prospector/Plots/Histograms/"
        f"{parameter}_histograms.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()