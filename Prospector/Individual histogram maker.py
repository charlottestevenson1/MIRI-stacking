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

params = ["logmass"]

# ==========================================
# SETTINGS
# ==========================================

for parameter in params:

    redshift_bins = [
        # {
        #     "label": "z = 8–9",
        #     "with_miri": "Prospector/Fits/z=8-9 with MIRI.h5",
        #     "without_miri": "Prospector/Fits/z=8-9 no MIRI.h5",
        # },
        {
            "label": "z = 9–10",
            "with_miri": "Prospector/Fits/z=9-10 with MIRI.h5",
            "without_miri": "Prospector/Fits/z=9-10 no MIRI.h5",
        },
        # {
        #     "label": "z = 10–11",
        #     "with_miri": "Prospector/Fits/z=10-11 with MIRI.h5",
        #     "without_miri": "Prospector/Fits/z=10-11 no MIRI.h5",
        # },
        # {
        #     "label": "z = 11–12",
        #     "with_miri": "Prospector/Fits/z=11-12 with MIRI.h5",
        #     "without_miri": "Prospector/Fits/z=11-12 no MIRI.h5",
        # },
        # {
        #     "label": "z = 12–15",
        #     "with_miri": "Prospector/Fits/z=12-15 with MIRI.h5",
        #     "without_miri": "Prospector/Fits/z=12-15 no MIRI.h5",
        # },
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

        index = list(
            result["theta_labels"]
        ).index(parameter)

        samples = np.asarray(
            result["chain"]
        )[:, index]

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
        # No MIRI
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

        readfile = zbin["with_miri"]

        result, _, _ = reader.results_from(
            readfile,
            dangerous=False
        )

        model_params = result["model_params"]

        entry = next(
            (
                p for p in result["model_params"]
                if p["name"] == parameter
            ),
            model_params[9]
        )

        try:
            prior = pickle.loads(
                entry["prior"]
            )

            priors.append(prior)

        except:
            print(entry["prior"])

    # ==========================================
    # FIND COMMON X RANGE
    # ==========================================

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

        # No MIRI, if available
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

    padding = 0.05 * (xmax - xmin)

    xmin -= padding
    xmax += padding

    bins = np.linspace(
        xmin,
        xmax,
        nbins + 1
    )

    # ==========================================
    # PLOT EACH VERSION
    # ==========================================

    for show_miri in [False, True]:

        # --------------------------------------
        # CREATE FIGURE
        # --------------------------------------

        nplots = len(data)

        fig, axes = plt.subplots(
            nplots,
            1,
            figsize=(10, 4 * nplots),
            sharex=True,
            facecolor="none"
        )

        if nplots == 1:
            axes = [axes]

        for ax in axes:
            ax.set_facecolor("none")

        fig.subplots_adjust(
            hspace=0.28,
            top=0.93,
            bottom=0.07,
            left=0.10,
            right=0.78
        )

        # --------------------------------------
        # PLOT EACH BIN
        # --------------------------------------

        for ax, d, prior in zip(
            axes,
            data,
            priors
        ):

            samples_miri = d["miri_samples"]
            weights_miri = d["miri_weights"]

            samples_nomiri = d["nomiri_samples"]
            weights_nomiri = d["nomiri_weights"]

            # --------------------------------------
            # PRIOR
            # --------------------------------------

            x = np.linspace(
                xmin,
                xmax,
                50
            )

            if not isinstance(
                prior(0),
                np.ndarray
            ):
                prior_pdf = np.exp(
                    prior(x)
                )
            else:
                values = [
                    prior(x)[i][0]
                    for i in range(len(x))
                ]

                prior_pdf = np.exp(
                    values
                )

            # --------------------------------------
            # WITH MIRI
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

            width_miri = (
                p84_miri - p16_miri
            )

            # --------------------------------------
            # NO MIRI
            # --------------------------------------

            p16_nomiri = None
            p50_nomiri = None
            p84_nomiri = None
            width_nomiri = None
            median_shift = None
            width_ratio = None

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

                width_nomiri = (
                    p84_nomiri
                    - p16_nomiri
                )

                median_shift = (
                    p50_miri
                    - p50_nomiri
                )

                width_ratio = (
                    width_nomiri
                    / width_miri
                )

            # --------------------------------------
            # PLOT NO MIRI
            # --------------------------------------

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

            if samples_nomiri is not None:

                ax.axvline(
                    p50_nomiri,
                    color=nomiri_edge,
                    linestyle="--",
                    linewidth=2
                )

            # --------------------------------------
            # PLOT WITH MIRI
            # --------------------------------------

            if show_miri:

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
            # PRIOR
            # --------------------------------------

            ax.step(
                x,
                prior_pdf,
                where="mid",
                color="white",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                zorder=1
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

            if show_miri and samples_nomiri is not None:

                # With MIRI
                miri_label = TextArea(
                    "With MIRI: ",
                    textprops={
                        "color": miri_edge,
                        "fontsize": 12,
                        "fontweight": "bold"
                    }
                )

                miri_value = TextArea(
                    f"{p50_miri:.2f}"
                    f"$^{{+{p84_miri-p50_miri:.2f}}}"
                    f"_{{-{p50_miri-p16_miri:.2f}}}$",
                    textprops={
                        "color": miri_edge,
                        "fontsize": 12
                    }
                )

                miri_line = HPacker(
                    children=[
                        miri_label,
                        miri_value
                    ],
                    align="center",
                    pad=0,
                    sep=2
                )

                # No MIRI
                nomiri_label = TextArea(
                    "No MIRI: ",
                    textprops={
                        "color": nomiri_edge,
                        "fontsize": 12,
                        "fontweight": "bold"
                    }
                )

                nomiri_value = TextArea(
                    f"{p50_nomiri:.2f}"
                    f"$^{{+{p84_nomiri-p50_nomiri:.2f}}}"
                    f"_{{-{p50_nomiri-p16_nomiri:.2f}}}$",
                    textprops={
                        "color": nomiri_edge,
                        "fontsize": 12
                    }
                )

                nomiri_line = HPacker(
                    children=[
                        nomiri_label,
                        nomiri_value
                    ],
                    align="center",
                    pad=0,
                    sep=2
                )

                # Median shift
                shift_label = TextArea(
                    "Median shift: ",
                    textprops={
                        "color": "black",
                        "fontsize": 12,
                        "fontweight": "bold"
                    }
                )

                shift_value = TextArea(
                    f"{median_shift:+.2f}",
                    textprops={
                        "color": "black",
                        "fontsize": 12
                    }
                )

                shift_line = HPacker(
                    children=[
                        shift_label,
                        shift_value
                    ],
                    align="center",
                    pad=0,
                    sep=2
                )

                # Width ratio
                width_label = TextArea(
                    "Width ratio: ",
                    textprops={
                        "color": "black",
                        "fontsize": 12,
                        "fontweight": "bold"
                    }
                )

                width_value = TextArea(
                    f"{width_ratio:.2f}$\\times$",
                    textprops={
                        "color": "black",
                        "fontsize": 12
                    }
                )

                width_line = HPacker(
                    children=[
                        width_label,
                        width_value
                    ],
                    align="center",
                    pad=0,
                    sep=2
                )

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

            elif show_miri:
                # Only With MIRI available
                miri_label = TextArea(
                    "With MIRI: ",
                    textprops={
                        "color": miri_edge,
                        "fontsize": 12,
                        "fontweight": "bold"
                    }
                )

                miri_value = TextArea(
                    f"{p50_miri:.2f}"
                    f"$^{{+{p84_miri-p50_miri:.2f}}}"
                    f"_{{-{p50_miri-p16_miri:.2f}}}$",
                    textprops={
                        "color": miri_edge,
                        "fontsize": 12
                    }
                )

                packed_text = HPacker(
                    children=[
                        miri_label,
                        miri_value
                    ],
                    align="center",
                    pad=0,
                    sep=2
                )

            else:
                # No MIRI only
                nomiri_label = TextArea(
                    "No MIRI: ",
                    textprops={
                        "color": nomiri_edge,
                        "fontsize": 12,
                        "fontweight": "bold"
                    }
                )

                nomiri_value = TextArea(
                    f"{p50_nomiri:.2f}"
                    f"$^{{+{p84_nomiri-p50_nomiri:.2f}}}"
                    f"_{{-{p50_nomiri-p16_nomiri:.2f}}}$",
                    textprops={
                        "color": nomiri_edge,
                        "fontsize": 12
                    }
                )

                packed_text = HPacker(
                    children=[
                        nomiri_label,
                        nomiri_value
                    ],
                    align="center",
                    pad=0,
                    sep=2
                )

            stats_box = AnchoredOffsetbox(
                loc="center left",
                child=packed_text,
                pad=0.6,
                frameon=True,
                bbox_to_anchor=(1.01, 0.5),
                bbox_transform=ax.transAxes,
                borderpad=0.5
            )

            stats_box.patch.set_boxstyle(
                "round,pad=0.5"
            )

            stats_box.patch.set_facecolor(
                "#fff5f7"
            )

            stats_box.patch.set_edgecolor(
                "0.7"
            )

            stats_box.patch.set_alpha(
                1.0
            )

            ax.add_artist(
                stats_box
            )

            # --------------------------------------
            # AXIS FORMATTING
            # --------------------------------------

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

            ax.set_ylabel(
                "Posterior density",
                color="white",
                fontsize=16
            )

            ax.grid(
                axis="y",
                color="white",
                alpha=0.12
            )

            # Redshift-bin label
            ax.text(
                0.02,
                0.98,
                d["label"],
                transform=ax.transAxes,
                color="white",
                fontsize=15,
                fontweight="bold",
                ha="left",
                va="top"
            )

        # ==========================================
        # FINAL FORMATTING
        # ==========================================

        axes[-1].set_xlabel(
            parameter,
            color="white",
            fontsize=18,
            labelpad=10
        )

        axes[-1].tick_params(
            axis="x",
            colors="white",
            labelsize=14
        )

        # ==========================================
        # SAVE
        # ==========================================

        if show_miri:
            suffix = "both"
        else:
            suffix = "no MIRI"

        output_file = (
            f"Prospector/Plots/Individual Histograms/"
            f"{parameter}_histograms_z=9-10_{suffix}.png"
        )

        plt.savefig(
            output_file,
            dpi=400,
            bbox_inches="tight",
            transparent=True
        )

        plt.show()
        plt.close()