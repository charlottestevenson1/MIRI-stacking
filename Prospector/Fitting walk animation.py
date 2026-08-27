import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from multiprocessing import Pool
import time

from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.gridspec import GridSpec

import prospect.io.read_results as reader


results_type = "dynesty"

experiment = True
generate_3D = False
generate_1D = False

# ============================================================
# LOAD RESULTS
# ============================================================

readfile = "Prospector/Fits/z=8-9 with MIRI.h5"

result, obs, model = reader.results_from(
    readfile,
    dangerous=False
)

labels = list(result["theta_labels"])

pnames = [
    "logmass",
    "zred",
    "logzsol"
]

idx = [
    labels.index(p)
    for p in pnames
]


# ============================================================
# EXPERIMENT: 3D + TOP TRACE PLOTS
# ============================================================

if experiment:

    chain = np.asarray(
        result["chain"]
    )

    logl = np.asarray(
        result["lnlikelihood"]
    )

    logvol = np.asarray(
        result["logvol"]
    )

    # Reduce number of frames if needed
    stride = 10

    chain = chain[::stride]
    logl = logl[::stride]
    logvol = logvol[::stride]

    xyz = chain[:, idx]
    nframes = len(xyz)

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(14, 10),
        facecolor="black"
    )

    gs = GridSpec(
        2,
        3,
        height_ratios=[1, 3],
        figure=fig,
        hspace=0.3
    )

    # --------------------------------------------------------
    # TOP PLOTS
    # --------------------------------------------------------

    ax1 = fig.add_subplot(
        gs[0, 0]
    )

    ax2 = fig.add_subplot(
        gs[0, 1]
    )

    ax3 = fig.add_subplot(
        gs[0, 2]
    )

    top_axes = [
        ax1,
        ax2,
        ax3
    ]

    for ax, name in zip(
        top_axes,
        pnames
    ):

        ax.set_facecolor("black")

        ax.set_xlabel(
            "Iteration",
            color="white",
            fontsize=14
        )

        ax.set_ylabel(
            name,
            color="white",
            fontsize=14
        )

        ax.set_title(
            name,
            color="white",
            fontsize=15
        )

        ax.tick_params(
            axis="both",
            colors="white",
            labelsize=14,
            width=1.5,
            length=5
        )

        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(1.2)

        ax.grid(
            color="white",
            alpha=0.15
        )

    # --------------------------------------------------------
    # PRE-SET TOP AXIS LIMITS
    # --------------------------------------------------------

    for k, ax in enumerate(top_axes):

        y = xyz[:, k]

        pad = (
            0.05
            * (y.max() - y.min() + 1e-12)
        )

        ax.set_xlim(
            0,
            nframes
        )

        ax.set_ylim(
            y.min() - pad,
            y.max() + pad
        )

    # --------------------------------------------------------
    # TRACE LINES
    # --------------------------------------------------------

    lines = []
    points = []

    for ax in top_axes:

        line, = ax.plot(
            [],
            [],
            lw=2,
            color="hotpink"
        )

        point, = ax.plot(
            [],
            [],
            "o",
            ms=7,
            color="white",
            markeredgecolor="hotpink",
            markeredgewidth=2
        )

        lines.append(line)
        points.append(point)

    # --------------------------------------------------------
    # 3D PLOT
    # --------------------------------------------------------

    ax3d = fig.add_subplot(
        gs[1, :],
        projection="3d"
    )

    ax3d.set_facecolor("black")

    ax3d.set_xlabel(
        pnames[0],
        color="white",
        fontsize=14
    )

    ax3d.set_ylabel(
        pnames[1],
        color="white",
        fontsize=14
    )

    ax3d.set_zlabel(
        pnames[2],
        color="white",
        fontsize=14
    )

    ax3d.tick_params(
        axis="x",
        colors="white",
        labelsize=12
    )

    ax3d.tick_params(
        axis="y",
        colors="white",
        labelsize=12
    )

    ax3d.tick_params(
        axis="z",
        colors="white",
        labelsize=12
    )

    ax3d.xaxis.pane.set_facecolor("black")
    ax3d.yaxis.pane.set_facecolor("black")
    ax3d.zaxis.pane.set_facecolor("black")

    ax3d.xaxis.pane.set_edgecolor("white")
    ax3d.yaxis.pane.set_edgecolor("white")
    ax3d.zaxis.pane.set_edgecolor("white")

    for axis in [
        ax3d.xaxis,
        ax3d.yaxis,
        ax3d.zaxis
    ]:
        axis._axinfo["grid"]["color"] = (
            1,
            1,
            1,
            0.15
        )

    # --------------------------------------------------------
    # INITIAL LIMITS
    # --------------------------------------------------------

    mins = xyz.min(
        axis=0
    )

    maxs = xyz.max(
        axis=0
    )

    pad = (
        0.1
        * (maxs - mins + 1e-12)
    )

    ax3d.set_xlim(
        mins[0] - pad[0],
        maxs[0] + pad[0]
    )

    ax3d.set_ylim(
        mins[1] - pad[1],
        maxs[1] + pad[1]
    )

    ax3d.set_zlim(
        mins[2] - pad[2],
        maxs[2] + pad[2]
    )

    # --------------------------------------------------------
    # DYNAMIC ZOOM SETTINGS
    # --------------------------------------------------------

    current_limits = np.array([
        ax3d.get_xlim(),
        ax3d.get_ylim(),
        ax3d.get_zlim()
    ], dtype=float)

    smooth = 0.03
    zoom_window = 500

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(i):

        # ----------------------------------------------------
        # TOP TRACE PLOTS
        # ----------------------------------------------------

        for k in range(3):

            y = xyz[
                :i + 1,
                k
            ]

            lines[k].set_data(
                np.arange(i + 1),
                y
            )

            points[k].set_data(
                [i],
                [xyz[i, k]]
            )

        # ----------------------------------------------------
        # RECENT 3D POINTS
        # ----------------------------------------------------

        start = max(
            0,
            i - zoom_window
        )

        recent = xyz[
            start:i + 1
        ]

        # Remove previous points
        for collection in ax3d.collections[:]:
            collection.remove()

        if len(recent) > 0:

            alpha = np.linspace(
                0.05,
                0.7,
                len(recent)
            )

            for j, p in enumerate(recent):

                ax3d.scatter(
                    p[0],
                    p[1],
                    p[2],
                    color="deeppink",
                    alpha=alpha[j],
                    s=14
                )

        # ----------------------------------------------------
        # CURRENT POINT
        # ----------------------------------------------------

        p = xyz[i]

        ax3d.scatter(
            p[0],
            p[1],
            p[2],
            color="white",
            edgecolor="hotpink",
            linewidth=2,
            s=65
        )

        # ----------------------------------------------------
        # SMOOTH ZOOM
        # ----------------------------------------------------

        if len(recent) > 20:

            centre = np.median(
                recent,
                axis=0
            )

            low = np.percentile(
                recent,
                5,
                axis=0
            )

            high = np.percentile(
                recent,
                95,
                axis=0
            )

            width = (
                high - low
            ) * 1.4

            width = np.maximum(
                width,
                1e-6
            )

            target = np.array([
                [
                    centre[0] - width[0] / 2,
                    centre[0] + width[0] / 2
                ],
                [
                    centre[1] - width[1] / 2,
                    centre[1] + width[1] / 2
                ],
                [
                    centre[2] - width[2] / 2,
                    centre[2] + width[2] / 2
                ]
            ])

            current_limits[:] = (
                (1 - smooth)
                * current_limits
                + smooth
                * target
            )

            ax3d.set_xlim(
                current_limits[0]
            )

            ax3d.set_ylim(
                current_limits[1]
            )

            ax3d.set_zlim(
                current_limits[2]
            )

        return (
            *lines,
            *points
        )

    # --------------------------------------------------------
    # ANIMATION
    # --------------------------------------------------------

    ani = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=30,
        blit=False
    )

    ani.save(
        "converging_walk.gif",
        writer=PillowWriter(
            fps=30
        ),
        dpi=150
    )

    plt.show()


# ============================================================
# STANDALONE 3D ANIMATION
# ============================================================

if generate_3D:

    chain = np.asarray(
        result["chain"]
    )

    xyz = chain[
        :,
        idx
    ][::20]

    fig = plt.figure(
        figsize=(8, 7),
        facecolor="black"
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.set_facecolor("black")

    ax.set_xlabel(
        pnames[0],
        color="white",
        fontsize=14
    )

    ax.set_ylabel(
        pnames[1],
        color="white",
        fontsize=14
    )

    ax.set_zlabel(
        pnames[2],
        color="white",
        fontsize=14
    )

    ax.tick_params(
        axis="x",
        colors="white",
        labelsize=12
    )

    ax.tick_params(
        axis="y",
        colors="white",
        labelsize=12
    )

    ax.tick_params(
        axis="z",
        colors="white",
        labelsize=12
    )

    ax.xaxis.pane.set_facecolor("black")
    ax.yaxis.pane.set_facecolor("black")
    ax.zaxis.pane.set_facecolor("black")

    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")

    for axis in [
        ax.xaxis,
        ax.yaxis,
        ax.zaxis
    ]:
        axis._axinfo["grid"]["color"] = (
            1,
            1,
            1,
            0.15
        )

    # --------------------------------------------------------
    # AXIS LIMITS
    # --------------------------------------------------------

    mins = xyz.min(
        axis=0
    )

    maxs = xyz.max(
        axis=0
    )

    pad = (
        0.08
        * (maxs - mins + 1e-12)
    )

    ax.set_xlim(
        mins[0] - pad[0],
        maxs[0] + pad[0]
    )

    ax.set_ylim(
        mins[1] - pad[1],
        maxs[1] + pad[1]
    )

    ax.set_zlim(
        mins[2] - pad[2],
        maxs[2] + pad[2]
    )

    # --------------------------------------------------------
    # TRAIL
    # --------------------------------------------------------

    trail_len = 200

    initial_segs = np.array([
        xyz[:2]
    ])

    trail = Line3DCollection(
        initial_segs,
        linewidths=3
    )

    ax.add_collection3d(
        trail
    )

    point = ax.scatter(
        [],
        [],
        [],
        color="white",
        edgecolor="hotpink",
        linewidth=2,
        s=65
    )

    def init():

        trail.set_segments([])

        point._offsets3d = (
            [],
            [],
            []
        )

        return trail, point

    smooth = 0.02

    current_limits = np.array([
        ax.get_xlim(),
        ax.get_ylim(),
        ax.get_zlim()
    ], dtype=float)

    def update(i):

        # ----------------------------------------------------
        # SMOOTH ZOOM
        # ----------------------------------------------------

        zoom_window = 200

        zoom_start = max(
            0,
            i - zoom_window
        )

        recent = xyz[
            zoom_start:i + 1
        ]

        if len(recent) > 20:

            centre = np.median(
                recent,
                axis=0
            )

            low = np.percentile(
                recent,
                5,
                axis=0
            )

            high = np.percentile(
                recent,
                95,
                axis=0
            )

            width = (
                high - low
            ) * 1.3

            width = np.maximum(
                width,
                1e-6
            )

            target_limits = np.array([
                [
                    centre[0] - width[0] / 2,
                    centre[0] + width[0] / 2
                ],
                [
                    centre[1] - width[1] / 2,
                    centre[1] + width[1] / 2
                ],
                [
                    centre[2] - width[2] / 2,
                    centre[2] + width[2] / 2
                ]
            ])

            current_limits[:] = (
                (1 - smooth)
                * current_limits
                + smooth
                * target_limits
            )

            ax.set_xlim(
                current_limits[0]
            )

            ax.set_ylim(
                current_limits[1]
            )

            ax.set_zlim(
                current_limits[2]
            )

        # ----------------------------------------------------
        # TRAIL
        # ----------------------------------------------------

        start = max(
            0,
            i - trail_len
        )

        pts = xyz[
            start:i + 1
        ]

        if len(pts) >= 2:

            segs = np.stack(
                [
                    pts[:-1],
                    pts[1:]
                ],
                axis=1
            )

            alphas = (
                np.linspace(
                    0.05,
                    0.9,
                    len(segs)
                ) ** 4
            )

            trail_colors = np.zeros(
                (len(segs), 4)
            )

            trail_colors[:, 0] = 1.0
            trail_colors[:, 1] = 0.08
            trail_colors[:, 2] = 0.55
            trail_colors[:, 3] = alphas

            trail.set_segments(
                segs
            )

            trail.set_color(
                trail_colors
            )

        else:

            trail.set_segments([])

        # ----------------------------------------------------
        # CURRENT POINT
        # ----------------------------------------------------

        x, y, z = xyz[i]

        point._offsets3d = (
            [x],
            [y],
            [z]
        )

        return trail, point

    ani = FuncAnimation(
        fig,
        update,
        frames=len(xyz),
        init_func=init,
        interval=30,
        blit=False
    )

    ani.save(
        "sampler_walk.gif",
        writer=PillowWriter(
            fps=30
        ),
        dpi=200
    )

    plt.show()


# ============================================================
# 1D ANIMATIONS
# ============================================================

if generate_1D:

    for pname in pnames:

        idx = labels.index(
            pname
        )

        y = np.asarray(
            result["chain"]
        )[:, idx]

        x = np.arange(
            len(y)
        )

        stride = 20

        x = x[::stride]
        y = y[::stride]

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 5),
            facecolor="black"
        )

        ax.set_facecolor(
            "black"
        )

        # ----------------------------------------------------
        # LIMITS
        # ----------------------------------------------------

        ax.set_xlim(
            x.min(),
            x.max()
        )

        pad = (
            0.05
            * (y.max() - y.min() + 1e-12)
        )

        ax.set_ylim(
            y.min() - pad,
            y.max() + pad
        )

        # ----------------------------------------------------
        # STYLING
        # ----------------------------------------------------

        ax.set_xlabel(
            "Iteration",
            color="white",
            fontsize=14
        )

        ax.set_ylabel(
            pname,
            color="white",
            fontsize=14
        )

        ax.tick_params(
            axis="both",
            colors="white",
            labelsize=12
        )

        for spine in ax.spines.values():
            spine.set_color(
                "white"
            )

        ax.grid(
            color="white",
            alpha=0.15
        )

        # ----------------------------------------------------
        # TRACE
        # ----------------------------------------------------

        line, = ax.plot(
            [],
            [],
            lw=2,
            color="hotpink"
        )

        point, = ax.plot(
            [],
            [],
            "o",
            color="white",
            markeredgecolor="hotpink",
            markeredgewidth=2,
            ms=7
        )

        def init():

            line.set_data(
                [],
                []
            )

            point.set_data(
                [],
                []
            )

            return (
                line,
                point
            )

        def update(i):

            line.set_data(
                x[:i + 1],
                y[:i + 1]
            )

            point.set_data(
                [x[i]],
                [y[i]]
            )

            return (
                line,
                point
            )

        # ----------------------------------------------------
        # ANIMATION
        # ----------------------------------------------------

        ani = FuncAnimation(
            fig,
            update,
            frames=len(x),
            init_func=init,
            interval=30,
            blit=True
        )

        ani.save(
            f"{pname} lower res.gif",
            writer=PillowWriter(
                fps=30
            ),
            dpi=200
        )

        plt.show()