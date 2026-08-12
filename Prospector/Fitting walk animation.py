import argparse
from prospect.io import write_results
from prospect.fitting import fit_model
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
results_type = 'dynesty'

experiment = True
generate_3D = False
generate_1D = False

# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
readfile = 'Prospector/z=8-9 with MIRI.h5'
result, obs, model = reader.results_from(readfile, dangerous=False)

labels = list(result["theta_labels"])

pnames = ["logmass", "zred", "logzsol"]
idx = [labels.index(p) for p in pnames]

if experiment:
    chain = np.asarray(result["chain"])
    logl = np.asarray(result["lnlikelihood"])
    logvol = np.asarray(result["logvol"])

    # Reduce number of frames if needed
    stride = 10

    chain = chain[::stride]
    logl = logl[::stride]
    logvol = logvol[::stride]

    xyz = chain[:, idx]
    nframes = len(xyz)

    # ------------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(14, 10))

    gs = GridSpec(
        2, 3,
        height_ratios=[1, 3],
        figure=fig,
        hspace=0.3
    )

    # Top plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    top_axes = [ax1, ax2, ax3]

    for ax, name in zip(top_axes, pnames):
        ax.set_xlabel("iteration")
        ax.set_ylabel(name)
        ax.set_title(name)

    # Pre-set top-axis limits
    for k, ax in enumerate(top_axes):
        y = xyz[:, k]

        pad = 0.05 * (y.max() - y.min() + 1e-12)

        ax.set_xlim(0, nframes)
        ax.set_ylim(y.min() - pad, y.max() + pad)

    # Trace lines
    lines = []
    points = []

    for ax in top_axes:
        line, = ax.plot([], [], lw=1.5)
        point, = ax.plot([], [], "o", ms=5)

        lines.append(line)
        points.append(point)

    # ------------------------------------------------------------
    # 3D PLOT
    # ------------------------------------------------------------

    ax3d = fig.add_subplot(gs[1, :], projection="3d")

    ax3d.set_xlabel(pnames[0])
    ax3d.set_ylabel(pnames[1])
    ax3d.set_zlabel(pnames[2])

    # Initial limits
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    pad = 0.1 * (maxs - mins + 1e-12)

    ax3d.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax3d.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax3d.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    # ------------------------------------------------------------
    # DYNAMIC ZOOM SETTINGS
    # ------------------------------------------------------------

    current_limits = np.array([
        ax3d.get_xlim(),
        ax3d.get_ylim(),
        ax3d.get_zlim()
    ], dtype=float)

    smooth = 0.03
    zoom_window = 500

    # ------------------------------------------------------------
    # ANIMATION
    # ------------------------------------------------------------

    def update(i):

        # --------------------------------------------------------
        # TOP TRACE PLOTS
        # --------------------------------------------------------

        for k in range(3):

            y = xyz[:i+1, k]

            lines[k].set_data(
                np.arange(i+1),
                y
            )

            points[k].set_data(
                [i],
                [xyz[i, k]]
            )

    # --------------------------------------------------------
    # DYNAMIC 3D POINTS
    # --------------------------------------------------------

    # Only recent points
    start = max(0, i - zoom_window)

    recent = xyz[start:i+1]

    # plot the recent samples
    ax3d.collections.clear()

    if len(recent) > 0:

        # Make older points fainter
        alpha = np.linspace(0.05, 0.7, len(recent))

        for j, p in enumerate(recent):

            ax3d.scatter(
                p[0],
                p[1],
                p[2],
                color="crimson",
                alpha=alpha[j],
                s=8
            )

    # Current point
    p = xyz[i]

    ax3d.scatter(
        p[0],
        p[1],
        p[2],
        color="black",
        s=45
    )

    # --------------------------------------------------------
    # SMOOTH ZOOM
    # --------------------------------------------------------

    if len(recent) > 20:

        centre = np.median(recent, axis=0)

        low = np.percentile(recent, 5, axis=0)
        high = np.percentile(recent, 95, axis=0)

        width = (high - low) * 1.4
        width = np.maximum(width, 1e-6)

        target = np.array([
            [
                centre[0] - width[0]/2,
                centre[0] + width[0]/2
            ],
            [
                centre[1] - width[1]/2,
                centre[1] + width[1]/2
            ],
            [
                centre[2] - width[2]/2,
                centre[2] + width[2]/2
            ]
        ])

        current_limits[:] = (
            (1 - smooth) * current_limits
            + smooth * target
        )

        ax3d.set_xlim(current_limits[0])
        ax3d.set_ylim(current_limits[1])
        ax3d.set_zlim(current_limits[2])

    return (*lines, *points)


    # ------------------------------------------------------------
    # ANIMATION
    # ------------------------------------------------------------

    ani = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=30,
        blit=False
    )

    ani.save(
        "converging_walk.gif",
        writer=PillowWriter(fps=30),
        dpi=150
    )

    plt.show()

if generate_3D:
    chain = np.asarray(result["chain"])
    xyz = chain[:, idx][::20]   # optional stride to speed things up

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(pnames[0])
    ax.set_ylabel(pnames[1])
    ax.set_zlabel(pnames[2])

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    pad = 0.08 * (maxs - mins + 1e-12)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    trail_len = 200  # how many past points stay visible

    # initialize with one real segment, not an empty list
    initial_segs = np.array([xyz[:2]])   # shape (1, 2, 3)
    trail = Line3DCollection(initial_segs, linewidths=2)
    ax.add_collection3d(trail)

    point = ax.scatter([], [], [], color="pink", s=45)

    def init():
        trail.set_segments([])
        point._offsets3d = ([], [], [])
        return trail, point

    smooth = 0.02   # smaller = smoother/slower

    current_limits = np.array([
        ax.get_xlim(),
        ax.get_ylim(),
        ax.get_zlim()
    ], dtype=float)

    def update(i):
        # Number of recent points used to determine zoom
        zoom_window = 200

        zoom_start = max(0, i - zoom_window)
        recent = xyz[zoom_start:i+1]

        # Only start zooming once we have enough points
        if len(recent) > 20:
            centre = np.median(recent, axis=0)

            # Use percentiles rather than min/max so one outlier
            # doesn't stop the plot from zooming
            low = np.percentile(recent, 5, axis=0)
            high = np.percentile(recent, 95, axis=0)

            width = high - low

            # add some padding
            width *= 1.3

            # prevent axes collapsing to zero width
            width = np.maximum(width, 1e-6)

            target_limits = np.array([
            [centre[0] - width[0]/2, centre[0] + width[0]/2],
            [centre[1] - width[1]/2, centre[1] + width[1]/2],
            [centre[2] - width[2]/2, centre[2] + width[2]/2]
            ])

            # Gradually approach the desired limits
            current_limits[:] = (
                (1 - smooth) * current_limits
                + smooth * target_limits
            )

            ax.set_xlim(current_limits[0])
            ax.set_ylim(current_limits[1])
            ax.set_zlim(current_limits[2])


        start = max(0, i - trail_len)
        pts = xyz[start:i+1]

        if len(pts) >= 2:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)

            # older segments are more transparent, newer ones more opaque
            alphas = np.linspace(0.05, 0.9, len(segs))**4
            colors = np.zeros((len(segs), 4))
            colors[:, 0] = 0.86   # red
            colors[:, 1] = 0.08
            colors[:, 2] = 0.24
            colors[:, 3] = alphas

            trail.set_segments(segs)
            trail.set_color(colors)
        else:
            trail.set_segments([])

        x, y, z = xyz[i]
        point._offsets3d = ([x], [y], [z])
        return trail, point

    ani = FuncAnimation(
        fig,
        update,
        frames=len(xyz),
        init_func=init,
        interval=30,
        blit=False,
    )

    ani.save("sampler_walk.gif", writer=PillowWriter(fps=30), dpi=200)

    plt.show()

if generate_1D:
    for pname in pnames:
        idx = labels.index(pname)
        y = np.asarray(result["chain"])[:, idx]
        x = np.arange(len(y))

        stride = 20
        x = x[::stride]
        y = y[::stride]
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(x.min(), x.max())

        pad = 0.05 * (y.max() - y.min())
        ax.set_ylim(y.min() - pad, y.max() + pad)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(pname)

        line, = ax.plot([], [], lw=1, color='pink')
        point, = ax.plot([], [], " ", color='pink')

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point
        
        def update(i):
            # progressively reveal chain
            line.set_data(x[:i+1], y[:i+1])

            # current sample
            point.set_data([x[i]], [y[i]])

            return line, point

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
            writer=PillowWriter(fps=30),
            dpi=200
        )

        plt.show()