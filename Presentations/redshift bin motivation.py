import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.colors as colors
import prospect.io.read_results as reader

# ============================================================
# LOAD FILTERS
# ============================================================

result, obs, _ = reader.results_from(
    "Prospector/Fits/z=8-9 with MIRI.h5",
    dangerous=False
)

filters = obs["filters"]
filternames = obs["filternames"]

# ============================================================
# REDSHIFT RANGE
# ============================================================

zmin = 8
zmax = 15

z_values = np.linspace(
    zmin,
    zmax,
    500
)

# ============================================================
# FILTER COLOURS
# ============================================================

# Colour filters by their observed wavelength, so the same
# filter keeps the same colour throughout the plot.

filter_centres = np.array([
    np.average(
        f.wavelength,
        weights=f.transmission
    )
    for f in filters
])

norm = colors.Normalize(
    vmin=filter_centres.min(),
    vmax=filter_centres.max()
)

cmap = colormaps["prism"]

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 10),
    facecolor="none"
)

ax.tick_params(
    axis="both",
    colors="white"
)

ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")

for spine in ax.spines.values():
    spine.set_color("white")

ax.set_facecolor("none")

for f, name, centre_obs in zip(
    filters,
    filternames,
    filter_centres
):

    colour = cmap(
        norm(centre_obs)
    )

    wave_obs = f.wavelength.copy()
    transmission = f.transmission.copy()

    # Normalise transmission
    transmission /= transmission.max()

    # --------------------------------------------------------
    # Effective wavelength as a function of redshift
    # --------------------------------------------------------

    lambda_eff = np.average(
        wave_obs,
        weights=f.transmission
    ) / (1 + z_values)

    # --------------------------------------------------------
    # Determine approximate filter width
    # using where transmission > 10%
    # --------------------------------------------------------

    significant = transmission > 0.1

    wave_low_obs = wave_obs[significant].min()
    wave_high_obs = wave_obs[significant].max()

    lambda_low = (
        wave_low_obs
        / (1 + z_values)
    )

    lambda_high = (
        wave_high_obs
        / (1 + z_values)
    )

    # --------------------------------------------------------
    # Draw filter band
    # --------------------------------------------------------

    ax.fill_betweenx(
        z_values,
        lambda_low,
        lambda_high,
        color=colour,
        alpha=0.22
    )

    # Central wavelength
    ax.plot(
        lambda_eff,
        z_values,
        color=colour,
        lw=2.5
    )

    # --------------------------------------------------------
    # Label at top of plot
    # --------------------------------------------------------

    ax.text(
        lambda_eff[-1],
        zmax + 0.15,
        name,
        color="white",
        fontsize=14,
        fontweight='bold',
        ha="center",
        va="bottom",
        rotation=45
    )

# ============================================================
# REDSHIFT BIN BOUNDARIES
# ============================================================

for z in [9, 10, 11, 12]:
    ax.axhline(
        z,
        color="white",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6
    )

# ============================================================
# IMPORTANT REST-FRAME WAVELENGTHS
# ============================================================

ax.axvline(
    10**3,
    color="white",
    linestyle=":",
    linewidth=2,
    alpha=0.8
)

ax.axvline(
    10**4,
    color="white",
    linestyle=":",
    linewidth=2,
    alpha=0.8
)

# ============================================================
# FORMAT
# ============================================================

ax.set_xscale("log")

ax.set_xlabel(
    "Rest-frame wavelength [Å]",
    fontsize=20,
    color="white",
    labelpad=12
)

ax.set_ylabel(
    "Redshift",
    fontsize=20,
    color="white",
    labelpad=12
)

ax.tick_params(
    axis="both",
    colors="white",
    labelsize=16,
    width=1.5,
    length=6
)

for spine in ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1.5)

ax.set_ylim(
    zmin,
    zmax
)

ax.grid(
    alpha=1,
    color='white',
    axis="x"
)

# ax.set_title(
#     "Rest-frame filter coverage as a function of redshift",
#     fontsize=15
# )

# ax.legend(
#     loc="lower left"
# )

plt.tight_layout()

plt.savefig(
    "Presentations/restframe_filter_coverage_continuous.png",
    dpi=400,
    bbox_inches="tight",
    transparent=True
)

plt.show()