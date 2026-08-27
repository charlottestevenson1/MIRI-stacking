import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.animation import FuncAnimation


# ============================================================
# STACK THE GALAXIES
# ============================================================

stacking_galaxy_IDs = [
    int(i)
    for i in open(
        "Stacking/Objects for stacking.txt"
    ).readlines()
]

bands = ["F150W"]


for i in range(1,15,1):

    for band in bands:

        band_galaxy_IDs = [
            int(i)
            for i in open(
                f"Filter objects/{band} objects.txt"
            )
        ]

        galaxy_IDs = [
            ID
            for ID in stacking_galaxy_IDs
            if ID in band_galaxy_IDs
        ]

        n_galaxies = min(
            370,
            len(galaxy_IDs)
        )

        galaxy_IDs = galaxy_IDs[:n_galaxies]

        SCI_array = np.stack([
            fits.getdata(
                f"Stacking/Masked inverse variance stacking/"
                f"Masked cutouts/SCI/"
                f"{ID}_{band}_MASKED.fits"
            )
            for ID in galaxy_IDs[:i]
        ])

        ERR_array = np.stack([
            fits.getdata(
                f"Cutouts/ERR/"
                f"{ID}_{band}_ERR.fits"
            )
            for ID in galaxy_IDs[:i]
        ])

        valid = (
            np.isfinite(SCI_array)
            &
            np.isfinite(ERR_array)
            &
            (ERR_array > 0)
        )

        weights = np.zeros_like(
            ERR_array
        )

        weights[valid] = (
            1.0
            /
            ERR_array[valid] ** 2
        )

        numerator = np.sum(
            np.where(
                valid,
                SCI_array * weights,
                0.0
            ),
            axis=0
        )

        denominator = np.sum(
            weights,
            axis=0
        )

        stack = np.divide(
            numerator,
            denominator,
            out=np.full_like(
                numerator,
                np.nan
            ),
            where=denominator > 0
        )

        #ENABLE FOR MEDIAN STACK
        #stack = np.nanmedian(SCI_array, axis=0)

        fits.writeto(
            f"Presentations/Stacking animation/"
            f"{band}_median_stack_masked_{i}.fits",
            stack,
            overwrite=True
        )

# # Central pixel
# cy = SCI_array.shape[1] // 2
# cx = SCI_array.shape[2] // 2

# # Flux and error for each galaxy at central pixel
# central_flux = SCI_array[:, cy, cx]
# central_err = ERR_array[:, cy, cx]

# # Inverse variance weight
# central_weight = 1.0 / central_err**2

# # Only keep valid values
# valid_centre = (
#     np.isfinite(central_flux)
#     & np.isfinite(central_weight)
#     & (central_err > 0)
# )

# central_flux = central_flux[valid_centre]
# central_weight = central_weight[valid_centre]


# # Plot
# fig, ax = plt.subplots(figsize=(7, 6))

# ax.scatter(
#     central_weight,
#     central_flux,
#     alpha=0.7
# )

# ax.set_xlabel(r"Inverse variance weight $1/\sigma^2$")
# ax.set_ylabel("Central pixel flux")

# plt.tight_layout()
# plt.show()


# ============================================================
# LOAD STACKS
# ============================================================

arrays = [
    fits.getdata(
        f"Presentations/Stacking animation/"
        f"{bands[0]}_median_stack_masked_{i}.fits"
    )
    for i in range(1,15,1)
]


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 6),
    facecolor="black"
)

ax.set_facecolor("black")
ax.axis("off")

fig.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1
)


# ============================================================
# FIRST IMAGE
# ============================================================

im = ax.imshow(
    arrays[0],
    cmap="viridis"
)

title = ax.text(
    0.5,
    0.95,
    "1 object",
    transform=ax.transAxes,
    ha="center",
    va="top",
    color="white",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# UPDATE
# ============================================================

def update(i):

    im.set_data(
        arrays[i]
    )

    n = i + 1

    if n == 1:
        title.set_text("1 object")
    else:
        title.set_text(
            f"{n} objects"
        )

    return im, title


# ============================================================
# ANIMATION
# ============================================================

anim = FuncAnimation(
    fig,
    update,
    frames=len(arrays),
    interval=333,
    blit=True
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "Presentations/Stacking animation/"
    "Masked animation.gif"
)

anim.save(
    output_file,
    writer="pillow",
    fps=3
)

plt.show()