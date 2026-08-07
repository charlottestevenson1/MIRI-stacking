import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

bands = [
    'F070W', 'F090W', 'F115W', 'F150W', 'F162M', 'F182M', 'F200W', 'F210M',
    'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F410M', 'F430M', 'F444W',
    'F460M', 'F480M', 'F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W',
    'F1800W', 'F2100W', 'F2550W'
]

fig, axes = plt.subplots(5, 6, figsize=(12, 10))
fig.patch.set_facecolor('#2b2b2b')
axes = axes.ravel()

fig.suptitle(
    "Median",
    fontsize=18,
    fontweight="bold",
    color="white"   # if you're using the dark background
)

# ----- Global stretch (comment out to use individual stretches) -----
images = [
    fits.getdata(f'Stacking/VI2 no BNs/Median stack/{band}_stack_MEDIAN.fits')
    for band in bands
]

global_vmin = min(np.nanpercentile(image, 5) for image in images)
global_vmax = max(np.nanpercentile(image, 99) for image in images)
#####

for i, (band, image) in enumerate(zip(bands, images)):

    axes[i].set_facecolor("#2b2b2b")

    axes[i].set_title(
        band,
        fontsize=8,
        color="white"
        )

    axes[i].set_axis_off()

    # ----- Individual stretch -----
    vmin = np.nanpercentile(image, 0)
    vmax = np.nanpercentile(image, 100)

    # # ----- Global stretch -----
    # vmin = global_vmin
    # vmax = global_vmax

    axes[i].imshow(
        image,
        origin='lower',
        cmap='inferno',
        vmin=vmin,
        vmax=vmax
    )

    axes[i].text(
        0.5, -0.08,
        f"[{vmin:.2e}, {vmax:.2e}]",
        transform=axes[i].transAxes,
        ha='center',
        va='top',
        fontsize=8,
        color='white'
    )

# Turn off unused panels
for ax in axes[len(bands):]:
    ax.set_axis_off()

plt.tight_layout()

fig.savefig(
    'Stacking/VI2 no BNs/All_stacks_median.pdf',
    bbox_inches='tight'
)

plt.show()