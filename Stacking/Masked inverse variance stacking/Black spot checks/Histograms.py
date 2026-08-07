import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import CircularAnnulus

bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M", "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

MIRI_bands = [
    "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

ptp_scale = 0.031  # arcsec/pixel

annuli_as = [
    (1e-10, 0.15), (0.15, 0.4), (0.4, 0.65), (0.65, 0.9),
    (0.9, 1.15), (1.15, 1.4), (1.65, 1.9), (1.9, 2.15), (2.15, 2.4)
]
annuli = np.array(annuli_as) / ptp_scale

output_dir = "Stacking/Masked inverse variance stacking/Black spot checks/Histograms"
os.makedirs(output_dir, exist_ok=True)

for band in bands:
    filename = f"Stacking/Masked inverse variance stacking/Stacked images/SCI/{band}_stack.fits"
    image = fits.getdata(filename)

    ny, nx = image.shape
    position = ((nx - 1) / 2, (ny - 1) / 2)

    fig, axes = plt.subplots(
        len(annuli),
        1,
        figsize=(10, 18),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    bins = np.linspace(image.min(), image.max(), 100)

    for ax, (rin, rout) in zip(axes, annuli):
        ann = CircularAnnulus(position, r_in=rin, r_out=rout)
        mask = ann.to_mask(method="center")
        cutout = mask.cutout(image)

        pixels = cutout[mask.data > 0]

        ax.hist(pixels, bins=bins, density=True)
        ax.set_title(f"{rin:.0f}–{rout:.0f} px", fontsize=10)

        if rout==0.15/ptp_scale:
            median = np.median(pixels)

        if band not in MIRI_bands:
            if rout==2.4/ptp_scale:
                xlim = np.percentile(pixels, 99)*1.1
        else:
            xlim = max(xlim, np.percentile(pixels, 99)*1.1)

        ax.axvline(
            x=0,
            color="black",
            linestyle="--",
            linewidth=1
        )   

        ax.axvline(
            x = np.median(pixels),
            color="blue",
            linestyle="--",
            linewidth=1
        )
        
        ax.text(
            0.95, 0.95, f'Median = {np.median(pixels):.3g}',
            transform=ax.transAxes,
            ha="right",
            va="top"
        )

    for ax in axes:
        ax.axvline(
            x=median,
            color="red",
            linestyle="--",
            linewidth=1
        )

    fig.suptitle(f"Masked IVS plot, {band}", fontsize=15)
    fig.supxlabel("Pixel value")
    fig.supylabel("Probability density")
    axes[0].set_xlim(None, max(xlim, median*1.1))

    #plt.show()
    outpath = os.path.join(output_dir, f"{band}_histograms.png")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {band}")