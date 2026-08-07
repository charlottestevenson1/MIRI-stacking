import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    ApertureStats,
    aperture_photometry
)

bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M", "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

ptp_scale = 0.031  # arcsec/pixel

radii_as = np.arange(0.1, 2.5, 0.1)
radii_px = np.array(radii_as) / ptp_scale

fig, axes = plt.subplots(
    4, 7,
    sharex=True
)

fig.subplots_adjust(hspace=0.5, wspace=0.2)

for i in range(len(bands)):
    band = bands[i]

    fluxes = []
    flux_errors = []

    filename = f"Stacking/Masked inverse variance stacking/Masked median stacking/Stacked images/SCI/{band}_stack.fits"
    image = fits.getdata(filename)

    ny, nx = image.shape
    position = ((nx - 1) / 2, (ny - 1) / 2)

    for radius in radii_px:
        aperture = CircularAperture(position, r = radius)
        flux, flux_err = aperture.do_photometry(image)
        fluxes.append(flux)

    ax = axes[i//7, i%7]

    ax.plot(radii_as, fluxes, marker='o', markersize=3)
    #ax.set_yscale('log')
    ax.set_title(band)

axes[3,5].set_axis_off()

axes[3,6].set_axis_off()

fig.supxlabel("Radius (arcsec)")
fig.supylabel("flux")
#fig.suptitle("Radius-flux for all bands")
fig.suptitle("Radius-flux for all bands")

plt.show()