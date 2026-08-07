### DOING STACK PHOTOMETRY ###

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

with open('Photometry/Background levels.txt') as f:
    bkgs = np.array([float(bkg) for bkg in f.readlines()])

with open('Photometry/Background levels MAD.txt') as f:
    bkg_errors = [1.4826*float(line) for line in f.readlines()]

with open('Photometry/Pivot wavelengths.txt') as f:
    pivot_waves = np.array([float(wl) for wl in f.readlines()])

ap_radius = 5     # in px

fluxes = []
errors = []

for i in range(len(bands)):
    band = bands[i]

    image = fits.getdata(f'Photometry/Stacked images/SCI/{band}_stack.fits')
    error = fits.getdata(f'Photometry/Stacked images/ERR/{band}_stack_ERR.fits')

    ny, nx = image.shape
    position = ((nx - 1) / 2, (ny - 1) / 2)

    aperture = CircularAperture(position, r = ap_radius)
    flux, flux_err = aperture.do_photometry(image, error=error)
    fluxes.append(flux[0])
    errors.append(flux_err[0])

fluxes_BSUB = np.array(fluxes) - np.array(bkgs)

plt.errorbar(
    pivot_waves, # microns
    fluxes_BSUB,
    yerr = bkg_errors,
    fmt = 'o',
    capsize = 3,
)

plt.title('Lorenzo SED')
plt.xscale('log')
plt.xlabel('Pivot wavelength (microns)')
plt.ylabel('Flux in 0.15" aperture (nJy)')

plt.savefig('Photometry/Stack SED.png')
plt.show()

# plt.plot(pivot_waves/1e4, errors, label='IVW error', marker='o', ls=' ')
# plt.plot(pivot_waves/1e4, bkg_errors, label='RAP error', marker='o', ls=' ')
# plt.legend()
# plt.show()