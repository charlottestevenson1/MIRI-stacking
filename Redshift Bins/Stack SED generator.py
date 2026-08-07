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

lya_rest = 0.1216 # microns
hbeta_rest = 0.4861 # microns
oiii1_rest = 0.4960 # microns
oiii2_rest = 0.5007 # microns
halpha_rest = 0.6563 # microns
oii_rest = 0.3727 # microns
neiii_rest = 0.3869 # microns
sii1_rest = 0.4069 # microns
sii2_rest = 0.4076 # microns

bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M", "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

MJysr_to_nJy = np.array([21.15398748 for i in range(18)] + [84.61594994 for i in range(8)])

with open('Photometry/Band data/Eff widths.txt') as f:
    band_widths = [float(line) for line in f.readlines()]

with open('Photometry/Pivot wavelengths.txt') as f:
    pivot_waves = np.array([float(wl) for wl in f.readlines()])

with open('Photometry/Band data/Aperture corrections.txt') as f:
    ACs = np.array([float(wl) for wl in f.readlines()])

z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for z_range in z_ranges:
    zlo = z_range[0]
    zup = z_range[1]
    with open(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/Background levels.txt') as f:
        all_bkgs = np.array([float(bkg) for bkg in f.readlines()]) * MJysr_to_nJy

    with open(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/Background levels MAD.txt') as f:
        all_bkg_errors = np.array([1.4826*float(line) for line in f.readlines()]) * MJysr_to_nJy


    ap_radius = 5     # in px

    fluxes_CORR = []
    x_errors = []
    #stack_errors = []
    pwaves = []
    bkgs_CORR = []
    bkg_errors_CORR = []

    # CORR means aperture corrected
    for i in range(len(bands)):
        band = bands[i]
        image = fits.getdata(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/SCI/{band}_stack.fits')
        #error = fits.getdata(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/ERR/{band}_stack_ERR.fits')

        ny, nx = image.shape
        position = ((nx - 1) / 2, (ny - 1) / 2)

        aperture = CircularAperture(position, r = ap_radius)
        #flux, flux_err = aperture.do_photometry(image, error=error)
        fluxMJy, flux_errMJy = aperture.do_photometry(image)
        flux = fluxMJy
        flux_err = flux_errMJy
        pwaves.append(pivot_waves[i])
        x_errors.append(band_widths[i])
        fluxes_CORR.append(flux[0]*ACs[i])
        #stack_errors.append(flux_err[0])
        bkgs_CORR.append(all_bkgs[i]*ACs[i])
        bkg_errors_CORR.append(all_bkg_errors[i]*ACs[i])

    fluxes_CORR = fluxes_CORR * MJysr_to_nJy
    fluxes_BSUB_CORR = np.array(fluxes_CORR) - np.array(bkgs_CORR)

    with open(f'Prospector/Stack data/{zlo}-{zup} Fluxes.txt', 'w') as f:
        for point in fluxes_BSUB_CORR:
            f.writelines(str(point)+'\n')

    with open(f'Prospector/Stack data/{zlo}-{zup} Errors.txt', 'w') as f:
            for point in bkg_errors_CORR:
                f.writelines(str(point)+'\n')

    plt.errorbar(
        np.array(pwaves)/1e4, # so that they are in units of microns
        fluxes_BSUB_CORR,
        xerr = x_errors,
        yerr = bkg_errors_CORR,
        ecolor = 'gray',
        markersize = 5,
        fmt = 'o',
        capsize = 1.5,
    )

    plt.title(f'Stack SED, z={zlo}-{zup}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pivot wavelength (microns)')
    plt.ylabel('Flux in 0.15" aperture (nJy)')

    plt.axvspan(
        lya_rest*zlo,
        lya_rest*(zup+1),
        color = 'yellow',
        alpha=0.3,
        label='Lyman-alpha'
    )

    plt.axvspan(
            hbeta_rest*(zlo+1),
            hbeta_rest*(zup+1),
            color = 'blue',
            alpha=0.3,
            label='H-beta'
    )

    plt.axvspan(
            oiii1_rest*(zlo+1),
            oiii1_rest*(zup+1),
            color = 'red',
            alpha=0.3,
            label='OIII'
    )

    plt.axvspan(
                oiii2_rest*(zlo+1),
                oiii2_rest*(zup+1),
                color = 'red',
                alpha=0.3
    )

    plt.axvspan(
                halpha_rest*(zlo+1),
                halpha_rest*(zup+1),
                color = 'green',
                alpha=0.3,
                label='H-alpha'
    )

    plt.axvspan(
        oii_rest*(zlo+1),
        oii_rest*(zup+1),
        color = 'purple',
        alpha = 0.3,
        label = 'OII'
    )

    plt.axvspan(
        neiii_rest*(zlo+1),
        neiii_rest*(zup+1),
        color = 'orange',
        alpha = 0.3,
        label = 'NeIII'
    )

    # plt.axvspan(
    #     sii1_rest*(zlo+1),
    #     sii1_rest*(zup+1),
    #     color = 'green',
    #     alpha = 0.3,
    #     label = 'SII'
    # )

    # plt.axvspan(
    #     sii2_rest*(zlo+1),
    #     sii2_rest*(zup+1),
    #     color = 'green',
    #     alpha = 0.3,
    # )

    plt.legend()
    plt.savefig(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/Stack SED.png')
    plt.show()

    # plt.plot(pivot_waves/1e4, errors, label='IVW error', marker='o', ls=' ')
    # plt.plot(pivot_waves/1e4, bkg_errors, label='RAP error', marker='o', ls=' ')
    # plt.legend()
    # plt.show()