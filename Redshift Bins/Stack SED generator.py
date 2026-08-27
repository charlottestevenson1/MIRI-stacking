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

# lya_rest = 0.1216 # microns
# hbeta_rest = 0.4861 # microns
# oiii1_rest = 0.4960 # microns
# oiii2_rest = 0.5007 # microns
# halpha_rest = 0.6563 # microns
# oii_rest = 0.3727 # microns
# neiii_rest = 0.3869 # microns
# sii1_rest = 0.4069 # microns
# sii2_rest = 0.4076 # microns

NIRCam_bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M"
]

MIRI_bands = [
    "F560W", "F770W", "F1000W", "F1280W", 
    "F1500W", "F1800W", "F2100W", "F2550W"
]

bands = NIRCam_bands + MIRI_bands

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

    fluxes_CORR = []
    x_errors = []
    #stack_errors = []
    pwaves = []
    bkgs_CORR = []
    bkg_errors_CORR = []

    # CORR means aperture corrected
    for i in range(len(bands)):
        band = bands[i]

        ap_radius = 5 if band in NIRCam_bands else 2.5 if band in MIRI_bands else print(f'ERROR! {band}')

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

    # Convert fluxes from nJy to AB magnitudes
    flux = np.asarray(fluxes_BSUB_CORR, dtype=float)
    flux_err = np.asarray(bkg_errors_CORR, dtype=float)

    valid = (
        np.isfinite(flux)
        & np.isfinite(flux_err)
        & (flux > 0)
    )

    mag = 31.4 - 2.5 * np.log10(flux)

    mag_bright = 31.4 - 2.5 * np.log10(flux + flux_err)
    mag_faint = 31.4 - 2.5 * np.log10(flux - flux_err)

    mag_err_lower = mag - mag_bright
    mag_err_upper = mag_faint - mag

    fig, ax = plt.subplots(
        figsize=(12, 7),
        facecolor="none"
    )
    ax.set_facecolor("none")

    # NIRCam
    ax.errorbar(
        np.array(pwaves)[:18][valid[:18]] / 1e4,
        mag[:18][valid[:18]],
        yerr=[
            mag_err_lower[:18][valid[:18]],
            mag_err_upper[:18][valid[:18]]
        ],
        ecolor="steelblue",
        color="steelblue",
        markersize=7,
        fmt="o",
        capsize=2,
        label="NIRCam"
    )

    # MIRI
    ax.errorbar(
        np.array(pwaves)[18:][valid[18:]] / 1e4,
        mag[18:][valid[18:]],
        yerr=[
            mag_err_lower[18:][valid[18:]],
            mag_err_upper[18:][valid[18:]]
        ],
        ecolor="crimson",
        color="crimson",
        markersize=7,
        fmt="o",
        capsize=2,
        label="MIRI"
    )

    ax.set_xlim(0.4, 30)

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.invert_yaxis()

    ax.set_title(
        f"Stack SED, z={zlo}-{zup}",
        color="white",
        fontsize=20,
        pad=12
    )

    ax.set_xlabel(
        "Observed wavelength (microns)",
        color="white",
        fontsize=18,
        labelpad=10
    )

    ax.set_ylabel(
        "AB magnitude",
        color="white",
        fontsize=18,
        labelpad=10
    )

    ax.tick_params(
        axis="both",
        which="both",
        colors="white",
        labelsize=14,
        width=1.5
    )

    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.5)

    ax.grid(
        color="white",
        alpha=0.12
    )

    legend = ax.legend(
        fontsize=14,
        facecolor="none",
        edgecolor="white"
    )

    for text in legend.get_texts():
        text.set_color("white")

    legend.get_frame().set_alpha(0)

    fig.tight_layout()

    fig.savefig(
        f"Presentations/z={zlo}-{zup} Stack SED.png",
        dpi=400,
        bbox_inches="tight",
        transparent=True
    )

    #plt.show()
    plt.close(fig)