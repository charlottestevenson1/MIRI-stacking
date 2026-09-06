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

# Load bands and define NIRCam and MIRI bands
with open('Final/Filter lists/filter list wide.txt') as f:
    bands = [band.strip() for band in f.readlines()]

NIRCam_bands = bands[:8]

MIRI_bands = bands[8:]

# Define MJysr to nJy conversion - it differs between NIRCam and MIRI bands
MJysr_to_nJy = np.array([21.15398748 for i in range(8)] + [84.61594994 for i in range(8)])
    
# Aperture correction factors for ALL BANDS
with open('Final/5. Photometry/Aperture corrections.txt') as f:
    ACs = np.array([float(wl) for wl in f.readlines()])
    
z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for (zlo, zup) in z_ranges:

    # Background levels
    with open(f'Final/5. Photometry/Background levels/Redshifts {zlo}-{zup} background levels.txt') as f:
        all_bkgs = np.array([float(bkg) for bkg in f.readlines()]) * MJysr_to_nJy

    # Background levels ERRORS
    with open(f'Final/5. Photometry/Background levels/Redshifts {zlo}-{zup} background levels MAD.txt') as f:
        all_bkg_errors = np.array([1.4826*float(line) for line in f.readlines()]) * MJysr_to_nJy

    # 'CORR'/'Corrected' here refers to aperture correction
    fluxes_CORR = []        # Corrected flux values
    bkgs_CORR = []          # Corrected background values
    bkg_errors_CORR = []    # Corrected background errors

    for i in range(len(bands)):

        # Perform aperture photometry
        band = bands[i]

        ap_radius = 5 if band in NIRCam_bands else 2.5 if band in MIRI_bands else print(f'ERROR! {band}')

        image = fits.getdata(f'Final/4. Stacking/Stacks/Redshifts {zlo}-{zup}/SCI/{band}_stack.fits')
        
        ny, nx = image.shape
        position = ((nx - 1) / 2, (ny - 1) / 2)

        aperture = CircularAperture(position, r = ap_radius)
        fluxMJy, flux_errMJy = aperture.do_photometry(image)
        flux = fluxMJy
        flux_err = flux_errMJy
        fluxes_CORR.append(flux[0]*ACs[i])
        bkgs_CORR.append(all_bkgs[i]*ACs[i])
        bkg_errors_CORR.append(all_bkg_errors[i]*ACs[i])

    fluxes_CORR = fluxes_CORR * MJysr_to_nJy
    fluxes_BSUB_CORR = np.array(fluxes_CORR) - np.array(bkgs_CORR)

    with open(f'Final/6. Prospector/Stack data/{zlo}-{zup} Fluxes.txt', 'w') as f:
        for point in fluxes_BSUB_CORR:
            f.writelines(str(point)+'\n')

    with open(f'Final/6. Prospector/Stack data/{zlo}-{zup} Errors.txt', 'w') as f:
            for point in bkg_errors_CORR:
                f.writelines(str(point)+'\n')