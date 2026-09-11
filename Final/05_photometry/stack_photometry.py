### DOING STACK PHOTOMETRY ###

import numpy as np
from astropy.io import fits

from photutils.aperture import CircularAperture

# Load bands and define NIRCam and MIRI bands
with open('final/filter_lists/filter_list_wide.txt') as f:
    bands = [band.strip() for band in f.readlines()]

nircam_bands = bands[:8]

miri_bands = bands[8:]

# Define MJysr to nJy conversion - it differs between NIRCam and MIRI bands
mjysr_to_njy = np.array([21.15398748 for i in range(8)] + [84.61594994 for i in range(8)])
    
# Aperture correction factors for ALL BANDS
with open('final/05_photometry/aperture_corrections.txt') as f:
    acs = np.array([float(wl) for wl in f.readlines()])

if not (len(bands) == len(mjysr_to_njy) == len(acs)):
    raise ValueError(
        'Filter list, MJy/sr conversion, and aperture-correction lengths must match.'
    )
    
z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for (z_lo, z_up) in z_ranges:

    # Background levels
    with open(f'final/05_photometry/background_levels/redshifts_{z_lo}_{z_up}_background_levels.txt') as f:
        all_bkgs = np.array([float(bkg) for bkg in f.readlines()]) * mjysr_to_njy

    # Background levels ERRORS
    with open(f'final/05_photometry/background_levels/redshifts_{z_lo}_{z_up}_background_levels_mad.txt') as f:
        all_bkg_errors = np.array([1.4826*float(line) for line in f.readlines()]) * mjysr_to_njy

    if len(all_bkgs) != len(bands) or len(all_bkg_errors) != len(bands):
        raise ValueError(f'Background data for z={z_lo}-{z_up} must contain one value per band.')

    # 'corr'/'Corrected' here refers to aperture correction
    fluxes_corr = []        # Corrected flux values
    bkgs_corr = []          # Corrected background values
    bkg_errors_corr = []    # Corrected background errors

    for i in range(len(bands)):

        # Perform aperture photometry
        band = bands[i]

        ap_radius = 5 if band in nircam_bands else 2.5 if band in miri_bands else print(f'ERROR! {band}')

        image = fits.getdata(f'final/04_stacking/stacks/redshifts_{z_lo}_{z_up}/sci/{band}_stack.fits')
        
        ny, nx = image.shape
        position = ((nx - 1) / 2, (ny - 1) / 2)

        aperture = CircularAperture(position, r = ap_radius)
        flux_mjy, _ = aperture.do_photometry(image)
        flux = flux_mjy
        fluxes_corr.append(flux[0]*acs[i])
        bkgs_corr.append(all_bkgs[i]*acs[i])
        bkg_errors_corr.append(all_bkg_errors[i]*acs[i])

    fluxes_corr = np.asarray(fluxes_corr) * mjysr_to_njy
    fluxes_bsub_corr = np.array(fluxes_corr) - np.array(bkgs_corr)

    with open(f'final/06_sed_fitting/stack_data/{z_lo}_{z_up}_fluxes.txt', 'w') as f:
        for point in fluxes_bsub_corr:
            f.writelines(str(point)+'\n')

    with open(f'final/06_sed_fitting/stack_data/{z_lo}_{z_up}_errors.txt', 'w') as f:
            for point in bkg_errors_corr:
                f.writelines(str(point)+'\n')