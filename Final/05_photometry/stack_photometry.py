### DOING STACK PHOTOMETRY ###

import numpy as np
from astropy.io import fits

from photutils.aperture import CircularAperture

# Load bands and define NIRCam and MIRI bands
bands = np.loadtxt('final/filter_lists/filter_list_wide.txt', dtype=str)

nircam_bands = bands[:8]

miri_bands = bands[8:]

# Define MJysr to nJy conversion - it differs between NIRCam and MIRI bands
mjysr_to_njy = np.array([21.15398748 for i in range(8)] + [84.61594994 for i in range(8)])
    
# Aperture correction factors for ALL BANDS
acs = np.loadtxt('final/05_photometry/aperture_corrections.txt', dtype=float)

if not (len(bands) == len(mjysr_to_njy) == len(acs)):
    raise ValueError(
        'Filter list, MJy/sr conversion, and aperture-correction lengths must match.'
    )
    
z_ranges = np.loadtxt('final/redshift_bins.txt', dtype=float)

for (z_lo, z_up) in z_ranges:

    # Background levels
    all_bkgs = np.loadtxt(f'final/05_photometry/background_levels/redshifts_{z_lo}_{z_up}_background_levels.txt', dtype=float) * mjysr_to_njy

    # Background levels ERRORS
    all_bkg_errors = np.loadtxt(f'final/05_photometry/background_levels/redshifts_{z_lo}_{z_up}_background_levels_mad.txt', dtype=float) * mjysr_to_njy

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

    np.savetxt(f'final/06_sed_fitting/stack_data/{z_lo}_{z_up}_fluxes.txt', fluxes_bsub_corr, fmt='%f')
    np.savetxt(f'final/06_sed_fitting/stack_data/{z_lo}_{z_up}_errors.txt', bkg_errors_corr, fmt='%f')
