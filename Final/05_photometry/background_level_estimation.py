import numpy as np
from astropy.io import fits

from photutils.aperture import (
    CircularAperture
)

# Load bands and define NIRCam and MIRI bands
with open('final/filter_lists/filter_list_wide.txt') as f:
    bands = [band.strip() for band in f.readlines()]

nircam_bands = bands[:8]
miri_bands = bands[8:]

z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for (z_lo, z_up) in z_ranges:

    flux_medians = []
    flux_mads = []

    for band in bands:
        image = fits.getdata(f'final/04_stacking/stacks/redshifts_{z_lo}_{z_up}/sci/{band}_stack.fits')
        nx, ny = image.shape

        # band multipler - pixels have twice the size in MIRI
        band_mult = 1 if band in nircam_bands else 0.5 if band in miri_bands else print(f'ERROR! {band} not assigned.')

        ap_radius = 5 * band_mult

        fluxes = []

        # No. of trials
        for i in range(10000):
            x_c = np.random.choice(int(nx-10*band_mult)) + 5*band_mult
            y_c = np.random.choice(int(ny-10*band_mult)) + 5*band_mult

            r = ((x_c - (nx-1)/2)**2 + (y_c-(ny-1)/2)**2)**0.5

            if r < 10*band_mult:
                continue
            
            aperture = CircularAperture((x_c, y_c), r = ap_radius)
            flux, _ = aperture.do_photometry(image)
            fluxes.append(flux[0])

        fluxes = np.asarray(fluxes)
        flux_median = np.nanmedian(fluxes)
        flux_medians.append(flux_median)
        flux_mads.append(np.nanmedian(np.abs(fluxes - flux_median)))

    with open(f'final/05_photometry/background_levels/redshifts_{z_lo}_{z_up}_background_levels.txt', 'w') as f:
        f.writelines(str(flux_medians[i])+'\n' for i in range(len(bands)))
    with open(f'final/05_photometry/background_levels/redshifts_{z_lo}_{z_up}_background_levels_mad.txt', 'w') as f:
        f.writelines(str(flux_mads[i])+'\n' for i in range(len(bands)))
    
    print(f'Completed z = {z_lo} to {z_up}.')