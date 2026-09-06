import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from photutils.aperture import (
    CircularAperture
)

# Load bands and define NIRCam and MIRI bands
with open('Final/Filter lists/filter list wide.txt') as f:
    bands = [band.strip() for band in f.readlines()]

NIRCam_bands = bands[:8]
MIRI_bands = bands[8:]

z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for (zlo, zup) in z_ranges:

    flux_medians = []
    flux_MADs = []

    for band in bands:
        image = fits.getdata(f'Final/4. Stacking/Stacks/Redshifts {zlo}-{zup}/SCI/{band}_stack.fits')
        nx, ny = image.shape

        # band multipler - pixels have twice the size in MIRI
        BM = 1 if band in NIRCam_bands else 0.5 if band in MIRI_bands else print(f'ERROR! {band} not assigned.')

        ap_radius = 5 * BM

        fluxes = []

        # No. of trials
        for i in range(10000):
            x_c = np.random.choice(int(nx-10*BM)) + 5*BM
            y_c = np.random.choice(int(ny-10*BM)) + 5*BM

            r = ((x_c - (nx-1)/2)**2 + (y_c-(ny-1)/2)**2)**0.5

            if r < 10*BM:
                continue
            
            aperture = CircularAperture((x_c, y_c), r = ap_radius)
            flux, flux_err = aperture.do_photometry(image)
            print(flux)
            fluxes.append(flux[0])

        flux_medians.append(np.nanmedian(fluxes))
        flux_MADs.append(np.nanmedian(abs(fluxes-np.nanmedian(fluxes))))

    with open(f'Final/5. Photometry/Background levels/Redshifts {zlo}-{zup} background levels.txt', 'w') as f:
        f.writelines(str(flux_medians[i])+'\n' for i in range(len(bands)))
    with open(f'Final/5. Photometry/Background levels/Redshifts {zlo}-{zup} background levels MAD.txt', 'w') as f:
        f.writelines(str(flux_MADs[i])+'\n' for i in range(len(bands)))
    
    print(f'Completed z = {zlo} to {zup}.')