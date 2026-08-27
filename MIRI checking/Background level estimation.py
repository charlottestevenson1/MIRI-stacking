import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from photutils.aperture import (
    CircularAperture
)

ID = 37925

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

flux_medians = []
flux_MADs = []

for band in bands:
    try:
        image = fits.getdata(f'Cutouts/SCI/{ID}_{band}.fits')

    except FileNotFoundError:
        print(f'{band} not present for {ID}')
        flux_medians.append(np.nan)
        flux_MADs.append(np.nan)

        continue
    
    
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
        fluxes.append(flux)

    flux_medians.append(np.nanmedian(fluxes))
    flux_MADs.append(np.nanmedian(abs(fluxes-np.nanmedian(fluxes))))

with open(f'MIRI checking/{ID} Background levels.txt', 'w') as f:
    f.writelines(str(flux_medians[i])+'\n' for i in range(len(bands)))
with open(f'MIRI checking/{ID} Background levels MAD.txt', 'w') as f:
    f.writelines(str(flux_MADs[i])+'\n' for i in range(len(bands)))