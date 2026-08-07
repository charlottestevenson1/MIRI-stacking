import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from photutils.aperture import (
    CircularAperture
)

bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M", "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

flux_medians = []
flux_MADs = []

ap_radius = 5   # in px

z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for z_range in z_ranges:
    zlo = z_range[0]
    zup = z_range[1]

    for band in bands:
        image = fits.getdata(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/SCI/{band}_stack.fits')
        nx, ny = image.shape

        fluxes = []

        # No. of trials
        for i in range(10000):
            x_c = np.random.choice(nx-10) + 5
            y_c = np.random.choice(ny-10) + 5

            r = ((x_c - (nx-1)/2)**2 + (y_c-(ny-1)/2)**2)**0.5

            if r < 10:
                continue
            
            aperture = CircularAperture((x_c, y_c), r = ap_radius)
            flux, flux_err = aperture.do_photometry(image)
            fluxes.append(flux)

        # for x_c in range(5, nx-5):
        #     for y_c in range(5, ny-5):
        #         r = ((x_c - (nx-1)/2)**2 + (y_c-(ny-1)/2)**2)**0.5
        #         radii.append(r)
                
        #         aperture = CircularAperture((x_c, y_c), r = ap_radius)
        #         flux, flux_err = aperture.do_photometry(image)
        #         fluxes.append(flux)

        flux_medians.append(np.nanmedian(fluxes))
        flux_MADs.append(np.nanmedian(abs(fluxes-np.nanmedian(fluxes))))

    with open(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/Background levels.txt', 'w') as f:
        f.writelines(str(flux_medians[i])+'\n' for i in range(len(bands)))
    with open(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/Background levels MAD.txt', 'w') as f:
        f.writelines(str(flux_MADs[i])+'\n' for i in range(len(bands)))