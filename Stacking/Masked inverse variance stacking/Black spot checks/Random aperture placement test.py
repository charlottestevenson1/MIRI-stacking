import os
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

ap_radius = 5   # in px


for band in bands:
    image = fits.getdata(f'Stacking/Masked inverse variance stacking/Stacked images/SCI/{band}_stack.fits')
    nx, ny = image.shape

    radii = []
    fluxes = []

    # No. of trials
    for i in range(1000):
        x_c = np.random.choice(nx-10) + 5
        y_c = np.random.choice(ny-10) + 5

        r = ((x_c - (nx-1)/2)**2 + (y_c-(ny-1)/2)**2)**0.5
        radii.append(r)
        
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

    radii.append(0)
    aperture = CircularAperture((83,83), r = ap_radius)
    flux, flux_err = aperture.do_photometry(image)
    fluxes.append(flux)

    plt.plot(radii, fluxes, marker='o', markersize=3, ls=' ')
    plt.title(f'{band} RAP test')
    plt.xlabel('Distance from central object')
    plt.ylabel('(Stacked) flux')
    #plt.show()
    plt.savefig(f"Stacking/Masked inverse variance stacking/Black spot checks/Random aperture placement plots/{band}_RAPT.png", bbox_inches="tight")
    plt.close()