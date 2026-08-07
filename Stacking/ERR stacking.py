import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

stacking_galaxy_IDs = [int(i) for i in open('Stacking/Objects for stacking.txt').readlines()]

bands = [
    'F070W', 'F090W', 'F115W', 'F150W', 'F162M', 'F182M', 'F200W', 'F210M',
    'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F410M', 'F430M', 'F444W',
    'F460M', 'F480M', 'F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W',
    'F1800W', 'F2100W', 'F2550W'
]

for band in bands:
    band_galaxy_IDs = [int(i) for i in open(f'Filter objects/{band} objects.txt')]

    galaxy_IDs = [ID for ID in stacking_galaxy_IDs if ID in band_galaxy_IDs]
    n_galaxies = min(370, len(galaxy_IDs))
    galaxy_IDs = galaxy_IDs[:n_galaxies]

    SCI_array = np.stack([
        fits.getdata(f'Cutouts/SCI/{ID}_{band}.fits')
        for ID in galaxy_IDs
    ])

    ERR_array = np.stack([
        fits.getdata(f'Cutouts/ERR/{ID}_{band}_ERR.fits')
        for ID in galaxy_IDs
    ])

    # Valid pixels only: finite science, finite error, positive error
    valid = np.isfinite(SCI_array) & np.isfinite(ERR_array) & (ERR_array > 0)

    weights = np.zeros_like(ERR_array)
    weights[valid] = 1.0 / (ERR_array[valid] ** 2)

    numerator = np.sum(np.where(valid, SCI_array * weights, 0.0), axis=0)
    denominator = np.sum(weights, axis=0)

    stack = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0
    )

    stack_err = np.divide(
        1.0,
        np.sqrt(denominator),
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > 0
    )

    #fits.writeto(f'Stacking/{band}_stack.fits', stack, overwrite = True)
    fits.writeto(f'Stacking/{band}_stack_ERR.fits', stack_err, overwrite=True)

    print(f'Saved {band}.')