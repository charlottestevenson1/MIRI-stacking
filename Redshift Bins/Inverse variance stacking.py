import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

zlo = 11
zup = 12

# Get band list
with open('Stacking/Masked inverse variance stacking/Bands to stack.txt') as f:
    bands = [band.strip() for band in f.readlines()]

# Get ID list
with open(f'Redshift Bins/Bin objects/Redshifts {zlo}-{zup}.txt') as f:
    all_IDs = [int(ID) for ID in f.readlines()]

for i in range(len(bands)):
    band = bands[i]

    with open(f'Filter objects/{band} objects.txt') as f:
        band_IDs = [int(i) for i in f.readlines()]
        IDs = [ID for ID in all_IDs if ID in band_IDs]
    
    try:
        SCI_array = np.stack([
            fits.getdata(f'Stacking/Masked inverse variance stacking/Masked cutouts/SCI/{ID}_{band}_MASKED.fits')
            for ID in IDs
        ])

        ERR_array = np.stack([
            fits.getdata(f'Cutouts/ERR/{ID}_{band}_ERR.fits')
            for ID in IDs
        ])

        # Valid pixels only: just admits pixels that are not nan
        valid = np.isfinite(SCI_array) & np.isfinite(ERR_array) & (ERR_array > 0)

        weights = np.zeros_like(ERR_array, dtype=float)
        weights[valid] = 1.0 / (ERR_array[valid] ** 2)

        numerator = np.nansum(np.where(valid, SCI_array * weights, 0.0), axis=0)
        denominator = np.nansum(weights, axis=0)

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

        fits.writeto(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/SCI/{band}_stack.fits', stack, overwrite = True)
        fits.writeto(f'Redshift Bins/Stacks/Redshifts {zlo}-{zup}/ERR/{band}_stack_ERR.fits', stack_err, overwrite=True)

        print(f'Saved {band}.')

    except:
        print(f'No valid objects for band {band}.')
        continue

print(f'{len(IDs)} stacked.')