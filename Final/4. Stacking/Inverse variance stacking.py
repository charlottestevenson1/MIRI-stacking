import numpy as np
from astropy.io import fits
import os

# Ensure Final/Stacks folder exists
os.makedirs('Final/4. Stacking/Stacks', exist_ok = True)

# Redshift bin boundaries
zreds = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for (zlo, zup) in zreds:

    print(f'Redshift bin: z = {zlo} - {zup}\n')

    os.makedirs(f'Final/4. Stacking/Stacks/Redshifts {zlo}-{zup}', exist_ok = True)

    # Get band list
    with open('Final/Filter lists/filter list wide.txt') as f:
        bands = [band.strip() for band in f.readlines()]

    # Get ID list for the selected redshift bin
    with open(f'Final/2. Redshift bins/Bin objects/Redshifts {zlo}-{zup}.txt') as f:
        all_IDs = [int(ID) for ID in f.readlines()]

    # Iterate through bands
    for i in range(len(bands)):
        band = bands[i]

        # Find the IDs in the selected redshift bin which have data in the selected band
        with open(f'Final/Filter objects/{band} objects.txt') as f:
            band_IDs = [int(i) for i in f.readlines()]
            IDs = [ID for ID in all_IDs if ID in band_IDs]
        
        if not len(IDs) == 0:
            # Generate 3D tables of mosaics, where the third dimension is ID. This means we can easily stack later.
            SCI_array = np.stack([
                fits.getdata(f'Final/3. Generate cutouts/Masked cutouts/SCI/{ID}_{band}_MASKED.fits')
                for ID in IDs
            ])

            ERR_array = np.stack([
                fits.getdata(f'Final/3. Generate cutouts/Unmasked cutouts/ERR/{ID}_{band}_ERR.fits')
                for ID in IDs
            ])

            # Valid pixels only: just admits pixels that are not nan
            valid = np.isfinite(SCI_array) & np.isfinite(ERR_array) & (ERR_array > 0)

            # Inverse variance stacking: w = 1/s^2
            weights = np.zeros_like(ERR_array, dtype=float)
            weights[valid] = 1.0 / (ERR_array[valid] ** 2)

            # Ignore nan values and perform weighted sum over ID dimension, ignoring nan values
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

            os.makedirs(f'Final/4. Stacking/Stacks/Redshifts {zlo}-{zup}/SCI', exist_ok = True)
            os.makedirs(f'Final/4. Stacking/Stacks/Redshifts {zlo}-{zup}/ERR', exist_ok = True)
            fits.writeto(f'Final/4. Stacking/Stacks/Redshifts {zlo}-{zup}/SCI/{band}_stack.fits', stack, overwrite = True)
            fits.writeto(f'Final/4. Stacking/Stacks/Redshifts {zlo}-{zup}/ERR/{band}_stack_ERR.fits', stack_err, overwrite=True)

            print(f'Saved {band}: {len(IDs)} objects stacked.')

        else:
            print(f'No valid objects for band {band}.')
            continue
