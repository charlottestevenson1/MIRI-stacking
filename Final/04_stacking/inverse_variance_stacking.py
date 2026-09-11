import numpy as np
from astropy.io import fits
import os

# Ensure final/stacks folder exists
os.makedirs('final/04_stacking/stacks', exist_ok = True)

# Redshift bin boundaries
zreds = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for (z_lo, z_up) in zreds:

    print(f'Redshift bin: z = {z_lo} - {z_up}\n')

    os.makedirs(f'final/04_stacking/stacks/Redshifts {z_lo}-{z_up}', exist_ok = True)

    # Get band list
    with open('final/filter_lists/filter_list_wide.txt') as f:
        bands = [band.strip() for band in f.readlines()]

    # Get ID list for the selected redshift bin
    with open(f'final/02_redshift_bins/bin_objects/redshifts_{z_lo}_{z_up}.txt') as f:
        all_IDs = [int(ID) for ID in f.readlines()]

    # Iterate through bands
    for i in range(len(bands)):
        band = bands[i]

        # Find the IDs in the selected redshift bin which have data in the selected band
        with open(f'final/filter_objects/{band.lower()}_objects.txt') as f:
            band_IDs = [int(i) for i in f.readlines()]
            IDs = [ID for ID in all_IDs if ID in band_IDs]
        
        if not len(IDs) == 0:
            # Generate 3D tables of mosaics, where the third dimension is ID. This means we can easily stack later.
            sci_array = np.stack([
                fits.getdata(f'final/03_generate_cutouts/masked_cutouts/sci/{ID}_{band}_masked.fits')
                for ID in IDs
            ])

            err_array = np.stack([
                fits.getdata(f'final/03_generate_cutouts/unmasked_cutouts/err/{ID}_{band}_err.fits')
                for ID in IDs
            ])

            # Valid pixels only: just admits pixels that are not nan
            valid = np.isfinite(sci_array) & np.isfinite(err_array) & (err_array > 0)

            # Inverse variance stacking: w = 1/s^2
            weights = np.zeros_like(err_array, dtype=float)
            weights[valid] = 1.0 / (err_array[valid] ** 2)

            # Ignore nan values and perform weighted sum over ID dimension, ignoring nan values
            numerator = np.nansum(np.where(valid, sci_array * weights, 0.0), axis=0)
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

            os.makedirs(f'final/04_stacking/stacks/Redshifts {z_lo}-{z_up}/sci', exist_ok = True)
            os.makedirs(f'final/04_stacking/stacks/Redshifts {z_lo}-{z_up}/err', exist_ok = True)
            fits.writeto(f'final/04_stacking/stacks/Redshifts {z_lo}-{z_up}/sci/{band}_stack.fits', stack, overwrite = True)
            fits.writeto(f'final/04_stacking/stacks/Redshifts {z_lo}-{z_up}/err/{band}_stack_err.fits', stack_err, overwrite=True)

            print(f'Saved {band}: {len(IDs)} objects stacked.')

        else:
            print(f'No valid objects for band {band}.')
            continue
