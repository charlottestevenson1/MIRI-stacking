### MASKED CUTOUT GENERATION ###

from astropy.io import fits
import numpy as np
from scipy.ndimage import binary_dilation
import os

os.makedirs('final/03_generate_cutouts/masked_cutouts', exist_ok = True)

# Get band list
with open('final/filter_lists/filter_list_wide.txt') as f:
    bands = [band.strip() for band in f.readlines()]

# Get MIRI bands
with open('final/filter_lists/filter_list_miri.txt') as f:
    miri_bands = [band.strip() for band in f.readlines()]

# Get ID list
with open('final/filter_objects/all_miri.txt') as f:
    all_ids = [int(id) for id in f.readlines()]

# Loop through bands
for i in range(len(bands)):
    band = bands[i]
    print(f'BAND: {band}------------------------------')
    
    # Check valid IDs
    with open(f"final/filter_objects/{band.lower()}_objects.txt") as f:
        band_ids = [int(id) for id in f.readlines()]

    ids = [id for id in all_ids if id in band_ids]

    for id in ids:
        # Load image and segmentation files
        image = fits.getdata(f'final/03_generate_cutouts/unmasked_cutouts/sci/{id}_{band}.fits')
        seg = fits.getdata(f'final/03_generate_cutouts/unmasked_cutouts/sci/{id}_segmentation.fits')

        # If we're in a MIRI band, we need to make the mask greedier and then collapse it before mapping it.
        if band in miri_bands:
            mask_hi = (seg != 0) & (seg != id)
            mask_hi = binary_dilation(mask_hi, iterations=1)
            mask_hi = mask_hi[:166, :166]
            mask = mask_hi.reshape(83, 2, 83, 2).any(axis=(1, 3))

        # If we're in a NIRCam band, we don't need to.
        else:
            mask = (seg != 0) & (seg != id)
        
        masked_image = np.where(mask, np.nan, image)

        fits.writeto(
            f"final/03_generate_cutouts/masked_cutouts/sci/{id}_{band}_masked.fits",
            masked_image,
            overwrite = True
        )
    
        print(f"{id} completed.")