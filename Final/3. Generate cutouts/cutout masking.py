### MASKED CUTOUT GENERATION ###

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
import os

os.makedirs('Final/3. Generate cutouts/Masked cutouts', exist_ok = True)

# Get band list
with open('Final/Filter lists/filter list wide.txt') as f:
    bands = [band.strip() for band in f.readlines()]

# Get MIRI bands
with open('Final/Filter lists/filter list miri.txt') as f:
    MIRI_bands = [band.strip() for band in f.readlines()]

# Get ID list
with open('Final/Filter objects/ALL MIRI.txt') as f:
    all_IDs = [int(ID) for ID in f.readlines()]

# Loop through bands
for i in range(len(bands)):
    band = bands[i]
    print(f'BAND: {band}------------------------------')
    
    # Check valid IDs
    with open(f"Final/Filter objects/{band} objects.txt") as f:
        band_IDs = [int(ID) for ID in f.readlines()]

    IDs = [ID for ID in all_IDs if ID in band_IDs]

    for ID in IDs:
        # Load image and segmentation files
        image = fits.getdata(f'Final/3. Generate cutouts/Unmasked cutouts/SCI/{ID}_{band}.fits')
        seg = fits.getdata(f'Final/3. Generate cutouts/Unmasked cutouts/SCI/{ID}_segmentation.fits')
        # plt.imshow(image)
        # plt.show()

        # If we're in a MIRI band, we need to make the mask greedier and then collapse it before mapping it.
        if band in MIRI_bands:
            mask_hi = (seg != 0) & (seg != ID)
            mask_hi = binary_dilation(mask_hi, iterations=1)
            mask_hi = mask_hi[:166, :166]
            mask = mask_hi.reshape(83, 2, 83, 2).any(axis=(1, 3))

        # If we're in a NIRCam band, we don't need to.
        else:
            mask = (seg != 0) & (seg != ID)
        
        masked_image = np.where(mask, np.nan, image)
        # plt.imshow(masked_image)
        # plt.show()

        fits.writeto(
            f"Final/3. Generate cutouts/Masked cutouts/SCI/{ID}_{band}_MASKED.fits",
            masked_image,
            overwrite = True
        )
    
        print(f"{ID} completed.")