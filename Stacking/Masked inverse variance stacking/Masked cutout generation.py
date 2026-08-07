### MASKED CUTOUT GENERATION ###

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

# Get band list
with open('Stacking/Masked inverse variance stacking/Bands to stack.txt') as f:
    bands = [band.strip() for band in f.readlines()]

# Get ID list
with open('Stacking/Masked inverse variance stacking/Objects to stack.txt') as f:
    all_IDs = [int(ID) for ID in f.readlines()]

# Loop through bands
for i in range(26):
    band = bands[i]
    print(f'BAND: {band}------------------------------')
    
    # Check valid IDs
    with open(f"Filter objects/{band} objects.txt") as f:
        band_IDs = [int(ID) for ID in f.readlines()]

    IDs = [ID for ID in all_IDs if ID in band_IDs]

    for ID in IDs:
        # Load image and segmentation files
        image = fits.getdata(f'Cutouts/SCI/{ID}_{band}.fits')
        seg = fits.getdata(f'Cutouts/SCI/{ID}_segmentation.fits')
        # plt.imshow(image)
        # plt.show()

        # If we're in a MIRI band, we need to make the mask greedier and then collapse it before mapping it.
        if i > 17:
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
            f"Stacking/Masked inverse variance stacking/Masked cutouts/SCI/{ID}_{band}_MASKED.fits",
            masked_image,
            overwrite = True
        )
    
        print(f"{ID} completed.")