### WORKING OUT PIXEL DILATION ###

from astropy.io import fits
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Get band list
with open('Stacking/Masked inverse variance stacking/Bands to stack.txt') as f:
    bands = [band.strip() for band in f.readlines()]

# Get ID list
with open('Stacking/Masked inverse variance stacking/Objects to stack.txt') as f:
    IDs = [int(ID) for ID in f.readlines()]

# Number of loops
for i in range(10):
    # Choose random ID and band
    band = np.random.choice(bands)
    ID = np.random.choice(IDs)
    
    # Load image and segmentation map files
    try:
        image = fits.getdata(f'Cutouts/SCI/{ID}_{band}.fits')

    except:
        continue
    
    seg = fits.getdata(f'Cutouts/SCI/{ID}_segmentation.fits')

    if image.shape != seg.shape:
        print("Shape mismatch:", image.shape, seg.shape)
        continue

    # Generate mask
    mask = (seg == 0) | (seg == ID)

    masked_image = np.where(mask, image, np.nan)
    plt.imshow(masked_image, origin='lower')
    plt.show()