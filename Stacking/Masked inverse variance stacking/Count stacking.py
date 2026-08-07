import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Get band list
with open('Stacking/Masked inverse variance stacking/Bands to stack.txt') as f:
    bands = [band.strip() for band in f.readlines()]

# Get ID list
with open('Stacking/Masked inverse variance stacking/Objects to stack.txt') as f:
    all_IDs = [int(ID) for ID in f.readlines()]

for i in range(len(bands)):
    band = bands[i]

    with open(f'Filter objects/{band} objects.txt') as f:
        band_IDs = [int(i) for i in f.readlines()]
        IDs = [ID for ID in all_IDs if ID in band_IDs]
    

    SCI_array = np.stack([
        fits.getdata(f'Stacking/Masked inverse variance stacking/Masked cutouts/SCI/{ID}_{band}_MASKED.fits')
        for ID in IDs
    ])
    
    # Valid pixels only: just admits pixels that are not nan
    valid = np.isfinite(SCI_array)

    count = valid.sum(axis=0).astype(int)

    plt.imshow(count, origin="lower", vmin=0, vmax=count.max())
    plt.colorbar()
    plt.show()

    fits.writeto(f'Stacking/Masked inverse variance stacking/Stacked images/Counts/{band}_counts.fits', count, overwrite = True)

    print(f'Saved {band}.')