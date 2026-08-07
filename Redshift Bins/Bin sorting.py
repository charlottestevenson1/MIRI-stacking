from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

hdul = fits.open('FITS files/JADES_DR5_z_gt_8_Catalog_Hainline.fits')
props = hdul[1].data

with open('Stacking/Objects for stacking.txt') as f:
    IDs = [int(ID) for ID in f.readlines()]

# Redshifts: z_spec or EAZY_z_a
redshifts = props[np.isin(props['ID'], IDs)]['EAZY_z_a']

z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for z_range in z_ranges:
    zlo = z_range[0]
    zup = z_range[1]
    bin_IDs = [str(IDs[j])+'\n' for j in range(len(IDs)) if (redshifts[j]>=zlo and redshifts[j]<zup)]
    print(zlo, len(bin_IDs))
    with open(f'Redshift Bins/Bin objects/Redshifts {zlo}-{zup}.txt', 'w') as f:
        f.writelines(bin_IDs)