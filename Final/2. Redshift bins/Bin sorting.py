from astropy.io import fits
import numpy as np
import os

os.makedirs('Final/2. Redshift bins/Bin objects', exist_ok=True)

with fits.open('Final/FITS files/JADES_DR5_z_gt_8_Catalog_Hainline.fits') as hdul:
    props = hdul[1].data

with open('Final/Filter objects/ALL MIRI.txt') as f:
    IDs = [int(ID) for ID in f.readlines()]

# Redshifts: z_spec or EAZY_z_a. Match each redshift explicitly to its ID
# rather than relying on catalogue order matching the order in ALL MIRI.txt.
redshifts_by_id = {
    int(row['ID']): row['EAZY_z_a']
    for row in props
    if int(row['ID']) in IDs
}

# The bin boundaries
z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

for z_range in z_ranges:
    zlo = z_range[0]
    zup = z_range[1]
    bin_IDs = [
        f'{ID}\n'
        for ID in IDs
        if ID in redshifts_by_id
        and zlo <= redshifts_by_id[ID] < zup
    ]
    print(zlo, len(bin_IDs))
    with open(f'Final/2. Redshift bins/Bin objects/Redshifts {zlo}-{zup}.txt', 'w') as f:
        f.writelines(bin_IDs)