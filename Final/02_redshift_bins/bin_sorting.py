from astropy.io import fits
import numpy as np
import os

os.makedirs('final/02_redshift_bins/bin_objects', exist_ok=True)

with fits.open('final/fits_files/JADES_DR5_z_gt_8_Catalog_Hainline.fits') as hdul:
    props = hdul[1].data

with open('final/filter_objects/all_miri.txt') as f:
    ids = [int(id) for id in f.readlines()]

# Redshifts: z_spec or EAZY_z_a. Match each redshift explicitly to its ID
# rather than relying on catalogue order matching the order in ALL MIRI.txt.
redshifts_by_id = {
    int(row['ID']): row['EAZY_z_a']
    for row in props
    if int(row['ID']) in ids
}

# The bin boundaries
z_ranges = np.loadtxt('final/redshift_bins.txt', dtype=float)

for z_range in z_ranges:
    z_lo = z_range[0]
    z_up = z_range[1]
    bin_ids = [
        f'{id}\n'
        for id in ids
        if id in redshifts_by_id
        and z_lo <= redshifts_by_id[id] < z_up
    ]
    print(z_lo, len(bin_ids))
    with open(f'final/02_redshift_bins/bin_objects/redshifts_{z_lo}_{z_up}.txt', 'w') as f:
        f.writelines(bin_ids)