from astropy.io import fits
import os

# Ensure that the 'Filter objects' folder exists:
os.makedirs("Final/Filter objects", exist_ok=True)

# Bands in goods-s (35)
with open('Final/Filter lists/filter list s.txt') as f:
    bands_s = [band.strip() for band in f.readlines()]

# Bands in goods-n (27)
with open('Final/Filter lists/filter list n.txt') as f:
    bands_n = [band.strip() for band in f.readlines()]

# Bands in MIRI HDU (8)
with open('Final/Filter lists/filter list miri.txt') as f:
    bands_miri = [band.strip() for band in f.readlines()]

# IDs of objects included in Hainline paper (2081 objects)
with open('Final/1. Build Filter objects folder/Hainline galaxy IDs.txt') as f:
    hainline_ids = [int(id) for id in f.readlines()]

# bands_n is a subset of bands_s, so we can just iterate through bands_s
for band in bands_s:

    # The final list to be written to the file
    final_list = []

    for j in ['goods-s', 'goods-n']:

        # Open the right catalog for n/s
        hdul = fits.open('Final/FITS files/'+j+' catalog.fits')

        # Pick the HDU that the band data is stored in
        if band in bands_miri:
            data = hdul[10].data # MIRI HDU
        else:
            data = hdul[4].data # NIRCam HDU

        # Select indices with non-zero flux entries for band in question - but skip bands which are not in GOODS-N if that's the GOODS file we are looking at
        if (j == 'goods-s') or (band in bands_n):
            fluxes = data[band+'_CIRC0']

        else:
            fluxes = []
        
        # Indices of non-zero flux entries
        non_zero_indices = [i for i in range(len(fluxes)) if fluxes[i]!=0]

        # Indices of Hainline galaxies in main file
        all_ids = data['ID'].tolist()
        hainline_indices = [i for i in range(len(all_ids)) if all_ids[i] in hainline_ids]

        # Intersection of these lists gives Hainline objects with non-zero flux values
        non_zero_hainline_indices = [i for i in non_zero_indices if i in hainline_indices]

        # Find corresponding IDs and add to list
        non_zero_hainline_objects = [all_ids[i] for i in non_zero_hainline_indices]
        final_list += non_zero_hainline_objects

    # Write final_list to a file in the filter objects folder
    with open('Final/Filter objects/'+band+' objects.txt', 'w') as f:
        for line in final_list:
            f.write(str(line)+'\n')
    
    print(f"{band} completed: {len(final_list)} objects")


# Generate list of objects with data in all MIRI bands

# MIRI bands
with open('Final/Filter lists/filter list miri.txt') as f:
    bands_miri = [band.strip() for band in f.readlines()]

# Setting up 2D array: 1 row for each wide filter 
objects = [0 for i in range(len(bands_miri))]

# For each band, put the list of objects which appear in that band in the corresponding 2D array entry
for i in range(len(bands_miri)):
    objects[i] = [int(ID) for ID in open(f'Final/Filter objects/{bands_miri[i]} objects.txt').readlines()]

# To get objects in all miri bands:
final_list = [ID for ID in hainline_ids if all(ID in sublist for sublist in objects)]

with open('Final/Filter objects/ALL MIRI.txt', 'w') as f:
    for line in final_list:
        f.write(str(line)+'\n')