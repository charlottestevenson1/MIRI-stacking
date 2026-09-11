from astropy.io import fits
import os

# Ensure that the 'filter_objects' folder exists:
os.makedirs("final/filter_objects", exist_ok=True)

# Bands in goods-s (35)
with open('final/filter_lists/filter_list_s.txt') as f:
    bands_s = [band.strip() for band in f.readlines()]

# Bands in goods-n (27)
with open('final/filter_lists/filter_list_n.txt') as f:
    bands_n = [band.strip() for band in f.readlines()]

# Bands in MIRI HDU (8)
with open('final/filter_lists/filter_list_miri.txt') as f:
    bands_miri = [band.strip() for band in f.readlines()]

# IDs of objects included in Hainline paper (2081 objects)
with open('final/01_build_filter_objects/hainline_galaxy_ids.txt') as f:
    hainline_ids = [int(id) for id in f.readlines()]

# bands_n is a subset of bands_s, so we can just iterate through bands_s
for band in bands_s:

    # The final list to be written to the file
    final_list = []

    for j in ['goods-s', 'goods-n']:

        # Open the right catalog for n/s
        with fits.open('final/fits_files/'+j+' catalog.fits') as hdul:

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
    with open(f'final/filter_objects/{band.lower()}_objects.txt', 'w') as f:
        for line in final_list:
            f.write(str(line)+'\n')
    
    print(f"{band} completed: {len(final_list)} objects")


# Generate list of objects with data in all MIRI bands

# MIRI bands
with open('final/filter_lists/filter_list_miri.txt') as f:
    bands_miri = [band.strip() for band in f.readlines()]

# Setting up 2D array: 1 row for each wide filter 
objects = [0 for i in range(len(bands_miri))]

# For each band, put the list of objects which appear in that band in the corresponding 2D array entry
for i in range(len(bands_miri)):
    objects[i] = [int(ID) for ID in open(f'final/filter_objects/{bands_miri[i].lower()}_objects.txt').readlines()]

# To get objects in all miri bands:
final_list = [ID for ID in hainline_ids if all(ID in sublist for sublist in objects)]

with open('final/filter_objects/all_miri.txt', 'w') as f:
    for line in final_list:
        f.write(str(line)+'\n')