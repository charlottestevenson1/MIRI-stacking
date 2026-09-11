from astropy.io import fits
import os
import numpy as np

# Ensure that the 'filter_objects' folder exists:
os.makedirs("final/filter_objects", exist_ok=True)

# Bands in goods-s (35)
bands_s = np.loadtxt('final/filter_lists/filter_list_s.txt', dtype=str)

# Bands in goods-n (27)
bands_n = np.loadtxt('final/filter_lists/filter_list_n.txt', dtype=str)

# Bands in MIRI HDU (8)
bands_miri = np.loadtxt('final/filter_lists/filter_list_miri.txt', dtype=str)

# IDs of objects included in Hainline paper (2081 objects)
hainline_ids = np.loadtxt('final/01_build_filter_objects/hainline_galaxy_ids.txt', dtype=int)

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
    np.savetxt(f'final/filter_objects/{band.lower()}_objects.txt', final_list, fmt='%d')
    
    print(f"{band} completed: {len(final_list)} objects")


# Generate list of objects with data in all MIRI bands

# Setting up 2D array: 1 row for each wide filter 
objects = [0 for i in range(len(bands_miri))]

# For each band, put the list of objects which appear in that band in the corresponding 2D array entry
for i in range(len(bands_miri)):
    objects[i] = [int(ID) for ID in open(f'final/filter_objects/{bands_miri[i].lower()}_objects.txt').readlines()]

# To get objects in all miri bands:
final_list = [ID for ID in hainline_ids if all(ID in sublist for sublist in objects)]

np.savetxt('final/filter_objects/all_miri.txt', final_list, fmt='%d')