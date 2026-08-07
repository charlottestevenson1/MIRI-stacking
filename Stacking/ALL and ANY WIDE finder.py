from astropy.io import fits

# Bands in MIRI HDU (8)
with open('Filter lists/filter list wide') as f:
    bands_wide = [band.strip() for band in f.readlines()]

with open('Stacking/Objects for stacking.txt') as f:
    stacking_ids = [int(id) for id in f.readlines()]

#f560w = open('Filter objects/f560w objects.txt')

objects = [0 for i in range(16)]

for i in range(len(bands_wide)):
    objects[i] = [int(ID) for ID in open(f'Filter objects/{bands_wide[i]} objects.txt').readlines()]

# To get objects in all MIRI bands:
final_list = [ID for ID in stacking_ids if all(ID in sublist for sublist in objects)]

with open('Filter objects/ALL WIDE.txt', 'w') as f:
    for line in final_list:
        f.write(str(line)+'\n')

# # To get objects in at least one MIRI band:
# final_list = [ID for ID in hainline_ids if any(ID in sublist for sublist in objects)]

# with open('Filter objects/ANY MIRI.py', 'w') as f:
#     for line in final_list:
#         f.write(str(line)+'\n')