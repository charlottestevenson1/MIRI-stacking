from astropy.io import fits

IDs = [int(id) for id in open('Filter objects/ALL MIRI.txt').readlines()]

hdul = fits.open('FITS files/goods-s catalog.fits')
data = hdul[4].data

final_list = []

for ID in IDs:
    regions = []

    RA = data[data['ID']==ID][0]['RA']
    DEC = data[data['ID']==ID][0]['DEC']

    if (53.01802 < RA) and (RA < 53.10729) and (-27.91344 < DEC) and (DEC < -27.83641):
        regions.append('deep')
    
    if (53.04812 < RA) and (RA < 53.09009) and (-27.91458 < DEC) and (DEC < -27.86637):
        regions.append('medium 1')

    if (52.97339 < RA) and (RA < 53.01576) and (-27.88316 < DEC) and (DEC < -27.83611):
        regions.append('medium 2')
    
    if (53.00319 < RA) and (RA < 53.04286) and (-27.81665 < DEC) and (DEC < -27.78126):
        regions.append('medium 3')
    
    if (52.94715 < RA) and (RA < 52.98718) and (-27.72036 < DEC) and (DEC < -27.68480):
        regions.append('medium 4')

    if (53.04094 < RA) and (RA < 53.09496) and (-27.70825 < DEC) and (DEC < -27.67440):
        regions.append('medium 5')
    
    if (53.08176 < RA) and (RA < 53.13630) and (-27.68821 < DEC) and (DEC < -27.65218):
        regions.append('medium 6')

    if (53.07031 < RA) and (RA < 53.23127) and (-27.88787 < DEC) and (DEC < -27.72866):
        regions.append('smiles')

    final_list.append(f"{str(ID)} {' '.join(regions)}")
    if len(regions) != 1:
        print(ID)

with open('MIRI region list.txt', 'w') as f:
    for line in final_list:
        f.write(line+'\n')