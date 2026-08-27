from astropy.io import fits

hdul = fits.open('/data/cs2399/FITS files/goods-s catalog.fits')

ID = 37925

NIRCam_bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M"
]

MIRI_bands = [
    "F560W", "F770W", "F1000W", "F1280W", 
    "F1500W", "F1800W", "F2100W", "F2550W"
]

bands = NIRCam_bands + MIRI_bands

#print(repr(hdul[4].header[:100]))

NIR_data = hdul[4].data

MIRI_data = hdul[10].data


with open(f'MIRI checking/{ID} catalog fluxes.txt', 'w') as f:

    for band in NIRCam_bands:
        f.write(f"{band}: {NIR_data[NIR_data['ID']==ID][0][f'{band}_CIRC2']}\n")

    for band in MIRI_bands:
        f.write(f"{band}: {MIRI_data[MIRI_data['ID']==ID][0][f'{band}_CIRC2']}\n")

with open(f'MIRI checking/{ID} catalog errors.txt', 'w') as f:

    for band in NIRCam_bands:
        f.write(f"{band}: {NIR_data[NIR_data['ID']==ID][0][f'{band}_CIRC2_e']}\n")

    for band in MIRI_bands:
        f.write(f"{band}: {MIRI_data[MIRI_data['ID']==ID][0][f'{band}_CIRC2_e']}\n")