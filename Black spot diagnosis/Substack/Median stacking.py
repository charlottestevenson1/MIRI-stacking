import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

go_galaxy_IDs = [int(i) for i in open('Stacking/Objects for stacking.txt').readlines()][:50]
stacking_galaxy_IDs = np.random.choice(go_galaxy_IDs, size = 200)

bands = [
    'F070W', 'F090W', 'F115W', 'F150W', 'F162M', 'F182M', 'F200W', 'F210M',
    'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F410M', 'F430M', 'F444W',
    'F460M', 'F480M', 'F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W',
    'F1800W', 'F2100W', 'F2550W'
]

for band in bands:
    band_galaxy_IDs = [int(i) for i in open(f'Filter objects/{band} objects.txt')]

    galaxy_IDs = [ID for ID in stacking_galaxy_IDs if ID in band_galaxy_IDs]
    n_galaxies = min(370, len(galaxy_IDs))
    galaxy_IDs = galaxy_IDs[:n_galaxies]

    try:
        SCI_array = np.stack([
            fits.getdata(f'Cutouts/SCI/{ID}_{band}.fits')
            for ID in galaxy_IDs
        ])
    except:
        SCI_array = np.full((167,167), np.nan)

    stack = np.nanmedian(SCI_array, axis=0)

    fits.writeto(f'Black spot diagnosis/Substack/Stacks/{band}_stack_MEDIAN_200.fits', stack, overwrite = True)

    print(f'Saved {band}.')