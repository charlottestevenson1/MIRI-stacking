from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdul = fits.open('FITS files/goods-s catalog.fits')

zlo = 8
zup = 9

with open(f'Redshift Bins/Bin objects/Redshifts {zlo}-{zup}.txt') as f:
    IDs = [int(ID) for ID in f.readlines()]

data = hdul[4].data
all_IDs = data['ID']
ID_indices = [i for i in range(len(all_IDs)) if all_IDs[i] in IDs]

NIRCam_bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M"
]

MIRI_bands = [
    "F560W", "F770W", "F1000W", "F1280W", 
    "F1500W", "F1800W", "F2100W", "F2550W"
]

flux_array = []
error_array = []

bands = NIRCam_bands + MIRI_bands

for band in bands:

    if band in NIRCam_bands:
        data = hdul[4].data
    elif band in MIRI_bands:
        data = hdul[10].data

    band_fluxes = [data[f'{band}_CIRC2']]
    band_errors = [data[f'{band}_CIRC2_e']]

    fluxes = [band_fluxes[0][i] for i in range(len(band_fluxes[0])) if i in ID_indices]
    errors = [band_errors[0][i] for i in range(len(band_errors[0])) if i in ID_indices]

    flux_array.append(fluxes)
    error_array.append(errors)

flux_array = np.array(flux_array)
flux_array[flux_array==0] = np.nan

error_array = np.array(error_array)

stacked_fluxes = []
stacked_errors = []

for i in range(len(flux_array)):
    fluxes = flux_array[i]
    errors = error_array[i]

    mask = np.isfinite(fluxes) & np.isfinite(errors) & (errors > 0)

    fluxes = fluxes[mask]
    errors = errors[mask]
    
    weights = 1/errors**2

    weighted_mean = np.average(fluxes, weights=weights)
    weighted_error = np.sqrt(1 / np.sum(weights))

    stacked_fluxes.append(weighted_mean)
    stacked_errors.append(weighted_error)

with open('Photometry/Pivot wavelengths.txt') as f:
    pivot_waves = np.array([float(wl) for wl in f.readlines()])
    pivot_waves = [pivot_waves[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

with open('Prospector/Stack data/8-9 Fluxes.txt') as f:
    mea_fluxes = [float(i) for i in f.readlines()]
    mea_fluxes = [mea_fluxes[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

with open('Prospector/Stack data/8-9 Errors.txt') as f:
    mea_errors = [float(i) for i in f.readlines()]
    mea_errors = [mea_errors[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

cat_fluxes = stacked_fluxes
cat_fluxes = [cat_fluxes[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

cat_errors = stacked_errors
cat_errors = [cat_errors[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

plt.figure(figsize=(10, 6))
plt.ylim(1, 200)
plt.yscale('log')

plt.errorbar(
    pivot_waves,
    cat_fluxes,
    yerr=cat_errors,
    fmt="o",
    markersize=7,
    capsize=3,
    color="steelblue",
    label='catalogue values'
)

plt.errorbar(
    pivot_waves,
    mea_fluxes,
    yerr=mea_errors,
    fmt="o",
    markersize=7,
    capsize=3,
    color="hotpink",
    label='measured values'
)

plt.xscale("log")
plt.xlabel("Wavelength [Å]")
plt.ylabel("Flux density")
plt.legend()

plt.tight_layout()
plt.show()