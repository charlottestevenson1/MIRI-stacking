from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdul = fits.open('jades_small.fits')

with open('Photometry/Pivot wavelengths.txt') as f:
    pivot_waves = np.array([float(wl) for wl in f.readlines()])

with open('Photometry/Band data/Aperture corrections.txt') as f:
    ACs = np.array([float(wl) for wl in f.readlines()])

band_medians = []

bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M", "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

z_ranges = [(8,9), (9,10), (10,11), (11,12), (12,15)]

IDs = hdul[2].data['ID']

z_range = (8,9)

zlo = z_range[0]
zup = z_range[1]

with open('Redshift Bins/Bin objects/Redshifts 8-9.txt') as f:
    bin_IDs = [int(ID) for ID in f.readlines()]

for band in bands[:18]:
    band_data = []
    data = hdul[2].data
    search_key = band+'_CIRC1'
    all_band_data = data[search_key]

    for ID in bin_IDs:
        ID_index = np.where(IDs == ID)
        band_data.append(all_band_data[ID_index][0])

    for i in range(len(band_data)):
        if band_data[i] == 0:
            band_data[i] = np.nan
    
    band_medians.append(np.nanmedian(band_data))
    print(f'{band} done')

for band in bands[18:]:
    band_data = []
    data = hdul[3].data
    search_key = band+'_CIRC1'
    all_band_data = data[search_key]

    for ID in bin_IDs:
        ID_index = np.where(IDs == ID)
        band_data.append(all_band_data[ID_index])

    for i in range(len(band_data)):
            if band_data[i] == 0:
                band_data[i] = np.nan

    band_medians.append(np.nanmedian(band_data))
    print(f'{band} done')

plt.errorbar(
    np.array(pivot_waves)/1e4, # so that they are in units of microns
    np.asarray(band_medians)/ACs,
    ecolor = 'gray',
    markersize = 5,
    fmt = 'o',
    capsize = 1.5,
)

plt.title(f'Stack SED, z={zlo}-{zup}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Pivot wavelength (microns)')
plt.ylabel('Flux in 0.15" aperture (nJy)')

plt.show()