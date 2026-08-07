from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np

# Important constants
colors = ['r', 'g', 'b', 'y']
filternames = ['nircam', 'miri', 'acs', 'wfc3']
snrlim = 2      # SNR limit for when we use flux vs error
id = 398355     # ID of the galaxy we want to plot the SED for

#Opening HDUL file and extracting data
filename = 'FITS files/jades_small.fits'
hdul = fits.open(filename)
ndata = hdul[2].data
mdata = hdul[3].data

# Separating filters by instrument
nircam_filters = ['F070W', 'F090W', 'F115W', 'F150W', 'F162M', 'F182M', 'F200W', 'F210M', 'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F410M', 'F430M', 'F444W', 'F460M', 'F480M']
miri_filters = ['F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
acs_filters = ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']
wfc3_filters = ['F105W', 'F125W', 'F140W', 'F160W']

filters = [nircam_filters, miri_filters, acs_filters, wfc3_filters]

# Importing pre-calculated pivot wavelengths from file, and converting to microns
pivotlist = [float(line.strip())/1e4 for line in open('Visual inspection figures/SED plotting code/pivot wave calculation/pivot_waves.txt', 'r').readlines()]

# Constructing lists of fluxes for each instrument
nircam_fluxes = []
miri_fluxes = []
acs_fluxes = []
wfc3_fluxes = []

nircam_errors = []
miri_errors = []
acs_errors = []
wfc3_errors = []

# For each instrument we go through each filter, adding either the flux or the uncertainty to fluxlist (depending on SNR).
for band in nircam_filters:
    flux = ndata[ndata['ID'] == id][0][band+'_CIRC0']
    unc = ndata[ndata['ID'] == id][0][band+'_CIRC0_e']
    nircam_errors.append(unc)
    if flux == 0:
        nircam_fluxes.append(0)
    elif flux/unc > snrlim:
        nircam_fluxes.append(flux)
    else:
        nircam_fluxes.append(snrlim*unc)

for band in miri_filters:
    flux = mdata[mdata['ID'] == id][0][band+'_CIRC0']
    unc = mdata[mdata['ID'] == id][0][band+'_CIRC0_e']
    miri_errors.append(unc)
    if flux == 0:
        miri_fluxes.append(0)
    elif flux/unc > snrlim:
        miri_fluxes.append(flux)
    else:
        miri_fluxes.append(snrlim*unc)

for band in acs_filters:
    flux = ndata[ndata['ID'] == id][0][band+'_CIRC0']
    unc = ndata[ndata['ID'] == id][0][band+'_CIRC0_e']
    acs_errors.append(unc)
    if flux == 0:
        acs_fluxes.append(0)
    elif flux/unc > snrlim:
        acs_fluxes.append(flux)
    else:
        acs_fluxes.append(snrlim*unc)

for band in wfc3_filters:
    flux = ndata[ndata['ID'] == id][0][band+'_CIRC0']
    unc = ndata[ndata['ID'] == id][0][band+'_CIRC0_e']
    wfc3_errors.append(unc)
    if flux == 0:
        wfc3_fluxes.append(0)
    elif flux/unc > snrlim:
        wfc3_fluxes.append(flux)
    else:
        wfc3_fluxes.append(snrlim*unc)

# Combine all fluxes into a single list
filterlist = nircam_filters + miri_filters + acs_filters + wfc3_filters
fluxlist = nircam_fluxes + miri_fluxes + acs_fluxes + wfc3_fluxes
errorlist = nircam_errors + miri_errors + acs_errors + wfc3_errors

# Selecting which instruments we'd like to display, and getting the relevant info
sel = [0, 1, 2, 3]
sel_filters = [filters[i] for i in range(4) if i in sel]
sel_filter_names = [filternames[i] for i in range(4) if i in sel]

# Iterating through the selected instrument flux lists, we select the flux and pivot values for non-zero and non-negative fluxes and plot them on the same graph in different colours
for i in range(len(sel_filters)):
    indices = [j for j in range(len(sel_filters[i])) if fluxlist[filterlist.index(sel_filters[i][j])] not in [0]]
    fluxes = [fluxlist[filterlist.index(sel_filters[i][j])] for j in range(len(sel_filters[i])) if j in indices]
    errors = [errorlist[filterlist.index(sel_filters[i][j])] for j in range(len(sel_filters[i])) if j in indices]
    pivots = [pivotlist[filterlist.index(sel_filters[i][j])] for j in range(len(sel_filters[i])) if j in indices]
    print(fluxes)
    plt.errorbar(pivots, fluxes, yerr = errors, capsize = 2, marker = 'o', ls = ' ', color = colors[i], label = sel_filter_names[i])

# Displaying the graph
plt.yscale('log')
plt.legend()
plt.show()