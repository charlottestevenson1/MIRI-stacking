import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
})

# Includes HST data - can remove
BANDS = [BAND.strip() for BAND in open('Filter lists/filter list s').readlines()[:26]]+['detection']

IDs = [int(id) for id in open('Stacking/Objects for stacking.txt').readlines()]

def plot_SED(ID):
    ax_sed = fig.add_subplot(bottom[0, 1])
    ax_sed.set_yscale("log")
    ax_sed.set_xlabel("Pivot wavelength (microns)")
    ax_sed.set_ylabel("Flux (nJy)")

    # Important constants
    colors = ['b', 'y']
    filternames = ['nircam', 'miri']
    snrlim = 2      # SNR limit for when we use flux vs error
    id = ID     # ID of the galaxy we want to plot the SED for

    #Opening HDUL file and extracting data
    filename = 'FITS files/jades_small.fits'
    hdul = fits.open(filename)
    ndata = hdul[2].data
    ndata_line = ndata[ndata['ID'] == id][0]
    mdata = hdul[3].data
    mdata_line = mdata[mdata['ID'] == id][0]

    # Separating filters by instrument
    nircam_filters = ['F070W', 'F090W', 'F115W', 'F150W', 'F162M', 'F182M', 'F200W', 'F210M', 'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F410M', 'F430M', 'F444W', 'F460M', 'F480M']
    miri_filters = ['F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
    
    # Might want to take out acs and wfc3?
    filters = [nircam_filters, miri_filters]

    # Importing pre-calculated pivot wavelengths from file, and converting to microns
    pivotlist = [float(line.strip())/1e4 for line in open('Visual inspection figures/SED plotting code/pivot wave calculation/pivot_waves.txt', 'r').readlines()]

    # Constructing lists of fluxes for each instrument
    nircam_fluxes = []
    miri_fluxes = []

    nircam_uplim = []
    miri_uplim = []

    nircam_errors = []
    miri_errors = []

    # For each instrument we go through each filter, adding either the flux or the uncertainty to fluxlist (depending on SNR).
    for band in nircam_filters:
        flux = ndata_line[band+'_CIRC0']
        unc = ndata_line[band+'_CIRC0_e']
        nircam_errors.append(unc)
        if flux == 0:
            nircam_fluxes.append(0)
            nircam_uplim.append(False)

        elif flux/unc > snrlim:
            nircam_fluxes.append(flux)
            nircam_uplim.append(False)

        else:
            nircam_fluxes.append(snrlim*unc)
            nircam_uplim.append(True)

    for band in miri_filters:
        flux = mdata_line[band+'_CIRC0']
        unc = mdata_line[band+'_CIRC0_e']
        miri_errors.append(unc)
        if flux == 0:
            miri_fluxes.append(0)
            miri_uplim.append(False)
        elif flux/unc > snrlim:
            miri_fluxes.append(flux)
            miri_uplim.append(False)
        else:
            miri_fluxes.append(snrlim*unc)
            miri_uplim.append(True)

    # Combine all fluxes into a single list
    filterlist = nircam_filters + miri_filters
    fluxlist = nircam_fluxes + miri_fluxes
    errorlist = nircam_errors + miri_errors
    uplimlist = nircam_uplim + miri_uplim
    nircam_pivots = pivotlist[0:18]
    miri_pivots = pivotlist[18:26]

    # Iterating through the selected instrument flux lists, we select the flux and pivot values for non-zero and non-negative fluxes and plot them on the same graph in different colours

    #NIRCam:
    nircam_uplim = np.array(nircam_uplim)
    ax_sed.errorbar(np.array(nircam_pivots[:18])[~nircam_uplim], np.array(nircam_fluxes)[~nircam_uplim], yerr=np.array(nircam_errors)[~nircam_uplim], capsize = 2, marker = 'o', markersize = 3, ls = ' ', color = 'b', label = 'NIRCam')
    ax_sed.errorbar(np.array(nircam_pivots[:18])[nircam_uplim], np.array(nircam_fluxes)[nircam_uplim], yerr=np.array(nircam_errors)[nircam_uplim], uplims=True, capsize = 2, marker = 'o', markersize = 3, ls = ' ', color = 'b')
    
    #MIRI:
    miri_uplim = np.array(miri_uplim)
    ax_sed.errorbar(np.array(miri_pivots[:18])[~miri_uplim], np.array(miri_fluxes)[~miri_uplim], yerr=np.array(miri_errors)[~miri_uplim], capsize = 2, marker = 'o', markersize = 3, ls = ' ', color = 'r', label = 'MIRI')
    ax_sed.errorbar(np.array(miri_pivots[:18])[miri_uplim], np.array(miri_fluxes)[miri_uplim], yerr=np.array(miri_errors)[miri_uplim], uplims=True, capsize = 2, marker = 'o', markersize = 3, ls = ' ', color = 'r')
    
    ax_sed.legend()

for ID in IDs:
    fig = plt.figure(figsize=(10,7), facecolor="#EBE5D9")
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.1)
    top = outer[0].subgridspec(3, 9, wspace=0.05, hspace=0.3)
    bottom = outer[1].subgridspec(1,3, width_ratios=[1,6,1])
    fig.suptitle(f'Galaxy ID: {ID}', fontsize = 15, fontweight="bold") 

    for i in range(27):
        # Finding location in grid
        x = i % 9
        y = i // 9
        BAND = BANDS[i]
        ax = fig.add_subplot(top[y, x])
        ax.set_axis_off()
        ax.set_box_aspect(1)

        if str(ID) in [ID.strip() for ID in open(f'Filter objects/{BAND} objects.txt')]:
            hdul = fits.open(f'Cutouts/SCI/{ID}_{BAND}.fits')
            bigdata = hdul[0].data
            nx, ny = bigdata.shape
            lim = int(np.floor(nx*0.3))
            data = bigdata[lim:(nx-lim), lim:(nx-lim)]

            #put back in for normal
            vmin, vmax = ZScaleInterval().get_limits(data)
            ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap='gray_r')
            ax.set_title(BAND, fontsize = 8)

            # nx, ny = data.shape

            # circle = Circle(
            # (nx/2, ny/2),
            # 0.1*nx,
            # edgecolor = 'red',
            # facecolor = 'none',
            # linewidth = 1
            # )

            # ax.add_patch(circle)

    plot_SED(ID)
    fig.canvas.draw()
    fig.savefig(f'Visual inspection figures/2" images/{ID}_figure.pdf', bbox_inches='tight', pad_inches=0.02)

    print(f"{ID} completed.")
    plt.close()