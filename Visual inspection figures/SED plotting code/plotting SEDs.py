from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np

# Important constants
colors = ['steelblue', 'crimson']
filternames = ['NIRCam', 'MIRI']
snrlim = -10000000      # SNR limit for when we use flux vs error
id = 73568     # ID of the galaxy we want to plot the SED for

#Opening HDUL file and extracting data
filename = 'FITS files/jades_small.fits'
hdul = fits.open(filename)
ndata = hdul[2].data
mdata = hdul[3].data

# Separating filters by instrument
nircam_filters = ['F070W', 'F090W', 'F115W', 'F150W', 'F162M', 'F182M', 'F200W', 'F210M', 'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F410M', 'F430M', 'F444W', 'F460M', 'F480M']
miri_filters = ['F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
# acs_filters = ['F435W', 'F606W', 'F775W', 'F814W', 'F850LP']
# wfc3_filters = ['F105W', 'F125W', 'F140W', 'F160W']

filters = [nircam_filters, miri_filters]#, acs_filters, wfc3_filters][:2]

# Importing pre-calculated pivot wavelengths from file, and converting to microns
pivotlist = [float(line.strip())/1e4 for line in open('Visual inspection figures/SED plotting code/pivot wave calculation/pivot_waves.txt', 'r').readlines()]

# Constructing lists of fluxes for each instrument
nircam_fluxes = []
miri_fluxes = []
# acs_fluxes = []
# wfc3_fluxes = []

nircam_errors = []
miri_errors = []
# acs_errors = []
# wfc3_errors = []

# For each instrument we go through each filter, adding either the flux or the uncertainty to fluxlist (depending on SNR).
for band in nircam_filters:
    flux = ndata[ndata['ID'] == id][0][band+'_CIRC2']
    unc = ndata[ndata['ID'] == id][0][band+'_CIRC2_e']
    nircam_errors.append(unc)
    if flux == 0:
        nircam_fluxes.append(0)
    elif flux/unc > snrlim:
        nircam_fluxes.append(flux)
    else:
        nircam_fluxes.append(snrlim*unc)

for band in miri_filters:
    flux = mdata[mdata['ID'] == id][0][band+'_CIRC2']
    unc = mdata[mdata['ID'] == id][0][band+'_CIRC2_e']
    miri_errors.append(unc)
    if flux == 0:
        miri_fluxes.append(0)
    elif flux/unc > snrlim:
        miri_fluxes.append(flux)
    else:
        miri_fluxes.append(snrlim*unc)

# for band in acs_filters:
#     flux = ndata[ndata['ID'] == id][0][band+'_CIRC0']
#     unc = ndata[ndata['ID'] == id][0][band+'_CIRC0_e']
#     acs_errors.append(unc)
#     if flux == 0:
#         acs_fluxes.append(0)
#     elif flux/unc > snrlim:
#         acs_fluxes.append(flux)
#     else:
#         acs_fluxes.append(snrlim*unc)

# for band in wfc3_filters:
#     flux = ndata[ndata['ID'] == id][0][band+'_CIRC0']
#     unc = ndata[ndata['ID'] == id][0][band+'_CIRC0_e']
#     wfc3_errors.append(unc)
#     if flux == 0:
#         wfc3_fluxes.append(0)
#     elif flux/unc > snrlim:
#         wfc3_fluxes.append(flux)
#     else:
#         wfc3_fluxes.append(snrlim*unc)

# Combine all fluxes into a single list
filterlist = nircam_filters + miri_filters #+ acs_filters + wfc3_filters
fluxlist = nircam_fluxes + miri_fluxes #+ acs_fluxes + wfc3_fluxes
errorlist = nircam_errors + miri_errors #+ acs_errors + wfc3_errors

# Selecting which instruments we'd like to display, and getting the relevant info
sel = [0, 1, 2, 3]
sel_filters = [filters[i] for i in range(2) if i in sel]
sel_filter_names = [filternames[i] for i in range(2) if i in sel]

# Create transparent figure
fig, ax = plt.subplots(figsize=(14, 8), facecolor="none")
ax.set_facecolor("none")

# Plot each instrument
for i in range(len(sel_filters)):
    indices = [
        j for j in range(len(sel_filters[i]))
        if fluxlist[filterlist.index(sel_filters[i][j])] != 0
    ]

    fluxes = np.array([
        fluxlist[filterlist.index(sel_filters[i][j])]
        for j in indices
    ])

    errors = np.array([
        errorlist[filterlist.index(sel_filters[i][j])]
        for j in indices
    ])

    pivots = np.array([
        pivotlist[filterlist.index(sel_filters[i][j])]
        for j in indices
    ])

    # Need F - sigma > 0 to calculate magnitude errors
    valid = (
        np.isfinite(fluxes)
        & np.isfinite(errors)
        & (fluxes > 0)
        & (errors > 0)
        #& (fluxes - errors > 0)
    )

    print(fluxes)
    print(errors)

    fluxes = fluxes[valid]
    errors = errors[valid]
    pivots = pivots[valid]

    print(fluxes)
    print(errors)

    mag = 31.4 - 2.5 * np.log10(fluxes)
    mag_bright = 31.4 - 2.5 * np.log10(fluxes + errors)
    mag_faint = 31.4 - 2.5 * np.log10(fluxes - errors)

    mag_err_lower = mag - mag_bright
    mag_err_upper = mag_faint - mag

    ax.errorbar(
        pivots,
        mag,
        yerr=[mag_err_lower, mag_err_upper],
        fmt="o",
        capsize=4,
        markersize=9,
        ls="",
        color=colors[i],
        ecolor=colors[i],
        elinewidth=2,
        capthick=2,
        label=sel_filter_names[i]
    )

# AB magnitude: brighter is upwards
ax.invert_yaxis()

ax.set_xlim(0.4, 30)

ax.set_xscale('log')

ax.set_xlabel(
    "Observed wavelength (microns)",
    fontsize=20,
    color="white",
    labelpad=12
)

# Optional subtle grid
ax.grid(
    color="white",
    alpha=0.12,
    which="both"
)

ax.set_ylabel(
    "AB magnitude",
    fontsize=20,
    color="white",
    labelpad=12
)

ax.tick_params(
    axis="both",
    colors="white",
    labelsize=16,
    width=1.5,
    length=6
)

# White axes
for spine in ax.spines.values():
    spine.set_color("white")
    spine.set_linewidth(1.5)

# Subtle grid
ax.grid(
    color="white",
    alpha=0.12
)

# Transparent legend with white text
legend = ax.legend(
    fontsize=16,
    frameon=True,
    facecolor="none",
    edgecolor="white"
)

for text in legend.get_texts():
    text.set_color("white")

legend.get_frame().set_alpha(0)

plt.tight_layout()

plt.savefig(
    f"Presentations/{id}_SED.png",
    dpi=400,
    bbox_inches="tight",
    transparent=True
)

plt.show()