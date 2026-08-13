import argparse
from prospect.io import write_results
from prospect.fitting import fit_model
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool
import time

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# import prospect.io.read_results as reader
# results_type = 'dynesty'

# # grab results (dictionary), the obs dictionary, and our corresponding models
# # When using parameter files set `dangerous=True`
# readfile = 'Prospector/z=9-10 with MIRI.h5'
# result, obs, model = reader.results_from(readfile, dangerous=False)

# print(obs['phot_mask'])

bands = ["F070W", "F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]

for zred in [(8,9), (9,10), (10,11), (11,12), (12,15)]:
    fluxes = []
    errors = []
    fluxname = f'Prospector/Stack data/{zred[0]}-{zred[1]} Fluxes.txt'
    errname = f'Prospector/Stack data/{zred[0]}-{zred[1]} Errors.txt'

    with open(fluxname) as f:
        for line in f:
            fluxes.append(float(line))
    
    with open(errname) as f:
        for line in f:
            errors.append(float(line))
    
    SNRs = np.array(fluxes)/np.array(errors)

    print(f'\n\nRedshifts {zred[0]}-{zred[1]}:')
    for i in range(8):
        print(f'{bands[i]}: SNR = {SNRs[i]:.2f}')

