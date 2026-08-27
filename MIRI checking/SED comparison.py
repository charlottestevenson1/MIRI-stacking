import argparse
from prospect.io import write_results
from prospect.fitting import fit_model
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from multiprocessing import Pool
import time


import prospect.io.read_results as reader

_, old_obs, _ = reader.results_from(
    'Prospector/z=8-9 with MIRI.h5',
    dangerous=False
)

with open('Photometry/Pivot wavelengths.txt') as f:
    pivot_waves = np.array([float(wl) for wl in f.readlines()])
    pivot_waves = [pivot_waves[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

old_fluxes = old_obs['maggies'] * 3631 * 1e9
old_errors = old_obs['maggies_unc'] * 3631 * 1e9

with open('Prospector/Stack data/8-9 Fluxes.txt') as f:
    new_fluxes = [float(i) for i in f.readlines()]
    new_fluxes = [new_fluxes[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

with open('Prospector/Stack data/8-9 Errors.txt') as f:
    new_errors = [float(i) for i in f.readlines()]
    new_errors = [new_errors[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

plt.figure(figsize=(10, 6))
plt.ylim(1, 200)
plt.yscale('log')

plt.errorbar(
    pivot_waves,
    old_fluxes,
    yerr=old_errors,
    fmt="o",
    markersize=7,
    capsize=3,
    color="steelblue",
    label='old'
)

plt.errorbar(
    pivot_waves,
    new_fluxes,
    yerr=new_errors,
    fmt="o",
    markersize=7,
    capsize=3,
    color="hotpink",
    label='new'
)

plt.xscale("log")
plt.xlabel("Wavelength [Å]")
plt.ylabel("Flux density")
plt.legend()

plt.tight_layout()
plt.show()