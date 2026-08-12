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

import prospect.io.read_results as reader
results_type = 'dynesty'

# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
readfile = 'Prospector/z=8-9 no MIRI.h5'
result, obs, model = reader.results_from(readfile, dangerous=False)

imax = np.argmax(result["lnprobability"])
print(result['chain'][imax][4:11])