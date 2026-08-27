import argparse
from prospect.io import write_results
from prospect.fitting import fit_model
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool
import time
from pprint import pprint
import pickle

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import prospect.io.read_results as reader
results_type = 'dynesty'

# grab results (dictionary), the obs dictionary, and our corresponding models
# When using parameter files set `dangerous=True`
readfile = 'Prospector/Fits/z=9-10 with MIRI.h5'
result, obs, model = reader.results_from(readfile, dangerous=False)

model_params = result['model_params']
prior = model_params[9]['prior']
pprint(pickle.loads(prior))

entry = next(
    p for p in result["model_params"] if p["name"] == "zred"
)

prior = pickle.loads(entry['prior'])
# samples = [prior.sample()[0] for i in range(10000)]
# plt.hist(samples, bins=50, density = True)
# plt.show()

x = np.linspace(6,9,1000)
lnp = prior(x)
pdf = np.exp(lnp)

plt.plot(x, pdf)
plt.show()