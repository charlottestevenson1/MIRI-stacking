#pivot_wave_calculations
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

filename = 'fits_files/jades_small.fits'
hdul = fits.open(filename)

filterlist = hdul[1].data['band'].tolist()

pivot_waves = []

for band in filterlist:
    filepath = 'development/visual_inspection_figures/sed_plotting_code/pivot_wave_calculation/profile_data/' + band + '.dat'

    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    profile = [[float(i) for i in line.split()] for line in lines]

    x_values = [i[0] for i in profile]
    y_values = [i[1] for i in profile]

    num = np.trapezoid(y_values, x=x_values)

    den_y_values = [y/x**2 for x,y in zip(x_values, y_values)]

    den = np.trapezoid(den_y_values, x=x_values)

    pivot_wave = np.sqrt(num/den)
    pivot_waves.append(pivot_wave)

#save this in file
with open('development/visual_inspection_figures/sed_plotting_code/pivot_wave_calculation/pivot_waves.txt', 'w') as f:
    for wave in pivot_waves:
        f.write(str(wave) + '\n')

