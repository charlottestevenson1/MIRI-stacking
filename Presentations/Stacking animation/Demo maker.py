import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

stacking_galaxy_IDs = [int(i) for i in open('Stacking/Objects for stacking.txt').readlines()]

bands = [
    'F150W'
]

# for i in range(5, 300, 5):
#     for band in bands:
#         band_galaxy_IDs = [int(i) for i in open(f'Filter objects/{band} objects.txt')]

#         galaxy_IDs = [ID for ID in stacking_galaxy_IDs if ID in band_galaxy_IDs]
#         n_galaxies = min(370, len(galaxy_IDs))
#         galaxy_IDs = galaxy_IDs[:n_galaxies]

#         SCI_array = np.stack([
#             fits.getdata(f'Cutouts/SCI/{ID}_{band}.fits')
#             for ID in galaxy_IDs[:i]
#         ])

#         ERR_array = np.stack([
#             fits.getdata(f'Cutouts/ERR/{ID}_{band}_ERR.fits')
#             for ID in galaxy_IDs[:i]
#         ])

#         # Valid pixels only: finite science, finite error, positive error
#         valid = np.isfinite(SCI_array) & np.isfinite(ERR_array) & (ERR_array > 0)

#         weights = np.zeros_like(ERR_array)
#         weights[valid] = 1.0 / (ERR_array[valid] ** 2)

#         numerator = np.sum(np.where(valid, SCI_array * weights, 0.0), axis=0)
#         denominator = np.sum(weights, axis=0)

#         stack = np.divide(
#             numerator,
#             denominator,
#             out=np.full_like(numerator, np.nan),
#             where=denominator > 0
#         )

#         stack_err = np.divide(
#             1.0,
#             np.sqrt(denominator),
#             out=np.full_like(denominator, np.nan, dtype=float),
#             where=denominator > 0
#         )

#         fits.writeto(f'Cutouts/Demo/{band}_stack_{i}.fits', stack, overwrite = True)
#         #fits.writeto(f'Stacking/{band}_stack_ERR.fits', stack_err, overwrite=True)

#         #print(f'Saved {band}.')

import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

arrays = [fits.getdata(f'Cutouts/Demo/{bands[0]}_stack_{i}.fits') for i in range(5,300, 5)]
print(arrays[0])

fig, ax = plt.subplots()
ax.axis("off")

img = arrays[0]
im = ax.imshow(img, cmap='inferno')

title = ax.set_title("5 objects")

def update(i):
    img = arrays[i]
    im.set_data(img)
    title.set_text(f"{(i+1)*5} objects")
    return [im]

anim = FuncAnimation(fig, update, frames=len(arrays), interval=200, blit=False)
plt.show()
anim.save("animation.gif", writer="pillow", fps=5)