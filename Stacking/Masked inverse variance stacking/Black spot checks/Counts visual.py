import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M", "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

plots = [fits.getdata(f'Stacking/Masked inverse variance stacking/Stacked images/Counts/{band}_counts.fits') for band in bands]

vmin = min(np.nanmin(a) for a in plots)
vmax = max(np.nanmax(a) for a in plots)

fig, axs = plt.subplots(5, 6, figsize=(18, 15), constrained_layout=True)
axs = axs.flat

for i, arr in enumerate(plots):
    axs[i].imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
    axs[i].set_title(f"{bands[i]}", fontsize=9)
    axs[i].axis("off")

# turn off unused subplots
for ax in axs[len(plots):]:
    ax.axis("off")

fig.suptitle("Counts", fontsize=12)

plt.show()