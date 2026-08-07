import os
import matplotlib.pyplot as plt

image_dir = "Stacking/Masked inverse variance stacking/Black spot checks/Random aperture placement plots"


bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M", "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

nrows = 4
ncols = 7

fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
axs = axs.ravel()

for ax, band in zip(axs, bands):
    img = plt.imread(image_dir+f'/{band}_RAPT.png')
    ax.imshow(img)
    ax.axis("off")

# Hide any unused axes
for ax in axs[len(bands):]:
    ax.axis("off")

plt.tight_layout()
plt.show()