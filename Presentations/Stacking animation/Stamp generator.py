from astropy.io import fits
import matplotlib.pyplot as plt

nircam_filters = ['F070W', 'F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
miri_filters = ['F560W', 'F770W', 'F1000W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']
bands = nircam_filters + miri_filters

for band in bands:
    mosaic = fits.open(f'Redshift Bins/Stacks/Redshifts 8-9/SCI/{band}_stack.fits')[0].data
    print(mosaic)
    plt.imshow(mosaic, origin='lower', cmap='viridis')
    plt.axis('off')
    plt.text(
        0.5, 0.95,
        band,
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
        color="white",
        fontsize=16,
        fontweight="bold"
    )
    plt.savefig(f'Presentations/Stack stamps/{band}.png', transparent=True)
    plt.close()