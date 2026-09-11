from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u
import os

os.makedirs('final/03_generate_cutouts/unmasked_cutouts/sci', exist_ok=True)
os.makedirs('final/03_generate_cutouts/unmasked_cutouts/err', exist_ok=True)


### IMPORTANT: enter mosaic directory path here!
# It should have the JADES and SMILES mosaics
mosaic_directory = ''

# Selecting IDs which have all WIDE bands.
with open("final/filter_objects/all_miri.txt") as f:
    ids = [int(id) for id in f.readlines()]

# Wide NIRCam bands and segmentation
nircam_bands = [i.strip() for i in open('final/filter_lists/filter_list_wide.txt', 'r').readlines()][:8]+['segmentation']

# MIRI bands
miri_bands = [i.strip() for i in open('final/filter_lists/filter_list_wide.txt', 'r').readlines()][8:16]

# Use the GOODS-S catalogue for source coordinates. Its NIRCam table is HDU 4.
coord_data = fits.getdata('final/fits_files/goods-s catalog.fits', ext=4)

size = 5 * u.arcsec

for band in nircam_bands:
    print(f'band: {band}------------------------------')

    # The ALL MIRI objects are only in GOODS-S
    filename = f'{mosaic_directory}/hlsp_jades_jwst_nircam_goods-s_{band.lower()}_v5.0_drz.fits'

    sci_data = fits.getdata(filename, ext=1, memmap=False)
    err_data = fits.getdata(filename, ext=2, memmap=False)
    w = WCS(fits.getheader(filename, ext=1))

    for id in ids:

        coords = [coord_data[coord_data['ID'] == id][0][i] for i in ['RA', 'DEC']]
        
        # Preparing WCS and coord parameters for cutout
        coord = SkyCoord(coords[0], coords[1], unit="deg")

        # Defining cutout and header
        sci_cutout = Cutout2D(sci_data, coord, size, wcs = w)
        err_cutout = Cutout2D(err_data, coord, size, wcs = w)
        
        new_sci_header = sci_cutout.wcs.to_header()
        new_err_header = err_cutout.wcs.to_header()

        fits.writeto(
            f"final/03_generate_cutouts/unmasked_cutouts/sci/{id}_{band}.fits",
            sci_cutout.data,
            header = new_sci_header,
            overwrite = True
        )

        fits.writeto(
            f"final/03_generate_cutouts/unmasked_cutouts/err/{id}_{band}.fits",
            err_cutout.data,
            header = new_err_header,
            overwrite = True
        )

        print(f"{id} completed.")



for band in miri_bands:
    print(f'band: {band}------------------------------')

    # The ALL MIRI objects are only in GOODS-S
    filename = f'{mosaic_directory}/hlsp_smiles_jwst_miri_goodss_{band.lower()}_v1.0_drz.fits'
        
    sci_data = fits.getdata(filename, ext=1, memmap=False)
    err_data = fits.getdata(filename, ext=2, memmap=False)
    w = WCS(fits.getheader(filename, ext=1))

    for id in ids:

        coords = [coord_data[coord_data['ID'] == id][0][i] for i in ['RA', 'DEC']]
        
        # Preparing WCS and coord parameters for cutout
        coord = SkyCoord(coords[0], coords[1], unit="deg")

        # Defining cutout and header
        sci_cutout = Cutout2D(sci_data, coord, size, wcs = w)
        err_cutout = Cutout2D(err_data, coord, size, wcs = w)
        
        new_sci_header = sci_cutout.wcs.to_header()
        new_err_header = err_cutout.wcs.to_header()

        fits.writeto(
            f"final/03_generate_cutouts/unmasked_cutouts/sci/{id}_{band}.fits",
            sci_cutout.data,
            header = new_sci_header,
            overwrite = True
        )

        fits.writeto(
            f"final/03_generate_cutouts/unmasked_cutouts/err/{id}_{band}.fits",
            err_cutout.data,
            header = new_err_header,
            overwrite = True
        )

        print(f"{id} completed.")