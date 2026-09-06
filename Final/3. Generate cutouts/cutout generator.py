from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u
import os

os.makedirs('Final/3. Generate cutouts/Unmasked cutouts/SCI', exist_ok=True)
os.makedirs('Final/3. Generate cutouts/Unmasked cutouts/ERR', exist_ok=True)


### IMPORTANT: enter mosaic directory path here!
# It should have the JADES and SMILES mosaics
MOSAIC_DIRECTORY = ''

# Selecting IDs which have all WIDE bands.
with open("Final/Filter objects/ALL MIRI.txt") as f:
    IDs = [int(ID) for ID in f.readlines()]

# Wide NIRCam bands and segmentation
NIRCam_BANDS = [i.strip() for i in open('Final/Filter lists/filter list wide.txt', 'r').readlines()][:8]+['segmentation']

# MIRI bands
MIRI_BANDS = [i.strip() for i in open('Final/Filter lists/filter list wide.txt', 'r').readlines()][8:16]

size = 5 * u.arcsec

for BAND in NIRCam_BANDS:
    print(f'BAND: {BAND}------------------------------')

    # The ALL MIRI objects are only in GOODS-S
    filename = f'{MOSAIC_DIRECTORY}/hlsp_jades_jwst_nircam_goods-s_{BAND.lower()}_v5.0_drz.fits'

    hdul = fits.open(filename)

    SCIdata = hdul[1].data
    ERRdata = hdul[2].data

    # Access locations from small catalog - can replace with the main GOODS-S catalog HDUL, HDU no. 2 (I think)
    coorddata = fits.open('Final/FITS files/jades_small.fits')[2].data

    for ID in IDs:

        coords = [coorddata[coorddata['ID'] == ID][0][i] for i in ['RA', 'DEC']]
        
        # Preparing WCS and coord parameters for cutout
        w = WCS(hdul[1].header)
        coord = SkyCoord(coords[0], coords[1], unit="deg")

        # Defining cutout and header
        SCIcutout = Cutout2D(SCIdata, coord, size, wcs = w)
        ERRcutout = Cutout2D(ERRdata, coord, size, wcs = w)
        
        new_SCI_header = SCIcutout.wcs.to_header()
        new_ERR_header = ERRcutout.wcs.to_header()

        fits.writeto(
            f"Final/3. Generate cutouts/Unmasked cutouts/SCI/{ID}_{BAND}.fits",
            SCIcutout.data,
            header = new_SCI_header,
            overwrite = True
        )

        fits.writeto(
            f"Final/3. Generate cutouts/Unmasked cutouts/ERR/{ID}_{BAND}.fits",
            ERRcutout.data,
            header = new_ERR_header,
            overwrite = True
        )

        print(f"{ID} completed.")



for BAND in MIRI_BANDS:
    print(f'BAND: {BAND}------------------------------')

    # The ALL MIRI objects are only in GOODS-S
    filename = f'{MOSAIC_DIRECTORY}/hlsp_smiles_jwst_miri_goodss_{BAND.lower()}_v1.0_drz.fits'
        
    hdul = fits.open(filename)

    SCIdata = hdul[1].data
    ERRdata = hdul[2].data

    coorddata = fits.open('Final/FITS files/jades_small.fits')[2].data

    for ID in IDs:

        coords = [coorddata[coorddata['ID'] == ID][0][i] for i in ['RA', 'DEC']]
        
        # Preparing WCS and coord parameters for cutout
        w = WCS(hdul[1].header)
        coord = SkyCoord(coords[0], coords[1], unit="deg")

        # Defining cutout and header
        SCIcutout = Cutout2D(SCIdata, coord, size, wcs = w)
        ERRcutout = Cutout2D(ERRdata, coord, size, wcs = w)
        
        new_SCI_header = SCIcutout.wcs.to_header()
        new_ERR_header = ERRcutout.wcs.to_header()

        fits.writeto(
            f"Final/3. Generate cutouts/Unmasked cutouts/SCI/{ID}_{BAND}.fits",
            SCIcutout.data,
            header = new_SCI_header,
            overwrite = True
        )

        fits.writeto(
            f"Final/3. Generate cutouts/Unmasked cutouts/ERR/{ID}_{BAND}.fits",
            ERRcutout.data,
            header = new_ERR_header,
            overwrite = True
        )

        print(f"{ID} completed.")