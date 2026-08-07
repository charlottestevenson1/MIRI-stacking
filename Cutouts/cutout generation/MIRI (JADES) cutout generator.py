from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u

BANDS = [i.strip() for i in open('Filter lists/filter list miri', 'r').readlines()]

size = 5 * u.arcsec

### IMPORTANT: enter mosaic directory path here!
MOSAIC_DIRECTORY = ''

for BAND in BANDS:
    print(f'BAND: {BAND}------------------------------')
    #Selecting IDs which have all MIRI bands and the band in question.
    with open("Filter objects/ALL MIRI.txt") as f:
        all_miri_IDs = [int(ID) for ID in f.readlines()]

    with open(f"Filter objects/{BAND} objects.txt") as f:
        band_IDs = [int(ID) for ID in f.readlines()]

    IDs = [int(ID) for ID in open('Overlap galaxies.txt')]  

    # The ALL MIRI objects are only in GOODS-S
    filename = f'{MOSAIC_DIRECTORY}/hlsp_jades_jwst_miri_goodss_deep_f770w_v1.0_drz.fits'

    hdul = fits.open(filename)

    # For SCI, this should be 1...
    # For ERR, this should be 2...
    # For WHT, this should be 3...
    datatype = 3
    dataaddons = [0, "", "_ERR", "_WHT"]

    data = hdul[datatype].data

    coorddata = fits.open('FITS files/jades_small.fits')[2].data

    for ID in IDs:

        coords = [coorddata[coorddata['ID'] == ID][0][i] for i in ['RA', 'DEC']]
        
        # Preparing WCS and coord parameters for cutout
        w = WCS(hdul[1].header)
        coord = SkyCoord(coords[0], coords[1], unit="deg")

        # Defining cutout and header
        cutout = Cutout2D(data, coord, size, wcs = w)
        
        new_header = cutout.wcs.to_header()

        fits.writeto(
            f"Cutouts/WHT/Deep/{ID}_{BAND}_WHT.fits",
            cutout.data,
            header = new_header,
            overwrite = True
        )

        print(f"{ID} completed.")
