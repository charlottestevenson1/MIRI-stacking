# Generate Cutouts

This stage generates and masks the 5-inch cutouts used for stacking.

## Requirements

The directory must contain both the JADES NIRCam mosaics and the SMILES MIRI mosaics.
Set `MOSAIC_DIRECTORY` in `cutout generator.py`. It may be an external hard-drive
location or a directory inside `Final/FITS files/` if the mosaics are stored there.

## Run Order

Run `cutout generator.py` first, followed by `cutout masking.py`.

The scripts create `Unmasked cutouts/` and `Masked cutouts/` in this folder.