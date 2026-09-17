# Generate cutouts

This stage generates and masks the 5-inch cutouts used for stacking.

## Requirements

The directory must contain both the JADES NIRCam mosaics and the SMILES MIRI mosaics.
Set `MOSAIC_DIRECTORY` in `cutout_generator.py`. It may be an external hard-drive
location or a directory inside `final/fits_files/` if the mosaics are stored there.

## Run Order

Run `cutout_generator.py` first, followed by `cutout_masking.py`.

The scripts create `unmasked_cutouts/` and `masked_cutouts/` in this folder.