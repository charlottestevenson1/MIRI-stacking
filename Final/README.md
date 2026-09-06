# Final Pipeline

## Requirements

- Place the catalogue FITS files listed in `FITS files/README.md` in `Final/FITS files/`.
- Provide the JADES NIRCam and SMILES MIRI mosaic files for the cutout stage.
- Set `MOSAIC_DIRECTORY` in `3. Generate cutouts/cutout generator.py`.

## Run Order

Run all scripts from the repository root, `MIRI-stacking`, in this order:

1. Build filter-object lists.
2. Assign objects to redshift bins.
3. Generate and mask cutouts.
4. Perform inverse-variance stacking.
5. Estimate backgrounds and perform aperture photometry.
6. Run Prospector or inspect existing Prospector results.

Read the `README.md` in each numbered folder before running its scripts.