# final Pipeline

## Requirements

- Place the catalogue fits_files listed in `fits_files/README.md` in `final/fits_files/`.
- Provide the JADES NIRCam and SMILES MIRI mosaic files for the cutout stage.
- Set `MOSAIC_DIRECTORY` in `3_generate_cutouts/cutout_generator.py`.

## Run Order

Run all scripts from the repository root, `MIRI-stacking`, in this order:

1. Build filter-object lists.
2. Assign objects to redshift bins.
3. Generate and mask cutouts.
4. Perform inverse-variance stacking.
5. Estimate backgrounds and perform aperture photometry.
6. Run prospector or inspect existing prospector results.

Read the `README.md` in each numbered folder before running its scripts.