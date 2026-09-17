# photometry

This stage generates the flux and error data used for SED fitting.

## Run Order

1. Run `background_level_estimation.py` to create background estimates in
   `final/05_photometry/background_levels/`.
2. Run `stack_photometry.py` to create flux and error files in
   `final/06_sed_fitting/stack_data/`.