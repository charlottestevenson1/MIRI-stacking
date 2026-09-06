# Photometry

This stage generates the flux and error data used for SED fitting.

## Run Order

1. Run `Background level estimation.py` to create background estimates in
   `Final/5. Photometry/Background levels/`.
2. Run `Stack photometry.py` to create flux and error files in
   `Final/6. Prospector/Stack data/`.