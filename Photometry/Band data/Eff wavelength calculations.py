# Calculates the effective wavelengths of each filter band

import numpy as np

bands = [
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
    "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M",
    "F430M", "F444W", "F460M", "F480M", "F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"
]

widths = []
pwaves = []

for band in bands:
    filepath = f'Photometry/Band data/profile data/{band}.dat'
    
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    profile = [[float(i) for i in line.split()] for line in lines]

    lam = [i[0] for i in profile]
    S = [i[1] for i in profile]

    # lam in Angstroms, nm, or any wavelength unit
    # S is the dimensionless throughput
    lam = np.asarray(lam)
    S = np.asarray(S)

    # Sort in increasing wavelength
    idx = np.argsort(lam)
    lam = lam[idx]
    S = S[idx]

    lnlam = np.log(lam)

    # Effective wavelength (replace with your preferred definition if needed)
    lam_eff = np.exp(
        np.trapezoid(S * lnlam, lnlam) /
        np.trapezoid(S, lnlam)
    )

    # Fractional RMS width
    sigma = np.sqrt(
        np.trapezoid(S * (lnlam - np.log(lam_eff))**2, lnlam) /
        np.trapezoid(S, lnlam)
    )

    width = 2.355*sigma*lam_eff 
    widths.append(width)

    pwaves.append(lam_eff)

    print(f"{band} lambda_eff = {lam_eff/1e4:.2f} microns")
    print(f"{band} sigma = {sigma:.4f}")
    print(f"{band} width = {2.355*sigma*lam_eff/1e4} microns\n\n")

with open('Photometry/Band data/pivot_waves.txt', 'w') as f:
    for line in pwaves:
        f.writelines(str(line/1e4)+'\n') # conversion to microns

with open('Photometry/Band data/Eff widths.txt', 'w') as f:
    for line in widths:
        f.writelines(str(line/1e4)+'\n')