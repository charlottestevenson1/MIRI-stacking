import argparse
from prospect.io import write_results
from prospect.fitting import fit_model
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from multiprocessing import Pool
import time
import prospect.io.read_results as reader

# -----------------------
# adjust_continuity_agebins
# -----------------------

def adjust_continuity_agebins(parset, nbins=8):
    '''
    Define the agebins. The first bin goes from 0-5 Myr, the next from
    5-10 Myr, the next from 10-30 Myr, and the rest are evenly spaced in
    logarithmic time
    '''
    from prospect.models import priors

    if nbins < 4:
        raise ValueError('Must have nbins >= 4, returning')

    tuniv = cosmo.age(parset['zred']['init']).value * 1e9

    lim1, lim2 = 6.6990, 7.0
    agelims = (
        [0, lim1]
        + np.linspace(
            lim2,
            np.log10(tuniv),
            nbins - 1
        ).tolist()
    )

    agebins = np.array([
        agelims[:-1],
        agelims[1:]
    ])

    ncomp = nbins
    mean = np.zeros(ncomp - 1)
    scale = np.ones_like(mean) * 0.5
    df = np.ones_like(mean) * 2

    rprior = priors.StudentT(
        mean=mean,
        scale=scale,
        df=df
    )

    parset['mass']['N'] = ncomp
    parset['agebins']['N'] = ncomp
    parset['agebins']['init'] = agebins.T
    parset['logsfr_ratios']['N'] = ncomp - 1
    parset['logsfr_ratios']['init'] = mean
    parset['logsfr_ratios']['prior'] = rprior

    return parset


# -----------------------
# zred_to_agebins
# -----------------------

def zred_to_agebins(zred=None, nbins_sfh=8, **extras):
    '''
    Construct `nbins_sfh` bins in lookback time from 0 to the age of the
    Universe at the redshift under consideration. The first bin goes from
    0-10 Myr, the next from 10-30 Myr, and the rest are evenly spaced in
    logarithmic time.
    '''
    tuniv = cosmo.age(zred[0]).value * 1e9

    lim1, lim2 = 6.6990, 7.0

    agelims = (
        [0, lim1]
        + np.linspace(
            lim2,
            np.log10(tuniv),
            nbins_sfh[0] - 1
        ).tolist()
    )

    agebins = np.array([
        agelims[:-1],
        agelims[1:]
    ])

    return agebins.T


# -----------------------
# zlogsfr_ratios_to_masses
# -----------------------

def zlogsfr_ratios_to_masses(
    logmass=None,
    logsfr_ratios=None,
    zred=None,
    **extras
):
    '''
    Convert log10(SFR_j / SFR_{j+1}) and log10(total mass)
    into the mass formed in each age bin.
    '''
    agebins = zred_to_agebins(zred, **extras)
    nbins = agebins.shape[0]

    sratios = 10 ** np.clip(
        logsfr_ratios,
        -100,
        100
    )

    dt = (
        10 ** agebins[:, 1]
        - 10 ** agebins[:, 0]
    )

    coeffs = np.array([
        (
            (1. / np.prod(sratios[:i]))
            * (
                np.prod(dt[1:i + 1])
                / np.prod(dt[:i])
            )
        )
        for i in range(nbins)
    ])

    m1 = (
        10 ** logmass
    ) / coeffs.sum()

    return m1 * coeffs


# -----------------------
# to_dust1
# -----------------------

def to_dust1(
    dust1_fraction=None,
    dust1=None,
    dust2=None,
    **extras
):
    return dust1_fraction * dust2


# -----------------------
# build_model
# -----------------------

def build_model(nbins_sfh=8, **kwargs):
    from prospect.models.sedmodel import SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    model_params = TemplateLibrary["continuity_sfh"]

    # Redshift
    model_params['zred']['isfree'] = True
    model_params['zred']['init'] = 0.5 * (zlo + zup)
    model_params['zred']['prior'] = priors.TopHat(
        mini=zlo,
        maxi=zup
    )

    # Age bins
    model_params['nbins_sfh'] = dict(
        N=1,
        isfree=False,
        init=nbins_sfh
    )

    model_params = adjust_continuity_agebins(
        model_params,
        nbins=nbins_sfh
    )

    model_params['agebins']['N'] = nbins_sfh
    model_params['agebins']['depends_on'] = zred_to_agebins

    # Mass
    model_params['logmass']['isfree'] = True
    model_params['logmass']['init'] = 9
    model_params['logmass']['prior'] = priors.Uniform(
        mini=6,
        maxi=12
    )

    model_params['mass']['isfree'] = False
    model_params['mass']['init'] = np.array([
        [
            10 ** model_params['logmass']['init']
            / nbins_sfh
        ] * nbins_sfh
    ])
    model_params['mass']['depends_on'] = zlogsfr_ratios_to_masses

    # IMF
    model_params['imf_type']['init'] = 1

    model_params['imf_upper_limit'] = {
        'name': 'imf_upper_limit',
        'N': 1,
        'isfree': False,
        'init': 300.0,
        'units': 'type'
    }

    model_params['imf_lower_limit'] = {
        'name': 'imf_lower_limit',
        'N': 1,
        'isfree': False,
        'init': 0.1,
        'units': 'type'
    }

    # Dust
    model_params['dust_type']['init'] = 4

    model_params['dust2']['isfree'] = True
    model_params['dust2']['prior'] = priors.ClippedNormal(
        mean=0.3,
        sigma=1,
        mini=0,
        maxi=4
    )

    model_params['dust_index'] = dict(
        N=1,
        isfree=True,
        init=0,
        prior=priors.TopHat(
            mini=-1,
            maxi=0.4
        )
    )

    model_params['dust1'] = {
        "N": 1,
        "isfree": False,
        "init": 0.0,
        "units": "optical depth towards young stars",
        "prior": None,
        "depends_on": to_dust1
    }

    model_params['dust1_index'] = {
        "N": 1,
        "isfree": False,
        "init": -1.0,
        "units": "power-law index of the birth-cloud attenuation"
    }

    model_params['dust1_fraction'] = dict(
        N=1,
        isfree=True,
        init=1.0,
        prior=priors.ClippedNormal(
            mean=1,
            sigma=0.3,
            mini=0,
            maxi=2
        )
    )

    # Metallicity
    model_params['logzsol']['isfree'] = True
    model_params['logzsol']['init'] = -1
    model_params['logzsol']['prior'] = priors.TopHat(
        mini=-2,
        maxi=0.19
    )

    # Nebular emission
    model_params.update(
        TemplateLibrary['nebular']
    )

    model_params['add_neb_emission']['init'] = True
    model_params['nebemlineinspec']['init'] = False
    model_params['gas_logu']['isfree'] = True
    model_params['gas_logu']['init'] = -2.5
    model_params['gas_logu']['prior'] = priors.TopHat(
        mini=-4,
        maxi=-1
    )

    # Ignore Lyman alpha
    lines_to_ignore = ['Ly-alpha 1215']
    model_params['elines_to_ignore'] = dict(
        init=lines_to_ignore,
        isfree=False
    )

    # IGM
    model_params.update(
        TemplateLibrary['igm']
    )

    model_params['igm_factor']['isfree'] = True
    model_params['igm_factor']['init'] = 1
    model_params['igm_factor']['prior'] = priors.ClippedNormal(
        mean=1,
        sigma=0.3,
        mini=0,
        maxi=2
    )

    return SpecModel(model_params)


# -----------------------
# build_sps
# -----------------------

def build_sps(**kwargs):
    from prospect.sources import FastStepBasis
    return FastStepBasis()


# ============================================================
# REDSHIFT BINS
# ============================================================

zreds = [
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 15)
][:2]


# ============================================================
# LOOP OVER REDSHIFT BINS
# ============================================================

for zlo, zup in zreds:

    print(f'Plotting for z={zlo}-{zup}...')

    start_time = time.perf_counter()

    model = build_model()

    print(
        f'model built after '
        f'{time.perf_counter() - start_time:.2f} seconds\n'
    )

    sps = build_sps()

    print(
        f'sps built after '
        f'{time.perf_counter() - start_time:.2f} seconds\n'
    )

    withMIRIfile = (
        f'Prospector/Fits/'
        f'z={zlo}-{zup} with MIRI.h5'
    )

    noMIRIfile = (
        f'Prospector/Fits/'
        f'z={zlo}-{zup} no MIRI.h5'
    )

    # ========================================================
    # LOAD RESULTS
    # ========================================================

    withMIRIresult, obs, _ = reader.results_from(
        withMIRIfile,
        dangerous=False
    )

    noMIRIresult, _, _ = reader.results_from(
        noMIRIfile,
        dangerous=False
    )

    # ========================================================
    # HELPER: POSTERIOR SED
    # ========================================================

    def posterior_sed(
        result,
        obs,
        model,
        sps,
        nsamples=500,
        n_wavelengths=5000
    ):
        chain = np.asarray(result["chain"])

        weights = np.asarray(
            result["weights"],
            dtype=float
        )

        weights /= weights.sum()

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(chain),
            size=min(
                nsamples,
                len(chain)
            ),
            replace=True,
            p=weights
        )

        theta_samples = chain[indices]

        z_index = list(
            result["theta_labels"]
        ).index("zred")

        z_samples = theta_samples[:, z_index]

        z_min = np.percentile(
            z_samples,
            1
        )

        z_max = np.percentile(
            z_samples,
            99
        )

        wave_min = (
            sps.wavelengths.min()
            * (1 + z_min)
        )

        wave_max = (
            sps.wavelengths.max()
            * (1 + z_max)
        )

        common_wave = np.logspace(
            np.log10(wave_min),
            np.log10(wave_max),
            n_wavelengths
        )

        spectra = []
        photometry = []

        for theta in theta_samples:

            mspec, mphot, _ = model.mean_model(
                theta,
                obs,
                sps=sps
            )

            zred = theta[z_index]

            wave = (
                sps.wavelengths
                * (1 + zred)
            )

            spectrum_interp = np.interp(
                common_wave,
                wave,
                mspec,
                left=np.nan,
                right=np.nan
            )

            spectra.append(
                spectrum_interp
            )

            photometry.append(
                mphot
            )

        spectra = np.asarray(
            spectra
        )

        photometry = np.asarray(
            photometry
        )

        sed_p16 = np.nanpercentile(
            spectra,
            16,
            axis=0
        )

        sed_p50 = np.nanpercentile(
            spectra,
            50,
            axis=0
        )

        sed_p84 = np.nanpercentile(
            spectra,
            84,
            axis=0
        )

        phot_p16 = np.nanpercentile(
            photometry,
            16,
            axis=0
        )

        phot_p50 = np.nanpercentile(
            photometry,
            50,
            axis=0
        )

        phot_p84 = np.nanpercentile(
            photometry,
            84,
            axis=0
        )

        return (
            common_wave,
            sed_p16,
            sed_p50,
            sed_p84,
            phot_p16,
            phot_p50,
            phot_p84
        )

    # ========================================================
    # COMPUTE POSTERIOR SEDS
    # ========================================================

    (
        withMIRI_wspec,
        withMIRI_sed_p16,
        withMIRI_sed_p50,
        withMIRI_sed_p84,
        withMIRI_phot_p16,
        withMIRI_phot_p50,
        withMIRI_phot_p84
    ) = posterior_sed(
        withMIRIresult,
        obs,
        model,
        sps,
        nsamples=500
    )

    (
        noMIRI_wspec,
        noMIRI_sed_p16,
        noMIRI_sed_p50,
        noMIRI_sed_p84,
        noMIRI_phot_p16,
        noMIRI_phot_p50,
        noMIRI_phot_p84
    ) = posterior_sed(
        noMIRIresult,
        obs,
        model,
        sps,
        nsamples=500
    )

    # ========================================================
    # MAGGIES -> AB MAG
    # ========================================================

    wphot = obs["phot_wave"]

    def maggies_to_ab(flux):
        flux = np.asarray(
            flux,
            dtype=float
        )

        mag = np.full_like(
            flux,
            np.nan
        )

        valid = (
            np.isfinite(flux)
            & (flux > 0)
        )

        mag[valid] = (
            -2.5
            * np.log10(flux[valid])
        )

        return mag

    # Flux percentile ordering reverses in magnitude space
    withMIRI_sed_mag_p16 = maggies_to_ab(
        withMIRI_sed_p84
    )
    withMIRI_sed_mag_p50 = maggies_to_ab(
        withMIRI_sed_p50
    )
    withMIRI_sed_mag_p84 = maggies_to_ab(
        withMIRI_sed_p16
    )

    noMIRI_sed_mag_p16 = maggies_to_ab(
        noMIRI_sed_p84
    )
    noMIRI_sed_mag_p50 = maggies_to_ab(
        noMIRI_sed_p50
    )
    noMIRI_sed_mag_p84 = maggies_to_ab(
        noMIRI_sed_p16
    )

    withMIRI_phot_mag_p16 = maggies_to_ab(
        withMIRI_phot_p84
    )
    withMIRI_phot_mag_p50 = maggies_to_ab(
        withMIRI_phot_p50
    )
    withMIRI_phot_mag_p84 = maggies_to_ab(
        withMIRI_phot_p16
    )

    noMIRI_phot_mag_p16 = maggies_to_ab(
        noMIRI_phot_p84
    )
    noMIRI_phot_mag_p50 = maggies_to_ab(
        noMIRI_phot_p50
    )
    noMIRI_phot_mag_p84 = maggies_to_ab(
        noMIRI_phot_p16
    )

    # ========================================================
    # OBSERVED PHOTOMETRY
    # ========================================================

    obs_flux = np.asarray(
        obs["maggies"],
        dtype=float
    )

    obs_unc = np.asarray(
        obs["maggies_unc"],
        dtype=float
    )

    snr = obs_flux / obs_unc

    detected = (
        (snr >= 1.5)
        & np.isfinite(obs_flux)
        & np.isfinite(obs_unc)
        & (obs_unc > 0)
        & (obs_flux > obs_unc)
    )

    upper_limits = (
        (snr < 1.5)
        & np.isfinite(obs_unc)
        & (obs_unc > 0)
    )

    obs_mag = np.full_like(
        obs_flux,
        np.nan
    )

    obs_mag_err_lower = np.full_like(
        obs_flux,
        np.nan
    )

    obs_mag_err_upper = np.full_like(
        obs_flux,
        np.nan
    )

    obs_mag[detected] = maggies_to_ab(
        obs_flux[detected]
    )

    mag_bright = maggies_to_ab(
        obs_flux[detected]
        + obs_unc[detected]
    )

    mag_faint = maggies_to_ab(
        obs_flux[detected]
        - obs_unc[detected]
    )

    obs_mag_err_lower[detected] = (
        obs_mag[detected]
        - mag_bright
    )

    obs_mag_err_upper[detected] = (
        mag_faint
        - obs_mag[detected]
    )

    upper_limit_mag = maggies_to_ab(
        1.5 * obs_unc[upper_limits]
    )

    # ========================================================
    # FIGURE FUNCTION
    # ========================================================

    def make_sed_plot(
        show_miri,
        filename
    ):

        fig, ax = plt.subplots(
            figsize=(16, 8),
            facecolor="none"
        )

        ax.set_facecolor("none")

        # ----------------------------------------------------
        # NO MIRI POSTERIOR
        # ----------------------------------------------------

        ax.fill_between(
            noMIRI_wspec,
            noMIRI_sed_mag_p16,
            noMIRI_sed_mag_p84,
            color="lightpink",
            alpha=0.35,
            label="No MIRI 16–84%"
        )

        ax.plot(
            noMIRI_wspec,
            noMIRI_sed_mag_p50,
            color="hotpink",
            lw=2.0,
            label="No MIRI posterior median"
        )

        # ----------------------------------------------------
        # WITH MIRI POSTERIOR
        # ----------------------------------------------------

        if show_miri:

            ax.fill_between(
                withMIRI_wspec,
                withMIRI_sed_mag_p16,
                withMIRI_sed_mag_p84,
                color="lightblue",
                alpha=0.35,
                label="With MIRI 16–84%"
            )

            ax.plot(
                withMIRI_wspec,
                withMIRI_sed_mag_p50,
                color="steelblue",
                lw=2.0,
                label="With MIRI posterior median"
            )

        # ----------------------------------------------------
        # MODEL PHOTOMETRY
        # ----------------------------------------------------

        ax.errorbar(
            wphot,
            noMIRI_phot_mag_p50,
            yerr=[
                noMIRI_phot_mag_p50
                - noMIRI_phot_mag_p16,
                noMIRI_phot_mag_p84
                - noMIRI_phot_mag_p50
            ],
            label="No MIRI median photometry",
            marker="s",
            markersize=10,
            ls="",
            markerfacecolor="none",
            markeredgecolor="hotpink",
            markeredgewidth=3,
            ecolor="hotpink",
            alpha=0.7,
            capsize=3
        )

        if show_miri:

            ax.errorbar(
                wphot,
                withMIRI_phot_mag_p50,
                yerr=[
                    withMIRI_phot_mag_p50
                    - withMIRI_phot_mag_p16,
                    withMIRI_phot_mag_p84
                    - withMIRI_phot_mag_p50
                ],
                label="With MIRI median photometry",
                marker="s",
                markersize=10,
                ls="",
                markerfacecolor="none",
                markeredgecolor="steelblue",
                markeredgewidth=3,
                ecolor="steelblue",
                alpha=0.7,
                capsize=3
            )

        # ----------------------------------------------------
        # OBSERVED PHOTOMETRY
        # ----------------------------------------------------

        ax.errorbar(
            wphot[detected],
            obs_mag[detected],
            yerr=[
                obs_mag_err_lower[detected],
                obs_mag_err_upper[detected]
            ],
            label="Observed photometry",
            ecolor="gray",
            marker="o",
            markersize=10,
            ls="",
            markerfacecolor="none",
            markeredgecolor="gray",
            markeredgewidth=3,
            capsize=3
        )

        ax.scatter(
            wphot[upper_limits],
            upper_limit_mag,
            marker="v",
            s=100,
            color="gray",
            label="Observed < 1.5σ",
            zorder=5
        )

        # ----------------------------------------------------
        # FORMATTING
        # ----------------------------------------------------

        ax.set_xlabel(
            "Wavelength [Å]",
            fontsize=20,
            color="white",
            labelpad=12
        )

        ax.set_ylabel(
            "AB magnitude",
            fontsize=20,
            color="white",
            labelpad=12
        )

        ax.set_xscale("log")
        ax.set_yscale("linear")

        ax.set_xlim(
            np.min(wphot) * 0.8,
            np.max(wphot) / 0.8 * 5
        )

        # Keep your existing presentation range
        ax.set_ylim(
            27,
            32
        )

        ax.invert_yaxis()

        ax.tick_params(
            axis="both",
            colors="white",
            labelsize=16,
            width=1.5,
            length=6
        )

        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(1.5)

        ax.grid(
            color="white",
            alpha=0.12
        )

        legend = ax.legend(
            loc="best",
            fontsize=14,
            frameon=True,
            facecolor="none",
            edgecolor="white"
        )

        for text in legend.get_texts():
            text.set_color("white")

        legend.get_frame().set_alpha(0)

        fig.suptitle(
            f"Posterior median SED: z={zlo}-{zup}",
            fontsize=20,
            color="white",
            fontweight="bold",
            y=0.94
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.93]
        )

        fig.savefig(
            filename,
            dpi=400,
            bbox_inches="tight",
            transparent=True
        )

        plt.close(fig)

    # ========================================================
    # SAVE NO-MIRI ONLY
    # ========================================================

    make_sed_plot(
        show_miri=False,
        filename=(
            f"Prospector/Plots/Posterior median SEDs/"
            f"z={zlo}-{zup} posterior median SED no MIRI.png"
        )
    )

    # ========================================================
    # SAVE BOTH
    # ========================================================

    make_sed_plot(
        show_miri=True,
        filename=(
            f"Prospector/Plots/Posterior median SEDs/"
            f"z={zlo}-{zup} posterior median SED both.png"
        )
    )

    print(
        f"Saved SED plots for z={zlo}-{zup}"
    )