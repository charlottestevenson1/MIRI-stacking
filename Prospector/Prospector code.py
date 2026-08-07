import argparse
from prospect.io import write_results
from prospect.fitting import fit_model
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from multiprocessing import Pool
import time

# Send email when fitting has completed
import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
load_dotenv()

print("SERVER:", os.environ.get("EMAIL"))
print("PORT:", os.environ.get("PASSWD"))

def send_email(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ['EMAIL']
    msg["To"] = os.environ['EMAIL']
    msg.set_content(body)

    with smtplib.SMTP(os.environ['SERVER'], 587) as server:
        server.starttls()
        server.login(
            os.environ['EMAIL'],
            os.environ['PASSWD']
        )
        server.send_message(msg)

# Lower and upper redshift limits
zlo = 8
zup = 9

# Set the fitting method to use: either "emcee" or "dynesty"
fit_method = "dynesty"  # "emcee" or "dynesty"

# -----------------------
# 1) run-time parameters
# -----------------------
run_params = {}

# Set the output file name
run_params["outfile"] = "CMD LINE TEST"

run_params["dynesty"] = fit_method == "dynesty"
run_params["emcee"] = fit_method == "emcee"

# Set whether optimization should be performed before sampling
run_params["optimization"] = True

# Set method and parameters for nested sampling (if using dynesty)
run_params["nested_sample"] = "slice"
run_params["nested_bound"] = "multi"
run_params["nested_nlive_init"] = 100  # No. live points for batch 0
run_params["nested_walks"] = 48 # Number of random walks to take for each live point in nested sampling
run_params["nested_nlive_batch"] = 100  # Number of live points to add in each batch for nested sampling
run_params["nested_dlogz_init"] = 0.05 # Target stopping criterion for batch 0
run_params["nested_maxcall"] = 50000000  # Only matters AFTER initial run
run_params["nested_maxiter"] = 1e6
run_params["nested_maxbatch"] = 10  # Maximum number of batches for nested sampling
run_params["verbose"] = True
run_params["nested_bootstrap"] = 0
run_params["nested_posterior_thresh"] = 0.05

# run_params["nwalkers"] = 32
# run_params["niter"] = 100

# -----------------------
# 2) build_obs
# -----------------------
def build_obs(zlo=zlo, zup=zup):
    """Build a dictionary of observational data.  In this example 
    the data consist of photometry for a single nearby dwarf galaxy 
    from Johnson et al. 2013.
    
    :param snr:
        The S/N to assign to the photometry, since none are reported 
        in Johnson et al. 2013
        
    :param ldist:
        The luminosity distance to assume for translating absolute magnitudes 
        into apparent magnitudes.
        
    :returns obs:
        A dictionary of observational data to use in the fit.
    """
    from prospect.utils.obsutils import fix_obs
    import sedpy

    # The obs dictionary, empty for now
    obs = {}

    # Filter names - loads the transmission curves of the filters
    NIRCam = [[f'jwst_{band.lower()}' for band in ["F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W",
        "F210M", "F250M", "F277W", "F300M", "F335M", "F356W", "F410M", "F430M", "F444W", "F460M", "F480M"]][i] for i in [0,1,2,3,6,9,12,15]] # Only W bands
    MIRI = [f'jwst_{band.lower()}' for band in ["F560W", "F770W", "F1000W",
    "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"]]
    filternames = NIRCam + MIRI

    # Instantiate 'Filter()' objects with sedpy, and put them in the filters key of obs
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # Load fluxes and errors
    with open(f'Prospector/Stack data/{zlo}-{zup} Fluxes.txt') as f:
        all_fluxes_nJy = [float(point) for point in f.readlines()]
        fluxes_nJy = [all_fluxes_nJy[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

    with open(f'Prospector/Stack data/{zlo}-{zup} Errors.txt') as f:
        all_errors_nJy = [float(point) for point in f.readlines()]
        errors_nJy = [all_errors_nJy[i] for i in [0,1,2,3,6,9,12,15,18,19,20,21,22,23,24,25]]

    # Convert to maggies
    fluxes = np.array(fluxes_nJy) / (3.631*1e12)
    errors = np.array(errors_nJy) / (3.631*1e12)

    obs["maggies"] = fluxes
    obs["maggies_unc"] = errors

    # Make a mask for the objects you want to consider (True) or ignore (False).
    obs["phot_mask"] = np.array([True for f in obs["filters"]])

    # Array of effective wavelengths for each filter
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # Since we do not have a spectrum, we have to set some required elements of obs to None:
    obs["wavelength"] = None
    obs["spectrum"] = None
    obs["unc"] = None
    obs["mask"] = None

    # Makes sure all required keys are present in obs
    obs = fix_obs(obs)

    return obs

# -----------------------
# 3) adjust_continuity_agebins
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
    # Limits of 5 Myr and 10 Myr
    lim1, lim2 = 6.6990, 7.0
    agelims = ([0, lim1] + np.linspace(lim2, np.log10(tuniv), nbins-1).tolist())
    agebins = np.array([agelims[:-1], agelims[1:]])

    ncomp = nbins
    mean = np.zeros(ncomp-1)
    scale = np.ones_like(mean)*0.5
    df = np.ones_like(mean)*2
    rprior = priors.StudentT(mean=mean, scale=scale, df=df)

    parset['mass']['N'] = ncomp
    parset['agebins']['N'] = ncomp
    parset['agebins']['init'] = agebins.T
    parset['logsfr_ratios']['N'] = ncomp-1
    parset['logsfr_ratios']['init'] = mean
    parset['logsfr_ratios']['prior'] = rprior

    return parset

# -----------------------
# 4) zred_to_agebins
# -----------------------
def zred_to_agebins(zred=None, nbins_sfh=8, **extras):
    '''
    Construct `nbins_sfh` bins in lookback time from 0 to the age of the
    Universe at the redshift under consideration. The first bin goes from
    0-10 Myr, the next from 10-30 Myr, and the rest are evenly spaced in
    logarithmic time
    '''

    tuniv = cosmo.age(zred[0]).value * 1e9
    lim1, lim2 = 6.6990, 7.0
    agelims = ([0, lim1] +
        np.linspace(lim2, np.log10(tuniv), nbins_sfh[0]-1).tolist())
    agebins = np.array([agelims[:-1], agelims[1:]])

    return agebins.T

# -----------------------
# 5) zlogsfr_ratios_to_masses
# -----------------------
def zlogsfr_ratios_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
    '''
    This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
    log10(Sum(M_i) over i) to values of M_i.  j=0 is the most recent bin in
    lookback time; it incorporates changes in the agebins due to changing
    redshift.
    '''
    agebins = zred_to_agebins(zred, **extras)
    nbins = agebins.shape[0]
    sratios = 10**np.clip(logsfr_ratios, -100, 100)  # numerical issues...
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    coeffs = np.array([
        (1. / np.prod(sratios[:i])) * (np.prod(dt[1: i+1]) / np.prod(dt[: i]))
        for i in range(nbins)
    ])
    m1 = (10**logmass) / coeffs.sum()

    return m1 * coeffs

# -----------------------
# 6) build_model
# -----------------------
# Set fixable parameters as params
def build_model(nbins_sfh=8, **kwargs):
    from prospect.models.sedmodel import SpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors

    model_params = TemplateLibrary["continuity_sfh"]

    # redshift:
    model_params['zred']['isfree'] = True
    model_params['zred']['init'] = 0.5*(zlo+zup)
    model_params['zred']['prior'] = priors.TopHat(mini=zlo, maxi=zup)
    print(f'Redshift range: z={zlo}-{zup}\n\n')

    # Deal with the age bins - let them vary based on the redshift being
    # considered
    model_params['nbins_sfh'] = dict(N=1, isfree=False, init=nbins_sfh)
    model_params = adjust_continuity_agebins(model_params, nbins=nbins_sfh)
    model_params['agebins']['N'] = nbins_sfh
    model_params['agebins']['depends_on'] = zred_to_agebins


    # Let log(mass) vary (not mass, Prospector calculates that)
    model_params['logmass']['isfree'] = True
    model_params['logmass']['init'] = 9
    model_params['logmass']['prior'] = priors.Uniform(mini=6, maxi=12)
    # Set up the mass parameter that Prospector calculates
    model_params['mass']['isfree'] = False
    model_params['mass']['init'] = np.array([nbins_sfh *[10**model_params['logmass']['init']/nbins_sfh]])
    model_params['mass']['depends_on'] = zlogsfr_ratios_to_masses

    # Choose the IMF and the upper and lower limits
    model_params['imf_type']['init'] = 1  # Chabrier
    model_params['imf_upper_limit'] = {'name': 'imf_upper_limit',
                                       'N': 1,
                                       'isfree': False,
                                       'init': 300.0,
                                       'units': 'type'}
    model_params['imf_lower_limit'] = {'name': 'imf_lower_limit',
                                       'N': 1,
                                       'isfree': False,
                                       'init': 0.1,
                                       'units': 'type'}

    # Choose the dust model and let diffuse dust optical depth and power law
    # modifier to Calzetti+2000 attenuateion curve vary
    model_params['dust_type']['init'] = 4  # Kriek & Conroy (2013)
    model_params['dust2']['isfree'] = True
    model_params['dust2']['prior'] = priors.ClippedNormal(
        mean=0.3, sigma=1, mini=0, maxi=4)
    model_params['dust_index'] = dict(N=1, isfree=True, init=0,
        prior=priors.TopHat(mini=-1, maxi=1))
    # Also allow differential attenuation of young (<10 Myr old) stars by their
    # birth cloud to vary
    model_params['dust1'] = dict(N=1, isfree=True, init=0,
        prior=priors.ClippedNormal(mean=1, sigma=0.3, mini=0, maxi=4))

    # Give stellar metallicity a log-uniform prior between 0.01-1 Zsun
    model_params['logzsol']['isfree'] = True
    model_params['logzsol']['init'] = -1
    model_params['logzsol']['prior'] = priors.TopHat(mini=-2, maxi=0)

    # Add in nebular emission and give log(U) a log-uniform prior between -4 and -1
    model_params.update(TemplateLibrary['nebular'])
    model_params['add_neb_emission']['init'] = True
    model_params['nebemlineinspec']['init'] = False
    model_params['gas_logu']['isfree'] = True
    model_params['gas_logu']['init'] = -2.5
    model_params['gas_logu']['prior'] = priors.TopHat(mini=-4, maxi=0)

    # Ignore Lyman alpha
    lines_to_ignore = ['Ly-alpha 1215']
    model_params['elines_to_ignore'] = dict(init=lines_to_ignore, isfree=False)

    # Add IGM attenuation and let the scaling (relative to the Madau+1995
    # model) range a bit
    model_params.update(TemplateLibrary['igm'])
    model_params['igm_factor']['isfree'] = True
    model_params['igm_factor']['init'] = 1
    model_params['igm_factor']['prior'] = priors.ClippedNormal(mean=1, sigma=0.3, mini=0, maxi=3)

    # Instantiate the model using this dictionary of parameter specifications
    model = SpecModel(model_params)
    # for i in model.config_dict:
    #     print(i)
    # print('\n\n\n')
    return model

# -----------------------
# 7) build_sps
# -----------------------
def build_sps(**kwargs):
    from prospect.sources import FastStepBasis
    sps = FastStepBasis()
    return sps

# -----------------------
# 8) minimize
# -----------------------
def minimize(obs, model, sps):
    params = run_params.copy()
    params["dynesty"] = False
    params["emcee"] = False
    params["optimize"] = True
    params["min_method"] = 'lm'

    params["nmin"] = 2

    output = fit_model(obs, model, sps, **params)

    print("Done optimization in {}s".format(output["optimization"][1]))

    return output

# -----------------------
# 9) plot results
# -----------------------
### LOAD FILE
def plot_results(run_params, model, sps):
    import prospect.io.read_results as reader
    results_type = fit_method
    # grab results (dictionary), the obs dictionary, and our corresponding models
    # When using parameter files set `dangerous=True`
    readfile = run_params["outfile"] if run_params["fitnewmodel"] else run_params["readfile"]
    result, obs, _ = reader.results_from(readfile, dangerous=False)

    #The following commented lines reconstruct the model and sps object, 
    # if a parameter file continaing the `build_*` methods was saved along with the results
    #model = reader.get_model(result)
    #sps = reader.get_sps(result)

    ### PLOT PARAMETER TRACES
    if results_type == "emcee":
        chosen = np.random.choice(result["run_params"]["nwalkers"], size=10, replace=False)
        tracefig = reader.traceplot(result, figsize=(20,10), chains=chosen)
    else:
        tracefig = reader.traceplot(result, figsize=(20,10))

    plt.show()

    ### PLOT CORNER PLOT
    # maximum a posteriori (of the locations visited by the MCMC sampler)
    imax = np.argmax(result['lnprobability'])
    if results_type == "emcee":
        i, j = np.unravel_index(imax, result['lnprobability'].shape)
        theta_max = result['chain'][i, j, :].copy()
        thin = 5
    else:
        theta_max = result["chain"][imax, :]
        thin = 1

    #print('Optimization value: {}'.format(theta_best))
    print('MAP value: {}'.format(theta_max))
    #cornerfig = reader.subcorner(result, start=0, thin=thin, truths=theta_best, fig=plt.subplots(5,5,figsize=(27,27))[0])
    
    fig = plt.figure(figsize=(24, 24))

    newlabels = [
    "zred",
    "logZsol",
    "dust2",
    "logM",
    "logSFRr1", "logSFRr2", "logSFRr3", "logSFRr4", "logSFRr5", "logSFRr6", "logSFRr7",
    "dust_index",
    "dust1",
    "gas_logu",
    "igm_factor"
    ]  

    cornerfig = reader.subcorner(
        result,
        start=0,
        thin=thin,
        fill_contours=True,
        color='steelblue',
        fig=fig,
        label_kwargs={"fontsize": 8},
        title_kwargs={"fontsize": 8},
        max_n_ticks=2,
        show_titles=True,
        title_fmt=".2f",
    )

    # Make figure roomier
    cornerfig.set_size_inches(24, 24)
    cornerfig.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=0.93, wspace=0.05, hspace=0.05)

    for ax in cornerfig.axes:
            # Make titles smaller and move them to the next line
            title = ax.get_title()
            if "=" in title:
                ax.set_title(title.replace(" = ", "\n= "), fontsize=8)

            # Smaller tick labels
            ax.tick_params(axis='both', labelsize=6)
    
            # Move x-labels further away from the ticks
            ax.xaxis.labelpad = 30
    
            # Smaller axis labels
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)

    for ax in cornerfig.axes:
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        ax.xaxis.set_label_coords(0.5, -0.45)   # move x-label down

        if xlabel.startswith("logsfr_ratios_"):
            num = xlabel.split("_")[-1]
            ax.set_xlabel(f"logSFRr{num}")

        if ylabel.startswith("logsfr_ratios_"):
            num = ylabel.split("_")[-1]
            ax.set_ylabel(f"logSFRr{num}")

    plt.show()



    ###PLOT SFH
    if results_type == "dynesty":
        # MAP sample
        import copy
        imax = np.argmax(result["lnprobability"])
        theta_map = result["chain"][imax].copy()

        # Temporary copy so the fitted model is unchanged
        plot_model = copy.deepcopy(model)
        plot_model.set_parameters(theta_map)

        # SFH from MAP parameters
        agebins = np.array(plot_model.params["agebins"])
        masses = np.array(plot_model.params["mass"]).ravel()

        dt = 10**agebins[:, 1] - 10**agebins[:, 0]   # years
        sfr = masses / dt                            # Msun / yr

        # Plot in linear time
        t0 = 10**agebins[:, 0] / 1e6   # Myr
        t1 = 10**agebins[:, 1] / 1e6   # Myr

        fig, ax = plt.subplots(figsize=(8, 5))

        for lo, hi, y in zip(t0, t1, sfr):
            ax.plot([lo, hi], [y, y], lw=2)
            ax.vlines([lo, hi], 0, y, alpha=0.2)

        ax.set_xlabel("Lookback time [Myr]")
        ax.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([1, 10**(np.log10(cosmo.age(zlo).value*1e3))])
        ax.invert_xaxis()
        plt.tight_layout()
        plt.show()



    ### PLOT SEDs AND RESIDUALS
    # randomly chosen parameters from chain
    randint = np.random.randint
    if results_type == "emcee":
        nwalkers, niter = result["run_params"]["nwalkers"], result["run_params"]['niter']
        theta = result['chain'][randint(nwalkers), randint(niter)]
    else:
        theta = result["chain"][randint(len(result["chain"]))]

    # generate models
    # sps = reader.get_sps(result)  # this works if using parameter files
    mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)
    mspec_map, mphot_map, _ = model.mean_model(theta_max, obs, sps=sps)

    # Make plot of data and model
    plt.figure(figsize=(16,8))

    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    # photometric effective wavelengths
    wphot = obs["phot_wave"]
    # spectroscopic wavelengths
    if obs["wavelength"] is None:
        # *restframe* spectral wavelengths, since obs["wavelength"] is None
        wspec = sps.wavelengths
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]

    # establish bounds
    # Can change this for visuals!
    ARTIFICIAL_Y_LIM = 1e-17

    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    temp = np.interp(np.linspace(xmin,xmax,10000), wspec, mspec)
    print(f'temp.min={temp.min()}, temp.max={temp.max()}')
    ymin, ymax = max(temp.min()*(0.1),ARTIFICIAL_Y_LIM), temp.max()/0.1

    mask = mphot > 0

    plt.loglog(wspec, mspec, label='Model spectrum (random draw)',
        lw=0.7, color='navy', alpha=0.7)
    plt.loglog(wspec, mspec_map, label='Model spectrum (MAP)',
        lw=0.7, color='green', alpha=0.7)
    plt.errorbar(wphot, mphot, label='Model photometry (random draw)',
            marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
            markerfacecolor='none', markeredgecolor='blue', 
            markeredgewidth=3)
    plt.errorbar(wphot, mphot_map, label='Model photometry (MAP)',
            marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
            markerfacecolor='none', markeredgecolor='green', 
            markeredgewidth=3)
    plt.errorbar(wphot[mask], obs['maggies'][mask], yerr=obs['maggies_unc'][mask], 
            label='Observed photometry', ecolor='red', 
            marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
            markerfacecolor='none', markeredgecolor='red', 
            markeredgewidth=3)

    # plot transmission curves
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        plt.loglog(w, t, lw=3, color='gray', alpha=0.7)

    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux Density [maggies]')
    # plt.xlim([xmin, xmax])
    plt.xlim([xmin, xmax*5])
    plt.ylim([ymin, ymax])
    # add vertical stripes on plot to show min and max wavelengths of Al V, Pf-5, Ca IV, Pf-delta
    cws = [29053, 30392, 32070, 32970][2:]
    labels = ['Ca IV', 'Pf-delta']
    colors = ['green', 'red']
    for i in range(2):
        plt.axvspan(cws[i]*9, cws[i]*10, color=colors[i], alpha=0.3)
        plt.axvline(cws[i]*9.32, color=colors[i], alpha=0.7, ls='--', lw=2, label=labels[i])
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()
    plt.show()

# -----------------------
# 10) main program
# -----------------------
if __name__ == "__main__":
    ## To add passable parameters:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=str, default=run_params["outfile"])
    parser.add_argument("--showplots", action="store_true")
    parser.add_argument("--fitnewmodel", action="store_true")
    parser.add_argument("--readfile", type=str, default=None)
    parser.add_argument("--minimize", action="store_true")
    args = parser.parse_args()
    run_params["outfile"] = args.outfile
    run_params["showplots"] = args.showplots
    run_params["fitnewmodel"] = args.fitnewmodel
    run_params["readfile"] = args.readfile
    run_params["minimize"] = args.minimize

    start_time = time.perf_counter()

    obs = build_obs(zlo=zlo, zup=zup)
    print(f'obs built after {time.perf_counter() - start_time:.2f} seconds \n\n')

    model = build_model()
    print(f'model built after {time.perf_counter() - start_time:.2f} seconds \n\n')

    sps = build_sps()
    print(f'sps built after {time.perf_counter() - start_time:.2f} seconds \n\n')


    if run_params["fitnewmodel"]:
        # Seems like fit_model can just minimize already, so maybe we don't need to do a separate minimize step. But if we do, we can uncomment the following lines:
        # if run_params["minimize"]:
        #     ### MINIMISE
        #     minimize_output = minimize(obs, model, sps)
        #     (results, topt) = minimize_output["optimization"]
        #     # Find which of the minimizations gave the best result, 
        #     # and use the parameter vector for that minimization
        #     ind_best = np.argmin([r.cost for r in results])
        #     theta_best = results[ind_best].x.copy()
        #     print("minimization done")
        
        print(f'Fitting model using {fit_method}...\n\n')

        # Use multiprocessing to run the model fitting in parallel
        with Pool(processes=40) as pool:
            print(f'Using {pool._processes} processes for parallel fitting.\n\n')
            run_params["pool"] = pool
            run_params["queue_size"] = 40
            results = fit_model(obs, model, sps, **run_params)

        run_params.pop("pool", None)
        run_params.pop("queue_size", None)

        #results = fit_model(obs, model, sps, **run_params)
        
        sampling_result, tsample = results["sampling"]
        opt_result, toptimize = results["optimization"]
        print(type(results))
        print(results.keys() if isinstance(results, dict) else results)
        print(f'model fit after {time.perf_counter() - start_time:.2f} seconds')

        # Make the SPS library names into strings so they can be JSON serialized
        # May need a try - but if so, catch the specific error and print a warning that the SPS library names are not being converted to strings
        try:
            run_params['sps_libraries'] = tuple([library.decode() for library in run_params['sps_libraries']])
        except KeyError:
            print("Warning: 'sps_libraries' key not found in run_params. Skipping conversion to strings.")

        # Cast initial values of model parameters to lists if they're numpy arrays
        # so they can be JSON serialized
        for param in model.config_list:
            if isinstance(param['init'], np.ndarray):
                param['init'] = param['init'].tolist()

        write_results.write_hdf5(
            hfile=run_params["outfile"], 
            run_params=run_params, 
            model=model, 
            obs=obs, 
            sampler=sampling_result, 
            optimize_result_list=opt_result,
            tsample=tsample,
            toptimize=toptimize,
            sps=sps
            )

        with open(f'Prospector/{run_params["outfile"]+" generation time"}.txt', 'w') as f:
            f.write(f'Fitting model using {fit_method}...\n\n')
            f.write(f'model fit after {time.perf_counter() - start_time:.2f} seconds\n\n')
            f.write(f'run_params: {run_params}\n\n')
            f.write(f'sampling_result: {sampling_result}\n\n')
            f.write(f'opt_result: {opt_result}\n\n')


        # comment out if you don't want to set this up!
        send_email(
            "Prospector run finished",
            f"Your prospector run finished in {(time.perf_counter() - start_time)/3600:.2f} hours."
        )

    if run_params["showplots"]:
        plot_results(run_params, model, sps)