"""
This code is modified from bilby example `Tutorial to demonstrate running
parameter estimation on GW150914`, https://git.ligo.org/lscsoft/bilby/-/blob/
master/examples/gw_examples/data_examples/GW150914.py, and `Tutorial to
demonstrate running parameter estimation on GW190425`, https://git.ligo.org/
lscsoft/bilby/-/blob/master/examples/gw_examples/data_examples/GW190425.py.
"""

import pickle
import numpy as np
import h5py
import bilby
from gwosc import datasets

logger = bilby.core.utils.logger
label = "GW150914"
outdir = label+"_result"

trigger_time = datasets.event_gps(label)

fmax = 512
fmin = 20

with open(label+"_data.pickle", "rb") as file:
    ifo_list = pickle.load(file)

logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass.

### Here the waveform is modified.
def modified_model(
        frequency_array, xi, Omega, mass_1, mass_2, luminosity_distance,
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, fiducial,
        **kwargs):
    strain = bilby.gw.source.lal_binary_black_hole_relative_binning(
        frequency_array, mass_1, mass_2, luminosity_distance,
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, fiducial,
        **kwargs)
    if (fiducial == 1):
        f = frequency_array
        where = np.all((f>fmin, f<fmax), axis=0)
        gamma = (1+xi*f[where]**(-2))*np.exp(1j*Omega*f[where]**(-1))
        for key in strain.keys():
            strain[key][where] *= gamma
    else:
        f = kwargs["frequency_bin_edges"]
        gamma = (1+xi*f**(-2))*np.exp(1j*Omega*f**(-1))
        for key in strain.keys():
            strain[key] *= gamma
    return strain

waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=modified_model,
    parameter_conversion=(
        bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    ),
    waveform_arguments={
        "waveform_approximant": "IMRPhenomXP",
        "reference_frequency": 50,
    },
)

# We now define the prior.
# We have defined our prior distribution in a local file, GW150914.prior
# The prior is printed to the terminal at run-time.
# You can overwrite this using the syntax below in the file,
# or choose a fixed value by just providing a float value as the prior.
priors = bilby.gw.prior.BBHPriorDict(filename="GW150914.prior")
priors["fiducial"] = 0

# Add geocent_time prior.
priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
)

### Add xi and Omega prior.
priors["xi"] = bilby.core.prior.Uniform(-1, 1, name="xi")
priors["Omega"] = bilby.core.prior.Uniform(-1, 1, name="Omega")

### The fiducial parameters are taken to me the mean of samples in posterior
### sample release of https://zenodo.org/api/files/
### ecf41927-9275-47da-8b37-e299693fe5cb/
### IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5, which is 
### renamed as GW150914_posterior.h5.

# The fiducial parameters should always be in provided in the same basis as
# the sampling basis. For example, if sampling in  `mass_1` and `mass_2`
# instead of `chirp_mass` and `mass_ratio`, the fiducial parameters should also
# be provided in `mass_1` and `mass_2` below.
fiducial_parameters = {}
with h5py.File("GW150914_posterior.h5", "r") as file:
    dset = file['C01:IMRPhenomXPHM']['posterior_samples']
    for key in priors.keys():
        if (key not in ("fiducial", "mass_1", "mass_2", "xi", "Omega")):
            fiducial_parameters[key] = dset[key].mean()
fiducial_parameters["xi"] = 0
fiducial_parameters["Omega"] = 0

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
    ifo_list,
    waveform_generator,
    priors=priors,
    fiducial_parameters=fiducial_parameters,
    phase_marginalization=True,
    distance_marginalization=True,
    time_marginalization=False,
)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data.
if (__name__ == "__main__"):
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="dynesty",
        outdir=outdir,
        label=label,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        npool=1,
        nlive=1000,
        check_point_delta_t=600,
        check_point_plot=True,
    )
    result.plot_corner()
