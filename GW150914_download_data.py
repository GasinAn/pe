"""
This code is modified from bilby example `Tutorial to demonstrate running parameter estimation on GW150914`, https://git.ligo.org/lscsoft/bilby/blob/master/examples/gw_examples/data_examples/GW150914.py.
"""

import pickle
import bilby
from gwosc import datasets
from gwpy.timeseries import TimeSeries

logger = bilby.core.utils.logger
label = "GW150914"
outdir = label+"_result"

# We now use gwpy to obtain analysis and psd data and create the ifo_list.
detectors = list(datasets.event_detectors(label))

trigger_time = datasets.event_gps(label)
post_trigger_duration = 2 
duration = 4
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s.
psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

maximum_frequency = 512
minimum_frequency = 20

ifo_list = bilby.gw.detector.InterferometerList([])
for det in detectors:
    ifo = bilby.gw.detector.get_empty_interferometer(det)

    logger.info("Downloading analysis data for ifo {}".format(det))
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )

    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency

    ifo_list.append(ifo)

with open(label+"_data.pickle", "wb") as file:
    pickle.dump(ifo_list, file)
