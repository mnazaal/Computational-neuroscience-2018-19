import numpy as np
from poisson import get_spike_train, Hz, sec, ms
from load import load_data
import matplotlib.pyplot as plt

# QUESTION 1
def fano_factor_q1():
    T_q1            = 1000*sec
    rate_q1         = 35*Hz
    ref_periods_q1  = [0, 5*ms]
    window_width_q1 = [10*ms, 50*ms, 100*ms]

    fano_factors = []

    for ref in ref_periods_q1:
        for wind in window_width_q1:
            bins        = [wind*i for i in range(int(T_q1))]
            
            #intervals    = int(T_q1/wind)
            spike_trains = get_spike_train(rate_q1, T_q1, ref)

            # Split into  intervals
            hist, bins  = np.histogram(spike_trains, bins)

            # Compute statistics
            fano_factors.append(np.var(hist)/np.mean(hist))

    return fano_factors


def coeff_variation_q1():
    T_q1           = 1000*sec
    rate_q1        = 35*Hz
    ref_periods_q1 = [0, 5*ms]

    coeffs_variation = []

    for ref in ref_periods_q1:
        spike_trains         = get_spike_train(rate_q1, T_q1, ref)
        interspike_intervals = [t2 - t1 for t1,
                                t2 in zip(spike_trains, spike_trains[1:])]

        coeffs_variation.append(
            np.std(interspike_intervals)/np.mean(interspike_intervals))

    return coeffs_variation


# fano_factor_q1()
# coeff_variation_q1()
print(fano_factor_q1())
print(coeff_variation_q1())
# QUESTION 2


spikes = load_data("rho.dat", int)
# First get data as array with time index when spiked


def time_indexer(vals):  # :: [Int] -> [Float]
    # Convert list of 0s and 1s to times of the spikes
    time_interval = 2*ms

    return list(filter(None, map(lambda pair: pair[0]*time_interval if (pair[1] == 1) else None, zip(range(len(vals)), vals))))


def fano_factor_q2():
    T_q1            = 1000*sec
    window_width_q1 = [10*ms, 50*ms, 100*ms]
    spike_times     = time_indexer(spikes)

    fano_factors = []

    for wind in window_width_q1:
        bins         = [wind*i for i in range(int(T_q1))]

        hist, bins  = np.histogram(spike_times, bins)

        # Compute statistics
        fano_factors.append(np.var(hist)/np.mean(hist))

    return fano_factors


def coeff_variation_q2():
    spike_trains = time_indexer(spikes)
    interspike_intervals = [t2 - t1 for t1,
                            t2 in zip(spike_trains, spike_trains[1:])]
    return np.std(interspike_intervals)/np.mean(interspike_intervals)


print(fano_factor_q2())
print(coeff_variation_q2())


def spike_trigg_avg_q3():
    window    = 100*ms
    intervals = int((20*60*sec)/(window*1000))
    stimulus  = load_data("stim.dat", float)
    bins         = [window*i for i in range(int(len(stimulus)/1000))]
    hist, bins   = np.histogram(stimulus, bins)

    stimulus_binned = np.array(np.array_split(stimulus, intervals))           # X in Wikipedia
    count_per_bin   = np.array(list(map(lambda l : len(l), stimulus_binned))) # y in Wikipedia
    total_count     = np.sum(count_per_bin)                                   # n_sp in Wikipedia

    STA             = (1/total_count)* (stimulus_binned.T @ count_per_bin)
    
    time_axis = [100*ms*i for i in range(len(STA))]
    plt.plot(time_axis, STA)
    plt.show()
    return time_axis, STA

spike_trigg_avg_q3()