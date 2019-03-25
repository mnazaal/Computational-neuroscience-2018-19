import numpy as np
from poisson import get_spike_train, Hz, sec, ms
from load import load_data
import matplotlib.pyplot as plt

# QUESTION 1


def interval_statistics(vals):  # :: [[Float]] -> (Float, Float)
    # Computes mean and variance for data in bins/intervals, assuming midpoints
    # are the arithmetic mean of the max and min value in the bin/interval,
    # as opposed to the mid points of the bin/interval
    total      = (len(vals)*len(vals[0]))
    mid_points = list(map(lambda l: ((l[0] + l[-1])/2), vals))
    counts     = list(map(lambda l: len(l), vals))

    mean         = sum([a*b for a, b in zip(mid_points, counts)])/total
    mean_squared = sum([a*a*b for a, b in zip(mid_points, counts)])/total

    return mean, (mean_squared - mean*mean)


def fano_factor_q1():
    T_q1            = 1000*sec
    rate_q1         = 35*Hz
    ref_periods_q1  = [0, 5*ms]
    window_width_q1 = [10*ms, 50*ms, 100*ms]

    fano_factors = []

    for ref in ref_periods_q1:
        for wind in window_width_q1:
            intervals    = int(T_q1/(wind*1000))
            spike_trains = get_spike_train(rate_q1, T_q1, ref)

            # Split into  intervals
            spike_trains = np.array_split(spike_trains, intervals)

            # Compute statistics
            mean, var = interval_statistics(spike_trains)
            fano_factors.append(var/mean)

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
        intervals = int(T_q1/(wind*1000))

        # Split into  intervals
        spike_trains = np.array_split(spike_times, intervals)

        # Compute statistics
        mean, var = interval_statistics(spike_trains)
        fano_factors.append(var/mean)

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

    stimulus_binned = np.array(np.array_split(stimulus, intervals))           # X in Wikipedia
    count_per_bin   = np.array(list(map(lambda l : len(l), stimulus_binned))) # y in Wikipedia
    total_count     = np.sum(count_per_bin)                                   # n_sp in Wikipedia
    STA             = (1/total_count)* (stimulus_binned.T @ count_per_bin)
    
    time_axis = [100*ms*i for i in range(len(STA))]
    plt.plot(time_axis, STA)
    plt.show()
    return time_axis, STA

spike_trigg_avg_q3()