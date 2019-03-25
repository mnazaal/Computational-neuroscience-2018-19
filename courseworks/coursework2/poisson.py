import random as rnd
import numpy as np

def get_spike_train(rate, big_t, tau_ref):

    if 1 <= rate*tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []

    exp_rate = rate/(1-tau_ref*rate)

    spike_train = []

    t = rnd.expovariate(exp_rate)

    while t < big_t:
        spike_train.append(t)
        t += tau_ref+rnd.expovariate(exp_rate)

    return spike_train


Hz = 1.0
sec = 1.0
ms = 0.001


rate = 35.0 * Hz
tau_ref = 5*ms
big_t = 5*sec

spike_train = get_spike_train(rate, big_t, tau_ref)

def fano_factor_of_provided_spike_train():
    # Assuming no intervals to bin the data
    return np.var(spike_train)/np.mean(spike_train)
