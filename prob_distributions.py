import numpy as np


def generate_dist(mean, thickness, sample_size):
    if len(mean) == 1:
        mu = [mean]; std = [thickness]
        return np.random.normal(mu[0], std[0], sample_size)
    else:
        this_dists = []
        for i in range(len(mean)):
            this_dists.append(np.random.normal(mean[i], thickness[i], sample_size))

        return np.concatenate(this_dists)
