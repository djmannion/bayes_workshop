
import collections

import numpy as np


def pred_to_dist(durations, pred):

    levels = np.sort(np.unique(durations))

    dist = []

    for level in levels:

        i_trials = (durations == level)

        n_trials = np.sum(i_trials)

        # sum over repeat trials
        n_longer_responses = np.sum(pred[:, i_trials], axis=1)

        # will have `n_pred` props after this
        props = n_longer_responses / n_trials

        # count up the unique instances
        prop_k = collections.Counter(props)

        # get a table of (response_prop, pred_prop) pairs
        prop = np.array(list(prop_k.items()))

        # prepend a column with the durations
        prop = np.concatenate(
            (
                np.ones((prop.shape[0], 1)) * level,
                prop
            ),
            axis=1
        )

        dist.append(prop)

    # put into one big array
    dist = np.concatenate(dist)

    return dist


def raw_to_prop(durations, responses):

    levels = np.sort(np.unique(durations))
    n_levels = len(levels)

    prop_data = np.full((n_levels, 2), np.nan)

    for (i_level, level) in enumerate(levels):

        i_trials = (durations == level)

        n_trials = np.sum(i_trials)

        n_longer_responses = np.sum(responses[i_trials])

        prop_data[i_level, 0] = level
        prop_data[i_level, 1] = n_longer_responses / n_trials

    assert np.sum(np.isnan(prop_data)) == 0

    return prop_data


def logistic(x, alpha, beta):

    theta = 1 / (1 + np.exp(-((x - alpha) / beta)))

    return theta
