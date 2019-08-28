import importlib
import sys

import numpy as np

import pymc3 as pm

import bayes_workshop.conf


def run_model(model_name):

    conf = bayes_workshop.conf.get_conf()

    model_def = importlib.import_module(
        "bayes_workshop.models." + model_name
    )

    model = model_def.get_model()

    with model:
        trace = pm.sample_prior_predictive(
            samples=conf.n_prior_samples,
            random_seed=conf.prior_seeds[model_name]
        )

    np.savez(file=get_path(model_name, conf), **trace)

    return trace


def load(model_name):

    conf = bayes_workshop.conf.get_conf()

    trace = np.load(file=get_path(model_name, conf))

    return trace


def get_path(model_name, conf):

    return str(conf.results_dir / (model_name + "_prior_pred.npz"))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Need to provide a model name.")

    (_, curr_model_name) = sys.argv

    run_model(model_name=curr_model_name)
