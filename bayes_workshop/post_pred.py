import importlib
import sys

import numpy as np

import pymc3 as pm

import bayes_workshop.conf
import bayes_workshop.trace


def run_model(model_name):

    conf = bayes_workshop.conf.get_conf()

    model_def = importlib.import_module(
        "bayes_workshop.models." + model_name
    )

    model = model_def.get_model()

    trace = bayes_workshop.trace.load_trace(model_name=model_name)

    with model:
        ppc = pm.sample_posterior_predictive(
            trace=trace,
            samples=conf.n_post_pred_samples,
            random_seed=conf.post_pred_seeds[model_name]
        )

    np.savez(
        file=get_path(model_name, conf),
        **ppc
    )

    return trace


def load(model_name):

    conf = bayes_workshop.conf.get_conf()

    ppc = np.load(file=get_path(model_name, conf))

    return ppc


def get_path(model_name, conf):

    return str(conf.results_dir / (model_name + "_ppc.npz"))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Need to provide a model name.")

    (*_, curr_model_name) = sys.argv

    run_model(model_name=curr_model_name)
