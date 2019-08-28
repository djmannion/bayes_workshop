import importlib
import sys

import pymc3 as pm

import bayes_workshop.conf


def run_model(model_name):

    conf = bayes_workshop.conf.get_conf()

    model_def = importlib.import_module(
        "bayes_workshop.models." + model_name
    )

    model = model_def.get_model()

    with model:
        trace = pm.sample(
            draws=conf.n_mcmc_samples,
            random_seed=conf.mcmc_seeds[model_name],
            cores=conf.n_mcmc_chains,
            chains=conf.n_mcmc_chains,
            nuts_kwargs={"target_accept": 0.99}
        )

    pm.save_trace(
        trace=trace,
        directory=str(conf.results_dir / model_name),
        overwrite=True
    )

    return trace


def load_trace(model_name):

    conf = bayes_workshop.conf.get_conf()

    model_def = importlib.import_module(
        "bayes_workshop.models." + model_name
    )

    model = model_def.get_model()

    trace = pm.load_trace(
        directory=str(conf.results_dir / model_name),
        model=model
    )

    return trace


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Need to provide a model name.")

    (*_, curr_model_name) = sys.argv

    run_model(model_name=curr_model_name)
