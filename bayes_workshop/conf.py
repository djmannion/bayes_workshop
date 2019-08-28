
import types
import pathlib

import numpy as np


def get_conf():

    conf = types.SimpleNamespace()

    base_dir = pathlib.Path(__file__).absolute().parent.parent.parent.parent

    conf.results_dir = base_dir / "results"

    conf.data_file = base_dir / "data" / "vanDrielData2015.mat"

    conf.n_subj = 19

    conf.modalities = ["auditory", "visual"]
    conf.n_modalities = len(conf.modalities)

    conf.n_trials_per_cond = 240
    conf.n_trials_per_block = 80
    conf.n_blocks = int(conf.n_trials_per_cond / conf.n_trials_per_block)

    conf.standard_ms = 500

    conf.n_total_data_rows = (
        conf.n_subj * conf.n_modalities * conf.n_trials_per_cond
    )

    conf.fine_x = np.linspace(100.0, 900.0, 101)

    conf.model_names = [
        "demo_subj",
        "demo_subj_vague",
        "lapses",
        "lapses_mixture",
        "group",
        "group_rev",
        "av_cmp",
        "common_cause",
    ]

    conf.n_mcmc_samples = 10000
    conf.n_mcmc_chains = 4
    mcmc_rand = np.random.RandomState(3325449526)
    conf.mcmc_seeds = {
        model_name: mcmc_rand.randint(low=0, high=2 ** 30 - 1)
        for model_name in conf.model_names
    }

    # prior prob settings
    conf.n_prior_samples = 100
    prior_rand = np.random.RandomState(374048569)
    conf.prior_seeds = {
        model_name: prior_rand.randint(low=0, high=2 ** 30 - 1)
        for model_name in conf.model_names
    }

    # post prob settings
    conf.n_post_pred_samples = 100
    ppc_rand = np.random.RandomState(988210739)
    conf.post_pred_seeds = {
        model_name: ppc_rand.randint(low=0, high=2 ** 30 - 1)
        for model_name in conf.model_names
    }

    return conf


if __name__ == "__main__":

    demo_conf = get_conf()

    print(demo_conf)
