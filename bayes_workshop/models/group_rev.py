import numpy as np
import scipy.stats

import pymc3 as pm

import bayes_workshop.conf
import bayes_workshop.data
import bayes_workshop.utils


def get_model():

    conf = bayes_workshop.conf.get_conf()

    (cols, data) = bayes_workshop.data.get_data()

    # from Lee
    i_subjects = (6, 8, 14, 17, 5, 2)
    modality = "visual"

    n_subj = len(i_subjects)

    i_trials = np.logical_and(
        np.isin(data[:, cols.index("i_subj")], i_subjects),
        data[:, cols.index("i_modality")] ==
        conf.modalities.index(modality)
    )

    data = data[i_trials, :]

    i_subj = scipy.stats.rankdata(
        data[:, cols.index("i_subj")],
        method="dense"
    ) - 1

    responses = data[:, cols.index("target_longer")]
    cmp_dur = data[:, cols.index("target_duration")]

    with pm.Model() as model:

        alpha_mu = pm.Normal(
            "alpha_mu",
            mu=conf.standard_ms,
            sd=50.0,
            testval=conf.standard_ms
        )

        alpha_sd = pm.Uniform(
            "alpha_sd",
            lower=0.01,
            upper=100.0,
            testval=5.0
        )

        alphas_offset = pm.Normal(
            "alphas_offset",
            mu=0.0,
            sd=1.0,
            shape=n_subj
        )

        alphas = pm.Deterministic(
            "alphas",
            alpha_mu + alpha_sd * alphas_offset
        )

        beta_mu = pm.TruncatedNormal(
            "beta_mu",
            mu=0.0,
            sd=100.0,
            lower=0.0,
            upper=10e6,
            testval=100.0
        )

        beta_sd = pm.Uniform(
            "beta_sd",
            lower=0.01,
            upper=100.0,
            testval=50.0
        )

        betas = pm.TruncatedNormal(
            "betas",
            mu=beta_mu,
            sd=beta_sd,
            lower=0.0,
            upper=10e6,
            shape=n_subj,
            testval=100.0
        )

        theta = bayes_workshop.utils.logistic(
            x=cmp_dur,
            alpha=alphas[i_subj],
            beta=betas[i_subj]
        )

        obs = pm.Bernoulli(
            "obs",
            p=theta,
            observed=responses
        )

    return model


if __name__ == "__main__":
    demo_model = get_model()

    print(vars(demo_model))
