import numpy as np

import pymc3 as pm
import theano.tensor as tt

import bayes_workshop.conf
import bayes_workshop.data
import bayes_workshop.utils


def get_model():

    conf = bayes_workshop.conf.get_conf()

    (cols, data) = bayes_workshop.data.get_data()

    demo_subj_id = 2
    demo_modality = "visual"

    i_demo_trials = np.logical_and(
        data[:, cols.index("i_subj")] == demo_subj_id,
        (
            data[:, cols.index("i_modality")] ==
            conf.modalities.index(demo_modality)
        )
    )

    demo_data = data[i_demo_trials, :]

    (n_trials, _) = demo_data.shape

    responses = demo_data[:, cols.index("target_longer")]
    cmp_dur = demo_data[:, cols.index("target_duration")]

    with pm.Model() as model:

        alpha = pm.Normal(
            "alpha",
            mu=conf.standard_ms,
            sd=50.0
        )

        beta = pm.HalfNormal("beta", sd=100.0)

        lapse_bias = pm.Uniform("psi", lower=0.0, upper=1.0)

        # probability of lapsing on each trial
        lapse_p = pm.Uniform("phi", lower=0.0, upper=1.0)

        lapse_ind = pm.Bernoulli("lapse_ind", p=lapse_p, shape=n_trials)

        theta_pf = bayes_workshop.utils.logistic(
            x=cmp_dur,
            alpha=alpha,
            beta=beta
        )

        obs_pf = pm.Bernoulli.dist(p=theta_pf)
        obs_lapse = pm.Bernoulli.dist(p=lapse_bias, shape=n_trials)

        mix = pm.Mixture(
            "obs",
            w=tt.stack([1.0 - lapse_ind, lapse_ind], axis=1),
            comp_dists=[obs_pf, obs_lapse],
            observed=responses
        )

    return model


if __name__ == "__main__":
    demo_model = get_model()

    print(vars(demo_model))
