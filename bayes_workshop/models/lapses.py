import numpy as np

import pymc3 as pm

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

        psi = pm.Uniform("psi", lower=0.0, upper=1.0)

        lapse_response = pm.Bernoulli(
            "lapse_response",
            p=psi,
            shape=n_trials
        )

        phi = pm.Uniform("phi", lower=0.0, upper=1.0)

        z = pm.Bernoulli("z", p=phi, shape=n_trials)

        theta_pf = bayes_workshop.utils.logistic(
            x=cmp_dur,
            alpha=alpha,
            beta=beta
        )

        theta = pm.math.switch(
            z,
            lapse_response,
            theta_pf
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
