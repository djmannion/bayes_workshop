import numpy as np

import pymc3 as pm

import bayes_workshop.conf
import bayes_workshop.data
import bayes_workshop.utils


def get_model():

    conf = bayes_workshop.conf.get_conf()

    (cols, data) = bayes_workshop.data.get_data()

    demo_subj_id = 8
    i_demo_trials = (data[:, cols.index("i_subj")] == demo_subj_id)
    demo_data = data[i_demo_trials, :]

    (n_trials, _) = demo_data.shape

    i_modality = demo_data[:, cols.index("i_modality")]
    responses = demo_data[:, cols.index("target_longer")]
    cmp_dur = demo_data[:, cols.index("target_duration")]

    i_audio = (i_modality == 0)
    i_visual = (i_modality == 1)

    with pm.Model() as model:

        alpha = pm.Normal(
            "alpha",
            mu=conf.standard_ms,
            sd=50.0
        )

        beta = pm.HalfNormal("beta", sd=100.0)

        theta_audio = bayes_workshop.utils.logistic(
            x=cmp_dur[i_audio],
            alpha=alpha,
            beta=beta
        )

        theta_visual = bayes_workshop.utils.logistic(
            x=cmp_dur[i_visual],
            alpha=alpha,
            beta=beta
        )

        obs_audio = pm.Bernoulli(
            "obs_audio",
            p=theta_audio,
            observed=responses[i_audio]
        )

        obs_visual = pm.Bernoulli(
            "obs_visual",
            p=theta_visual,
            observed=responses[i_visual]
        )

    return model


if __name__ == "__main__":
    demo_model = get_model()

    print(vars(demo_model))
