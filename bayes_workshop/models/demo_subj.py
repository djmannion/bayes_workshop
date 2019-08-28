import numpy as np

import pymc3 as pm

import bayes_workshop.conf
import bayes_workshop.data
import bayes_workshop.utils


def get_model():

    conf = bayes_workshop.conf.get_conf()

    (cols, data) = bayes_workshop.data.get_data()

    demo_subj_id = 6
    demo_modality = "visual"

    i_demo_trials = np.logical_and(
        data[:, cols.index("i_subj")] == demo_subj_id,
        (
            data[:, cols.index("i_modality")] ==
            conf.modalities.index(demo_modality)
        )
    )

    demo_data = data[i_demo_trials, :]

    responses = demo_data[:, cols.index("target_longer")]
    cmp_dur = demo_data[:, cols.index("target_duration")]

    with pm.Model() as model:

        alpha = pm.Normal(
            "alpha",
            mu=conf.standard_ms,
            sd=50.0
        )

        beta = pm.HalfNormal("beta", sd=100.0)

        theta = bayes_workshop.utils.logistic(
            x=cmp_dur,
            alpha=alpha,
            beta=beta
        )

        obs = pm.Bernoulli(
            "obs",
            p=theta,
            observed=responses
        )

        pf = pm.Deterministic(
            "pf",
            bayes_workshop.utils.logistic(
                x=conf.fine_x,
                alpha=alpha,
                beta=beta
            )
        )

    return model


if __name__ == "__main__":
    demo_model = get_model()

    print(vars(demo_model))
