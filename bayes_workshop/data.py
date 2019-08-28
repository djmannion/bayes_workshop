
import numpy as np
import scipy.io

import bayes_workshop.conf


def get_data():

    conf = bayes_workshop.conf.get_conf()

    raw = scipy.io.loadmat(
        file_name=str(conf.data_file),
        squeeze_me=True,
        struct_as_record=False
    )["d"]

    cols = get_columns()

    data = np.full((conf.n_total_data_rows, len(cols)), np.nan)

    i_row = 0

    for i_subj in range(conf.n_subj):

        for (i_modality, modality) in enumerate(conf.modalities):

            for i_trial in range(conf.n_trials_per_cond):

                target_duration = (
                    getattr(raw, modality + "Stimulus")[i_subj, i_trial]
                )

                target_longer = (
                    getattr(raw, modality + "Decision")[i_subj, i_trial]
                )

                i_block = int(np.floor(i_trial / conf.n_trials_per_block))

                data[i_row, cols.index("i_subj")] = i_subj
                data[i_row, cols.index("i_modality")] = i_modality
                data[i_row, cols.index("i_block")] = i_block
                data[i_row, cols.index("i_trial")] = i_trial
                data[i_row, cols.index("target_duration")] = target_duration
                data[i_row, cols.index("target_longer")] = target_longer

                i_row += 1

    assert i_row == conf.n_total_data_rows

    # remove the NaNs
    nan_resp = np.isnan(data[:, cols.index("target_longer")])
    data = data[np.logical_not(nan_resp), :]

    # can now safely convert to integers
    data = data.astype("int")

    return (cols, data)


def get_columns():

    cols = [
        "i_subj",
        "i_modality",
        "i_block",
        "i_trial",
        "target_duration",
        "target_longer"
    ]

    return cols


if __name__ == "__main__":

    (demo_cols, demo_data) = get_data()

    print(demo_cols)
    print(demo_data)
