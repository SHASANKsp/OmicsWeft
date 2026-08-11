from omicsweft.bench import robustness_curve


def test_robustness_noise(synth):
    df = robustness_curve(
        synth, "concat_pca", "clustering", "ari",
        levels=(0.0, 0.5), kind="noise", n_repeats=2,
        integrator_params={"n_components": 10},
        task_params={"label_key": "subtype"},
    )
    assert list(df["level"]) == [0.0, 0.5]
    assert "ari_mean" in df.columns
    assert (df["n"] == 2).all()


def test_robustness_dropout_degrades(synth):
    df = robustness_curve(
        synth, "concat_pca", "clustering", "ari",
        levels=(0.0, 0.8), kind="dropout", n_repeats=2,
        integrator_params={"n_components": 10},
        task_params={"label_key": "subtype"},
    )
    clean = df.loc[df["level"] == 0.0, "ari_mean"].iloc[0]
    heavy = df.loc[df["level"] == 0.8, "ari_mean"].iloc[0]
    # heavy dropout should not improve clustering agreement
    assert heavy <= clean + 1e-6


def test_copy_is_independent(synth):
    # perturbation must not mutate the original dataset
    import numpy as np

    before = np.array(synth.get_omic("rna").X, copy=True)
    robustness_curve(
        synth, "concat_pca", "clustering", "ari",
        levels=(0.0, 1.0), kind="noise", n_repeats=1,
        integrator_params={"n_components": 5},
        task_params={"label_key": "subtype"},
    )
    after = np.asarray(synth.get_omic("rna").X)
    assert np.allclose(before, after)
