import numpy as np

from omicsweft.preprocess import (
    BatchLinear,
    SelectVariance,
    Standardize,
    compute_qc,
    run_pipeline,
)


def test_standardize(synth):
    out = Standardize().apply(synth)
    X = np.asarray(out.get_omic("rna").X)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X.std(axis=0), 1, atol=1e-6)


def test_feature_selection_reduces_width(synth):
    before = synth.get_omic("rna").shape[1]
    out = SelectVariance(k=50).apply(synth)
    assert out.get_omic("rna").shape[1] == min(50, before)


def test_batch_linear_removes_mean_shift(synth):
    # inject a batch shift then correct it
    ad = synth.get_omic("rna")
    batch = synth.labels("batch").reindex(ad.obs_names).to_numpy()
    X = np.array(ad.X, dtype=float, copy=True)
    X[batch == "batch_B"] += 10.0
    ad.X = X
    out = BatchLinear(batch_key="batch").apply(synth)
    Xc = np.asarray(out.get_omic("rna").X)
    m_a = Xc[batch == "batch_A"].mean(axis=0)
    m_b = Xc[batch == "batch_B"].mean(axis=0)
    assert np.allclose(m_a, m_b, atol=1e-6)


def test_pipeline_chaining(synth):
    out = run_pipeline(synth, [Standardize(), SelectVariance(k=100)])
    assert out.get_omic("rna").shape[1] == 100


def test_qc_report(synth):
    rep = compute_qc(synth)
    assert set(rep["modality"]) == {"rna", "methylation", "protein"}
    assert (rep["n_samples"] == 90).all()
