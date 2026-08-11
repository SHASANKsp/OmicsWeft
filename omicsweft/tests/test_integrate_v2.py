import numpy as np
import pytest

from omicsweft.core.registry import INTEGRATORS
from omicsweft.integrate import JointNMF
from omicsweft.tasks import ClusteringTask


def _ari(emb, data):
    return ClusteringTask(label_key="subtype").evaluate(emb, data)["ari"]


def test_jnmf_recovers_structure(synth):
    emb = JointNMF(n_components=8, max_iter=100).fit_transform(synth)
    assert emb.X.shape == (90, 8)
    assert (emb.X >= 0).all()  # non-negative coefficients
    assert "loadings" in emb.meta
    assert _ari(emb, synth) > 0.5


def test_mofa_runs_and_exposes_loadings(synth):
    pytest.importorskip("mofapy2")
    emb = INTEGRATORS.create("mofa", n_factors=8).fit_transform(synth)
    assert emb.X.shape[0] == 90
    assert set(emb.meta["loadings"].keys()) == set(synth.modalities)
    assert _ari(emb, synth) > 0.5


def test_mvae_runs(synth):
    pytest.importorskip("torch")
    emb = INTEGRATORS.create("mvae", n_components=16, epochs=40).fit_transform(synth)
    assert emb.X.shape == (90, 16)
    assert np.isfinite(emb.X).all()


def test_mogonet_supervised(synth):
    pytest.importorskip("torch")
    emb = INTEGRATORS.create(
        "mogonet", label_key="subtype", epochs=60
    ).fit_transform(synth)
    assert emb.X.shape[0] == 90
    assert emb.meta.get("supervised") is True


def test_mvae_handles_partial_overlap(synth_partial):
    pytest.importorskip("torch")
    emb = INTEGRATORS.create("mvae", n_components=8, epochs=30).fit_transform(synth_partial)
    # union keeps all samples; common keeps the shared set — default is common
    assert emb.n_samples == len(synth_partial.common_samples())
