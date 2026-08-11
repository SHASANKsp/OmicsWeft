import pytest

from omicsweft.core.registry import INTERPRETERS
from omicsweft.integrate import ConcatPCA, JointNMF


def test_factor_loadings_from_jnmf(synth):
    emb = JointNMF(n_components=6, max_iter=80).fit_transform(synth)
    out = INTERPRETERS.create("factor_loadings", top_n=10).interpret(emb, synth)
    assert set(out["top_features"].keys()) == set(synth.modalities)
    # each factor yields a ranked series
    rna = out["top_features"]["rna"]
    assert len(rna["factor_0"]) == 10


def test_factor_loadings_requires_loadings(synth):
    emb = ConcatPCA(n_components=6).fit_transform(synth)  # no loadings in meta
    with pytest.raises(ValueError):
        INTERPRETERS.create("factor_loadings").interpret(emb, synth)


def test_embedding_correlation_model_agnostic(synth):
    emb = ConcatPCA(n_components=8).fit_transform(synth)
    out = INTERPRETERS.create("embedding_correlation", top_n=15).interpret(emb, synth)
    assert "rna" in out["top_features"]
    assert list(out["top_features"]["rna"].columns) == ["feature", "max_abs_corr"]


def test_permutation_importance(synth):
    emb = ConcatPCA(n_components=8).fit_transform(synth)
    out = INTERPRETERS.create(
        "permutation_importance", label_key="subtype", n_repeats=5
    ).interpret(emb, synth)
    imp = out["dimension_importance"]
    assert len(imp) == 8
    assert imp["importance"].is_monotonic_decreasing
