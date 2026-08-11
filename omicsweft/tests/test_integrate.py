from omicsweft.core.registry import INTEGRATORS
from omicsweft.integrate import SNF, ConcatPCA


def test_concat_pca_shape(synth):
    emb = ConcatPCA(n_components=8).fit_transform(synth)
    assert emb.X.shape == (90, 8)
    assert emb.n_samples == len(emb.samples)
    assert emb.method == "concat_pca"


def test_snf_shape(synth):
    emb = SNF(n_components=6, k=15, n_iters=10).fit_transform(synth)
    assert emb.X.shape[0] == 90
    assert emb.X.shape[1] == 6
    assert "fused_affinity" in emb.meta


def test_integrators_on_partial_overlap(synth_partial):
    # integrators should fall back to the shared sample set
    emb = ConcatPCA(n_components=5).fit_transform(synth_partial)
    common = synth_partial.common_samples()
    assert emb.n_samples == len(common)


def test_registry_create(synth):
    emb = INTEGRATORS.create("concat_pca", n_components=4).fit_transform(synth)
    assert emb.X.shape[1] == 4
    assert set(["concat_pca", "snf"]).issubset(set(INTEGRATORS.list()))
