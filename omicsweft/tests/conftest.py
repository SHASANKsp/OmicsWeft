import pytest

from omicsweft.datasets import make_synthetic, write_generic


@pytest.fixture
def synth():
    return make_synthetic(n_samples=90, n_clusters=3, seed=1)


@pytest.fixture
def synth_partial():
    return make_synthetic(n_samples=90, n_clusters=3, partial_overlap=0.15, seed=2)


@pytest.fixture
def generic_dir(tmp_path):
    data = make_synthetic(n_samples=90, n_clusters=3, seed=3)
    return write_generic(data, tmp_path / "study")
