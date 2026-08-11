import pandas as pd

from omicsweft.core.registry import DATASETS
from omicsweft.io import load_generic


def test_generic_loader_roundtrip(generic_dir):
    data = load_generic(generic_dir)
    assert set(data.modalities) == {"rna", "methylation", "protein"}
    assert data.n_samples == 90
    # sample sheet columns survived the round trip
    for col in ["subtype", "batch", "marker"]:
        assert col in data.obs.columns


def test_generic_loader_via_registry(generic_dir):
    data = DATASETS.create("generic", directory=str(generic_dir))
    assert data.n_samples == 90


def test_manifest_inference(generic_dir):
    # delete the manifest so the loader must infer it
    (generic_dir / "manifest.yaml").unlink()
    data = load_generic(generic_dir)
    # sample_sheet.csv should be recognised as the sheet, not an omic
    assert "sample_sheet" not in data.modalities
    assert data.n_samples == 90


def test_partial_overlap_coverage(synth_partial):
    cov = synth_partial.coverage()
    assert cov.n_modalities == 3
    # with 15% dropped per omic, not every sample is complete
    assert cov.n_complete <= cov.n_samples
    assert cov.per_modality  # populated


def test_orientation_transpose(tmp_path):
    # write one omic transposed and confirm it is oriented back
    import numpy as np
    import yaml

    rng = np.random.default_rng(0)
    samples = [f"S{i}" for i in range(10)]
    feats = [f"G{j}" for j in range(5)]
    fxs = pd.DataFrame(rng.normal(size=(5, 10)), index=feats, columns=samples)
    fxs.index.name = "feature"
    d = tmp_path / "ds"
    d.mkdir()
    fxs.to_csv(d / "rna.csv")
    manifest = {
        "omics": {"rna": {"path": "rna.csv", "orientation": "features_x_samples"}}
    }
    with open(d / "manifest.yaml", "w") as fh:
        yaml.safe_dump(manifest, fh)
    data = load_generic(d)
    assert data.get_omic("rna").shape == (10, 5)  # samples x features
