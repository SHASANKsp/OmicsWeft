from omicsweft.bench import run, run_and_save
from omicsweft.core.config import Config


def _config(generic_dir):
    return Config(
        name="e2e",
        seed=0,
        dataset={"name": "generic", "params": {"directory": str(generic_dir)}},
        preprocess=[
            {"name": "standardize"},
            {"name": "select_variance", "params": {"k": 100}},
        ],
        integrators=[
            {"name": "concat_pca", "params": {"n_components": 10}},
            {"name": "snf", "params": {"n_components": 6, "n_iters": 10}},
        ],
        tasks=[
            {"name": "clustering", "params": {"label_key": "subtype"}},
            {"name": "classification", "params": {"label_key": "subtype"}},
        ],
    )


def test_benchmark_runs_end_to_end(generic_dir):
    results = run(_config(generic_dir))
    assert not results.empty
    assert set(results["integrator"].unique()) >= {"concat_pca", "snf"}
    # at least one real ARI value came through
    ari = results[(results["task"] == "clustering") & (results["metric"] == "ari")]
    assert (ari["value"].astype(float) > 0.3).any()


def test_config_roundtrip_and_hash(generic_dir, tmp_path):
    cfg = _config(generic_dir)
    p = tmp_path / "config.yaml"
    cfg.to_yaml(p)
    cfg2 = Config.from_yaml(p)
    assert cfg.hash() == cfg2.hash()


def test_run_and_save_writes_manifest(generic_dir, tmp_path):
    results = run_and_save(_config(generic_dir), tmp_path / "out")
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    assert (tmp_path / "out" / "config.yaml").exists()
    assert not results.empty
