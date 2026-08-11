import pytest

from omicsweft.integrate import ConcatPCA
from omicsweft.tasks import ClassificationTask, ClusteringTask, RegressionTask


@pytest.fixture
def emb(synth):
    return ConcatPCA(n_components=10).fit_transform(synth)


def test_clustering_recovers_structure(emb, synth):
    res = ClusteringTask(label_key="subtype").evaluate(emb, synth)
    # synthetic clusters are well separated -> high agreement
    assert res["ari"] > 0.5
    assert res["nmi"] > 0.5
    assert "silhouette" in res


def test_classification(emb, synth):
    res = ClassificationTask(label_key="subtype", model="logreg").evaluate(emb, synth)
    assert res["macro_f1"] > 0.6
    assert 0.0 <= res.get("auroc", 0.0) <= 1.0


def test_regression(emb, synth):
    res = RegressionTask(target_key="marker", model="ridge").evaluate(emb, synth)
    assert "rmse" in res and "r2" in res


def test_survival_is_optional(emb, synth):
    # Should either compute (if lifelines present) or raise a clear ImportError.
    from omicsweft.tasks import SurvivalTask

    task = SurvivalTask(time_key="os_time", event_key="os_event")
    try:
        res = task.evaluate(emb, synth)
    except ImportError as exc:
        assert "survival" in str(exc).lower()
    else:
        assert "logrank_p" in res and "c_index" in res


def test_agnostic_dataset_never_needs_survival(synth):
    # A dataset with no time-to-event columns runs the full core pipeline
    # (integrate -> classify) without touching survival at all.
    data = synth
    data._sheet = data._sheet.drop(columns=["os_time", "os_event"])
    emb = ConcatPCA(n_components=6).fit_transform(data)
    res = ClusteringTask(label_key="subtype").evaluate(emb, data)
    assert "ari" in res
