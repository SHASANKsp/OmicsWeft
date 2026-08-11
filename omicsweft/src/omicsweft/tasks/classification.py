"""Supervised classification task on a joint embedding (domain-free)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

from ..core.base import Embedding, Task
from ..core.omicsdata import OmicsData
from ..core.registry import TASKS


def _make_model(name: str, random_state: int):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=random_state)
    raise ValueError(f"unknown model {name!r}; use 'logreg' or 'rf'")


@TASKS.register("classification")
class ClassificationTask(Task):
    """Cross-validated classification predicting a sample-sheet label."""

    kind = "classification"

    def __init__(
        self,
        label_key: str,
        model: str = "logreg",
        n_splits: int = 5,
        random_state: int = 0,
    ) -> None:
        self.label_key = label_key
        self.model = model
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, embedding: Embedding, data: OmicsData) -> dict:
        y_raw = data.labels(self.label_key).reindex(embedding.samples)
        mask = y_raw.notna().to_numpy()
        X = embedding.X[mask]
        y = LabelEncoder().fit_transform(y_raw[mask].astype(str))
        classes = np.unique(y)
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError("classification needs at least 2 classes with labels")

        n_splits = min(self.n_splits, int(np.bincount(y).min()))
        n_splits = max(n_splits, 2)
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        y_pred = np.empty_like(y)
        y_proba = np.zeros((len(y), n_classes))
        for tr, te in skf.split(X, y):
            clf = _make_model(self.model, self.random_state)
            clf.fit(X[tr], y[tr])
            y_pred[te] = clf.predict(X[te])
            proba = clf.predict_proba(X[te])
            # align columns to global class order
            for j, c in enumerate(clf.classes_):
                y_proba[te, list(classes).index(c)] = proba[:, j]

        out = {
            "n_classes": float(n_classes),
            "macro_f1": float(f1_score(y, y_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        }
        try:
            if n_classes == 2:
                out["auroc"] = float(roc_auc_score(y, y_proba[:, 1]))
                out["auprc"] = float(average_precision_score(y, y_proba[:, 1]))
            else:
                Yb = label_binarize(y, classes=classes)
                out["auroc"] = float(
                    roc_auc_score(Yb, y_proba, average="macro", multi_class="ovr")
                )
                out["auprc"] = float(average_precision_score(Yb, y_proba, average="macro"))
        except ValueError:
            pass  # degenerate fold composition; skip probabilistic metrics
        return out
