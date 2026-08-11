"""Plugin contracts.

These abstract base classes define the interface every plugin of a given kind
must satisfy. Keeping them small is deliberate: a narrow contract is what lets
the benchmark runner treat every method uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .omicsdata import OmicsData


@dataclass
class Embedding:
    """A joint sample representation produced by an Integrator.

    Attributes
    ----------
    X
        Array of shape ``(n_samples, k)`` — the joint latent representation.
    samples
        Sample identifiers, one per row of ``X`` (a pandas Index). Tasks use
        this to align labels drawn from the sample sheet.
    method
        Name of the integrator that produced the embedding.
    meta
        Optional free-form metadata (e.g. variance explained, affinity matrix).
    """

    X: np.ndarray
    samples: pd.Index
    method: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X)
        if not isinstance(self.samples, pd.Index):
            self.samples = pd.Index(self.samples)
        if self.X.shape[0] != len(self.samples):
            raise ValueError(
                f"embedding has {self.X.shape[0]} rows but {len(self.samples)} "
                "sample ids were given"
            )

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_dims(self) -> int:
        return self.X.shape[1] if self.X.ndim == 2 else 1


class Preprocessor(ABC):
    """Transforms an OmicsData in place-ish and returns it (chainable)."""

    #: modalities this step applies to; ``None`` means all.
    modalities: list[str] | None = None

    @abstractmethod
    def apply(self, data: OmicsData) -> OmicsData:  # pragma: no cover - abstract
        ...

    def _targets(self, data: OmicsData) -> list[str]:
        return list(self.modalities) if self.modalities else data.modalities

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        name = getattr(self, "_registry_name", type(self).__name__)
        return f"<Preprocessor {name}>"


class Integrator(ABC):
    """Produces a joint :class:`Embedding` from an OmicsData."""

    @abstractmethod
    def fit_transform(self, data: OmicsData) -> Embedding:  # pragma: no cover
        ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        name = getattr(self, "_registry_name", type(self).__name__)
        return f"<Integrator {name}>"


class Task(ABC):
    """Evaluates an :class:`Embedding` and returns a dict of metrics.

    A Task pulls any labels it needs from ``data`` by column name, so the core
    never hard-codes disease-specific fields. Survival (Kaplan-Meier, log-rank)
    is just one Task among several and is optional.
    """

    #: human-readable task family, e.g. "clustering".
    kind: str = "task"

    @abstractmethod
    def evaluate(self, embedding: Embedding, data: OmicsData) -> dict:  # pragma: no cover
        ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        name = getattr(self, "_registry_name", type(self).__name__)
        return f"<Task {name}>"


class Interpreter(ABC):
    """Produces an interpretation artifact from an embedding / model."""

    @abstractmethod
    def interpret(self, embedding: Embedding, data: OmicsData) -> dict:  # pragma: no cover
        ...
