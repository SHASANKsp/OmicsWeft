"""Config + run manifest.

A run is fully described by a :class:`Config`: a dataset, an ordered list of
preprocessing steps, one or more integrators, and one or more tasks — each named
+ parameterised. :class:`RunManifest` captures the environment and a hash of the
config so any result can be traced back and regenerated.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StepSpec:
    """A named plugin invocation: ``name`` + constructor ``params``."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, obj: StepSpec | str | dict) -> StepSpec:
        if isinstance(obj, StepSpec):
            return obj
        if isinstance(obj, str):
            return cls(name=obj)
        if isinstance(obj, dict):
            return cls(name=obj["name"], params=obj.get("params", {}))
        raise TypeError(f"cannot coerce {obj!r} to StepSpec")


@dataclass
class Config:
    """A complete, serialisable description of a run."""

    dataset: StepSpec | None = None
    preprocess: list[StepSpec] = field(default_factory=list)
    integrators: list[StepSpec] = field(default_factory=list)
    tasks: list[StepSpec] = field(default_factory=list)
    seed: int = 0
    name: str = "run"

    def __post_init__(self) -> None:
        if self.dataset is not None:
            self.dataset = StepSpec.coerce(self.dataset)
        self.preprocess = [StepSpec.coerce(s) for s in self.preprocess]
        self.integrators = [StepSpec.coerce(s) for s in self.integrators]
        self.tasks = [StepSpec.coerce(s) for s in self.tasks]

    # ---- (de)serialisation ------------------------------------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            dataset=d.get("dataset"),
            preprocess=d.get("preprocess", []),
            integrators=d.get("integrators", []),
            tasks=d.get("tasks", []),
            seed=d.get("seed", 0),
            name=d.get("name", "run"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class RunManifest:
    """Environment + config fingerprint for one execution."""

    config_hash: str
    config: dict
    seed: int
    timestamp: str
    python: str
    platform: str
    package_version: str

    @classmethod
    def capture(cls, config: Config) -> RunManifest:
        from .. import __version__

        return cls(
            config_hash=config.hash(),
            config=config.to_dict(),
            seed=config.seed,
            timestamp=datetime.now(UTC).isoformat(),
            python=platform.python_version(),
            platform=platform.platform(),
            package_version=__version__,
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)
