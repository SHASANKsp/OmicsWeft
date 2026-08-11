"""OmicsWeft — an end-to-end, disease-agnostic multi-omics integration toolkit.

The package is organised as a domain-free core plus swappable plugins:

    core        OmicsData container, plugin registries, config, seeding, manifests
    io          dataset loaders (generic matrix-per-omic + sample sheet is the default)
    preprocess  normalize / impute / batch-correct / feature-select / QC
    integrate   integration methods that produce a joint sample embedding
    tasks       clustering / classification / regression (+ optional survival)
    bench       a runner that ties any integrator x dataset x task into a results table

Nothing in the core knows about any disease. Cancer-specific analyses such as
survival / Kaplan-Meier live in an optional task plugin, loaded only when a
dataset actually carries time-to-event labels.
"""

from __future__ import annotations

from . import integrate as _integrate  # noqa: F401
from . import interpret as _interpret  # noqa: F401

# Importing these modules triggers their @register decorators so the plugins
# are discoverable by name through the registries.
from . import io as _io  # noqa: F401
from . import preprocess as _preprocess  # noqa: F401
from . import tasks as _tasks  # noqa: F401
from ._version import __version__
from .core.base import Embedding, Integrator, Preprocessor, Task
from .core.config import Config, RunManifest
from .core.omicsdata import OmicsData
from .core.registry import (
    DATASETS,
    INTEGRATORS,
    INTERPRETERS,
    PREPROCESSORS,
    TASKS,
)
from .core.utils import get_logger, set_seed

__all__ = [
    "DATASETS",
    "INTEGRATORS",
    "INTERPRETERS",
    "PREPROCESSORS",
    "TASKS",
    "Config",
    "Embedding",
    "Integrator",
    "OmicsData",
    "Preprocessor",
    "RunManifest",
    "Task",
    "__version__",
    "get_logger",
    "set_seed",
]
