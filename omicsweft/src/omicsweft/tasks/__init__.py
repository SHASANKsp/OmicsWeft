"""Task + evaluator plugins.

clustering / classification / regression are core and dependency-free. survival
is optional (needs the 'survival' extra) and is the only task that knows about
time-to-event data — the disease-agnostic hinge of the package.
"""

from ..core.registry import TASKS
from .classification import ClassificationTask
from .clustering import ClusteringTask
from .regression import RegressionTask
from .survival import SurvivalTask

__all__ = [
    "TASKS",
    "ClassificationTask",
    "ClusteringTask",
    "RegressionTask",
    "SurvivalTask",
]
