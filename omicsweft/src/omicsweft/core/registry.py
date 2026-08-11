"""A tiny name-based plugin registry.

Every extensible piece of the package (datasets, preprocessors, integrators,
tasks, interpreters) registers itself into one of the registries below via a
decorator. The benchmark runner and any user code can then instantiate a plugin
purely from a name string + keyword args, which is what makes config-driven,
reproducible runs possible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string name to a class of a given plugin kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Class decorator: ``@REGISTRY.register("my_name")``."""

        def _decorator(cls: type[T]) -> type[T]:
            key = name.lower()
            if key in self._items and self._items[key] is not cls:
                raise ValueError(
                    f"{self.kind} name {name!r} is already registered to "
                    f"{self._items[key].__name__}"
                )
            self._items[key] = cls
            cls._registry_name = name  # type: ignore[attr-defined]
            return cls

        return _decorator

    def get(self, name: str) -> type[T]:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}. available: {', '.join(self.list())}"
            )
        return self._items[key]

    def create(self, name: str, **kwargs) -> T:
        """Instantiate a registered plugin by name."""
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Registry({self.kind!r}, {self.list()})"


DATASETS: Registry = Registry("dataset")
PREPROCESSORS: Registry = Registry("preprocessor")
INTEGRATORS: Registry = Registry("integrator")
TASKS: Registry = Registry("task")
INTERPRETERS: Registry = Registry("interpreter")
