"""A module as one importer sees it: a test's fakes in place, the shared module left real."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any


class _ModuleView(SimpleNamespace):
    def __getattr__(self, name: str) -> Any:  # only a name the fakes do not hold
        return getattr(self.__dict__["_module"], name)


def module_view(module: ModuleType, **fakes: Any) -> Any:
    """`module` with `fakes` in place, for one importer: `setattr(owner, "psutil", view)`.

    A fake set on the shared module itself is read by every thread for the test's whole length,
    spawn's memory watch included, and that watch then fails the run's later setups.
    """
    return _ModuleView(_module=module, **fakes)
