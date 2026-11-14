"""Each test body called twice over the same fixtures, as CodSpeed's simulation calls it.

Loaded with ``-p devtools.testing.repeat`` by the cost job's pilot-slice check, so a test that only
passes on fresh fixtures fails there by name, before the measurement reads it as a red shard.
"""

from __future__ import annotations

import inspect
from collections.abc import Generator

import pytest

REPAIR = (
    "repeat: this body passed once and failed when called again over the same fixtures, as "
    "CodSpeed calls it; give each call its own folder (tmp_path_factory.mktemp inside the body) "
    "or read only what this call wrote"
)


@pytest.hookimpl(wrapper=True)  # type: ignore[untyped-decorator]
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> Generator[None, object, object]:
    """Call the body once, then let pytest call it again; the second call is the verdict."""
    function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(function):  # an async body is driven by its own plugin
        function(**{name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames})
    try:
        return (yield)
    except Exception as error:
        error.add_note(REPAIR)
        raise
