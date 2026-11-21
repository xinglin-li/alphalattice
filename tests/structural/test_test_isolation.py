"""The test guards that keep one test's interpreter and roots from reaching another's or the
person's: each fails the offending test by name, in memory, through the hooks pytest runs."""

from __future__ import annotations

import importlib
import os
import site
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.control.task_control import upgrade
from alphalattice.kernel.data_layer import history
from devtools.testing import isolation
from tests import conftest
from tests import pytest_environment as environment
from tests.structural.isolation_support import check_support_roots

TEST_SCOPE = "whole-tree"  # tests/conftest.py runs these guards around every test
PROBE = "tests/unit/test_probe.py::test_case"
LEAKS = {  # what a test body does to the interpreter, and what its teardown then names
    "clean": (lambda _place: None, ""),
    "path-gained": (lambda place: site.addsitedir(str(place)), "left sys.path gained"),
    "cwd-moved": (os.chdir, "left the working directory moved from"),
}
OUTSIDE = Path(sys.executable).parent  # a folder no run owns, never written or read here
ROOTS = {  # the named variable, the platform default's base, the resolved root; refused?
    "named-owned": ("owned", None, "owned", False),
    "default-owned": (None, "owned", "owned", False),
    "default-outside": (None, "outside", None, True),  # refused before the resolver runs
    "named-resolves-outside": ("owned", None, "outside", True),
    "link-leads-outside": ("owned", None, "link", True),  # a link under an owned root, judged
}
STORES = {  # an inherited store, the test host's kit; admitted only when they are one folder
    "store-inherited-alone": (OUTSIDE / "models", None, False),
    "store-at-the-kit-suffix": (Path("shared-kits", "model-store"), None, False),
    "store-named-as-the-kit": (Path("shared-kits", "model-store"), "same", True),
    "store-another-kit": (Path("shared-kits", "model-store"), Path("kit"), False),
}

ENTRIES = {  # a helper a plain script can run, and whether its run names roots of its own
    "builder-unnamed": ("build_real_risk_workspace", False),
    "builder-named": ("build_real_risk_workspace", True),
    "warm-unnamed": ("warm_real_risk_golden", False),
    "warm-named": ("warm_real_risk_golden", True),
}
PACKAGES = (
    "_support",
    "alternative_evidence_desk",
    "feature_engine",
    "feature_input_gateway",
    "portfolio_strategy_lab",
    "researcher_methodology_surface",
    "structural",
    "workspace_maintenance",
    "workspace_readiness",
    "workspace_task_runner",
)
SUPPORT_ROOTS = {
    f"support-{package}-{root}-{mode}": (package, root, mode)
    for package in PACKAGES
    for root in ("DATA", "BACKUP")
    for mode in ("unset", "default", "owned")
}


def _hooks(body):
    """Run a test body between conftest's setup and teardown wrappers; the teardown's failure."""
    factory = SimpleNamespace(_retention_policy="all")
    item = SimpleNamespace(
        stash=pytest.Stash(),
        nodeid=PROBE,
        config=SimpleNamespace(_tmp_path_factory=factory),
    )
    setup = conftest.pytest_runtest_setup(item)
    next(setup)
    with pytest.raises(StopIteration):
        next(setup)
    body()
    teardown = conftest.pytest_runtest_teardown(item, None)
    next(teardown)
    try:
        next(teardown)
    except AssertionError as error:
        return str(error)
    except StopIteration:
        return ""
    raise AssertionError("the teardown wrapper yielded twice")


@pytest.mark.proves("testing.a_test_leaking_its_interpreter_or_a_person_root_fails_by_name")
@pytest.mark.parametrize("case", [*LEAKS, *ROOTS, *STORES, *ENTRIES, *SUPPORT_ROOTS], ids=str)
def test_a_test_that_leaks_its_interpreter_or_reaches_a_person_root_fails_by_name(
    case, tmp_path, monkeypatch
):
    """A leaking body fails at its own teardown, by name, with the state put back; a resolver's
    default outside the run's roots fails before it reads, its resolved result after; a helper a
    script can run refuses, by name, before anything, unless roots of its own are named."""
    if case in SUPPORT_ROOTS:
        from tests import _support

        package, root, named = SUPPORT_ROOTS[case]
        module = _support if package == "_support" else importlib.import_module(f"tests.{package}")
        check_support_roots(module, root, named, tmp_path, monkeypatch)
        return
    if case in ENTRIES:
        from tests.researcher_methodology_surface import real_workspace

        entry, named = ENTRIES[case]
        for _module, _resolver, variable, _home in environment.PERSON_RESOLVERS:
            if named:
                monkeypatch.setenv(variable, str(tmp_path / variable))
            else:
                monkeypatch.delenv(variable, raising=False)

        def reached() -> None:  # the helper's first work after its roots are judged
            raise LookupError("reached its work")

        monkeypatch.setattr(real_workspace.FeatureCatalog, "load", reached)
        expected = (
            (LookupError, "^reached its work$")
            if named
            else (RuntimeError, rf"^tests\.person_root_unnamed: {entry} would open ")
        )
        with pytest.raises(expected[0], match=expected[1]):
            getattr(real_workspace, entry)(**({"root": tmp_path} if "build" in entry else {}))
        return
    if case in LEAKS:
        leak, named = LEAKS[case]
        before = isolation.interpreter_state()
        failure = _hooks(lambda: leak(tmp_path))
        assert isolation.interpreter_state() == before  # put back
        assert (
            failure.startswith(f"tests.interpreter_state_leaked: {PROBE} {named}")
            if named
            else not failure
        )
        return
    if case in STORES:
        named, kit, admitted = STORES[case]
        named = named if named.is_absolute() else tmp_path / named
        kit = "" if kit is None else str(named if kit == "same" else tmp_path / kit)
        store = environment.model_store_root(str(named), kit, lambda: tmp_path / "own")
        assert store == (Path(os.path.realpath(named)) if admitted else tmp_path / "own")
        return
    variable, base, found, refused = ROOTS[case]
    place = {"owned": tmp_path, "outside": OUTSIDE, "link": tmp_path / "link"}
    real = os.path.realpath
    monkeypatch.setattr(  # the link's target, as the file system would resolve it
        environment.os.path,
        "realpath",
        lambda p: str(OUTSIDE) if Path(p) == place["link"] else real(p),
    )
    for key in ("LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.setenv(key, str(place[base] if base else tmp_path))
    if variable:
        monkeypatch.setenv("ALPHALATTICE_PROBE_ROOT", str(place[variable]))
    else:
        monkeypatch.delenv("ALPHALATTICE_PROBE_ROOT", raising=False)
    read = []
    guarded = environment._owned_only(
        lambda: read.append(1) or place[found or "owned"],
        "probe.resolve",
        "ALPHALATTICE_PROBE_ROOT",
        False,
        environment.OWNED_ROOTS,
    )
    if refused:
        with pytest.raises(AssertionError, match=r"^tests\.person_root_resolved: .* reached "):
            guarded()
    else:
        assert guarded() == tmp_path
    assert read == ([] if found is None else [1])
    assert history.data_root in environment.GUARDED.values()  # each setup wraps the Host's own
    assert upgrade.data_root is history.data_root  # and a module's imported name for it
