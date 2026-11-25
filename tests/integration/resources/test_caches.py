"""The cache register's rule: every cache the product holds is registered, with a bound.

One table holds the register's inputs, each run against the product as it is:
- a cache whose owner holds the value is deleted;
- one with no bound is registered with a bound;
- one keyed without its invalidation input is keyed by it;
- a new unbounded cache is refused;
- a cache at its bound evicts and reads the value again.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alphalattice.kernel.resources import cache_bound, cache_register
from devtools.architecture.registries import CACHES_REGISTRY, _unbounded_cache
from devtools.pipeline.facets import CacheBound, cache_holdings

TEST_SOURCE = "src/alphalattice/kernel/resources/caches.py"
ROOT = Path(__file__).resolve().parents[3]
_WEAK_OWNER_DIRECTORIES = {
    # Callers retain the pools. The weak keys only track the names of live owners.
    "src/alphalattice/kernel/execution/pool.py::_POOLS",
}
# Module-level context state that holds a mapping but caches nothing: the scan reads past it.
_CONTEXT_STATE = {
    "src/alphalattice/interface/local_application/cli_contract.py::BINDING_NOTICE",  # a notice
    "src/alphalattice/interface/local_application/cli_contract.py::READ_VIEW",  # a view binding
    "src/alphalattice/interface/local_application/call_span.py::_REPLACED",  # span names
    "src/alphalattice/foundation/market_data_ops/storage/world_reads.py::_UPDATES",  # callbacks
    "src/alphalattice/foundation/data_layer/history_alpha_arrays.py::_ALPHA_PARTS",  # parts
}

type Declared = CacheBound


def _entries() -> dict[str, dict[str, object]]:
    return json.loads((ROOT / CACHES_REGISTRY).read_text(encoding="utf-8"))["entries"]


def _held(relative: str, source: str) -> dict[str, Declared]:
    """Each cache a module holds, named as the registry's ``defined_by`` names it, with the bound
    its source declares (the census's own facet, ``cache_holdings``), less an owner's weak pool
    directory while it stays a WeakKeyDictionary and the context state that caches nothing."""
    exempt = {"WeakKeyDictionary": _WEAK_OWNER_DIRECTORIES, "ContextVar": _CONTEXT_STATE}
    return {
        f"{relative}::{name}": bound
        for name, bound, kind in cache_holdings(source)
        if f"{relative}::{name}" not in exempt.get(kind, ())
    }


def _unregistered(held: dict[str, Declared]) -> list[str]:
    defined = {name for row in _entries().values() for name in row.get("defined_by", [])}
    return sorted(held.keys() - defined)


def _unreconciled(held: dict[str, Declared], entries: dict[str, dict[str, object]]) -> list[str]:
    """Each register row whose bound its source does not hold: a row that names a bound over a
    cache whose source declares none, or a count other than the one its source spells."""
    refused = []
    for key, row in entries.items():
        named = cache_bound(row.get("size_bound"))
        for name in row.get("defined_by", []):
            declared = held.get(name, "capacity")
            if named is None or declared in {"capacity", "hand"}:
                continue
            if declared is None:
                refused.append(f"{key}: names {row['size_bound']!r}, but its source holds no bound")
            elif named.count is not None and named.count != declared:
                refused.append(f"{key}: names {named.count}, but its source holds {declared}")
    return refused


def _product_caches() -> dict[str, Declared]:
    held: dict[str, Declared] = {}
    for path in sorted((ROOT / "src/alphalattice").rglob("*.py")):
        held |= _held(path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8"))
    return held


@pytest.mark.proves("resources.every_cache_is_registered_with_a_bound")
def test_every_cache_the_product_holds_is_registered_with_the_bound_its_source_holds() -> None:
    """Each functools cache, module-level stateful map and context variable holding one under
    src/alphalattice has a register row, and no row names a bound its source does not hold."""
    held = _product_caches()
    assert _unregistered(held) == []
    assert _unreconciled(held, _entries()) == []


def _deleted() -> None:
    from alphalattice.interface.local_application import labels

    relative = "src/alphalattice/interface/local_application/labels.py"
    assert _held(relative, (ROOT / relative).read_text(encoding="utf-8")) == {
        f"{relative}::_installed": 1
    }
    first = labels.label_table().labels[0]
    assert labels.label(first.id) is first


def _bounded() -> None:
    register = cache_register(_entries())
    bounds = {
        key.rsplit("::", 1)[1]: row.bound.count
        for key, row in register.items()
        if row.bound is not None
        and key.rsplit("::", 1)[1]
        in {
            "lightgbm-binned-dataset-scope",
            "lightgbm-parsed-booster-scope",
            "world-sector-names-at-cutoff",
        }
    }
    assert bounds == {
        "lightgbm-binned-dataset-scope": 8,
        "lightgbm-parsed-booster-scope": 64,
        "world-sector-names-at-cutoff": 64,
    }


def _keyed(tmp_path: Path) -> None:
    from alphalattice.capabilities.domain_contracts.project_layout import json_file

    resource = tmp_path / "resource.json"
    resource.write_text('{"words": "one"}', encoding="utf-8")
    assert json_file(resource) == {"words": "one"}
    resource.write_text('{"words": "two, rewritten"}', encoding="utf-8")
    assert json_file(resource) == {"words": "two, rewritten"}


_NEW_MODULE = "src/alphalattice/new_module.py"
_NEW_SOURCE = """from contextvars import ContextVar
from functools import cache, lru_cache

_HELD = 32
_UNSET = None
_SIZES = (None,)


@cache
def keyed(key):
    return key


@lru_cache(maxsize=None)
def unsized(key):
    return key


@lru_cache(maxsize=_HELD)
def sized(key):
    return key


@lru_cache(maxsize=_UNSET)
def unset(key):
    return key


@lru_cache(maxsize=_held())
def unresolved(key):
    return key


@lru_cache(**{"maxsize": None})
def unpacked(key):
    return key


@lru_cache(*_SIZES)
def starred(key):
    return key


_SCOPE: ContextVar[dict[str, object] | None] = ContextVar("scope", default=None)
_CALLBACK: ContextVar[Callable[[dict[str, str]], None] | None] = ContextVar("callback")
_WEAK_CACHE = WeakKeyDictionary()
"""


def _refused() -> None:
    assert _unbounded_cache(CACHES_REGISTRY, "entries:new", {"size_bound": "unbounded"})
    assert _unbounded_cache(CACHES_REGISTRY, "entries:new", {})
    pools = (
        "src/alphalattice/kernel/execution/pool.py"  # its weak directory turned strong is a cache
    )
    assert _held(pools, "_POOLS = WeakKeyDictionary()") == {}
    assert _held(pools, "_POOLS = OrderedDict()") == {f"{pools}::_POOLS": "hand"}
    held = _held(_NEW_MODULE, _NEW_SOURCE)
    names = [
        "_SCOPE",
        "_WEAK_CACHE",
        "keyed",
        "sized",
        "starred",
        "unpacked",
        "unresolved",
        "unset",
        "unsized",
    ]
    assert _unregistered(held) == [f"{_NEW_MODULE}::{name}" for name in names]
    rows = {
        name: {"defined_by": [f"{_NEW_MODULE}::{name}"], "size_bound": "64 entries"}
        for name in names
    }
    assert _unreconciled(held, rows) == [
        "_SCOPE: names '64 entries', but its source holds no bound",
        "keyed: names '64 entries', but its source holds no bound",
        "sized: names 64, but its source holds 32",
        "starred: names '64 entries', but its source holds no bound",
        "unpacked: names '64 entries', but its source holds no bound",
        "unresolved: names '64 entries', but its source holds no bound",
        "unset: names '64 entries', but its source holds no bound",
        "unsized: names '64 entries', but its source holds no bound",
    ]


def _evicted() -> None:
    from alphalattice.capabilities.alpha_modeling.adapters import lightgbm_chronological as owner

    class Backend:
        calls = 0

        class Dataset:
            def __init__(self, *_args, **_keywords) -> None:
                Backend.calls += 1

            def construct(self):
                return self

    adapter = owner.ChronologicalLightGBMAdapter()
    parameters = adapter._dataset_params(
        owner.ChronologicalLightGBMParameters(
            seed=1729, num_leaves=31, learning_rate=0.05, max_depth=3, min_child_samples=20
        )
    )

    def dataset(shift: float):
        return adapter._dataset(
            Backend,
            features=np.arange(12, dtype=np.float64).reshape(4, 3) + shift,
            targets=np.arange(4, dtype=np.float64),
            ordered_feature_ids=("a", "b", "c"),
            dataset_params=parameters,
            reference_key="training",
        )

    with owner.chronological_lightgbm_dataset_cache():
        first, key = dataset(0.0)
        assert dataset(0.0) == (first, key) and Backend.calls == 1
        for shift in range(1, owner._DATASETS_HELD + 1):
            dataset(float(shift))
        again, same = dataset(0.0)  # evicted at the bound: built again, under the same key
    assert same == key and again is not first
    assert Backend.calls == owner._DATASETS_HELD + 2


_TABLE = {
    "a cache whose owner already holds the value is deleted": _deleted,
    "a cache with no bound is registered with a bound": _bounded,
    "a cache keyed without its invalidation input is keyed by it": _keyed,
    "a new unbounded cache is refused": _refused,
    "a cache at its bound evicts and reads the value again": _evicted,
}


@pytest.mark.proves("resources.every_cache_is_registered_with_a_bound")
@pytest.mark.parametrize("case", list(_TABLE), ids=list(_TABLE))
def test_the_cache_register_holds_each_cache_to_its_bound(case: str, tmp_path: Path) -> None:
    """Each input of the register's table, run against the product's own caches."""
    run = _TABLE[case]
    if case == "a cache keyed without its invalidation input is keyed by it":
        run(tmp_path)
    else:
        run()
