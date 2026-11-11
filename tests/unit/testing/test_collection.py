from types import SimpleNamespace

import pytest

from devtools.testing import collection

TEST_SOURCE = "src/devtools/testing/collection.py"


@pytest.mark.proves("testing.worker_collection_refusal_is_named")
def test_a_workers_collection_refusal_is_named_before_it_ends_the_worker(capsys):
    """pytest sends a worker's collection in a finally, so xdist's controller sees only a crash on
    a test that never ran; the worker names its own refusal first, and the refusal still ends it."""
    for config, named in (
        (SimpleNamespace(workerinput={"workerid": "gw3"}), True),
        (SimpleNamespace(), False),  # the controller or a plain run: pytest itself reports it
    ):
        hook = collection.pytest_collection(SimpleNamespace(config=config))
        next(hook)
        with pytest.raises(pytest.UsageError, match=r"test_size.lane_unknown"):
            hook.throw(pytest.UsageError("test_size.lane_unknown"))
        named_line = "ERROR: [gw3] test_size.lane_unknown" in capsys.readouterr().err
        assert named_line is named
    hook = collection.pytest_collection(SimpleNamespace(config=SimpleNamespace()))
    next(hook)
    with pytest.raises(StopIteration) as finished:
        hook.send(["collected"])
    assert finished.value.value == ["collected"]  # an accepted collection passes through unchanged
