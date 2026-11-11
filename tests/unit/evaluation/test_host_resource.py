"""Activity snapshots keep surviving observations when process discovery races."""

from types import SimpleNamespace

import pytest

from devtools.evaluation.host_resource import activity_snapshot

TEST_SOURCE = "src/devtools/evaluation/host_resource.py"
pytestmark = pytest.mark.proves("evaluation.native_snapshot_tolerates_process_exit")


class Gone(Exception):
    pass


class Denied(Exception):
    pass


@pytest.mark.parametrize("failure", [Gone, Denied])
def test_snapshot_skips_process_that_cannot_be_opened(failure):
    def open_process(pid):
        if pid == 1:
            raise failure()
        return SimpleNamespace(pid=pid)

    api = SimpleNamespace(Process=open_process, NoSuchProcess=Gone, AccessDenied=Denied)

    def sample(processes):
        return {"processes": [{"private": 42} for p in processes if p.pid == 2]}

    assert activity_snapshot([1, 2], api, sample) == {"processes": 1, "commit_bytes": 42}


@pytest.mark.parametrize("failure", [Gone, Denied])
@pytest.mark.parametrize("stage", ["open", "birth", "reread"])
def test_census_keeps_survivors_and_marks_failed_observation(failure, stage):
    from devtools.evaluation.host_resource import process_census

    opens = {}

    class FakeProcess:
        def __init__(self, pid):
            opens[pid] = opens.get(pid, 0) + 1
            if pid == 1 and (stage == "open" or (stage == "reread" and opens[pid] > 1)):
                raise failure()
            self.pid = pid

        def create_time(self):
            if self.pid == 1 and stage == "birth":
                raise failure()
            return float(self.pid)

    api = SimpleNamespace(
        Process=FakeProcess, NoSuchProcess=Gone, AccessDenied=Denied, Error=(Gone, Denied)
    )
    owned = SimpleNamespace(process_ids=lambda: [1, 2])
    processes, error = process_census(owned, None, None, {}, api)
    assert [p.pid for p in processes] == [2]
    assert error and failure.__name__ in error
    from devtools.evaluation.resource_budget import peak

    sample = {
        "host_tree_observed": True,
        "host_tree_complete": not bool(error),
        "host_private_commit_bytes": 123,
    }
    assert peak(None, rows=[sample]) is None

    from devtools.evaluation.host_resource import owned_inventory

    def failed_birth():
        raise failure()

    children = [
        SimpleNamespace(pid=1, create_time=failed_birth),
        SimpleNamespace(pid=2, create_time=lambda: 2.0),
    ]
    root = SimpleNamespace(create_time=lambda: 3.0, children=lambda **_: children)
    api.Process = lambda _pid: root
    ids, inventory_error = owned_inventory({3: 3.0}, api)
    assert set(ids) == {2, 3} and inventory_error
