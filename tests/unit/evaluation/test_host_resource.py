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

        def ppid(self):
            return 0

    api = SimpleNamespace(
        Process=FakeProcess, NoSuchProcess=Gone, AccessDenied=Denied, Error=(Gone, Denied)
    )
    owned = SimpleNamespace(process_ids=lambda: [1, 2])
    processes, error = process_census(owned, None, None, {}, api)
    assert [p.pid for p in processes] == [2]
    if failure is Gone:
        assert error is None
    else:
        assert error and failure.__name__ in error
        assert "pid=1" in error and f"stage={stage}" in error
    from devtools.evaluation.resource_budget import peak

    sample = {
        "host_tree_observed": True,
        "host_tree_complete": not bool(error),
        "host_private_commit_bytes": 123,
    }
    assert peak(None, rows=[sample]) == (123 if failure is Gone else None)

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
    assert set(ids) == {2, 3}
    if failure is Gone:
        assert "pid=1" in inventory_error and "stage=lineage" in inventory_error
    else:
        assert "pid=1" in inventory_error and "stage=birth" in inventory_error

    def failed_children(**_kwargs):
        raise failure()

    root.children = failed_children
    _, inventory_error = owned_inventory({3: 3.0}, api)
    assert "pid=3" in inventory_error and failure.__name__ in inventory_error
    assert "stage=children" in inventory_error


@pytest.mark.parametrize("failure", [Gone, Denied])
def test_census_seen_process_exit_or_unreadable(failure):
    from devtools.evaluation.host_resource import process_census

    def open_process(pid):
        if pid == 1:
            raise failure()
        return SimpleNamespace(pid=pid, create_time=lambda: float(pid), ppid=lambda: 0)

    api = SimpleNamespace(Process=open_process, NoSuchProcess=Gone, Error=(Gone, Denied))
    seen = {1: 1.0, 2: 2.0}
    current, error = process_census(None, None, None, seen, api)
    assert [p.pid for p in current] == [2]
    if failure is Gone:
        assert seen == {2: 2.0} and error is None
    else:
        assert seen == {1: 1.0, 2: 2.0}
        assert "pid=1" in error and "Denied" in error and "stage=reread" in error


def test_census_refuses_changed_identity_and_unknown_root_lineage():
    from devtools.evaluation.host_resource import process_census

    def bad_children(**_kwargs):
        raise Gone()

    root = SimpleNamespace(pid=1, create_time=lambda: 2.0, children=bad_children)
    api = SimpleNamespace(Process=lambda _pid: root, NoSuchProcess=Gone, Error=(Gone, Denied))
    seen = {1: 1.0}
    current, error = process_census(None, root, None, seen, api)
    assert current == [] and seen == {}
    assert "pid=1 error=IdentityChanged stage=birth" in error
    assert "pid=1 error=Gone stage=children" in error


def test_census_refuses_surviving_child_of_vanished_parent():
    from devtools.evaluation.host_resource import process_census

    def open_process(pid):
        if pid == 1:
            raise Gone()
        return SimpleNamespace(pid=pid, create_time=lambda: 2.0, ppid=lambda: 1)

    api = SimpleNamespace(Process=open_process, NoSuchProcess=Gone, Error=(Gone, Denied))
    current, error = process_census(None, None, None, {1: 1.0, 2: 2.0}, api)
    assert [process.pid for process in current] == [2]
    assert "pid=1 error=Gone stage=lineage" in error

    from devtools.evaluation.host_resource import HostSeen

    dead, denied, births = set(), set(), {1: 1.0, 2: 2.0, 3: 3.0}
    parents = {1: 0, 2: 1, 3: 2}

    class Process:
        def __init__(self, pid):
            if pid in dead:
                raise Gone()
            if pid in denied:
                raise Denied()
            self.pid = pid

        def create_time(self):
            return births[self.pid]

        def ppid(self):
            return parents.get(self.pid, 0)

    api = SimpleNamespace(
        Process=Process,
        NoSuchProcess=Gone,
        Error=(Gone, Denied),
        _psplatform=SimpleNamespace(ppid_map=lambda: dict(parents)),
    )
    seen = HostSeen()
    current, error = process_census(None, None, lambda: [Process(1)], seen, api)
    assert error is None and {p.pid for p in current} == {1, 2, 3}
    assert current.host_snapshot.parents[3] == 2
    dead.add(2)
    parents.clear()
    parents.update({1: 0, 3: 0})  # the surviving grandchild has already been reparented
    current, error = process_census(None, None, lambda: [Process(1)], seen, api)
    assert error is None and {p.pid for p in current} == {1, 3}
    assert current.host_snapshot.exited == {2} and current.host_snapshot.parents[3] == 2
    assert current.host_snapshot.identities[2] == 2.0 and not current.host_snapshot.errors

    births[3] = 4.0  # the same PID now names a different process
    current, error = process_census(None, None, lambda: [Process(1)], seen, api)
    assert any("pid=3 error=IdentityChanged" in row for row in current.host_snapshot.errors)
    assert 3 not in {p.pid for p in current}

    births[3] = 3.0
    dead.clear()
    denied.add(3)
    parents.update({2: 1, 3: 2})
    current, error = process_census(None, None, lambda: [Process(1)], HostSeen(), api)
    assert any("pid=3 error=Denied stage=birth" in row for row in current.host_snapshot.errors)

    denied.clear()
    dead.add(2)  # no last birth observation exists for this intermediate
    current, error = process_census(None, None, lambda: [Process(1)], HostSeen(), api)
    assert any("pid=2 error=IdentityUnmeasured" in row for row in current.host_snapshot.errors)

    parents.clear()
    parents.update({1: 0, 3: 0})
    ambiguous = HostSeen()
    ambiguous.update({1: 1.0, 2: 2.0, 3: 3.0})  # legacy census knew births but no parent edges
    current, error = process_census(None, None, lambda: [Process(1)], ambiguous, api)
    assert any("pid=1 error=PriorLineageUnmeasured" in row for row in current.host_snapshot.errors)
    assert any(
        "pid=3 error=PriorMembershipUnmeasured" in row for row in current.host_snapshot.errors
    )
    current, error = process_census(None, None, lambda: [Process(1)], ambiguous, api)
    assert any(
        "pid=3 error=PriorMembershipUnmeasured" in row for row in current.host_snapshot.errors
    )
    denied.add(3)
    current, error = process_census(None, None, lambda: [Process(1)], ambiguous, api)
    assert any(
        "pid=3 error=Denied stage=previous_census" in row for row in current.host_snapshot.errors
    )
    denied.clear()
    births[3] = 4.0
    current, error = process_census(None, None, lambda: [Process(1)], ambiguous, api)
    assert any(
        "pid=3 error=IdentityChanged stage=previous_census" in row
        for row in current.host_snapshot.errors
    )
    births[3] = 3.0
    ambiguous = HostSeen()
    ambiguous.update({1: 1.0, 2: 2.0, 3: 3.0})
    process_census(None, None, lambda: [Process(1)], ambiguous, api)
    dead.add(3)
    current, error = process_census(None, None, lambda: [Process(1)], ambiguous, api)
    assert not current.host_snapshot.errors

    def unavailable():
        raise OSError("snapshot unavailable")

    api.Error = Exception
    api._psplatform.ppid_map = unavailable
    current, error = process_census(None, None, lambda: [Process(1)], HostSeen(), api)
    assert any("OSError stage=parent_snapshot" in row for row in current.host_snapshot.errors)


@pytest.mark.parametrize("case", ["replacement", "unresolved", "unresolved-child"])
def test_census_retains_only_proved_bound_host_lineage(tmp_path, case):
    from devtools.evaluation.host_resource import HostSeen, process_census, sample_memory
    from devtools.evaluation.resource_budget import peak

    parents = {1: 0, 2: 1, 3: 0}
    denied, dead = set(), set()

    class Process:
        def __init__(self, pid):
            if pid in dead:
                raise Gone()
            if pid in denied:
                raise Denied()
            self.pid = pid

        def create_time(self):
            return float(self.pid)

        def ppid(self):
            return parents[self.pid]

        def is_running(self):
            return True

        def memory_info(self):
            return SimpleNamespace(rss=100, private=100)

        def cmdline(self):
            return []

        def name(self):
            return "host"

    api = SimpleNamespace(
        Process=Process,
        NoSuchProcess=Gone,
        Error=(Gone, Denied),
        _psplatform=SimpleNamespace(ppid_map=lambda: dict(parents)),
    )
    state = HostSeen()
    selected = [1, "old"]

    def roots():
        roots.binding = (selected[0], 1234, selected[1], float(selected[0]))
        return [Process(selected[0])]

    def sample():
        observed, _ = process_census(None, None, roots, state, api)
        return sample_memory(
            observed,
            workspaces=[tmp_path],
            bindings_of=lambda endpoints, pids: [
                {"pid": selected[0], "port": 1234, "instance": selected[1]}
            ],
            processes_api=api,
            connection_type=SimpleNamespace(
                read=lambda _: SimpleNamespace(url="http://127.0.0.1:1234", instance=selected[1])
            ),
            connection_error=OSError,
        )

    if case == "replacement":
        assert sample()["host_private_commit_bytes"] == 200
        parents[2] = 0
        assert sample()["host_private_commit_bytes"] == 200
        selected[:] = [1, "new-instance"]
        assert sample()["host_private_commit_bytes"] == 100
        selected[:] = [3, "new"]
        result = sample()
        assert result["host_tree_complete"] and result["host_private_commit_bytes"] == 100
        selected[:] = [1, "old"]
        assert sample()["host_private_commit_bytes"] == 200
    else:
        if case == "unresolved-child":
            parents[3] = 2
        denied.add(2)
        assert "pid=2 error=Denied" in sample()["host_tree_error"]
        assert "pid=2" in sample()["host_tree_error"]
        denied.clear()
        parents[2] = 0
        result = sample()
        assert not result["host_tree_complete"] and "pid=2" in result["host_tree_error"]
        assert peak(None, rows=[result]) is None
        dead.add(2)
        parents.pop(2)
        result = sample()
        if case == "unresolved-child":
            assert not result["host_tree_complete"] and "pid=2" in result["host_tree_error"]
            dead.add(3)
            parents.pop(3)
            result = sample()
            assert not result["host_tree_complete"]  # the final exit remains unproved this interval
            result = sample()
        assert result["host_tree_complete"] and result["host_private_commit_bytes"] == 100


@pytest.mark.parametrize("surviving_descendant", [False, True])
def test_inventory_preserves_unknown_lineage_for_census(tmp_path, surviving_descendant):
    from devtools.evaluation.host_resource import owned_inventory, process_census, sample_memory
    from devtools.evaluation.resource_budget import peak

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            if self.pid == 2:
                raise Gone()
            return float(self.pid)

        def children(self, recursive):
            assert recursive
            return [Process(2), *([Process(3)] if surviving_descendant else [])]

        def ppid(self):
            return {1: 0, 3: 2}[self.pid]

        def is_running(self):
            return True

        def memory_info(self):
            return SimpleNamespace(rss=100, private=100)

        def cmdline(self):
            return []

        def name(self):
            return "host"

    api = SimpleNamespace(Process=Process, NoSuchProcess=Gone, Error=(Gone, Denied))
    seen = {1: 1.0}
    owned = SimpleNamespace(process_inventory=lambda: owned_inventory(seen, api))
    current, error = process_census(owned, None, None, {}, api)
    assert {p.pid for p in current} == ({1, 3} if surviving_descendant else {1})
    if surviving_descendant:
        assert "pid=2 error=Gone stage=lineage" in error
    else:
        assert error is None
    sample = sample_memory(
        current,
        workspaces=[tmp_path],
        bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
        processes_api=api,
        connection_type=SimpleNamespace(
            read=lambda _: SimpleNamespace(url="http://127.0.0.1:1234", instance="i")
        ),
        connection_error=OSError,
    )
    if error:
        sample.update(host_tree_complete=False, host_tree_error=error)
    assert peak(None, rows=[sample]) == (None if surviving_descendant else 100)


def _captured_snapshot_checks(tmp_path, Process, Error, NoSuchProcess, connection):
    from devtools.evaluation.host_resource import sample_memory
    from devtools.evaluation.resource_budget import peak

    class Captured(list):
        host_snapshot = SimpleNamespace(
            roots={1},
            parents={1: 0, 2: 1, 3: 2},
            members={1, 2, 3},
            exited={2},
            identities={1: 1.0, 2: 2.0, 3: 3.0},
            binding=(1, 1234, "i", 1.0),
            errors=[],
        )

    checked_api = SimpleNamespace(
        Error=Error,
        NoSuchProcess=NoSuchProcess,
        Process=lambda pid: Process(pid, {1: 0, 3: 2}[pid]),
    )
    captured = sample_memory(
        Captured([Process(1, 0), Process(2, 1), Process(3, 2)]),
        workspaces=[tmp_path],
        bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
        processes_api=checked_api,
        connection_type=connection,
        connection_error=OSError,
    )
    assert captured["host_tree_complete"] is True
    assert captured["host_private_commit_bytes"] == 200
    assert peak(None, rows=[captured]) == 200
    assert captured["host_exits"] == [{"pid": 2, "birth": 2.0, "parent": 1}]

    class ExitedAgent(Process):
        def ppid(self):
            raise NoSuchProcess()

    separate = sample_memory(
        Captured([Process(1, 0), Process(3, 2), ExitedAgent(9, 0)]),
        workspaces=[tmp_path],
        bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
        processes_api=checked_api,
        connection_type=connection,
        connection_error=OSError,
    )
    assert separate["host_tree_complete"] and separate["host_private_commit_bytes"] == 200
    assert "pid=9 error=NoSuchProcess stage=ppid" in separate["activity_tree_error"]

    for binding, reason in (
        (None, "HostBindingUnmeasured"),
        ((1, 1234, "different-instance", 1.0), "HostIdentityChanged"),
        ((1, 1234, "i", 9.0), "HostIdentityChanged"),
    ):
        Captured.host_snapshot.binding = binding
        changed = sample_memory(
            Captured([Process(1, 0), Process(3, 2)]),
            workspaces=[tmp_path],
            bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
            processes_api=checked_api,
            connection_type=connection,
            connection_error=OSError,
        )
        assert not changed["host_tree_complete"] and reason in changed["host_tree_error"]
    Captured.host_snapshot.binding = (1, 1234, "i", 1.0)

    checked_api.Process = lambda pid: SimpleNamespace(create_time=lambda: 99.0)
    reused = sample_memory(
        Captured([Process(1, 0), Process(3, 2)]),
        workspaces=[tmp_path],
        bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
        processes_api=checked_api,
        connection_type=connection,
        connection_error=OSError,
    )
    assert reused["host_tree_complete"] is False
    assert "IdentityChanged stage=birth_before" in reused["host_tree_error"]

    checks = []

    def open_identity(pid):
        if pid == 3:
            checks.append(pid)
        return SimpleNamespace(
            create_time=lambda: 99.0 if len(checks) > 1 and pid == 3 else float(pid)
        )

    checked_api.Process = open_identity
    reused_after = sample_memory(
        Captured([Process(1, 0), Process(3, 2)]),
        workspaces=[tmp_path],
        bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
        processes_api=checked_api,
        connection_type=connection,
        connection_error=OSError,
    )
    assert not reused_after["host_tree_complete"]
    assert "pid=3 error=IdentityChanged stage=birth_after" in reused_after["host_tree_error"]

    from devtools.evaluation.host_resource import BoundHostMeter, HostSeen, process_census

    pinned_api = SimpleNamespace(
        Error=Error,
        NoSuchProcess=NoSuchProcess,
        Process=lambda pid: Process(pid, 0),
        _psplatform=SimpleNamespace(ppid_map=lambda: {1: 0}),
    )

    def binding_reader(endpoints, pids=None):
        return [{"pid": 1, "port": 1234, "instance": "i"}]

    def client(workspace, timeout):
        return SimpleNamespace(
            connection=SimpleNamespace(instance="i"), activity=lambda limit: None
        )

    meter = BoundHostMeter(
        lambda: tmp_path, binding_reader, pinned_api, connection, client, OSError
    )
    observed, error = process_census(None, None, meter.roots, HostSeen(), pinned_api)
    measured = sample_memory(
        observed,
        workspaces=meter.workspaces(),
        bindings_of=binding_reader,
        processes_api=pinned_api,
        connection_type=connection,
        connection_error=OSError,
    )
    assert error is None and measured["host_tree_complete"]
    assert measured["host_tree_observed"] and measured["host_private_commit_bytes"] == 100


@pytest.mark.parametrize(
    ("failure", "survivor_parent", "complete"),
    [("gone", 1, True), ("gone", 2, False), ("denied", 1, False)],
)
@pytest.mark.parametrize("stage", ["memory_info", "cmdline", "ppid", "name"])
def test_host_sample_distinguishes_exited_processes_from_unknown_membership(
    tmp_path, failure, survivor_parent, complete, stage
):
    from devtools.evaluation.host_resource import sample_memory
    from devtools.evaluation.resource_budget import peak

    class Error(Exception):
        pass

    class NoSuchProcess(Error):
        pass

    class AccessDenied(Error):
        pass

    class Process:
        def __init__(self, pid, parent):
            self.pid, self.parent = pid, parent

        def check(self, where):
            if self.pid == 2 and stage == where:
                raise NoSuchProcess() if failure == "gone" else AccessDenied()

        def create_time(self):
            return float(self.pid)

        def memory_info(self):
            self.check("memory_info")
            return SimpleNamespace(rss=100, private=100, peak_wset=100)

        def cmdline(self):
            self.check("cmdline")
            return ["host", "--workspace", str(tmp_path)] if self.pid == 1 else []

        def ppid(self):
            self.check("ppid")
            return self.parent

        def name(self):
            self.check("name")
            return "test"

        def is_running(self):
            return True

    connection = SimpleNamespace(
        read=lambda _: SimpleNamespace(url="http://127.0.0.1:1234", instance="i")
    )
    result = sample_memory(
        iter([Process(1, 0), Process(2, 1), Process(3, survivor_parent)]),
        workspaces=[tmp_path],
        bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
        processes_api=SimpleNamespace(Error=Error, NoSuchProcess=NoSuchProcess),
        connection_type=connection,
        connection_error=OSError,
    )
    # A missing parent reading cannot prove whether the vanished process had surviving children.
    complete = complete and not (failure == "gone" and stage == "ppid")
    assert result["host_tree_complete"] is complete
    assert result["host_private_commit_bytes"] == (200 if complete else None)
    assert peak(None, rows=[result]) == (200 if complete else None)
    assert 2 not in {p["pid"] for p in result["processes"]}
    if failure == "gone" and survivor_parent == 2 and stage == "ppid":
        _captured_snapshot_checks(tmp_path, Process, Error, NoSuchProcess, connection)
    if not complete:
        assert "2" in result["host_tree_error"]
        assert ("NoSuchProcess" if failure == "gone" else "AccessDenied") in result[
            "host_tree_error"
        ]
        assert f"stage={stage}" in result["host_tree_error"]
    else:
        missing_listener = sample_memory(
            [Process(1, 0)],
            workspaces=[tmp_path],
            bindings_of=lambda endpoints, pids: [],
            processes_api=SimpleNamespace(Error=Error, NoSuchProcess=NoSuchProcess),
            connection_type=connection,
            connection_error=OSError,
        )
        assert missing_listener["host_tree_complete"] is False
        assert "ListenerUnmeasured" in missing_listener["host_tree_error"]

    if failure == "gone" and survivor_parent == 2 and stage == "memory_info":
        exited = False

        class ReparentingProcess(Process):
            def check(self, where):
                nonlocal exited
                if self.pid == 2 and where == "memory_info":
                    exited = True
                super().check(where)

            def ppid(self):
                return 0 if self.pid == 3 and exited else super().ppid()

        reparented = sample_memory(
            [ReparentingProcess(1, 0), ReparentingProcess(2, 1), ReparentingProcess(3, 2)],
            workspaces=[tmp_path],
            bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
            processes_api=SimpleNamespace(Error=Error, NoSuchProcess=NoSuchProcess),
            connection_type=connection,
            connection_error=OSError,
        )
        assert reparented["host_tree_complete"] is False

        exited = False

        class SnapshotRace(Process):
            def ppid(self):
                nonlocal exited
                parent = 0 if self.pid == 3 and exited else super().ppid()
                if self.pid == 2:
                    exited = True
                return parent

            def is_running(self):
                return not (self.pid == 2 and exited)

        raced = sample_memory(
            [SnapshotRace(1, 0), SnapshotRace(2, 1), SnapshotRace(3, 2)],
            workspaces=[tmp_path],
            bindings_of=lambda endpoints, pids: [{"pid": 1, "port": 1234, "instance": "i"}],
            processes_api=SimpleNamespace(Error=Error, NoSuchProcess=NoSuchProcess),
            connection_type=connection,
            connection_error=OSError,
        )
        assert raced["host_tree_complete"] is False
