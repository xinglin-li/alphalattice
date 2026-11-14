"""Observe the bound workspace Host and its process tree."""

import re
from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit


def listener_bindings(
    endpoints: dict[int, str], pids: set[int] | None = None, *, processes
) -> list[dict[str, Any]]:
    return [
        dict(pid=c.pid, port=c.laddr.port, instance=endpoints[c.laddr.port])
        for c in processes.net_connections(kind="tcp")
        if c.pid is not None
        and (pids is None or c.pid in pids)
        and c.status == processes.CONN_LISTEN
        and c.laddr
        and c.laddr.ip == "127.0.0.1"
        and c.laddr.port in endpoints
    ]


def activity_snapshot(pids, processes_api, sample_memory):
    """Sample processes still accessible after discovery without losing surviving rows."""
    observed = []
    for pid in pids:
        try:
            observed.append(processes_api.Process(pid))
        except (processes_api.NoSuchProcess, processes_api.AccessDenied):
            continue
    rows = sample_memory(observed)["processes"]
    complete = all(type(row.get("private")) is int for row in rows)
    return dict(
        processes=len(rows), commit_bytes=sum(r["private"] for r in rows) if complete else None
    )


def owned_inventory(seen, processes_api):
    """Inventory owned roots; an ambiguous surviving sibling may conservatively mark a gap."""
    held, errors, vanished = {}, [], []
    for pid, birth in list(seen.items()):
        stage = "open"
        try:
            process = processes_api.Process(pid)
            stage = "birth"
            if process.create_time() != birth:
                errors.append(f"pid={pid} error=IdentityChanged stage=birth")
                continue
            held[pid] = birth
            stage = "children"
            for child in process.children(recursive=True):
                try:
                    held[child.pid] = child.create_time()
                except processes_api.Error as error:
                    if isinstance(error, processes_api.NoSuchProcess):
                        seen.pop(child.pid, None)
                        held.pop(child.pid, None)
                        vanished.append((child.pid, pid, error))
                    else:
                        errors.append(f"pid={child.pid} error={type(error).__name__} stage=birth")
        except processes_api.Error as error:
            if isinstance(error, processes_api.NoSuchProcess) and stage != "children":
                seen.pop(pid, None)
                held.pop(pid, None)
                vanished.append((pid, pid, error))
            else:
                errors.append(f"pid={pid} error={type(error).__name__} stage={stage}")
    for pid, root, error in vanished:
        # The recursive snapshot cannot prove another survivor was not reparented after this exit.
        if any(survivor != root for survivor in held):
            errors.append(f"pid={pid} error={type(error).__name__} stage=lineage")
    seen.update(held)
    return list(held), ", ".join(errors) or None


def _discover_processes(
    owned, parent, observed_roots, seen, processes_api, record, vanished, errors
):
    observed, owned_pids = [], set()
    try:
        if owned is not None:
            inventory = getattr(owned, "process_inventory", None)
            pids, error = inventory() if inventory else (owned.process_ids(), None)
            if error:
                errors.append(str(error))
            for pid in pids:
                owned_pids.add(pid)
                try:
                    observed.append(processes_api.Process(pid))
                except processes_api.Error as error:
                    if isinstance(error, processes_api.NoSuchProcess):
                        seen.pop(pid, None)
                        vanished[pid] = error
                    else:
                        record(error, pid, "open")
        elif parent is not None:
            observed.append(parent)
            try:
                observed.extend(parent.children(recursive=True))
            except processes_api.Error as error:
                record(error, parent.pid, "children")
    except (processes_api.Error, OSError, ValueError) as error:
        record(error)
    if observed_roots is not None:
        try:
            for root in observed_roots():
                observed.append(root)
                try:
                    observed.extend(root.children(recursive=True))
                except processes_api.Error as error:
                    record(error, root.pid, "children")
        except (processes_api.Error, OSError, ValueError) as error:
            record(error, stage="roots")
    return observed, owned_pids


class HostSeen(dict):
    """Separate the owned agent census from previously identity-bound Host members."""

    def __init__(self):
        super().__init__()
        self.hosts = {}
        self.unresolved = {}
        self.unresolved_roots = set()


class HostObservation(list):
    def __init__(self, processes, snapshot):
        super().__init__(processes)
        self.host_snapshot = snapshot


def _pin_roots(roots, processes_api, identities, errors):
    pinned = {}
    for root in roots:
        try:
            pinned[root.pid] = root.create_time()
            identities[root.pid] = pinned[root.pid]
        except processes_api.Error as error:
            errors.append(f"pid={root.pid} error={type(error).__name__} stage=root_birth")
    return pinned


def _parent_graph(processes_api):
    # psutil's native Windows ppid_map uses the OS process snapshot. Its portable
    # Python fallback is a series of reads and cannot establish one lineage boundary.
    snapshot = processes_api._psplatform.ppid_map
    started = time()
    parents = snapshot()
    if not isinstance(parents, dict) or any(
        type(pid) is not int or type(ppid) is not int or pid < 0 or ppid < 0
        for pid, ppid in parents.items()
    ):
        raise ValueError("process parent snapshot is not a mapping")
    return parents, started


def _host_members(parents, pinned, previous, unknown):
    members = set(pinned) | set(previous) | set(unknown)
    # A previous member stays attributable after reparenting while its birth matches.
    for pid, (_, prior_parent) in previous.items():
        if pid not in parents or parents[pid] not in members:
            parents[pid] = prior_parent
    while True:
        expanded = members | {
            pid for pid, ppid in parents.items() if ppid in members and pid != ppid
        }
        if expanded == members:
            return members
        members = expanded


def _live_members(
    members, parents, pinned, previous, unknown, processes_api, started, identities, errors
):
    live, exited = [], set()
    next_seen = dict(previous)
    for pid in members:
        try:
            process = processes_api.Process(pid)
            birth = process.create_time()
            expected = pinned.get(pid, previous.get(pid, (birth, None))[0])
            if (
                birth != expected
                or (pid in previous and birth != previous[pid][0])
                or birth > started
            ):
                errors.append(f"pid={pid} error=IdentityChanged stage=birth")
                next_seen.pop(pid, None)
                unknown.setdefault(pid, set())
                continue
            if pid not in parents:
                errors.append(f"pid={pid} error=LineageUnmeasured stage=parent_snapshot")
                unknown.setdefault(pid, set())
                continue
            next_seen[pid] = (birth, parents[pid])
            identities[pid] = birth
            live.append(process)
        except processes_api.Error as error:
            if isinstance(error, processes_api.NoSuchProcess):
                exited.add(pid)
                next_seen.pop(pid, None)
                if pid not in previous and any(
                    parents.get(child) == pid for child in members - {pid}
                ):
                    errors.append(f"pid={pid} error=IdentityUnmeasured stage=parent_snapshot")
            else:
                errors.append(f"pid={pid} error={type(error).__name__} stage=birth")
                if pid not in previous:
                    unknown.setdefault(pid, set())
    return live, exited, next_seen


def _lineage_errors(live, members, pinned, parents, identities, errors):
    for process in live:
        pid, visited = process.pid, set()
        while pid in members and pid not in pinned:
            if pid in visited:
                errors.append(f"pid={pid} error=LineageCycle stage=parent_snapshot")
                break
            visited.add(pid)
            ppid = parents.get(pid)
            if ppid in identities and pid in identities and identities[pid] < identities[ppid]:
                errors.append(f"pid={pid} error=IdentityChanged stage=lineage")
                break
            pid = ppid


def _unresolved_members(unknown, parents, processes_api, errors):
    for pid, dependents in list(unknown.items()):
        dependents.update(_host_members(dict(parents), {pid: None}, {}, {}))
        dependents.discard(pid)
        alive = set()
        for member in {pid, *dependents}:
            try:
                processes_api.Process(member)
            except processes_api.NoSuchProcess:
                dependents.discard(member)
            except processes_api.Error:
                alive.add(member)
            else:
                alive.add(member)
        if not alive:
            unknown.pop(pid)
        if pid in unknown:
            errors.append(f"pid={pid} error=PriorMembershipUnmeasured stage=parent_snapshot")


def _host_snapshot(roots, previous, unknown, processes_api):
    """Pin roots, then retain a single native parent graph and birth-checked live members."""
    errors, exited, live = [], set(), []
    identities = {pid: birth for pid, (birth, _) in previous.items()}
    pinned = _pin_roots(roots, processes_api, identities, errors)
    try:
        parents, started = _parent_graph(processes_api)
    except (AttributeError, OSError, RuntimeError, ValueError, processes_api.Error) as error:
        errors.append(f"pid=unknown error={type(error).__name__} stage=parent_snapshot")
        return live, SimpleNamespace(
            roots=set(pinned),
            parents={},
            members=set(),
            exited=exited,
            identities=identities,
            errors=errors,
        )

    members = _host_members(parents, pinned, previous, unknown)
    live, exited, next_seen = _live_members(
        members, parents, pinned, previous, unknown, processes_api, started, identities, errors
    )
    _unresolved_members(unknown, parents, processes_api, errors)
    _lineage_errors(live, members, pinned, parents, identities, errors)
    previous.clear()
    previous.update(next_seen)
    return live, SimpleNamespace(
        roots=set(pinned),
        parents=parents,
        members=members,
        exited=exited,
        identities=identities,
        errors=errors,
    )


def _ordinary_census(owned, parent, observed_roots, seen, processes_api):
    errors, vanished = [], {}

    def record(error, pid="unknown", stage="inventory"):
        detail = type(error).__name__ if isinstance(error, BaseException) else str(error)
        errors.append(f"pid={pid} error={detail} stage={stage}")

    observed, owned_pids = _discover_processes(
        owned, parent, observed_roots, seen, processes_api, record, vanished, errors
    )
    for process in {p.pid: p for p in observed}.values():
        try:
            birth = process.create_time()
            if process.pid in seen and seen[process.pid] != birth:
                seen.pop(process.pid)
                record("IdentityChanged", process.pid, "birth")
            else:
                seen[process.pid] = birth
        except processes_api.Error as error:
            if isinstance(error, processes_api.NoSuchProcess) and (
                process.pid in seen or process.pid in owned_pids
            ):
                seen.pop(process.pid, None)
                vanished[process.pid] = error
            else:
                record(error, process.pid, "birth")
    current = []
    for pid, birth in list(seen.items()):
        try:
            process = processes_api.Process(pid)
            if process.create_time() != birth:
                seen.pop(pid, None)
                record("IdentityChanged", pid, "reread")
                continue
            current.append(process)
        except processes_api.Error as error:
            if isinstance(error, processes_api.NoSuchProcess):
                seen.pop(pid, None)
                vanished[pid] = error
            else:
                record(error, pid, "reread")
    if vanished:
        for process in current:
            try:
                parent_pid = process.ppid()
                if parent_pid in vanished:
                    record(vanished[parent_pid], parent_pid, "lineage")
            except processes_api.Error as error:
                record(error, process.pid, "ppid")
    return current, ", ".join(errors) or None


def _prior_lineage(seen, host, snapshot, processes_api):
    """Do not turn a pre-snapshot identity-only observation into a lineage fact."""
    live_host = {process.pid for process in host}
    for pid, birth in list(seen.unresolved.items()):
        if pid in live_host and snapshot.identities.get(pid) == birth:
            seen.unresolved.pop(pid)
            continue
        try:
            if processes_api.Process(pid).create_time() != birth:
                snapshot.errors.append(f"pid={pid} error=IdentityChanged stage=previous_census")
                seen.unresolved.pop(pid)
        except processes_api.NoSuchProcess:
            seen.unresolved.pop(pid)
        except processes_api.Error as error:
            snapshot.errors.append(f"pid={pid} error={type(error).__name__} stage=previous_census")
    if seen.unresolved:
        for pid in sorted(seen.unresolved_roots):
            snapshot.errors.append(f"pid={pid} error=PriorLineageUnmeasured stage=previous_census")
        for pid in sorted(seen.unresolved):
            snapshot.errors.append(
                f"pid={pid} error=PriorMembershipUnmeasured stage=previous_census"
            )
    else:
        seen.unresolved_roots.clear()


def process_census(owned, parent, observed_roots, seen, processes_api):
    """Keep every observable descendant and carry any incomplete discovery to its sample."""
    if not isinstance(seen, HostSeen) or observed_roots is None:
        return _ordinary_census(owned, parent, observed_roots, seen, processes_api)
    try:
        roots = observed_roots()
        provider = getattr(observed_roots, "__self__", observed_roots)
        binding = getattr(provider, "binding", None)
        key = (
            binding
            if isinstance(binding, tuple)
            and len(binding) == 4
            and all(isinstance(value, (int, float, str)) for value in binding)
            else None
        )
        previous, unknown = seen.hosts.setdefault(key, ({}, {}))
        late = {root.pid for root in roots if root.pid in seen and root.pid not in previous}
        if late:
            seen.unresolved_roots.update(late)
            seen.unresolved.update({pid: birth for pid, birth in seen.items() if pid not in late})
        host, snapshot = _host_snapshot(roots, previous, unknown, processes_api)
        snapshot.binding = binding
        _prior_lineage(seen, host, snapshot, processes_api)
    except (processes_api.Error, OSError, ValueError) as error:
        host, snapshot = (
            [],
            SimpleNamespace(
                roots=set(),
                parents={},
                members=set(),
                exited=set(),
                identities={},
                binding=None,
                errors=[f"pid=unknown error={type(error).__name__} stage=roots"],
            ),
        )
    agent, snapshot.agent_error = _ordinary_census(owned, parent, None, seen, processes_api)
    return HostObservation({p.pid: p for p in [*agent, *host]}.values(), snapshot), None


class BoundHostMeter:
    """Find only the authenticated listener for this run's bound workspace."""

    def __init__(self, workspace, bindings_of, processes, connection, client, client_error):
        self.workspace = workspace
        self.bindings_of = bindings_of
        self.processes = processes
        self.connection = connection
        self.client = client
        self.client_error = client_error
        self.measured = []
        self.verified = set()
        self.binding = None

    def roots(self):
        self.measured.clear()
        self.binding = None
        workspace = self.workspace()
        if workspace is None:
            return []
        try:
            connection = self.connection.read(workspace)
        except self.client_error as error:
            if "service_not_running" in str(error):
                return []
            raise
        port = urlsplit(connection.url).port
        if port is None:
            raise ValueError("native_meter.host_port_unmeasured")
        listeners = self.bindings_of({port: connection.instance})
        if not listeners:
            if self.verified and all(
                not self.processes.pid_exists(pid) for _, pid, _ in self.verified
            ):
                return []
            raise ValueError("native_meter.host_listener_unmeasured")
        if len(listeners) != 1:
            raise ValueError("native_meter.host_listener_ambiguous")
        process = self.processes.Process(listeners[0]["pid"])
        identity = (connection.instance, process.pid, process.create_time())
        if identity not in self.verified:
            client = self.client(workspace, timeout=5)
            if client.connection.instance != connection.instance:
                raise ValueError("native_meter.host_instance_changed")
            client.activity(limit=1)
            self.verified.add(identity)
        self.measured.append(workspace)
        self.binding = (process.pid, port, connection.instance, identity[2])
        return [process]

    def workspaces(self):
        return list(self.measured)


def _process_parents(observed, processes_api, parent_snapshot, failures):
    parents = {}
    for process in observed:
        if parent_snapshot is not None and process.pid not in parent_snapshot:
            failures.append((process.pid, "LineageUnmeasured", "parent_snapshot"))
            continue
        try:
            parents[process.pid] = (
                parent_snapshot[process.pid] if parent_snapshot is not None else process.ppid()
            )
        except processes_api.Error as error:
            failures.append((process.pid, error, "ppid"))
    for process in observed:
        if process.pid not in parents:
            continue
        try:
            if not process.is_running():
                raise processes_api.NoSuchProcess(process.pid)
        except processes_api.Error as error:
            parents.pop(process.pid)
            failures.append((process.pid, error, "parent_snapshot"))
    return parents


def _process_row(process, parent, processes_api, workspaces, identities, failures):
    if identities is not None and process.pid not in identities:
        failures.append((process.pid, "IdentityUnmeasured", "birth"))
        return None
    stage = "memory_info"
    try:
        if identities is not None:
            stage = "birth_before"
            if processes_api.Process(process.pid).create_time() != identities[process.pid]:
                failures.append((process.pid, "IdentityChanged", stage))
                return None
        stage = "memory_info"
        memory = process.memory_info()
        stage = "cmdline"
        command = process.cmdline()
        for index, arg in enumerate(command):
            value = (
                command[index + 1]
                if arg == "--workspace" and index + 1 < len(command)
                else arg.removeprefix("--workspace=")
                if arg.startswith("--workspace=")
                else None
            )
            if value:
                path = Path(value)
                stage = "cwd"
                workspaces.add(
                    (path if path.is_absolute() else Path(process.cwd()) / path).resolve()
                )
        stage = "name"
        name = process.name()
        if identities is not None:
            stage = "birth_after"
            if processes_api.Process(process.pid).create_time() != identities[process.pid]:
                failures.append((process.pid, "IdentityChanged", stage))
                return None
        return dict(
            pid=process.pid,
            parent=parent,
            name=name,
            host=False,
            rss=memory.rss,
            private=getattr(memory, "private", None),
            peak_working_set=getattr(memory, "peak_wset", None),
        )
    except processes_api.Error as error:
        failures.append((process.pid, error, stage))
        return None


def _process_rows(
    observed, processes_api, workspaces, parent_snapshot=None, exited=None, identities=None
):
    observed = list(observed)
    processes, failures = [], []
    parents = _process_parents(observed, processes_api, parent_snapshot, failures)
    for process in observed:
        if process.pid in parents:
            row = _process_row(
                process, parents[process.pid], processes_api, workspaces, identities, failures
            )
            if row is not None:
                processes.append(row)
    surviving_parents = {parents[row["pid"]] for row in processes}
    if exited is not None:
        exited.update(
            pid for pid, error, _ in failures if isinstance(error, processes_api.NoSuchProcess)
        )
    errors = [
        f"pid={pid} error={type(error).__name__ if isinstance(error, BaseException) else error} "
        f"stage={stage}"
        for pid, error, stage in failures
        if not isinstance(error, processes_api.NoSuchProcess)
        or (
            parent_snapshot is None
            and (stage in {"ppid", "parent_snapshot"} or pid in surviving_parents)
        )
    ]
    return processes, errors


def _workspace_endpoints(workspaces, explicit, connection_type, connection_error):
    endpoints: dict[int, str] = {}
    complete = True
    required_ports: set[int] = set()
    for workspace in workspaces:
        try:
            connection = connection_type.read(workspace)
            port = urlsplit(connection.url).port
            if port is not None:
                endpoints[port] = connection.instance
                if workspace in explicit:
                    required_ports.add(port)
            elif workspace in explicit:
                complete = False
        except (connection_error, OSError):
            if workspace in explicit:
                complete = False
    return endpoints, required_ports, complete


def _host_tree(processes, hosts):
    for row in processes:
        row["host"] = row["pid"] in hosts
    members = set(hosts)
    while True:
        expanded = members | {p["pid"] for p in processes if p["parent"] in members}
        if expanded == members:
            break
        members = expanded
    private = [p["private"] for p in processes if p["pid"] in members]
    errors = [
        f"pid={p['pid']} error=PrivateCommitUnmeasured stage=memory_info"
        for p in processes
        if p["pid"] in members and (type(p["private"]) is not int or p["private"] < 0)
    ]
    return private, errors


def _sample_processes(observed, processes_api, workspaces, snapshot):
    if snapshot is None:
        processes, errors = _process_rows(observed, processes_api, workspaces)
        other_errors = []
    else:
        host = [p for p in observed if p.pid in snapshot.members and p.pid not in snapshot.exited]
        other = [p for p in observed if p.pid not in snapshot.members]
        processes, errors = _process_rows(
            host, processes_api, workspaces, snapshot.parents, snapshot.exited, snapshot.identities
        )
        other_rows, other_errors = _process_rows(other, processes_api, workspaces)
        processes.extend(other_rows)
        errors.extend(snapshot.errors)
    return processes, errors, other_errors


def _snapshot_details(snapshot, other_errors):
    if snapshot is None:
        return {}
    details = {
        "host_exits": [
            dict(pid=pid, birth=snapshot.identities.get(pid), parent=snapshot.parents.get(pid))
            for pid in sorted(snapshot.exited)
        ]
    }
    activity_errors = list(filter(None, [getattr(snapshot, "agent_error", None), *other_errors]))
    if activity_errors:
        details["activity_tree_error"] = "; ".join(activity_errors)
    return details


def _binding_errors(snapshot, hosts, bindings):
    if snapshot is None:
        return []
    errors = []
    if hosts != snapshot.roots:
        errors.append("pid=unknown error=HostIdentityChanged stage=listener_bindings")
    binding = getattr(snapshot, "binding", None)
    if (snapshot.roots and binding is None) or (
        binding is not None
        and (
            not isinstance(binding, tuple)
            or len(binding) != 4
            or type(binding[0]) is not int
            or type(binding[1]) is not int
            or not isinstance(binding[2], str)
            or type(binding[3]) not in (int, float)
        )
    ):
        errors.append("pid=unknown error=HostBindingUnmeasured stage=listener_bindings")
    elif binding is not None and (
        binding[0] not in snapshot.roots
        or snapshot.identities.get(binding[0]) != binding[3]
        or not any((b["pid"], b["port"], b["instance"]) == binding[:3] for b in bindings)
    ):
        errors.append("pid=unknown error=HostIdentityChanged stage=listener_bindings")
    return errors


def sample_memory(
    observed, *, workspaces=(), bindings_of, processes_api, connection_type, connection_error
) -> dict:
    """Read private commit once per process; identify Hosts by their declared listeners."""
    explicit = {Path(path).resolve() for path in workspaces}
    workspaces = set(explicit)
    snapshot = getattr(observed, "host_snapshot", None)
    processes, errors, other_errors = _sample_processes(
        observed, processes_api, workspaces, snapshot
    )
    endpoints, required_ports, endpoints_complete = _workspace_endpoints(
        workspaces, explicit, connection_type, connection_error
    )
    if not endpoints_complete:
        errors.append("pid=unknown error=WorkspaceUnmeasured stage=workspace_connection")
    try:
        pids = {p["pid"] for p in processes}
        if snapshot is not None:
            pids &= snapshot.roots
        bindings = bindings_of(endpoints, pids) if endpoints else []
    except processes_api.Error as error:
        bindings = []
        errors.append(f"pid=unknown error={type(error).__name__} stage=listener_bindings")
    if not required_ports <= {binding["port"] for binding in bindings}:
        errors.append("pid=unknown error=ListenerUnmeasured stage=listener_bindings")
    hosts = {b["pid"] for b in bindings}
    errors.extend(_binding_errors(snapshot, hosts, bindings))
    private, tree_errors = _host_tree(processes, hosts)
    if snapshot is not None:
        private, tree_errors = _host_tree(processes, snapshot.members)
        for row in processes:
            row["host"] = row["pid"] in hosts
    errors.extend(tree_errors)
    complete = not errors
    return dict(
        processes=processes,
        host_bindings=bindings,
        rss_sum=sum(p["rss"] for p in processes),
        host_private_commit_bytes=sum(private) if complete else None,
        host_tree_complete=complete,
        host_tree_observed=bool(hosts),
        **_snapshot_details(snapshot, other_errors),
        **({"host_tree_error": "; ".join(errors)} if errors else {}),
    )


_HOST_REPORT = re.compile(r"\bThis Host \(process (\d+)\) runs apart from the shell")


def pin_reported_host(result, processes):
    """Bind the reported Host at launch, before its PID can be reused."""
    if match := _HOST_REPORT.search(result.get("stdout", "")):
        try:
            process = processes.Process(int(match.group(1)))
            result["reported_host"] = dict(pid=process.pid, birth=process.create_time())
        except processes.Error as error:
            result["reported_host_error"] = f"{type(error).__name__}: {error}"


def reported_host(result, workspace, owned, processes, now):
    """Resolve the launch-bound Host identity without adopting a reused or foreign PID."""
    match = _HOST_REPORT.search(result.get("stdout", ""))
    if not match:
        return None
    identity = result.get("reported_host") or {}
    pid, birth = int(match.group(1)), identity.get("birth")
    if (
        identity.get("pid") != pid
        or not isinstance(birth, (int, float))
        or birth < float(result["birth"])
    ):
        raise ValueError("reported Host birth was not proved at launch")
    row = dict(workspace=str(workspace), pid=pid, birth=birth, process_gone=False)
    try:
        process = processes.Process(pid)
        if process.create_time() != birth:
            row.update(process_gone=True, exit_observed_at=now())
            return row, None
        words = " ".join(process.cmdline()) + " " + process.exe()
        if not owned.search(words):
            raise ValueError("reported Host identity is not owned by this launch")
        return row, process
    except processes.NoSuchProcess:
        row.update(process_gone=True, exit_observed_at=now())
        return row, None


def finish_listener_stop(row, process, port, deadline, processes, monotonic):
    try:
        process.wait(timeout=max(0, deadline - monotonic()))
        process_gone = not process.is_running()
    except processes.NoSuchProcess:
        process_gone = True
    except processes.Error as error:
        process_gone = False
        row["reason"] = f"process {process.pid}: {type(error).__name__}"
    try:
        port_closed = not any(
            c.status == processes.CONN_LISTEN and c.laddr and c.laddr.port == port
            for c in processes.net_connections(kind="tcp")
        )
    except (OSError, processes.Error) as error:
        port_closed = False
        row["reason"] = f"port {port}: {type(error).__name__}"
    row.update(
        process_gone=process_gone, port_closed=port_closed, stopped=process_gone and port_closed
    )
    if not row["stopped"] and "reason" not in row:
        row["reason"] = "process or port still live"


def finish_reported_stop(row, process, deadline, processes, monotonic, now):
    try:
        process.wait(timeout=max(0, deadline - monotonic()))
        row["process_gone"] = not process.is_running()
    except processes.NoSuchProcess:
        row["process_gone"] = True
    except processes.Error as error:
        row["reason"] = f"process {process.pid}: {type(error).__name__}"
    if row["process_gone"]:
        row["exit_observed_at"] = now()
    else:
        row.setdefault("reason", "reported Host process still live")
