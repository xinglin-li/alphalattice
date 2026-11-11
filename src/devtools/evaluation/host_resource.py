"""Observe the bound workspace Host and its process tree."""

from pathlib import Path
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
    """Inventory owned roots independently, retaining a failure for this interval."""
    held, errors = {}, []
    for pid, birth in list(seen.items()):
        try:
            process = processes_api.Process(pid)
            if process.create_time() != birth:
                errors.append("process identity changed")
                continue
            held[pid] = birth
            for child in process.children(recursive=True):
                try:
                    held[child.pid] = child.create_time()
                except processes_api.Error as error:
                    errors.append(type(error).__name__)
        except processes_api.Error as error:
            errors.append(type(error).__name__)
            if isinstance(error, processes_api.NoSuchProcess):
                seen.pop(pid, None)
    seen.update(held)
    return list(held), ", ".join(errors) or None


def process_census(owned, parent, observed_roots, seen, processes_api):
    """Keep every observable descendant and carry any incomplete discovery to its sample."""
    errors, observed = [], []

    def record(error):
        errors.append(type(error).__name__ if isinstance(error, BaseException) else str(error))

    try:
        if owned is not None:
            inventory = getattr(owned, "process_inventory", None)
            pids, error = inventory() if inventory else (owned.process_ids(), None)
            if error:
                record(error)
            for pid in pids:
                try:
                    observed.append(processes_api.Process(pid))
                except processes_api.Error as error:
                    record(error)
        elif parent is not None:
            observed.append(parent)
            observed.extend(parent.children(recursive=True))
    except (processes_api.Error, OSError, ValueError) as error:
        record(error)
    if observed_roots is not None:
        try:
            for root in observed_roots():
                observed.append(root)
                try:
                    observed.extend(root.children(recursive=True))
                except processes_api.Error as error:
                    record(error)
        except (processes_api.Error, OSError, ValueError) as error:
            record(error)
    for process in {p.pid: p for p in observed}.values():
        try:
            seen[process.pid] = process.create_time()
        except processes_api.Error as error:
            record(error)
    current = []
    for pid, birth in list(seen.items()):
        try:
            process = processes_api.Process(pid)
            if process.create_time() != birth:
                seen.pop(pid, None)
                record("process identity changed")
                continue
            current.append(process)
        except processes_api.Error as error:
            record(error)
            if isinstance(error, processes_api.NoSuchProcess):
                seen.pop(pid, None)
    return current, ", ".join(errors) or None


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

    def roots(self):
        self.measured.clear()
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
        return [process]

    def workspaces(self):
        return list(self.measured)


def _process_rows(observed, processes_api, workspaces):
    processes, complete = [], True
    for process in observed:
        try:
            memory = process.memory_info()
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
                    workspaces.add(
                        (path if path.is_absolute() else Path(process.cwd()) / path).resolve()
                    )
            processes.append(
                dict(
                    pid=process.pid,
                    parent=process.ppid(),
                    name=process.name(),
                    host=False,
                    rss=memory.rss,
                    private=getattr(memory, "private", None),
                    peak_working_set=getattr(memory, "peak_wset", None),
                )
            )
        except processes_api.Error:
            complete = False
    return processes, complete


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


def sample_memory(
    observed, *, workspaces=(), bindings_of, processes_api, connection_type, connection_error
) -> dict:
    """Read private commit once per process; identify Hosts by their declared listeners."""
    explicit = {Path(path).resolve() for path in workspaces}
    workspaces = set(explicit)
    processes, complete = _process_rows(observed, processes_api, workspaces)
    endpoints, required_ports, endpoints_complete = _workspace_endpoints(
        workspaces, explicit, connection_type, connection_error
    )
    complete &= endpoints_complete
    try:
        pids = {p["pid"] for p in processes}
        bindings = bindings_of(endpoints, pids) if endpoints else []
    except processes_api.Error:
        bindings, complete = [], False
    complete &= required_ports <= {binding["port"] for binding in bindings}
    hosts = {b["pid"] for b in bindings}
    for row in processes:
        row["host"] = row["pid"] in hosts
    members = set(hosts)
    while True:
        expanded = members | {p["pid"] for p in processes if p["parent"] in members}
        if expanded == members:
            break
        members = expanded
    private = [p["private"] for p in processes if p["pid"] in members]
    complete &= all(type(value) is int and value >= 0 for value in private)
    return dict(
        processes=processes,
        host_bindings=bindings,
        rss_sum=sum(p["rss"] for p in processes),
        host_private_commit_bytes=sum(private) if complete else None,
        host_tree_complete=complete,
        host_tree_observed=bool(hosts),
    )
