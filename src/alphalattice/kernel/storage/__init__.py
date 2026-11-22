"""Persistent storage sessions and objects: every DuckDB open of this process, through one port.

`port` holds the injected driver, counts each instance open by its target, keeps the process's one
in-memory instance whose cursors serve the short in-memory computations, keeps each data root's
table schemas until that root's writer revision changes, and runs the teardowns its owners
register when the process ends. The kernel imports only the standard library: the product's
DuckDB driver is installed from outside it, at each process's entry, and an open with none is
refused by name.
"""

from .port import (
    MEMORY,
    Driver,
    DriverMissing,
    StoragePort,
    attaching,
    close,
    connect,
    install_driver,
    memory,
    on_teardown,
    opens,
    schema,
)

__all__ = [
    "MEMORY",
    "Driver",
    "DriverMissing",
    "StoragePort",
    "attaching",
    "close",
    "connect",
    "install_driver",
    "memory",
    "on_teardown",
    "opens",
    "schema",
]
