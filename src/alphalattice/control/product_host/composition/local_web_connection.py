"""Host-owned ephemeral client discovery; neither a Task nor artifact authority."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from alphalattice.interface.local_application.client import (
    LocalResearchClientError,
    LocalResearchConnection,
)


def publish_client_connection(connection: LocalResearchConnection) -> None:
    """Atomically publish an exact local connection after checking workspace ownership.

    Args:
        connection: Explicit live service connection metadata.

    Raises:
        LocalResearchClientError: Existing connection belongs to another workspace.
        OSError: Temporary write, fsync or atomic replacement fails; temporary cleanup is attempted.
    """
    root = Path(connection.workspace)
    path = connection.path(root)
    if path.exists():
        prior = LocalResearchConnection.read(root)
        if prior.workspace_id != connection.workspace_id:
            raise LocalResearchClientError("local_client.connection_owner_mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".client-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(connection.to_json().encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_client_connection(connection: LocalResearchConnection) -> None:
    """Remove local connection metadata only when it still names this service instance.

    Unknown, changed or unreadable metadata is retained; cleanup does not strand writer-lease
    shutdown.

    Args:
        connection: Exact service connection whose instance may be cleaned up.
    """
    root = Path(connection.workspace)
    try:
        path = connection.path(root)
        if path.exists() and LocalResearchConnection.read(root).instance == connection.instance:
            path.unlink()
    except (LocalResearchClientError, OSError):
        # Preserve unknown/changed metadata without stranding the writer lease.
        # A later client still has to authenticate to a live service instance.
        return
