"""Shared Git blob input for registry and test-shape guards."""

import io
import subprocess
from pathlib import Path


def read_blobs(specs: list[str], root: Path | None = None) -> dict[str, str]:
    """Read distinct Git specifications in one batch, omitting missing blobs."""
    specs = list(dict.fromkeys(specs))
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=("\n".join(specs) + "\n").encode(),
        cwd=root or Path(__file__).resolve().parents[3],
        capture_output=True,
        check=True,
    )
    stream, found = io.BytesIO(result.stdout), {}
    for spec in specs:
        header = stream.readline().split()
        if header[-1:] == [b"missing"]:
            continue
        data = stream.read(int(header[-1]))
        stream.read(1)
        found[spec] = data.decode("utf-8", errors="surrogateescape")
    return found
