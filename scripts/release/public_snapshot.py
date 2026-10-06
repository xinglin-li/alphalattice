"""Build a release's public snapshot from the private repository (RR2, the user, 2026-10-01).

The public repository is a new one. Its files are the public manifest's
(`config/release/public-manifest.json`), read as the accepted commit's blobs, never the working
tree, each checked against the hash the manifest took; its one first commit is made under the
user's identity, given explicitly. The internal history stays in the private repository and is
never imported. The provenance -- the private source SHA, the manifest's hash, every public
file with its SHA-256 and size, and the snapshot's commit and tree -- is written where the
private repository keeps it. Nothing is pushed: publishing is the user's act (RR9).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "alphalattice.release.public-manifest"
PROVENANCE_SCHEMA = "alphalattice.release.snapshot-provenance"
_IDENTITY = re.compile(r"^(?P<name>[^<>]+?) <(?P<email>[^<>@\s]+@[^<>\s]+)>$")


class SnapshotRefused(ValueError):
    """A snapshot the builder will not make, named by its reason."""


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments], capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        raise SnapshotRefused(
            f"release.git_failed:{arguments[0]}:{detail[-1] if detail else completed.returncode}"
        )
    return completed.stdout


def load_manifest(path: Path) -> dict[str, Any]:
    """The public manifest, refused when it is not one the builder can make a snapshot from.

    Args:
        path: The manifest file.

    Returns:
        The manifest.

    Raises:
        SnapshotRefused: An unknown schema, a file no rule settles, or a file list that is not
            sorted and unique.
    """
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema"), manifest.get("schema_version")) != (MANIFEST_SCHEMA, 1):
        raise SnapshotRefused("release.manifest_schema_unknown")
    if manifest.get("unsettled"):
        unsettled = sorted(str(entry["path"]) for entry in manifest["unsettled"])
        raise SnapshotRefused("release.manifest_unsettled:" + ",".join(unsettled)[:500])
    paths = [str(entry["path"]) for entry in manifest.get("public", ())]
    if not paths or paths != sorted(set(paths)):
        raise SnapshotRefused("release.manifest_paths_not_sorted_unique")
    return manifest


def identity(author: str) -> tuple[str, str]:
    """The name and email of `Name <email>`.

    Args:
        author: The user's identity for the first commit.

    Returns:
        The name and the email.

    Raises:
        SnapshotRefused: When it is not `Name <email>`.
    """
    matched = _IDENTITY.match(author.strip())
    if matched is None:
        raise SnapshotRefused("release.author_not_name_and_email")
    return matched["name"].strip(), matched["email"]


def build(
    repository: Path, manifest_path: Path, out: Path, *, author: str, message: str
) -> dict[str, Any]:
    """Make the public snapshot of the manifest's commit in a new repository at `out`.

    Args:
        repository: The private repository.
        manifest_path: The public manifest.
        out: A directory that does not exist or is empty; the new repository.
        author: The user's identity, `Name <email>`, author and committer of the one commit.
        message: The first commit's message.

    Returns:
        The provenance: the source SHA, the manifest's hash, each file's path, SHA-256 and
        size, the totals and the snapshot's commit and tree.

    Raises:
        SnapshotRefused: A manifest the builder refuses, a source it cannot read, a blob whose
            hash is not the manifest's, a link, an output that holds files, or a snapshot whose
            files do not read back as the manifest's.
    """
    manifest = load_manifest(manifest_path)
    name, email = identity(author)
    source = str(manifest["source_sha"])
    resolved = _git(repository, "rev-parse", "--verify", f"{source}^{{commit}}").decode().strip()
    if resolved != source:
        raise SnapshotRefused("release.source_sha_not_a_full_commit_id")
    if out.exists() and any(out.iterdir()):
        raise SnapshotRefused("release.output_not_empty")
    listed = _git(repository, "ls-tree", "-r", "-z", source).split(b"\0")
    modes = {
        line.split(b"\t", 1)[1].decode("utf-8"): line.split(b" ", 1)[0].decode()
        for line in listed
        if line
    }
    files: list[dict[str, Any]] = []
    for entry in manifest["public"]:
        path = str(entry["path"])
        mode = modes.get(path)
        if mode is None:
            raise SnapshotRefused(f"release.file_absent_at_source:{path}")
        if mode == "120000":
            raise SnapshotRefused(f"release.link_not_published:{path}")
        blob = _git(repository, "cat-file", "blob", f"{source}:{path}")
        digest = hashlib.sha256(blob).hexdigest()
        if digest != entry["sha256"]:
            raise SnapshotRefused(f"release.hash_mismatch:{path}")
        target = out.joinpath(*path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        files.append({"path": path, "sha256": digest, "bytes": len(blob), "mode": mode})
    _git(out, "init", "-q")
    for key, value in (
        ("user.name", name),
        ("user.email", email),
        ("core.autocrlf", "false"),
        ("core.safecrlf", "false"),
    ):
        _git(out, "config", key, value)
    _git(out, "add", "--all")
    for row in files:
        if row["mode"] == "100755":
            _git(out, "update-index", "--chmod=+x", row["path"])
    _git(out, "commit", "-q", "-m", message)
    commit = _git(out, "rev-parse", "HEAD").decode().strip()
    published = sorted(_git(out, "ls-files", "-z").decode("utf-8").split("\0")[:-1])
    if published != [row["path"] for row in files]:
        raise SnapshotRefused("release.snapshot_files_not_the_manifest")
    for row in files:
        if (
            hashlib.sha256(_git(out, "cat-file", "blob", f"HEAD:{row['path']}")).hexdigest()
            != (row["sha256"])
        ):
            raise SnapshotRefused(f"release.snapshot_changed_bytes:{row['path']}")
    return {
        "schema": PROVENANCE_SCHEMA,
        "schema_version": 1,
        "source_sha": source,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "author": f"{name} <{email}>",
        "snapshot_commit": commit,
        "snapshot_tree": _git(out, "rev-parse", "HEAD^{tree}").decode().strip(),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--manifest", type=Path, default=Path("config/release/public-manifest.json")
    )
    parser.add_argument("--out", type=Path, required=True, help="the new repository's directory")
    parser.add_argument("--author", required=True, help="the user's identity, Name <email>")
    parser.add_argument("--message", required=True, help="the first commit's message")
    parser.add_argument("--provenance", type=Path, required=True, help="where it is kept")
    args = parser.parse_args(argv)
    try:
        provenance = build(
            args.repository,
            args.repository / args.manifest if not args.manifest.is_absolute() else args.manifest,
            args.out,
            author=args.author,
            message=args.message,
        )
    except SnapshotRefused as error:
        print(json.dumps({"status": "REFUSED", "failure_code": str(error)}))
        return 2
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": "BUILT",
                "snapshot_commit": provenance["snapshot_commit"],
                "file_count": provenance["file_count"],
                "provenance": str(args.provenance),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
