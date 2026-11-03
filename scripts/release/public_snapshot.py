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
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "alphalattice.release.public-manifest"
PROVENANCE_SCHEMA = "alphalattice.release.snapshot-provenance"
_IDENTITY = re.compile(r"^(?P<name>[^<>]+?) <(?P<email>[^<>@\s]+@[^<>\s]+)>$")


class SnapshotRefused(ValueError):
    """A snapshot the builder will not make, named by its reason."""


def _git(repository: Path, *arguments: str, stdin: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments], input=stdin, capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        raise SnapshotRefused(
            f"release.git_failed:{arguments[0]}:{detail[-1] if detail else completed.returncode}"
        )
    return completed.stdout


def _tree(repository: Path, revision: str) -> dict[str, tuple[str, str]]:
    """Each tracked path's mode and object id at `revision`."""
    tree = {}
    for line in _git(repository, "ls-tree", "-r", "-z", revision).split(b"\0"):
        if line:
            info, name = line.split(b"\t", 1)
            mode, _kind, oid = info.decode().split()
            tree[name.decode("utf-8")] = (mode, oid)
    return tree


def _blobs(repository: Path, oids: list[str]) -> dict[str, bytes]:
    """The blobs' bytes by one `git cat-file --batch`: one process for the whole snapshot, never
    one per file (a release has thousands)."""
    wanted = sorted(set(oids))
    names = "".join(oid + "\n" for oid in wanted).encode()
    raw = _git(repository, "cat-file", "--batch", stdin=names)
    blobs, cursor = {}, 0
    for oid in wanted:
        end = raw.find(b"\n", cursor)
        header = raw[cursor:end].decode("utf-8", "replace").split() if end >= 0 else []
        if len(header) != 3 or header[:2] != [oid, "blob"] or not header[2].isdigit():
            raise SnapshotRefused(f"release.git_failed:cat-file:{oid}")
        size = int(header[2])
        blobs[oid] = raw[end + 1 : end + 1 + size]
        cursor = end + 2 + size
    return blobs


SCANNER = "scripts/release/public_manifest.py"
Projection = Callable[[str, bytes, set[str], Any], tuple[bytes, list[str]]]


def source_projection(repository: Path, source: str) -> Projection | None:
    """The source's own registry projection, read from its scanner at that commit, as the export
    runs that scanner to hash the manifest: what is published is what was hashed, whichever of
    the source and this builder is newer. A source older than the projection has none, and its
    files publish as they are."""
    listed = _git(repository, "ls-tree", source, "--", SCANNER)
    if not listed:
        return None
    code = _git(repository, "cat-file", "blob", f"{source}:{SCANNER}")
    module = types.ModuleType("source_public_manifest")
    module.__file__ = str(repository / SCANNER)  # its ROOT is the source repository
    sys.modules[module.__name__] = module  # dataclasses and annotations resolve their module
    try:
        exec(compile(code, f"{source}:{SCANNER}", "exec"), module.__dict__)
    except Exception as error:  # the source's own code; refuse by name, never publish unprojected
        raise SnapshotRefused(f"release.source_scanner_unloadable:{type(error).__name__}") from None
    finally:
        sys.modules.pop(module.__name__, None)
    found = getattr(module, "project_public", None)
    return found if callable(found) else None


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
    tree = _tree(repository, source)
    modes = {path: mode for path, (mode, _oid) in tree.items()}
    public = [str(entry["path"]) for entry in manifest["public"]]
    # read once up front; the loop below still refuses an absent path or a link in manifest order
    blobs = _blobs(repository, [tree[p][1] for p in public if p in tree and tree[p][0] != "120000"])
    files: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    private = set(modes) - {str(entry["path"]) for entry in manifest["public"]}
    project = source_projection(repository, source)
    for entry in manifest["public"]:
        path = str(entry["path"])
        mode = modes.get(path)
        if mode is None:
            raise SnapshotRefused(f"release.file_absent_at_source:{path}")
        if mode == "120000":
            raise SnapshotRefused(f"release.link_not_published:{path}")
        source_blob = blobs[tree[path][1]]
        blob, dropped = (
            project(path, source_blob, private, modes.keys())
            if project
            else (
                source_blob,
                [],
            )
        )
        if dropped:  # the manifest hashed the same projection; the check below proves both agree
            projections.append(
                {
                    "path": path,
                    "source_sha256": hashlib.sha256(source_blob).hexdigest(),
                    "dropped": dropped,
                }
            )
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
    written = _tree(out, "HEAD")
    kept = _blobs(out, [written[row["path"]][1] for row in files])
    for row in files:
        if hashlib.sha256(kept[written[row["path"]][1]]).hexdigest() != row["sha256"]:
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
        "registry_projections": projections,
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
