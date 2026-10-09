"""The public snapshot's builder (RR2): the manifest's files read from the source commit, one
commit under the user's identity, the provenance kept, and nothing else."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from scripts.release.public_snapshot import SnapshotRefused, build


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # The machine's own Git configuration (its signing, hooks or line endings) stays out.
    empty = tmp_path / "gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repository = tmp_path / "private"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "tool.py").write_bytes(b"print('public')\r\n")
    (repository / "LICENSE").write_bytes(b"Apache License\n")
    (repository / "board.md").write_bytes(b"internal\n")
    _git(repository, "init", "-q")
    for key, value in (
        ("user.name", "Someone Else"),
        ("user.email", "else@example.com"),
        ("core.autocrlf", "false"),
    ):
        _git(repository, "config", key, value)
    _git(repository, "add", "--all")
    _git(repository, "commit", "-q", "-m", "internal history")
    return repository


def _manifest(repository: Path, path: Path, files: list[str]) -> str:
    source = _git(repository, "rev-parse", "HEAD")
    public = [
        {
            "path": name,
            "sha256": hashlib.sha256((repository / name).read_bytes()).hexdigest(),
            "rule": "test",
        }
        for name in sorted(files)
    ]
    document = {
        "schema": "alphalattice.release.public-manifest",
        "schema_version": 1,
        "source_sha": source,
        "rules": [],
        "public": public,
        "private_count": 1,
        "unsettled": [],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return source


def test_the_snapshot_holds_the_manifests_files_in_one_commit_by_the_user(
    private: Path, tmp_path: Path
) -> None:
    """A user-created snapshot holds every manifest file from one committed revision."""

    manifest = tmp_path / "manifest.json"
    source = _manifest(private, manifest, ["LICENSE", "src/tool.py"])
    out = tmp_path / "public"
    provenance = build(
        private,
        manifest,
        out,
        author="Xinglin Li <xinglin@example.com>",
        message="AlphaLattice: first public release",
    )
    assert (provenance["source_sha"], provenance["file_count"]) == (source, 2)
    assert [row["path"] for row in provenance["files"]] == ["LICENSE", "src/tool.py"]
    assert provenance["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert _git(out, "log", "--format=%an <%ae>|%cn <%ce>") == (
        "Xinglin Li <xinglin@example.com>|Xinglin Li <xinglin@example.com>"
    )
    assert _git(out, "rev-list", "--count", "HEAD") == "1"
    assert _git(out, "ls-files").split() == ["LICENSE", "src/tool.py"]
    assert (out / "src" / "tool.py").read_bytes() == b"print('public')\r\n"
    assert not (out / "board.md").exists()
    assert provenance["snapshot_commit"] == _git(out, "rev-parse", "HEAD")


def test_the_builder_refuses_what_it_cannot_prove(private: Path, tmp_path: Path) -> None:
    """requirement (RR2): a blob whose hash is not the manifest's, a file no rule settles, an
    identity that is not `Name <email>` and an output that already holds files are refused by
    name, and nothing is committed."""

    manifest = tmp_path / "manifest.json"
    _manifest(private, manifest, ["LICENSE"])
    document = json.loads(manifest.read_text(encoding="utf-8"))
    for change, code in (
        (
            {"public": [{**document["public"][0], "sha256": "0" * 64}]},
            "release.hash_mismatch:LICENSE",
        ),
        (
            {"unsettled": [{"path": "board.md", "evidence": "no rule"}]},
            "release.manifest_unsettled:board.md",
        ),
    ):
        manifest.write_text(json.dumps({**document, **change}), encoding="utf-8")
        with pytest.raises(SnapshotRefused, match=code):
            build(
                private, manifest, tmp_path / "refused", author="X Y <x@example.com>", message="m"
            )
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SnapshotRefused, match="author_not_name_and_email"):
        build(private, manifest, tmp_path / "refused", author="Xinglin Li", message="m")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "x").write_text("x", encoding="utf-8")
    with pytest.raises(SnapshotRefused, match="output_not_empty"):
        build(private, manifest, occupied, author="X Y <x@example.com>", message="m")
    assert not (tmp_path / "refused" / ".git").exists()
