"""The public snapshot's builder (RR2): the manifest's files read from the source commit, one
commit under the user's identity, the provenance kept, and nothing else."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from scripts.release.public_manifest import project_public
from scripts.release.public_snapshot import SnapshotRefused, build

SCANNER = "scripts/release/public_manifest.py"  # each source's own scanner and projection

REGISTRY = "config/registries/sample.json"
REGISTRY_SOURCE = (
    json.dumps(
        {"schema": "sample", "rows": [{"pattern": "src/tool.py"}, {"pattern": "board.md"}]},
        indent=2,
    )
    + "\n"
)
REGISTRY_PUBLIC = (
    json.dumps({"schema": "sample", "rows": [{"pattern": "src/tool.py"}]}, indent=2) + "\n"
)


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
    (repository / "config" / "registries").mkdir(parents=True)
    (repository / REGISTRY).write_text(REGISTRY_SOURCE, encoding="utf-8", newline="\n")
    (repository / SCANNER).parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[3] / SCANNER, repository / SCANNER)
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


@pytest.mark.proves("pipeline.public_snapshot_holds_only_proved_bytes")
def test_the_snapshot_holds_the_manifests_files_in_one_commit_by_the_user(
    private: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: Public snapshots contain exact manifest bytes and one commit by the user, read
    and read back by one Git batch each, never a process per file (a release has thousands)."""

    manifest = tmp_path / "manifest.json"
    source = _manifest(private, manifest, ["LICENSE", "src/tool.py"])
    out = tmp_path / "public"
    reads, run = [], subprocess.run

    def counted(argv: list[str], **options: object) -> subprocess.CompletedProcess:
        reads.extend([argv[3:5]] if "cat-file" in argv else [])
        return run(argv, **options)

    monkeypatch.setattr(subprocess, "run", counted)
    provenance = build(
        private,
        manifest,
        out,
        author="Example Author <author@example.com>",
        message="AlphaLattice: first public release",
    )
    monkeypatch.setattr(subprocess, "run", run)
    assert reads.count(["cat-file", "--batch"]) == 2 and len(reads) == 3  # 3: the scanner's own
    assert (provenance["source_sha"], provenance["file_count"]) == (source, 2)
    assert [row["path"] for row in provenance["files"]] == ["LICENSE", "src/tool.py"]
    assert provenance["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert _git(out, "log", "--format=%an <%ae>|%cn <%ce>") == (
        "Example Author <author@example.com>|Example Author <author@example.com>"
    )
    assert _git(out, "rev-list", "--count", "HEAD") == "1"
    assert _git(out, "ls-files").split() == ["LICENSE", "src/tool.py"]
    assert (out / "src" / "tool.py").read_bytes() == b"print('public')\r\n"
    assert not (out / "board.md").exists()
    assert provenance["snapshot_commit"] == _git(out, "rev-parse", "HEAD")
    # A public registry publishes without its rows naming a private file, as the manifest hashed
    # it; the private registry stays as committed.
    document = json.loads(manifest.read_text(encoding="utf-8"))
    published = hashlib.sha256(REGISTRY_PUBLIC.encode()).hexdigest()
    document["public"] = [
        *document["public"],
        {"path": REGISTRY, "sha256": published, "rule": "test"},
    ]
    document["public"].sort(key=lambda row: row["path"])
    manifest.write_text(json.dumps(document), encoding="utf-8")
    # A projected registry loads through its declared owner in the public tree, or is refused.
    from scripts.release import public_snapshot

    for loader, refused_as in ((None, "unloaded"), ("raise ValueError('rule')", "unloadable")):
        with monkeypatch.context() as declared:
            declared.setattr(
                public_snapshot, "PROJECTION_LOADERS", {REGISTRY: loader} if loader else {}
            )
            with pytest.raises(
                SnapshotRefused, match=f"release.projection_{refused_as}:{REGISTRY}"
            ):
                build(
                    private,
                    manifest,
                    tmp_path / refused_as,
                    author="X Y <x@example.com>",
                    message="m",
                )
    # A private checkout on PYTHONPATH cannot supply a missing public owner.
    with monkeypatch.context() as declared:
        declared.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[3] / "src"))
        declared.setitem(
            public_snapshot.PROJECTION_LOADERS, REGISTRY, "import devtools.testing.sizes"
        )
        with pytest.raises(SnapshotRefused, match=r"release\.projection_unloadable"):
            public_snapshot._load_projected(tmp_path / "unloaded", REGISTRY)
    loads = "import json\njson.loads((root / 'config/registries/sample.json').read_text())"
    monkeypatch.setitem(public_snapshot.PROJECTION_LOADERS, REGISTRY, loads)
    try:
        projected = build(
            private, manifest, tmp_path / "projected", author="X Y <x@example.com>", message="m"
        )
    except SnapshotRefused as refused:  # the raw registry is not the bytes the manifest hashed
        projected = {"refused": str(refused)}
    assert projected.get("registry_projections") == [
        {
            "path": REGISTRY,
            "source_sha256": hashlib.sha256(REGISTRY_SOURCE.encode()).hexdigest(),
            "dropped": ["rows[1]"],
        }
    ]
    assert (tmp_path / "projected" / REGISTRY).read_text(encoding="utf-8") == REGISTRY_PUBLIC
    assert _git(private, "show", f"HEAD:{REGISTRY}") + "\n" == REGISTRY_SOURCE
    # A source from before the projection: its scanner hashed the raw registry, and the builder
    # publishes those bytes; a projected hash for that source is still a named mismatch.
    (private / SCANNER).write_text('"""An older scanner."""\n', encoding="utf-8")
    _git(private, "commit", "-qam", "an older scanner")
    for hashed, ends in ((REGISTRY_SOURCE, "historical"), (REGISTRY_PUBLIC, "mismatch")):
        document["source_sha"] = _git(private, "rev-parse", "HEAD")
        for row in document["public"]:
            if row["path"] == REGISTRY:
                row["sha256"] = hashlib.sha256(hashed.encode()).hexdigest()
        manifest.write_text(json.dumps(document), encoding="utf-8")
        try:
            older = build(private, manifest, tmp_path / ends, author="X Y <x@y.z>", message="m")
            assert older["registry_projections"] == [] and ends == "historical"
            assert (tmp_path / ends / REGISTRY).read_text(encoding="utf-8") == REGISTRY_SOURCE
        except SnapshotRefused as refused:
            assert (ends, str(refused)) == ("mismatch", f"release.hash_mismatch:{REGISTRY}")
    # Every valid form a registry is written in projects in that form; nothing dropped keeps its
    # exact bytes.
    forms = {
        b'{"rows":[{"p":"board.md"},{"p":"src/tool.py"}]}': b'{"rows":[{"p":"src/tool.py"}]}',
        b'{"rows": [{"p": "board.md"}, {"p": "src/tool.py"}]}\n': (
            b'{"rows": [{"p": "src/tool.py"}]}\n'
        ),
        b'[{"p":"board.md"},{"p":"src/tool.py"}]\n': b'[{"p":"src/tool.py"}]\n',
        b'\xef\xbb\xbf{\r\n\r\n    "rows": {"a": "board.md", "b": "\xc3\xa9"}\r\n}\r\n': (
            b'\xef\xbb\xbf{\r\n    "rows": {\r\n        "b": "\xc3\xa9"\r\n    }\r\n}\r\n'
        ),
        b'{"rows":[{"p":"src/tool.py"}]}': b'{"rows":[{"p":"src/tool.py"}]}',
    }
    for raw, published in forms.items():
        found = project_public(REGISTRY, raw, {"board.md"}, {"board.md", "src/tool.py"})
        assert found[0] == published, raw
    # Test-size reach is still public when only one of its inputs is private.
    sizes = "config/registries/test-sizes.json"
    private_paths = {
        "tests/example_private/private.py",
        "tests/structural/private.py",
        "example_data/hidden.json",
    }
    public_paths = {"tests/structural/public.py", "tests/structural/shared.py"}
    rows = {
        "schema": "alphalattice.test-sizes.v1",
        "defaults": [{"pattern": "tests/example_private/*"}, {"pattern": "tests/structural/*"}],
        "large": [
            {"pattern": "tests/structural/private.py::*"},
            {"pattern": "tests/structural/public.py::*", "reason": "retained"},
        ],
        "reach": [
            {
                "pattern": "tests/structural/public.py::*",
                "inputs": ["tests/structural/shared.py", "example_data/hidden.json"],
                "reason": "retained",
            },
            {"pattern": "tests/structural/private.py::*", "inputs": []},
        ],
    }
    rows["reach"].append(
        {"pattern": "tests/structural/public.py::malformed", "inputs": "example_data/hidden.json"}
    )
    raw = (json.dumps(rows, indent=2) + "\n").encode()
    published, dropped = project_public(sizes, raw, private_paths, private_paths | public_paths)
    expected = {
        **rows,
        "defaults": rows["defaults"][1:],
        "large": rows["large"][1:],
        "reach": [{**rows["reach"][0], "inputs": ["tests/structural/shared.py"]}],
    }
    assert json.loads(published) == expected
    assert set(dropped) == {"defaults[0]", "large[0]", "reach[0].inputs[1]", "reach[1]", "reach[2]"}
    clean = (
        b'{"schema":"alphalattice.test-sizes.v1",'
        b'"reach":[{"pattern":"tests/structural/public.py::*"}]}'
    )
    assert project_public(sizes, clean, private_paths, private_paths | public_paths) == (clean, [])


@pytest.mark.proves("pipeline.public_snapshot_holds_only_proved_bytes")
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
        build(private, manifest, tmp_path / "refused", author="Example Author", message="m")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "x").write_text("x", encoding="utf-8")
    with pytest.raises(SnapshotRefused, match="output_not_empty"):
        build(private, manifest, occupied, author="X Y <x@example.com>", message="m")
    assert not (tmp_path / "refused" / ".git").exists()
