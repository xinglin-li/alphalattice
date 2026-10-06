"""A public snapshot's source dependencies close over its supplied manifest."""

import json

import pytest

from release.public_closure import references, violations


def _failures(blobs, public):
    manifest = {"public": [{"path": path} for path in sorted(public)]}
    return {reference.target for reference in violations(manifest, blobs)}


@pytest.mark.parametrize("imported", ["hidden", "missing"])
def test_python_import_requires_a_public_local_module(imported):
    reader = "src/demo/consumer.py"
    target = f"src/demo/{imported}.py"
    blobs = {
        reader: f"import demo.{imported}\n".encode(),
        "src/demo/__init__.py": b"",
        "src/demo/hidden.py": b"value = 1\n",
    }
    public = {reader, "src/demo/__init__.py"}
    assert _failures(blobs, public) == {target}
    if imported == "hidden":
        assert _failures(blobs, public | {target}) == set()


@pytest.mark.parametrize(
    "read",
    [
        '(ROOT / "config" / "hidden.json").read_text()',
        'open(ROOT / "config" / "hidden.json").read()',
        'read_json(ROOT / "config" / "hidden.json")',
    ],
)
def test_checkout_reads_include_a_read_helpers_supplied_path(read):
    reader = "scripts/check.py"
    target = "config/hidden.json"
    source = (
        "import json\nfrom pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "def read_json(path):\n"
        "    return json.loads(path.read_text())\n"
        f"value = {read}\n"
    )
    blobs = {reader: source.encode(), target: b"{}"}
    assert _failures(blobs, {reader}) == {target}
    assert _failures(blobs, {reader, target}) == set()


def test_policy_strings_and_external_parameter_paths_are_not_checkout_reads():
    reader = "scripts/check.py"
    blobs = {
        reader: (
            b"from pathlib import Path\nROOT = Path(__file__).parents[1]\n"
            b'POLICY = "config/hidden.json"\n'
            b'# (ROOT / "config" / "hidden.json").read_text()\n'
            b"def read_external(ROOT):\n    return ROOT.read_text()\n"
        ),
        "config/hidden.json": b"{}",
    }
    assert _failures(blobs, {reader}) == set()


@pytest.mark.parametrize(
    "command",
    [
        'subprocess.run([str(ROOT / "scripts" / "hidden.py")])',
        'subprocess.run([sys.executable, str(ROOT / "scripts" / "hidden.py")])',
        'command = [sys.executable, str(ROOT / "scripts" / "hidden.py")]\n'
        "subprocess.check_output(command)",
    ],
)
def test_subprocess_reads_first_argument_and_a_named_command_list(command):
    reader = "scripts/check.py"
    target = "scripts/hidden.py"
    source = (
        "import subprocess\nimport sys\nfrom pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[1]\n" + command + "\n"
    )
    blobs = {reader: source.encode(), target: b"print(1)\n"}
    assert _failures(blobs, {reader}) == {target}
    assert _failures(blobs, {reader, target}) == set()


def test_cjs_require_ignores_comments_and_quoted_example_source():
    reader = "tools/reader.cjs"
    blobs = {
        reader: (
            b"// require('./comment.cjs')\n"
            b"/* require('./block.cjs') */\n"
            b"const example = \"require('./string.cjs')\";\n"
            b"const actual = require('./hidden.cjs');\n"
        ),
        "tools/comment.cjs": b"",
        "tools/block.cjs": b"",
        "tools/string.cjs": b"",
        "tools/hidden.cjs": b"",
    }
    found = references(reader, blobs[reader], set(blobs))
    assert {(ref.reader, ref.line, ref.target) for ref in found} == {
        (reader, 4, "tools/hidden.cjs")
    }
    assert _failures(blobs, {reader}) == {"tools/hidden.cjs"}
    assert _failures(blobs, {reader, "tools/hidden.cjs"}) == set()


@pytest.mark.parametrize(
    "body",
    [
        "[the test](../tests/hidden.py)\n",
        "Run `python tests/hidden.py`.\n",
        "```powershell\npython tests/hidden.py\n```\n",
    ],
)
def test_markdown_links_inline_paths_and_fenced_commands_need_public_files(body):
    reader = "docs/guide.md"
    target = "tests/hidden.py"
    blobs = {reader: body.encode(), target: b""}
    assert _failures(blobs, {reader}) == {target}
    assert _failures(blobs, {reader, target}) == set()


def test_markdown_directory_link_requires_a_public_descendant():
    reader = "docs/guide.md"
    blobs = {
        reader: b"[examples](../examples/)\n",
        "examples/public.py": b"",
        "examples/hidden.py": b"",
    }
    assert _failures(blobs, {reader}) == {"examples"}
    assert _failures(blobs, {reader, "examples/public.py"}) == set()


def test_copytree_requires_every_source_file_even_when_the_directory_is_public():
    reader = "scripts/check.py"
    blobs = {
        reader: (
            b"import shutil\nfrom pathlib import Path\n"
            b"ROOT = Path(__file__).resolve().parents[1]\n"
            b'shutil.copytree(ROOT / "kit", destination)\n'
        ),
        "kit/public.py": b"",
        "kit/hidden.py": b"",
    }
    public = {reader, "kit/public.py"}
    assert _failures(blobs, public) == {"kit"}
    assert _failures(blobs, public | {"kit/hidden.py"}) == set()


@pytest.mark.parametrize(
    ("handler", "private_cache", "expected"),
    [
        ('value = "{}"', False, set()),
        ("raise", False, {"config/cache.json"}),
        ('value = "{}"', True, {"config/cache.json"}),
    ],
)
def test_only_an_absent_optional_cache_with_a_fallback_is_not_required(
    handler, private_cache, expected
):
    reader = "scripts/check.py"
    source = (
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "try:\n"
        '    value = (ROOT / "config" / "cache.json").read_text()\n'
        "except FileNotFoundError:\n"
        f"    {handler}\n"
    )
    blobs = {reader: source.encode(), "config/public.json": b"{}"}
    if private_cache:
        blobs["config/cache.json"] = b"{}"
    assert _failures(blobs, {reader, "config/public.json"}) == expected


@pytest.mark.parametrize("package", ["browser-kit", "@tools/browser-kit"])
@pytest.mark.parametrize(
    ("declared", "locked", "publish_declaration", "publish_lock", "accepted"),
    [
        ("1.2.3", "1.2.3", True, True, True),
        ("^1.2.3", "1.2.3", True, True, False),
        ("1.2.3", "1.2.4", True, True, False),
        ("1.2.3", "1.2.3", False, True, False),
        ("1.2.3", "1.2.3", True, False, False),
    ],
)
def test_installed_node_dependency_requires_its_exact_public_declaration_and_lock(
    package, declared, locked, publish_declaration, publish_lock, accepted
):
    reader = "tools/reader.cjs"
    target = f"tools/node_modules/{package}/cli.js"
    declaration = "tools/package.json"
    lock = "tools/package-lock.json"
    blobs = {
        reader: f"require('./node_modules/{package}/cli.js');\n".encode(),
        declaration: json.dumps({"devDependencies": {package: declared}}).encode(),
        lock: json.dumps({"packages": {f"node_modules/{package}": {"version": locked}}}).encode(),
    }
    public = {reader}
    if publish_declaration:
        public.add(declaration)
    if publish_lock:
        public.add(lock)
    assert _failures(blobs, public) == (set() if accepted else {target})


def test_a_public_manifest_entry_requires_its_source_blob():
    reader = "scripts/missing.py"
    assert _failures({}, {reader}) == {reader}


def test_a_missing_private_rule_target_is_not_an_optional_cache():
    reader = "scripts/check.py"
    target = "private/secret.json"
    blobs = {
        reader: (
            b"from pathlib import Path\nROOT = Path(__file__).parents[1]\n"
            b"def load():\n    try:\n        return (ROOT / 'private/secret.json').read_text()\n"
            b"    except FileNotFoundError:\n        return {}\n"
        )
    }
    manifest = {
        "public": [{"path": reader}],
        "rules": [
            {
                "kind": "PRIVATE",
                "patterns": ["private/**"],
            }
        ],
    }
    assert {ref.target for ref in violations(manifest, blobs)} == {target}


def test_a_document_command_glob_requires_all_its_matching_files():
    reader = "docs/guide.md"
    blobs = {
        reader: b"Run `pytest tests/kit/*.py`.\n",
        "tests/kit/public.py": b"",
        "tests/kit/private.py": b"",
    }
    public = {reader, "tests/kit/public.py"}
    assert _failures(blobs, public) == {"tests/kit/*.py"}
    assert _failures(blobs, public | {"tests/kit/private.py"}) == set()


def _generated_fixture(family):
    if family == "assets":
        producer = "scripts/build_local_web_ui.py"
        blobs = {
            producer: (
                b"from pathlib import Path\n"
                b"ROOT = Path(__file__).resolve().parents[1]\n"
                b'ASSETS = ROOT / "src/alphalattice/interface/local_application/assets"\n'
                b'MANIFEST = "workbench-manifest.json"\n'
                b"def build():\n"
                b"    outputs = {}\n"
                b'    outputs["workbench.html"] = b"page"\n'
                b'    outputs[MANIFEST] = b"{}"\n'
                b"    for name, data in outputs.items():\n"
                b"        (ASSETS / name).write_bytes(data)\n"
            )
        }
        root = "src/alphalattice/interface/local_application/assets/"
        targets = {root + "workbench.html", root + "workbench-manifest.json"}
        dependency = producer
    else:
        producer = "src/alphalattice/interface/local_application/native_setup.py"
        dependency = "src/alphalattice/interface/local_application/native_bridge.py"
        blobs = {
            producer: (
                b"from alphalattice.interface.local_application.native_bridge import BINDING_NAME\n"
                b"def bind_session(project):\n"
                b"    _create_or_match(BINDING_NAME, {}, project)\n"
                b"def _create_or_match(name, document, project):\n"
                b"    root = project\n"
                b'    path = root / ".codex" / name\n'
                b'    with path.open("xb") as stream:\n'
                b'        stream.write(b"{}")\n'
            ),
            dependency: b'BINDING_NAME = "native-research.local.json"\n',
        }
        targets = {".codex/native-research.local.json"}
    reader = "README.md"
    blobs[reader] = "".join(
        f"[generated output]({target})\n" for target in sorted(targets)
    ).encode()
    return blobs, set(blobs), producer, dependency, targets


@pytest.mark.parametrize("family", ["assets", "native"])
def test_a_generated_output_requires_its_public_declaration_and_actual_write(family):
    from release.public_outputs import generated_outputs

    blobs, public, producer, _, targets = _generated_fixture(family)
    evidence = generated_outputs(blobs, public)
    assert set(evidence) == targets
    assert all(producer + ":" in value for value in evidence.values())
    assert _failures(blobs, public) == set()
    undeclared = next(iter(sorted(targets))).rsplit("/", 1)[0] + "/undeclared.json"
    blobs["README.md"] += f"[not generated]({undeclared})\n".encode()
    assert _failures(blobs, public) == {undeclared}


@pytest.mark.parametrize("availability", ["absent", "private"])
@pytest.mark.parametrize(
    ("family", "part"),
    [("assets", "producer"), ("native", "producer"), ("native", "constant_dependency")],
)
def test_a_generated_output_loses_permission_when_a_required_source_is_unpublished(
    family, availability, part
):
    from release.public_outputs import generated_outputs

    blobs, public, producer, dependency, _ = _generated_fixture(family)
    source = producer if part == "producer" else dependency
    public.remove(source)
    if availability == "absent":
        del blobs[source]
    assert generated_outputs(blobs, public) == {}


@pytest.mark.parametrize(
    ("family", "before", "after", "target_name"),
    [
        ("assets", b'"workbench.html"', b'"other.html"', "workbench.html"),
        (
            "assets",
            b'"workbench-manifest.json"',
            b'"other-manifest.json"',
            "workbench-manifest.json",
        ),
        (
            "native",
            b'"native-research.local.json"',
            b'"other.local.json"',
            "native-research.local.json",
        ),
    ],
)
def test_an_output_declaration_change_does_not_authorize_the_previous_path(
    family, before, after, target_name
):
    from release.public_outputs import generated_outputs

    blobs, public, producer, dependency, targets = _generated_fixture(family)
    source = dependency if family == "native" else producer
    blobs[source] = blobs[source].replace(before, after)
    expected = {target for target in targets if target.endswith("/" + target_name)}
    assert expected.isdisjoint(generated_outputs(blobs, public))
    assert _failures(blobs, public) == expected


@pytest.mark.parametrize(
    ("family", "write"),
    [
        ("assets", b"        (ASSETS / name).write_bytes(data)\n"),
        ("native", b'        stream.write(b"{}")\n'),
    ],
)
def test_an_output_declaration_without_a_write_is_still_an_omitted_dependency(family, write):
    from release.public_outputs import generated_outputs

    blobs, public, producer, _, targets = _generated_fixture(family)
    blobs[producer] = blobs[producer].replace(write, b"        pass\n")
    assert generated_outputs(blobs, public) == {}
    assert _failures(blobs, public) == targets


@pytest.mark.parametrize("family", ["assets", "native"])
def test_a_producer_never_authorizes_an_existing_private_output_path(family):
    from release.public_outputs import generated_outputs

    blobs, public, _, _, targets = _generated_fixture(family)
    private = sorted(targets)[0]
    blobs[private] = b"kept outside the public snapshot\n"
    assert private not in generated_outputs(blobs, public)
    assert _failures(blobs, public) == {private}


def test_a_native_binding_output_requires_bind_to_pass_the_imported_constant():
    from release.public_outputs import generated_outputs

    blobs, public, producer, _, targets = _generated_fixture("native")
    blobs[producer] = blobs[producer].replace(
        b"    _create_or_match(BINDING_NAME, {}, project)\n", b"    pass\n"
    )
    assert generated_outputs(blobs, public) == {}
    assert _failures(blobs, public) == targets
