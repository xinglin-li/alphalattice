"""Codex's trust of a project, read as Codex documents it, names the person's one step."""

import json
import os

import pytest

from alphalattice.interface.local_application import native_bridge

TEST_SOURCE = "src/alphalattice/interface/local_application/native_bridge.py"


@pytest.mark.proves("local_application.codex_first_use_names_project_trust")
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(None, id="no-entry"),
        pytest.param("untrusted", id="marked-untrusted"),
        pytest.param("trusted", id="trusted"),
        pytest.param("trusted-folded", id="trusted-other-case"),
        pytest.param("trusted-subfolder", id="a-trusted-subfolder-is-not-the-root"),
        pytest.param("unreadable", id="an-unreadable-config-is-carried-as-unknown-by-its-path"),
        pytest.param("system-untrusted", id="a-system-entry-names-the-system-layer"),
    ],
)
def test_codex_trust_is_read_at_the_project_root_with_the_persons_step(
    tmp_path, monkeypatch, entry
):
    """Only the root Codex finds by `.git` counts; untrusted names the step or the setting and
    the layer that holds it; an unreadable configuration is unknown, named, and never a gate."""

    project, home = tmp_path / "project", tmp_path / "codex-home"
    system = tmp_path / "system" / "OpenAI" / "Codex" / "config.toml"
    (project / ".git").mkdir(parents=True)
    (project / "research").mkdir()
    home.mkdir()
    system.parent.mkdir(parents=True)
    monkeypatch.setattr(native_bridge, "codex_system_config", lambda _env: system, raising=False)
    root = str(project.resolve())
    keyed = {
        "trusted-folded": root.swapcase(),
        "trusted-subfolder": str((project / "research").resolve()),
    }.get(entry or "", root)
    layer = system if entry == "system-untrusted" else home / "config.toml"
    if entry:
        level = "untrusted" if entry in {"untrusted", "system-untrusted"} else "trusted"
        text = f'[projects.{json.dumps(keyed)}]\ntrust_level = "{level}"\n'
        layer.write_text("[[not toml" if entry == "unreadable" else text, encoding="utf-8")
    environ = {"CODEX_HOME": str(home), "ProgramData": str(tmp_path / "system")}

    trust = native_bridge.codex_project_trust(project / "research", environ)

    folds = os.name == "nt"  # Windows paths are case-insensitive; Codex folds them there only
    if entry == "unreadable":
        named = trust["config"] in str(trust.get("words"))
        assert (trust["present"], trust["config"], named) == (None, str(layer), True)
        return
    if entry == "trusted" or (entry == "trusted-folded" and folds):
        assert trust == {"present": True, "project_root": root}
        return
    assert (trust["present"], trust["who_decides"], trust["project_root"]) == (
        False,
        "PERSON",
        root,
    )
    if entry in {"untrusted", "system-untrusted"}:
        assert (trust["commands"], trust["config"]) == ([], str(layer))
        assert trust["config"] in trust["words"]
    else:
        (step,) = trust["commands"]
        assert "codex exec --trust" in step and root in step and step in trust["words"]
