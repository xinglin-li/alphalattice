import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools.pipeline import receipt, stage
from devtools.pipeline.definition import DefinitionError

ROOT = Path(__file__).resolve().parents[2]
ROW = 'owner = "devtools.pipeline"\nreason = "r"\nhosts = '
BOTH = ROW + '["workshop", "public"]\nkind = "command"'
DEFINITION = f"""schema = "alphalattice.pipeline.v1"\nowner = "devtools.pipeline"
[stages.commit]\norder = 1\nreason = "r"\ntrigger = "line"\nrange = "card"\nbound_minutes = 1
rows = ["ok", "red", "local", "job"]
[rows.ok]\n{BOTH}\nrun = ["{{python}}", "-c", "pass"]
[rows.red]\n{BOTH}\nrun = ["{{python}}", "-c", "raise SystemExit(3)"]
[rows.local]\n{ROW}["workshop"]\nkind = "workshop"
[rows.job]\n{ROW}["public"]\nkind = "ci-job"\njob = "w"
"""
SHOP = (  # the adapter records the merge base its context carries
    'PROVIDES = {"local": lambda row, at: (at.out / "base").write_text(at.base)'
    ' and ({"status": "PASS", "exit": 0, "seconds": 0}, None)}'
)
CASES = {  # host, plugin bodies, the providers or the refusal's code
    "public": ("public", [], {"ok": "product", "local": "skipped", "job": "ci.yml:w"}),
    "a workshop row with no plugin": ("workshop", [], "row_unprovided"),
    "a plugin's row": ("workshop", [SHOP], {"local": "shop0", "job": "skipped"}),
    "two plugins for a row": ("workshop", [SHOP, SHOP], "row_provided_twice"),
    "a plugin for no row": ("workshop", [SHOP.replace('"local"', '"spare"')], "row_unregistered"),
    "a plugin with no PROVIDES": ("workshop", ["X = 1"], "plugin_unloadable"),
    "two plugins of one name": ("workshop", [SHOP + "\nNAME = 'x'"] * 2, "plugin_unloadable"),
}


@pytest.mark.proves("pipeline.stage_runs_each_row_by_one_provider")
@pytest.mark.proves("pipeline.stage_target_is_separate_from_stop")
@pytest.mark.parametrize("case", CASES)
def test_a_stage_runs_each_row_by_its_one_provider_into_a_receipt_that_verifies(
    tmp_path, case, monkeypatch
):
    host, bodies, wanted = CASES[case]
    (tmp_path / "config/registries").mkdir(parents=True)
    for name in ("owners-map.json", "test-sizes.json"):
        shutil.copy(ROOT / "config/registries" / name, tmp_path / "config/registries" / name)
    config = tmp_path / "config/pipeline.toml"
    config.write_text(DEFINITION, "utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for words in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "c"]):
        subprocess.run([*git, *words], check=True)
    paths = [tmp_path / f"shop{i}.py" for i in range(len(bodies))]
    for path, body in zip(paths, bodies, strict=True):
        path.write_text(body, "utf-8")
    out = tmp_path / "out"
    asked = dict(host=host, base="HEAD", workers=1, command=["t"], paths=paths)
    with pytest.raises(DefinitionError) as refused:  # a run ends on its rerun into its --out
        summary = stage.run(tmp_path, "commit", out=out, **asked)
        assert (summary["result"], summary["red_nodes"]) == ("FAIL", [])
        lines = (out / "receipt.jsonl").read_text("utf-8")
        assert all(f'"{row}": "{provider}"' in lines for row, provider in wanted.items())
        assert receipt.verify(out / "receipt.jsonl", host=host)["sha256"] == summary["sha256"]
        assert not bodies or (out / "base").read_text("utf-8") == summary["base"]  # the merge base
        stage.run(tmp_path, "commit", out=out, **asked)
    assert refused.value.code == "pipeline." + (wanted if isinstance(wanted, str) else "out_used")
    if case == "public":
        paths.append(tmp_path / "budget.py")
        paths[0].write_text(SHOP.replace('"local"', '"red"'), "utf-8")
        for stop, wall in ((2, 59), (2, 61), (2, 121), (3, 121)):
            result = "PASS" if wall <= stop * 60 else "ERROR"
            budget = f"target_minutes = 1\nstop_minutes = {stop}"
            config.write_text(DEFINITION.replace("bound_minutes = 1", budget), "utf-8")
            ticks = iter([0, 0, 0, wall])
            monkeypatch.setattr(stage, "time", SimpleNamespace(monotonic=ticks.__next__))
            target = tmp_path / f"budget-{stop}-{wall}"
            assert stage.run(tmp_path, "commit", out=target, **asked)["result"] == result
            records = map(json.loads, (target / "receipt.jsonl").read_text("utf-8").splitlines())
            seen = next(row["configuration"] for row in records if "configuration" in row["id"])
            metrics = next(row["metrics"] for row in records if row["id"] == {"metrics": {}})
            assert (seen["bound_seconds"], metrics["wall_seconds"] > 60) == (stop * 60, wall > 60)
