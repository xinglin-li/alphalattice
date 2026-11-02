import shutil
import subprocess
from pathlib import Path

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
SHOP = 'PROVIDES = {"local": lambda row, at: ({"status": "PASS", "exit": 0, "seconds": 0}, None)}'
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
@pytest.mark.parametrize("case", CASES)
def test_a_stage_runs_each_row_by_its_one_provider_into_a_receipt_that_verifies(tmp_path, case):
    host, bodies, expected = CASES[case]
    (tmp_path / "config/registries").mkdir(parents=True)
    for name in ("owners-map.json", "test-sizes.json"):
        shutil.copy(ROOT / "config/registries" / name, tmp_path / "config/registries" / name)
    (tmp_path / "config/pipeline.toml").write_text(DEFINITION, "utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    for words in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "c"]):
        subprocess.run([*git, *words], check=True)
    paths = [tmp_path / f"shop{i}.py" for i in range(len(bodies))]
    for path, body in zip(paths, bodies, strict=True):
        path.write_text(body, "utf-8")
    out, asked = (
        tmp_path / "out",
        dict(host=host, base="HEAD", workers=1, command=["t"], paths=paths),
    )
    with pytest.raises(DefinitionError) as refused:  # a run ends on its rerun into its --out
        summary = stage.run(tmp_path, "commit", out=out, **asked)
        assert (summary["result"], summary["red_nodes"]) == ("FAIL", [])
        lines = (out / "receipt.jsonl").read_text("utf-8")
        assert all(f'"{row}": "{provider}"' in lines for row, provider in expected.items())
        assert receipt.verify(out / "receipt.jsonl", host=host)["sha256"] == summary["sha256"]
        stage.run(tmp_path, "commit", out=out, **asked)
    code = expected if isinstance(expected, str) else "out_used"
    assert refused.value.code == "pipeline." + code
