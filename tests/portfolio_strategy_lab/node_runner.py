"""Node invocations of the test suites: one runner and the Workbench catalog reader.

Kept apart from local_web_support's Portfolio fixtures, so a change to those fixtures does not
select every test that only runs Node.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from alphalattice.kernel.shared_kernel import spawn


def run_node(
    arguments: list[str] | None,
    *,
    missing: str = "Node.js development runtime required",
    required: bool = False,
    pinned_browser: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any] | None:
    """Run one Node invocation with its caller's availability policy and process options."""
    node = shutil.which("node")
    if node is None:
        if required:
            raise AssertionError(missing)
        pytest.skip(missing)
    if arguments is None:
        return None
    supplied = kwargs.get("env")
    environment = dict(os.environ) if pinned_browser or supplied is None else {}
    if supplied is not None:
        environment.update(supplied)
    environment["TZ"] = "UTC"
    if pinned_browser:
        root = Path(kwargs.get("cwd") or Path(__file__).resolve().parents[2])
        environment.update(
            NODE_PATH=str(root / "third_party/playwright/node_modules"),
            PLAYWRIGHT_BROWSERS_PATH=str(root / "third_party/playwright/.browsers"),
        )
    browser = environment.get("PLAYWRIGHT_BROWSERS_PATH")
    if isinstance(browser, str):
        directory = Path(browser)
        # Leave relative paths and default modes for Playwright to interpret.
        if directory.is_absolute():
            environment["PLAYWRIGHT_BROWSERS_PATH"] = str(directory.resolve())
    kwargs["env"] = environment
    return spawn.run([node, *arguments], **kwargs)


_CATALOG_READER = r"""
const fs=require('fs'),vm=require('vm');
const library = require(process.argv[2]);
const root=process.argv[1], {said,messages}=JSON.parse(fs.readFileSync(0,'utf8'));
const c={window:{},document:{documentElement:{}},console};library.context(c);
vm.runInContext(messages,c);
vm.runInContext(fs.readFileSync(root+'/data/zh.js','utf8'),c);
vm.runInContext(fs.readFileSync(root+'/app/i18n.js','utf8')+';globalThis.I18N=I18N;',c);
c.I18N.set('zh');
console.log(JSON.stringify({read:said.map((s)=>c.I18N.t(s)),missing:c.I18N.untranslated()}));
"""


def read_in_chinese(said: list[str], *, missing: str, required: bool) -> dict[str, list[str]]:
    """The sentences as the Workbench's catalog reads them in Chinese, and any left English.

    The sentences go on stdin: the door's table outgrew a Windows command line (WinError 206).
    """
    root = Path(__file__).resolve().parents[2] / (
        "src/alphalattice/interface/local_application/assets/workbench-source/js"
    )
    from scripts import build_local_web_ui

    answer = run_node(
        [
            "-e",
            _CATALOG_READER,
            str(root),
            str(Path(__file__).with_name("workbench_library.cjs")),
        ],
        missing=missing,
        required=required,
        input=json.dumps(
            {"said": said, "messages": getattr(build_local_web_ui, "messages_js", lambda: "")()}
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(answer.stdout)
