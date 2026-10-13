"""Read and bind one Workbench asset snapshot, validating source-checkout builds before serving."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, cast

from .web import LocalWebApplication

TYPES = {
    "workbench.html": "text/html; charset=utf-8",
    "workbench-prelude.js": "text/javascript; charset=utf-8",
    "workbench.css": "text/css; charset=utf-8",
    "workbench.js": "text/javascript; charset=utf-8",
    "workbench.zh.js": "text/javascript; charset=utf-8",
    "fonts/geist-latin.woff2": "font/woff2",
    "fonts/geist-mono-latin.woff2": "font/woff2",
}


def serve_assets(web: LocalWebApplication, root: Path) -> None:
    """Bind bytes once; source checkouts refuse stale outputs, wheels use their packaged build."""
    built = {
        name: (root / name).read_bytes()
        for name in (*TYPES, "workbench-manifest.json")
        if (root / name).is_file()
    }
    script = Path(__file__).resolve().parents[4] / "scripts/build_local_web_ui.py"
    if script.is_file():
        if not all(name in built for name in (*TYPES, "workbench-manifest.json")):
            raise ValueError("local_web.assets_stale")
        spec = importlib.util.spec_from_file_location("workbench_source_build", script)
        assert spec is not None and spec.loader is not None
        renderer = cast(Any, importlib.util.module_from_spec(spec))
        spec.loader.exec_module(renderer)
        renderer.ASSETS = root
        expected = renderer.render_outputs(
            fonts={name: built[name] for name in TYPES if name.startswith("fonts/")}
        )
        if any(built.get(name) != data for name, data in expected.items()):
            raise ValueError("local_web.assets_stale")
    try:
        manifest = json.loads(built.get("workbench-manifest.json", b"{}"))
    except ValueError:
        manifest = {}
    web.add_asset("/", built["workbench.html"], TYPES["workbench.html"])
    for name, content_type in TYPES.items():
        if name not in built:
            continue
        web.add_asset(f"/{name}", built[name], content_type)
        hashed = manifest.get(name) if isinstance(manifest, dict) else None
        if isinstance(hashed, str) and hashed != f"/{name}" and name != "workbench.html":
            web.add_asset(hashed, built[name], content_type, immutable=True)
    web.add_asset("/favicon.ico", b"", "image/x-icon")
