"""Build the Local Web workbench assets from workbench-source, served under the
Host's strict CSP. `--product` is accepted for the callers that pass it; it is
the only build since the prototype entry (the synthetic reference with
fixtures) retired.

The build writes, beside the sources (round 96):

- `workbench.html`      the host page: the prelude, the font preloads, the stylesheet
                        and the deferred app, each by its content-hashed path
- `workbench-prelude.js` the few lines that run before the first paint (the
                        appearance, the language, the dock, the text size)
- `workbench.css`       the cascade, its font URLs content-hashed; its first part is
                        `design/parameters.json` written as custom properties (law 105:
                        every design value has that one owner)
- `workbench.js`        the engines and the app -- without the dictionary
- `workbench.zh.js`     the Chinese dictionary, loaded only by a zh reader
- `workbench-manifest.json`  logical name -> hashed path, read by the host so each
                        asset is served immutably under its hashed path and
                        revalidated under its plain one
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from alphalattice.interface.local_application.failure_codes import setup_failure  # noqa: E402
from alphalattice.kernel.shared_kernel.persistence import replace_with_retry  # noqa: E402

ASSETS = ROOT / "src/alphalattice/interface/local_application/assets"
REPLACE_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
"""A built asset replaces the served one, retried while a request reads it (RH, V170)."""
# The workbench sources and their built files are siblings under the
# product's own asset directory; nothing here reaches above the checkout.
SRC = ASSETS / "workbench-source"

PARAMETERS = SRC / "design/parameters.json"
# the product's names for what it installs (V451, U72): one table the CLI, the docs and the
# Workbench read
LABELS = ROOT / "src/alphalattice/interface/local_application/labels.json"
CSS_FILES = sorted(
    p.relative_to(SRC).as_posix() for p in (SRC / "css").glob("*.css")
)  # name order = cascade order, after the parameters
ENGINE_FILES = [
    "js/engines/static-mark.js",
]
APP_FILES = [
    "js/app/html.js",
    "js/app/i18n.js",
    "js/app/icons.js",
    "js/app/data.js",
    "js/app/status.js",
    "js/app/state.js",
    "js/app/components.js",
    "js/app/dialogs.js",
    "js/app/router.js",
    "js/app/window.js",
    "js/app/inspect.js",
    "js/app/controls.js",
    "js/app/lab-editor.js",
    "js/app/pages-portfolio.js",
    "js/app/pages-history.js",
    "js/app/pages-settings.js",
    "js/app/pages-kit.js",
    "js/app/live-views.js",
    "js/app/live-workarea.js",
    "js/app/live-workspace.js",
    "js/app/live-research.js",
    "js/app/live-features.js",
    "js/app/live-goals.js",
    "js/app/live-models.js",
    "js/app/live-feature-research.js",
    "js/app/live-activation.js",
    "js/app/live-activity.js",
    "js/app/live-team.js",
    "js/app/live-tasks.js",
    "js/app/live-review.js",
    "js/app/live-study.js",
    "js/app/actions.js",
    "js/app/boot.js",
]
FONT_FILES = ["fonts/geist-latin.woff2", "fonts/geist-mono-latin.woff2"]
MANIFEST = "workbench-manifest.json"


class AssetsUnwritable(OSError):
    """The built assets could not be written beside their sources; its message is the code
    (`local_web.assets_unwritable:<error>`), and the served assets stay as they were (V539)."""


def read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def concat(files, banner):
    parts = []
    for rel in files:
        parts.append(f"/* ---- {banner}: {rel} ---- */\n" + read(rel).rstrip() + "\n")
    return "\n".join(parts)


def parameters() -> dict:
    return json.loads(PARAMETERS.read_text(encoding="utf-8"))


def parameters_css(source: dict | None = None) -> str:
    """The parameter source as the first stylesheet: one block per mode, in the modes' order.
    That order is part of the cascade -- where two modes' selectors tie, the later one wins -- so
    it is the source's, never re-sorted here."""
    p = parameters() if source is None else source
    by_mode: dict[str, list[str]] = {m["name"]: [] for m in p["modes"]}
    for group in p["groups"].values():
        for name, values in group["tokens"].items():
            for mode, value in values.items():
                if mode == "about":
                    continue
                if mode not in by_mode:
                    raise SystemExit(f"parameters.json: --{name} names an unknown mode {mode!r}")
                by_mode[mode].append(f"  --{name}: {value};")
    parts = ["/* ---- css: design/parameters.json (written by the build; edit the JSON) ---- */"]
    for mode in p["modes"]:
        lines = by_mode[mode["name"]]
        if not lines:
            continue
        block = mode["selector"] + " {\n" + "\n".join(lines) + "\n}"
        if mode.get("media"):
            block = "@media " + mode["media"] + " {\n" + block + "\n}"
        parts.append(block)
    return "\n".join(parts) + "\n"


# the tones a meaning may take (role.tone's marks; `stopped` is the stop's grey dot)
TONE_NAMES = frozenset(
    {"good", "warning", "danger", "review", "accent", "neutral", "stopped", "cyan"}
)


def parameters_js(source: dict | None = None) -> str:
    """What the scripts read of the parameter source (Q2): the names by group (the workshop page,
    #page=kit, shows each one's value), the breakpoints by kind, and the values the source lists
    under `scripts` -- a length in px, as a number (a picker's edge, a chart's pads, the side's
    widths) -- under `counts`, a whole number (a lobby group's rows, a title's characters) -- and
    under `tones`, a tone's name (a meaning's colour: `danger`, `warning` ...). A listed value of
    the wrong kind is refused here."""
    p = parameters() if source is None else source
    groups = {group: list(body["tokens"]) for group, body in p["groups"].items()}
    steps = {
        kind: {
            name: v.get("width", v.get("height")) for name, v in body.items() if isinstance(v, dict)
        }
        for kind, body in p.get("breakpoints", {}).items()
        if kind != "about"
    }
    tokens = {name: v for body in p["groups"].values() for name, v in body["tokens"].items()}
    values = {}
    for name in p.get("scripts", []):
        base = tokens.get(name, {}).get("base", "")
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)px", base)
        if not m:
            raise SystemExit(
                f"parameters.json: `scripts` lists --{name}, "
                f"whose base value {base!r} is not a px length"
            )
        values[name] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    for name in p.get("counts", []):
        base = tokens.get(name, {}).get("base", "")
        if not re.fullmatch(r"\d+", base):
            raise SystemExit(
                f"parameters.json: `counts` lists --{name}, "
                f"whose base value {base!r} is not a whole number"
            )
        values[name] = int(base)
    for name in p.get("tones", []):
        base = tokens.get(name, {}).get("base", "")
        if base not in TONE_NAMES:
            raise SystemExit(
                f"parameters.json: `tones` lists --{name}, "
                f"whose base value {base!r} is not a tone ({', '.join(sorted(TONE_NAMES))})"
            )
        values[name] = base

    def dump(v: object) -> str:
        return json.dumps(v, separators=(",", ":"))

    return (
        f"const PARAMETER_GROUPS = {dump(groups)};\n"
        f"const BREAKPOINTS = {dump(steps)};\n"
        f"const PARAMETER_VALUES = {dump(values)};\n"
    )


def labels_js() -> str:
    """The label table as the scripts read it (U72): by the id the code keeps, its kind, its
    titles and its summaries, never a copy kept by hand."""
    table = json.loads(LABELS.read_text(encoding="utf-8"))
    keep = ("kind", "title", "title_zh", "summary", "summary_zh")
    labels = {label["id"]: {k: label[k] for k in keep} for label in table["labels"]}
    return f"const LABELS = {json.dumps(labels, ensure_ascii=False, separators=(',', ':'))};\n"


def hashed(name: str, data: bytes) -> str:
    """`workbench.css` -> `/workbench.<10 hex of sha256>.css`: a path a browser may keep."""
    stem, _dot, ext = name.rpartition(".")
    return f"/{stem}.{hashlib.sha256(data).hexdigest()[:10]}.{ext}"


def dictionary(source: str) -> str:
    """The dictionary as a file that may load before or after the app: it fills the one
    catalog object `i18n.js` reads (`window.ALPHA_ZH`), never replaces it, and says when it
    is complete."""
    head, tail = "window.ALPHA_ZH = {", "};"
    body = source.rstrip()
    if not (body.startswith(head) or ("\n" + head) in body) or not body.endswith(tail):
        raise SystemExit("zh.js: the dictionary is not `window.ALPHA_ZH = {...};`")
    body = body.replace(head, "Object.assign(window.ALPHA_ZH || (window.ALPHA_ZH = {}), {", 1)
    body = body[: -len(tail)] + "});"
    return body + "\nwindow.ALPHA_ZH_READY = true;\n"


def build(*, product: bool = True):
    """Assemble the product's workbench assets from the named sources.

    Source CSS order and JavaScript semantics are preserved; trusted code is
    served as local assets under the Host's strict CSP. `product` is kept for
    the callers that pass it; there is no other build since the prototype
    entry retired.
    """
    del product
    manifest: dict[str, str] = {}
    outputs: dict[str, bytes] = {}

    # the fonts: their bytes are the product's, so their paths carry their hash
    for rel in FONT_FILES:
        data = (ASSETS / rel).read_bytes()
        manifest[rel] = hashed(rel, data)

    css = parameters_css() + "\n" + concat(CSS_FILES, "css")
    for rel in FONT_FILES:
        css = css.replace(f"url(/{rel})", f"url({manifest[rel]})")
    outputs["workbench.css"] = css.encode("utf-8")

    zh = dictionary(read("js/data/zh.js"))
    outputs["workbench.zh.js"] = zh.encode("utf-8")

    engines = concat(ENGINE_FILES, "engine")
    app = parameters_js() + labels_js() + concat(APP_FILES, "app")
    scripts = (
        'window.ALPHA_PRODUCT=true; window.ALPHA_ENTRY={page:"history"};',
        engines,
        '(()=>{"use strict";\n' + app + "\n})();",
    )
    outputs["workbench.js"] = "\n".join(scripts).encode("utf-8")

    for name, data in list(outputs.items()):
        manifest[name] = hashed(name, data)
    prelude = read("js/app/prelude.js").replace("{{zh_src}}", manifest["workbench.zh.js"])
    outputs["workbench-prelude.js"] = prelude.encode("utf-8")
    manifest["workbench-prelude.js"] = hashed(
        "workbench-prelude.js", outputs["workbench-prelude.js"]
    )

    head = "\n".join(
        [f'<script src="{manifest["workbench-prelude.js"]}"></script>']
        + [
            f'<link rel="preload" href="{manifest[rel]}" as="font" type="font/woff2" crossorigin>'
            for rel in FONT_FILES
        ]
        + [f'<link rel="stylesheet" href="{manifest["workbench.css"]}">']
    )
    html = (
        read("shell.html")
        .replace("{{head}}", head)
        .replace("{{scripts}}", f'<script src="{manifest["workbench.js"]}" defer></script>')
    )
    outputs["workbench.html"] = html.encode("utf-8")

    manifest_text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    outputs[MANIFEST] = manifest_text.encode("utf-8")
    try:
        for name, data in outputs.items():
            # each file whole or not at all: two test workers may build at once, and a reader
            # (the served page, a test reading the bundle) must never see half a file
            partial = ASSETS / f"{name}.{os.getpid()}.partial"
            partial.write_bytes(data.replace(b"\r\n", b"\n"))
            # Windows refuses to replace a file a request holds open; the Host reads it briefly.
            replace_with_retry(partial, ASSETS / name, delays=REPLACE_DELAYS)
    except OSError as error:
        raise AssetsUnwritable(f"local_web.assets_unwritable:{type(error).__name__}") from error
    return {"workbench.html": html}


def check():
    """Syntax-check the assembled scripts with node, then refuse a design value written outside
    the parameter source (scripts/check_ui_parameters.py)."""
    import sys

    subprocess.run([sys.executable, str(ROOT / "scripts/check_ui_parameters.py")], check=True)
    engines = concat(ENGINE_FILES, "engine")
    app = '(()=>{"use strict";\n' + parameters_js() + concat(APP_FILES, "app") + "\n})();"
    prelude = read("js/app/prelude.js").replace("{{zh_src}}", "/workbench.zh.js")
    zh = dictionary(read("js/data/zh.js"))
    with TemporaryDirectory(prefix="ui-syntax-") as directory:
        for label, code in [("engines", engines), ("app", app), ("prelude", prelude), ("zh", zh)]:
            probe = Path(directory) / f"{label}.js"
            probe.write_text(code, encoding="utf-8")
            subprocess.run(["node", "--check", str(probe)], check=True)


def main(argv: list[str] | None = None) -> int:
    """Build the assets, printing each page's size, or the refusal in words (V539).

    Args:
        argv: The command line, else the process's.

    Returns:
        0 once built; 2 when the assets could not be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Also check JavaScript syntax with node."
    )
    parser.add_argument("--product", action="store_true", help="Accepted; the only build.")
    arguments = parser.parse_args(argv)
    try:
        out = build(product=arguments.product)
        if arguments.check:
            check()
    except AssetsUnwritable as error:
        refusal = {
            **setup_failure(error),
            "status": "REFUSED",
            "failure_code": str(error),
            "detail": (
                "The build writes the Local Web assets beside their sources in the checkout, and "
                "this process could not write there, so the served assets are as they were. Build "
                "where the checkout can be written: serving reads the built assets."
            ),
            "next_action": "BUILD_WHERE_THE_CHECKOUT_CAN_BE_WRITTEN",
            "assets": str(ASSETS),
        }
        print(json.dumps(refusal, indent=1))
        return 2
    except Exception as error:
        # Its way on is its help: the build itself is the check, and it writes (V590).
        help_command = [sys.executable, "scripts/build_local_web_ui.py", "--help"]
        print(
            json.dumps(
                {
                    **setup_failure(error),
                    "assets": str(ASSETS),
                    "next_commands": {"help": help_command},
                }
            )
        )
        return 2
    for name, html in out.items():
        print(f"{name}: {len(html.encode('utf-8')):,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
