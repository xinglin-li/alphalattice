# Contributing to AlphaLattice
Date: 2026-10-08

contributions open once the CLA bot is active

The maintainer must also complete the contribution infrastructure and repository
settings before announcing that contributions are open. See [governance](GOVERNANCE.md)
for final authority, contribution credit and maintainer permissions.

## Propose a change

Open an issue before starting nontrivial work so the maintainer can confirm its
scope. Describe the problem and proposed change with synthetic or redacted
examples. Report security vulnerabilities privately through [SECURITY.md](SECURITY.md).

On your first pull request, sign the [Contributor License Agreement](CLA.md)
through CLA Assistant. Link the agreed issue, describe the change and include
the check results. The maintainer reviews and decides whether to accept it.

## Verification

Install the locked environment and build Local Web before public verification:

```powershell
uv sync --locked --all-extras
uv run python scripts/build_local_web_ui.py --product
uv run python scripts/check_playpen.py --all-python --fast
uv run python scripts/build_local_web_ui.py --product --check
```

The last two commands are the public checks. The build's `--check` compares
the generated output with its sources and checks JavaScript syntax. Tests do
this at session start and build once when output is missing or stale, including
parallel runs; a build failure fails the run with its error. Node.js must be on
PATH for the syntax check. Test temporary directories must be outside the
checkout, for example:

```powershell
$env:ALPHALATTICE_NETWORK_DISABLED = '1'
uv run python -m pytest tests/portfolio_strategy_lab/test_local_web_product.py -n auto -m "not real_evidence" --basetemp "$env:TEMP/alphalattice-public-tests"
```

Full and real-evidence gates require the maintainer's private checkout and
refuse when its inputs are absent.

## How accepted changes ship

The maintainer imports accepted public pull requests into private development
while retaining the contributor's Git authorship. After the required verification,
the maintainer exports each shipped change to this repository and publishes
releases. The public pull request remains the record of the contribution.
