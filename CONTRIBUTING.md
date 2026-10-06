# Contributing to AlphaLattice
Date: 2026-10-01

The first release takes no outside contributions.

Others may join after the first release once the ownership of the existing
work, the rights to new contributions, credit and permissions are agreed.

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

Full and real-evidence gates require the maintainers' private checkout and
refuse when its inputs are absent.

Contact Xinglin Li through [LinkedIn](https://www.linkedin.com/in/xinglin-li-381571139/)
or [xinglin789@outlook.com](mailto:xinglin789@outlook.com) to discuss those terms.
