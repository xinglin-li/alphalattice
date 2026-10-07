# Change and extend the workspace
Date: 2026-10-07

Extensions begin as declared research proposals. Inspect each plan, contract, run and review before requesting the person's activation. Activation and qualification are different decisions. [Formula factors](formula-factors.md) owns the complete language, parameters, supported recipes and trial evidence; this page summarizes the extension paths.

## Change the code

The checkout is the person's workspace to change. Your agent is encouraged to
fix runtime bugs and add strategies, models and features in source. Use the
editable checkout and locked environment from [setup](../../.agents/skills/alphalattice-research/references/operating.md#setup-and-launch).
Before changing code, follow the [failure and recovery procedure](../../.agents/skills/alphalattice-research/references/operating.md).
Keep the original Goal, Task and error; a lawful refusal is not a defect to remove.
The person chooses the lead agent's model; retain the inexpensive specialist
defaults in the shipped cards.

Extend the existing declarations and their execution path:

- **Strategies:** [installed_strategies.py](../../src/alphalattice/investment/portfolio_strategy_lab/policies/installed_strategies.py)
  declares installed packages and their score sources through `install_frozen_strategies`.
  [strategy_package.py](../../src/alphalattice/investment/portfolio_strategy_lab/application/strategy_package.py)
  owns `FrozenStrategyPackage`, `StrategyScoreSource` and `InstalledPackageBinding`.
  A package must install every declared capability with its matching evidence
  identity; adding a file alone does not register a strategy.
- **Alpha models:** [`model_scaffold.py`](../../src/alphalattice/capabilities/alpha_modeling/model_scaffold.py)
  writes an adapter and its `.model.yaml` under `capabilities/alpha_modeling/extensions/`,
  plus a contract test in the extension directory it creates under
  `tests/alpha_research/`. Implement its `fit` and `predict`, then follow the
  check and sandbox procedure below.
- **Features:** [factors/registry.py](../../src/alphalattice/foundation/feature_engine/producers/factors/registry.py)
  owns `RegisteredFeatureKernel`; [factors/catalog.py](../../src/alphalattice/foundation/feature_engine/producers/factors/catalog.py)
  explicitly registers kernels and recipes. Declare the method-family owners in
  [arithmetic_identity.py](../../src/alphalattice/foundation/feature_engine/producers/arithmetic_identity.py)
  and the method's formula, boundaries, clock and admission in
  [factors/specifications.py](../../src/alphalattice/foundation/feature_engine/producers/factors/specifications.py).
  A new registration does not add columns to an already sealed Panel.

Run the tests that answer for the owner and its consumers, using the checkout's
locked environment. For example, choose the affected nodes or files under
`tests/portfolio_strategy_lab/test_public_portfolio_finalization.py`,
`tests/alpha_research/test_model_contract.py`, your scaffolded model test,
`tests/feature_engine/test_factor_arithmetic_identity.py` or
`tests/researcher_methodology_surface/test_factor_formula_specification.py`:

```powershell
$env:ALPHALATTICE_NETWORK_DISABLED = '1'
.venv/Scripts/python.exe -m pytest tests/feature_engine/test_factor_arithmetic_identity.py
```

In the editable install, source edits are live; restart only an idle Host you own
in the same workspace, open its new launch link and recheck the session binding.
Read the original Task before taking its current owner-offered resume or replan.
Keep the earlier sealed result to compare
with the new run. Computation identities bind the method's numerical source,
declaration and parameters; execution evidence records the Program, method and
input bindings and artifact references. Changing the computation produces a new
method/Program identity and new result evidence. Earlier results keep their
original method, inputs and artifacts and can be read as historical records
when the required artifacts and integrity bindings remain valid. Current reuse
or replay may be refused when the recorded method is no longer admitted.
When an identity changes without changing meaning, compatibility requires an
explicitly recorded successor; it is never inferred from a successful command.

Fix an integrity refusal, including an identity or evidence-binding mismatch,
at its source or input cause. Never bypass the verifier, weaken a binding check,
or edit stored results, receipts, hashes or bindings to make it pass. Run a new
computation when its method or inputs have changed. Source changes grant no
activation or external-publication authority; the person's decisions below
still apply.

## Formula extension

The bounded expression language does not run Python imports or arbitrary callables. Each time-series window counts a listing's own rows and cannot read a later row. The author selects preprocessing; cross-sectional rank, scaling and Sector adjustment belong there, not in the formula.

A Feature plan canonicalizes the declaration and states the trial's baseline requirements. Build on that input, choose an exact compatible completed Alpha study handed off from Factor evidence (or a Portfolio built on it) for the trial, then inspect the review packet's standing, contract, coverage, trials and search counts. The trial's `COMPLETED` state does not activate the factor; `NOT_COMPARED` claims no metric change. A person alone activates an eligible definition through Local Web. The next data update binds it in the daily catalog; older Panels retain their recorded catalog. Point-in-time formulas can be admitted with an eligible recipe; Sector-return formulas remain research-only until the daily build carries their Sector input.

## Alpha model extension

Use the checkout's model scaffold to create the declared adapter, model declaration and contract test. For this model path, implement the adapter's `fit` and `predict`; `model check` checks its protocol, axes, routing and deterministic behavior, including a row predicted alone. The locked environment is required. A new dependency needs its own project dependency decision.

```powershell
alphalattice model scaffold --save-declaration "<out>/model.yaml"
```

Edit that declaration for your model, then scaffold its source and contract test:

```powershell
alphalattice model scaffold --file "<out>/model.yaml"
alphalattice model check <declared-model-id>
```

Stop the workspace Host before `model sandbox`: its workspace must be at rest. The sandbox uses a copy, runs the supplied Alpha study with the model and recipe, checks that earlier stored results retain their verdicts and records the trial. Without a supplied study file it uses the latest published Alpha development study that declares a model, substituting the model and recipe being checked. `--keep` retains the copy for inspection. A sandbox is not access to a protected qualification population.

Read the contract and sandbox through `model list`. A person can activate only after a passed contract and sandbox for the same declaration and numerical binding. Changing the model does not transfer an old pass to its new identity. Deactivation removes catalog admission; saved studies keep their exact readback.

An activated agent model is qualified with its question's family. Its current refit seals the state projected by its adapter, bound to that adapter, its numerical binding and projection; it does not reuse an installed family's state. For kinds other than `LINEAR` or `TREE`, current-stability uses model-agnostic training error and current-score mean, spread and coverage against development folds. An agent-projected tree is refused as `ALPHA_CURRENT_REFIT_AGENT_TREE_UNSUPPORTED` because tree diagnostics read the installed family.

Activation authority belongs to the person. Neither a model check, sandbox, trial, review packet nor successful research command performs it.
