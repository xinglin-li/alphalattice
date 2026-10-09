# Change and extend the workspace
Date: 2026-10-08

Ask your agent to fix a runtime defect or propose a strategy, Alpha model or Feature. The editable checkout is your workspace to change. Explain the intended behavior and inspect the resulting contract, research evidence and eligibility before deciding on activation. The [research-agent guide](../../AGENTS.md#change-the-checkout) owns code recovery and the [person's decisions](../../AGENTS.md#what-only-a-person-decides).

## Source owners and expected changes

Extend the existing declaration and execution path. A new file alone does not register a capability.

- **Strategies:** [installed_strategies.py](../../src/alphalattice/investment/portfolio_strategy_lab/policies/installed_strategies.py) declares packages and score sources through `install_frozen_strategies`. [strategy_package.py](../../src/alphalattice/investment/portfolio_strategy_lab/application/strategy_package.py) owns the package, score-source and installed-binding contracts. Every declared capability needs its matching evidence identity. Installation produces a package; its reviewed whole-support book and forward positions follow the [strategy research path](research-flows.md#read-an-installed-strategys-book-and-positions).
- **Alpha models:** [model_scaffold.py](../../src/alphalattice/capabilities/alpha_modeling/model_scaffold.py) creates an adapter and `.model.yaml` under `capabilities/alpha_modeling/extensions/`, plus its contract test under `tests/alpha_research/`. The author implements `fit` and `predict`; the model's contract and sandbox determine its eligibility.
- **Features:** [factors/registry.py](../../src/alphalattice/foundation/feature_engine/producers/factors/registry.py) owns `RegisteredFeatureKernel`; [factors/catalog.py](../../src/alphalattice/foundation/feature_engine/producers/factors/catalog.py) registers kernels and recipes. [arithmetic_identity.py](../../src/alphalattice/foundation/feature_engine/producers/arithmetic_identity.py) declares method-family owners, and [factors/specifications.py](../../src/alphalattice/foundation/feature_engine/producers/factors/specifications.py) declares formula, boundaries, clock and admission. Registration adds no columns to a sealed Panel.

Your agent checks the changed owner and its consumers in the checkout's locked environment. Keep the original Goal, Task and error for recovery; a lawful refusal remains a limit to respect. A changed computation produces new method and Program identities and result evidence. Earlier sealed results retain their method, inputs and artifacts when their integrity bindings remain valid; current reuse or replay can be refused if the recorded method is no longer admitted. Equal meaning across an identity change requires an explicit successor record.

An identity or evidence-binding refusal is resolved at its source or input cause. Stored results, receipts, hashes and bindings are never edited to force a pass. A new computation supplies evidence for changed methods or inputs.

## Formula extension

[Formula factors](formula-factors.md) describes the bounded numerical language and preprocessing recipes. Ask for a plan on the chosen input and a trial against compatible Factor-handoff Alpha evidence, or a Portfolio built on it. Inspect the review's contract, coverage, comparisons and search counts. Completion does not activate the factor; `NOT_COMPARED` claims no metric change.

An eligible factor's activation on **Features** adds a daily catalog entry. The next data update rebuilds the Panel; older Panels retain their recorded catalog. Point-in-time formulas can be eligible with an admitted recipe. Sector-return formulas remain research-only until the daily build carries their Sector input.

## Alpha model extension

Create an editable model declaration with the scaffold command:

```powershell
alphalattice model scaffold --save-declaration "<out>/model.yaml"
```

Edit the declaration for your model, then create its adapter and contract test:

```powershell
alphalattice model scaffold --file "<out>/model.yaml"
```

Read the model's contract and sandbox through `model list`. `model check <model-id>` checks protocol, axes, routing and deterministic behavior, including a row predicted alone. A dependency outside the lock needs the project dependency decision described in the guide.

`model sandbox <model-id>` copies the workspace before running its trial, so the source workspace must be at rest with its Host stopped. It runs the Alpha study supplied with `--file` using the model and recipe, checks earlier stored verdicts and records the trial. Without a supplied study, it uses the latest completed Alpha development study that names a model. Add `--keep` to retain the copy for inspection.

Only a person may activate a model on **Models**. Activation requires a passed contract and sandbox for the same declaration and numerical binding. A changed model inherits no old pass. Deactivation removes catalog admission; saved studies keep their exact readback. Model activation and family qualification are distinct: neither a check, sandbox nor favorable research result establishes future performance.

An activated agent model can be qualified with its question's family. Its current refit seals the state projected by its adapter, bound to that adapter, its numerical binding and projection; it does not reuse an installed family's state. For kinds other than `LINEAR` and `TREE`, stability uses training error and the current scores' mean, spread and coverage against development folds. An agent-projected tree is refused as `ALPHA_CURRENT_REFIT_AGENT_TREE_UNSUPPORTED` because the tree diagnostics read the installed family.
