# AlphaLattice first release
Date: 2026-10-03

The `0.1.0` first release is a local equity research workstation for a person and a Codex or Claude Code agent. Factor evidence and curation lead to Alpha; Risk runs on the same input. Portfolio consumes Alpha and only uses Risk when its allocation reads it. Evidence and the CRO review the exact book and retain coverage gaps. A report-only Risk association does not mean weights used a Risk model.

## Included capabilities

The maintained CLI and Local Web reach the same calculation, permission, Task and publication owners. Local Web includes Chinese; `--lang zh` selects available Chinese CLI detail with English fallback. Editable declarations go through installed schemas. Answers record outcomes, refusals and continuations; saved reads retain their full selection, `--wait` follows named Task state, and Portfolio position metrics may include units. See [CLI](cli.md).

Native role cards and the research Skill let external agents interpret product evidence. Goals bind work to agent sessions; the product checks submissions against recorded Tasks and references but does not certify an agent's interpretation. Default research and Team need no product hooks. Team separates public messages, product receipts, sealed outputs, usage and optional native observations. See [Agents](agents.md).

Formula features have a bounded language, explicit preprocessing and separate plan, build, trial and review. A person can activate only an eligible daily definition. Sector-return leaves remain research-only until daily inputs are admitted. Alpha models have a contract check and copy-based sandbox before person-only activation; a passed sandbox is not qualification. See [Formula factors](formula-factors.md) and [Extending](extending.md).

Data updates preserve earlier inputs; issues expose evidence and permitted choices. Backup stores held state; restore targets a new directory and identifies values, inputs or packages that need rebuilding, recapture or installation. Native usage reading keeps whitelisted facts and drops conversation content; `--usage off` controls that reader, while the CLI launch counter remains separate. See [Data and care](data-and-care.md) and [Privacy](privacy.md).

Readbacks carry temporal statements for evaluated window, earlier input history, T0 membership backfill, survivorship, Sector treatment and price basis. A later observation does not rewrite a published session. Research publication does not establish future returns or an operating trading service. See [Temporal statements](temporal-statements.md).

## Platform and terms

Windows is verified for this source-checkout release; macOS and Linux are unverified. On 2026-09-30 at checkout `c109a7bc`, an empty-workspace preparation through its data decision took 407.1 seconds and 3.99 GB peak memory, using eight logical processors on four P-cores. The machine was Windows 11 Pro, Intel Core i9-13900K (24 cores, 32 logical processors) and 128 GB memory; a comparison ran concurrently on disjoint cores. This observation is neither a hardware minimum nor a runtime guarantee. [Getting started](getting-started.md) has the measurement context.

Xinglin Li created AlphaLattice alone and made this release. It is Apache-2.0; [NOTICE](../../NOTICE) lists included third-party material and separately installed software/model licenses. The release takes no outside contributions; see the [contributing note](../../CONTRIBUTING.md).

## 0.1.1

The seven shipped Codex specialist role cards run at `high` reasoning effort instead of `xhigh`:
in the agent evaluation, `high` completed the same tasks faster. A project configured from 0.1.0
keeps its copied cards; set `model_reasoning_effort` in its `.codex/agents/` files to choose.
